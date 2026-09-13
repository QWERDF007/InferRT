/**
 * @file DinoSimilarityCuda.cu
 * @brief 相似度归约的设备端实现。
 *
 * 输入是索引里的紧凑描述块（INT8 码 + 每条描述子的还原因子），解码在 kernel 内完成。
 * 每次调用只上传有界描述块和查询矩阵；结果在块级返回，宿主端只维护 Top-K 所需状态。
 */

#include "DinoSimilarity.hpp"

#include <inferrt/core/Exception.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace irt::features::priv {

namespace {

constexpr int kThreadsPerBlock = 256;

/** @brief 设备探测结果只算一次；探测失败时永久回退到 CPU。 */
bool probeCuda() noexcept
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
    {
        cudaGetLastError();
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

bool cudaAvailable() noexcept
{
    static const bool available = probeCuda();
    return available;
}

/** @brief 进程已用显存（字节）；查询失败时返回 0。 */
uint64_t deviceUsedBytes() noexcept
{
    std::size_t free_bytes  = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    {
        cudaGetLastError();
        return 0U;
    }
    return static_cast<uint64_t>(total_bytes) - static_cast<uint64_t>(free_bytes);
}

void checkCuda(const cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        cudaGetLastError();
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", operation, cudaGetErrorString(status));
    }
}

/** @brief 一块按需增长的设备缓冲，附带用于 D2H 的宿主影子缓冲。 */
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void ensure(const std::size_t bytes)
    {
        if (bytes <= bytes_)
        {
            return;
        }
        release();
        checkCuda(cudaMalloc(&device_, bytes), "cudaMalloc");
        bytes_ = bytes;
    }

    void upload(const void *source, const std::size_t bytes)
    {
        if (bytes == 0U)
        {
            return;
        }
        ensure(bytes);
        checkCuda(cudaMemcpy(device_, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
    }

    /** @brief 把指定长度的设备内容拷回宿主影子缓冲并等待完成。 */
    void download(const std::size_t bytes, cudaStream_t stream)
    {
        if (bytes > bytes_)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                 "CUDA download size %zu exceeds device buffer capacity %zu", bytes, bytes_);
        }
        host_.resize(bytes);
        if (bytes == 0U)
        {
            return;
        }
        checkCuda(cudaMemcpyAsync(host_.data(), device_, bytes, cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpy(D2H)");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    void *      data() noexcept { return device_; }
    void *      hostData() noexcept { return host_.data(); }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void release() noexcept
    {
        if (device_ != nullptr)
        {
            cudaFree(device_);
            device_ = nullptr;
        }
        bytes_ = 0U;
    }

    void                *device_{nullptr};
    std::size_t          bytes_{0U};
    std::vector<uint8_t> host_{};
};

/** @brief 每个 CUDA block 负责一个视图和一个查询 token 的最大相似度。 */
__global__ void viewMaxKernel(const int8_t *codes, const float *factors, const size_t *view_offsets,
                              const size_t *view_counts, const int view_count, const int token_count,
                              const int dimension, const float *tokens, float *out)
{
    const int view  = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (view >= view_count || token >= token_count)
    {
        return;
    }

    const size_t begin = view_offsets[view];
    const size_t end   = begin + view_counts[view];
    const float *query = tokens + static_cast<size_t>(token) * static_cast<size_t>(dimension);
    float        best  = -CUDART_INF_F;
    for (size_t descriptor = begin + threadIdx.x; descriptor < end; descriptor += blockDim.x)
    {
        const float factor = factors[descriptor];
        const int8_t *code = codes + descriptor * static_cast<size_t>(dimension);
        float       accumulator = 0.0F;
        for (int channel = 0; channel < dimension; ++channel)
        {
            // 与 CPU 解码路径一致：每个 INT8 分量先乘还原因子，再参与点积。
            accumulator = fmaf(static_cast<float>(code[channel]) * factor, query[channel], accumulator);
        }
        best = fmaxf(best, accumulator);
    }

    __shared__ float shared[kThreadsPerBlock];
    shared[threadIdx.x] = best;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[static_cast<size_t>(view) * static_cast<size_t>(token_count) + static_cast<size_t>(token)] = shared[0];
    }
}

/** @brief 一个有界紧凑块内，输出每条描述子对每个查询 ROI 向量的点积。 */
__global__ void regionScoreKernel(const int8_t *codes, const float *factors, const size_t descriptor_count,
                                  const int query_view_count, const int dimension, const float *roi_vectors,
                                  float *out)
{
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = descriptor_count * static_cast<size_t>(query_view_count);
    if (flat >= total)
    {
        return;
    }

    const size_t descriptor = flat / static_cast<size_t>(query_view_count);
    const int    query_view = static_cast<int>(flat % static_cast<size_t>(query_view_count));
    const int8_t *code      = codes + descriptor * static_cast<size_t>(dimension);
    const float  *roi       = roi_vectors + static_cast<size_t>(query_view) * static_cast<size_t>(dimension);
    float         score     = 0.0F;
    for (int channel = 0; channel < dimension; ++channel)
    {
        score = fmaf(static_cast<float>(code[channel]) * factors[descriptor], roi[channel], score);
    }
    out[flat] = score;
}

/** @brief 设备端紧凑描述归约后端。 */
class CudaSimilarityEngine final : public DinoSimilarityEngine
{
public:
    explicit CudaSimilarityEngine(const int dimension)
        : dimension_(dimension), baseline_used_(deviceUsedBytes())
    {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        {
            cudaGetLastError();
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to create the similarity CUDA stream");
        }
    }

    ~CudaSimilarityEngine() override
    {
        if (stream_ != nullptr)
        {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    void reduceViewGroup(const DinoCompactBlock &block, const std::size_t *view_offsets,
                         const std::size_t *view_counts, const std::size_t view_count, const float *tokens,
                         const int token_count, float *out) override
    {
        validateBlock(block);
        validateInputs(view_offsets, view_counts, tokens, token_count, out);
        if (block.count == 0U || view_count == 0U || token_count <= 0)
        {
            return;
        }
        if (view_count > 65535U || static_cast<unsigned int>(token_count) > 65535U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "CUDA view reduction grid dimensions are too large");
        }

        const auto descriptor_bytes = block.count * block.dimension * sizeof(int8_t);
        const auto factor_bytes     = block.count * sizeof(float);
        const auto offset_bytes     = view_count * sizeof(std::size_t);
        const auto token_bytes      = static_cast<std::size_t>(token_count) * block.dimension * sizeof(float);
        const auto output_bytes     = view_count * static_cast<std::size_t>(token_count) * sizeof(float);
        codes_.upload(block.codes, descriptor_bytes);
        factors_.upload(block.factors, factor_bytes);
        offsets_.upload(view_offsets, offset_bytes);
        counts_.upload(view_counts, offset_bytes);
        tokens_.upload(tokens, token_bytes);
        output_.ensure(output_bytes);
        updateAllocatedPeak();

        const dim3 grid(static_cast<unsigned int>(view_count), static_cast<unsigned int>(token_count));
        viewMaxKernel<<<grid, kThreadsPerBlock, 0, stream_>>>(
            static_cast<const int8_t *>(codes_.data()), static_cast<const float *>(factors_.data()),
            static_cast<const size_t *>(offsets_.data()), static_cast<const size_t *>(counts_.data()),
            static_cast<int>(view_count), token_count, dimension_, static_cast<const float *>(tokens_.data()),
            static_cast<float *>(output_.data()));
        checkCuda(cudaGetLastError(), "viewMaxKernel launch");
        output_.download(output_bytes, stream_);
        std::copy_n(static_cast<const float *>(output_.hostData()), view_count * static_cast<std::size_t>(token_count),
                    out);
        uploaded_bytes_ += descriptor_bytes + factor_bytes + offset_bytes * 2U + token_bytes;
        updateReservedPeak();
    }

    void reduceRegionScores(const DinoCompactBlock &block, const float *roi_vectors, const int query_view_count,
                            float *out) override
    {
        validateBlock(block);
        if (query_view_count <= 0 || block.count == 0U)
        {
            return;
        }
        if (roi_vectors == nullptr || out == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "CUDA region reduction received a null buffer");
        }

        const auto descriptor_bytes = block.count * block.dimension * sizeof(int8_t);
        const auto factor_bytes     = block.count * sizeof(float);
        const auto roi_bytes        = static_cast<std::size_t>(query_view_count) * block.dimension * sizeof(float);
        const auto output_count     = block.count * static_cast<std::size_t>(query_view_count);
        const auto output_bytes     = output_count * sizeof(float);
        codes_.upload(block.codes, descriptor_bytes);
        factors_.upload(block.factors, factor_bytes);
        roi_vectors_.upload(roi_vectors, roi_bytes);
        output_.ensure(output_bytes);
        updateAllocatedPeak();

        const auto grid_size = (output_count + static_cast<std::size_t>(kThreadsPerBlock) - 1U)
                             / static_cast<std::size_t>(kThreadsPerBlock);
        regionScoreKernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock, 0, stream_>>>(
            static_cast<const int8_t *>(codes_.data()), static_cast<const float *>(factors_.data()), block.count,
            query_view_count, dimension_, static_cast<const float *>(roi_vectors_.data()),
            static_cast<float *>(output_.data()));
        checkCuda(cudaGetLastError(), "regionScoreKernel launch");
        output_.download(output_bytes, stream_);
        std::copy_n(static_cast<const float *>(output_.hostData()), output_count, out);
        uploaded_bytes_ += descriptor_bytes + factor_bytes + roi_bytes;
        updateReservedPeak();
    }

    uint64_t deviceAllocatedBytes() const noexcept override { return allocated_bytes_; }
    uint64_t deviceReservedDeltaBytes() const noexcept override { return reserved_peak_; }
    uint64_t uploadedBytes() const noexcept override { return uploaded_bytes_; }

private:
    void validateBlock(const DinoCompactBlock &block) const
    {
        if (block.count > 0U && (block.codes == nullptr || block.factors == nullptr))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "CUDA reduction received a null descriptor buffer");
        }
        if (block.count > 0U && block.dimension != static_cast<std::size_t>(dimension_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Compact descriptor dimension %zu differs from the configured dimension %d",
                                 block.dimension, dimension_);
        }
    }

    void validateInputs(const std::size_t *view_offsets, const std::size_t *view_counts, const float *tokens,
                        const int token_count, const float *out) const
    {
        if (view_offsets == nullptr || view_counts == nullptr || tokens == nullptr || out == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "CUDA view reduction received a null buffer");
        }
        if (token_count <= 0)
        {
            return;
        }
    }

    void updateAllocatedPeak() noexcept
    {
        allocated_bytes_ = std::max(allocated_bytes_, tokens_.bytes() + roi_vectors_.bytes() + codes_.bytes()
                                                          + factors_.bytes() + offsets_.bytes() + counts_.bytes()
                                                          + output_.bytes());
    }

    void updateReservedPeak() noexcept
    {
        const auto used = deviceUsedBytes();
        if (used > baseline_used_)
        {
            reserved_peak_ = std::max(reserved_peak_, used - baseline_used_);
        }
    }

    int          dimension_{0};
    cudaStream_t stream_{nullptr};
    DeviceBuffer tokens_{};
    DeviceBuffer roi_vectors_{};
    DeviceBuffer codes_{};
    DeviceBuffer factors_{};
    DeviceBuffer offsets_{};
    DeviceBuffer counts_{};
    DeviceBuffer output_{};
    uint64_t     baseline_used_{0U};
    uint64_t     reserved_peak_{0U};
    uint64_t     allocated_bytes_{0U};
    uint64_t     uploaded_bytes_{0U};
};

} // namespace

bool dinoCudaSimilarityAvailable() noexcept
{
    return cudaAvailable();
}

std::unique_ptr<DinoSimilarityEngine> dinoMakeCudaSimilarityEngine(const int dimension)
{
    if (!cudaAvailable())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "No usable CUDA device for the similarity kernel");
    }
    if (dimension <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Similarity dimension must be positive");
    }
    return std::make_unique<CudaSimilarityEngine>(dimension);
}

} // namespace irt::features::priv
