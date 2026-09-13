/**
 * @file DinoSimilarity.cpp
 * @brief 相似度归约：CPU 后端、后端选择与归约器门面。
 */

#include "DinoSimilarity.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/core.hpp>

#include <atomic>
#include <vector>

namespace irt::features::priv {

void dinoAccumulateSimilarityMaximaFast(const float *descriptors, std::size_t descriptor_count, const float *tokens,
                                        int token_count, int dimension, const int *slot_of_token, float *maxima);

namespace {

/**
 * @brief 标量参考实现：逐描述子、逐 token 累加点积。
 *
 * 这条路径同时是语义基准（其余实现只允许改变累加顺序）与非 AVX2 机器上的回退。
 */
void dinoAccumulateSimilarityMaximaScalar(const float *descriptors, const std::size_t descriptor_count,
                                          const float *tokens, const int token_count, const int dimension,
                                          const int *slot_of_token, float *maxima)
{
    for (std::size_t index = 0; index < descriptor_count; ++index)
    {
        const float *descriptor = descriptors + index * static_cast<std::size_t>(dimension);
        for (int token = 0; token < token_count; ++token)
        {
            const float *query = tokens + static_cast<std::size_t>(token) * static_cast<std::size_t>(dimension);
            float        score = 0.0F;
            for (int channel = 0; channel < dimension; ++channel)
            {
                score += descriptor[channel] * query[channel];
            }
            const int slot = slot_of_token[token];
            if (score > maxima[slot])
            {
                maxima[slot] = score;
            }
        }
    }
}

using DinoCpuSimilarityKernel = void (*)(const float *, std::size_t, const float *, int, int, const int *, float *);

DinoCpuSimilarityKernel selectCpuKernel() noexcept
{
    if (cv::checkHardwareSupport(CV_CPU_AVX2))
    {
        return &dinoAccumulateSimilarityMaximaFast;
    }
    return &dinoAccumulateSimilarityMaximaScalar;
}

const DinoCpuSimilarityKernel kCpuKernel = selectCpuKernel();

/** @brief 进程级后端覆盖：-1 自动、0 强制 CPU、1 强制 CUDA。 */
std::atomic<int> gBackendOverride{-1};

/**
 * @brief CPU 后端：把紧凑块还原成 FP32 后交给 CPU 内核。
 *
 * 还原缓冲在多次调用之间复用；局部通道的每组视图只需还原一次，区域通道同理。
 */
class CpuSimilarityEngine final : public DinoSimilarityEngine
{
public:
    explicit CpuSimilarityEngine(const int dimension)
        : dimension_(dimension)
    {
    }

    void reduceViewGroup(const DinoCompactBlock &block, const std::size_t *view_offsets,
                         const std::size_t *view_counts, const std::size_t view_count, const float *tokens,
                         const int token_count, float *out) override
    {
        validateBlock(block);
        if (token_count <= 0 || view_count == 0U)
        {
            return;
        }
        decode(block);
        identity_slots_.resize(static_cast<std::size_t>(token_count));
        for (int token = 0; token < token_count; ++token)
        {
            identity_slots_[static_cast<std::size_t>(token)] = token;
        }
        for (std::size_t view = 0; view < view_count; ++view)
        {
            const auto count = view_counts[view];
            if (count == 0U)
            {
                continue;
            }
            kCpuKernel(decoded_.data() + view_offsets[view] * block.dimension, count, tokens, token_count,
                       dimension_, identity_slots_.data(), out + view * static_cast<std::size_t>(token_count));
        }
    }

    void reduceRegionScores(const DinoCompactBlock &block, const float *roi_vectors, const int query_view_count,
                            float *out) override
    {
        validateBlock(block);
        if (query_view_count <= 0)
        {
            return;
        }
        decode(block);
        for (std::size_t index = 0; index < block.count; ++index)
        {
            const float *descriptor = decoded_.data() + index * block.dimension;
            auto        *row         = out + index * static_cast<std::size_t>(query_view_count);
            for (int query_view = 0; query_view < query_view_count; ++query_view)
            {
                const float *roi    = roi_vectors + static_cast<std::size_t>(query_view) * block.dimension;
                float        score = 0.0F;
                for (std::size_t channel = 0; channel < block.dimension; ++channel)
                {
                    score += descriptor[channel] * roi[channel];
                }
                row[query_view] = score;
            }
        }
    }

    uint64_t deviceAllocatedBytes() const noexcept override { return 0U; }
    uint64_t deviceReservedDeltaBytes() const noexcept override { return 0U; }
    uint64_t uploadedBytes() const noexcept override { return 0U; }

private:
    void validateBlock(const DinoCompactBlock &block) const
    {
        if (block.count > 0U && block.dimension != static_cast<std::size_t>(dimension_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Compact descriptor dimension %zu differs from the configured dimension %d",
                                 block.dimension, dimension_);
        }
    }

    void decode(const DinoCompactBlock &block)
    {
        decoded_.resize(block.count * block.dimension);
        for (std::size_t index = 0; index < block.count; ++index)
        {
            const float factor = block.factors[index];
            const auto *code   = block.codes + index * block.dimension;
            auto       *target = decoded_.data() + index * block.dimension;
            for (std::size_t channel = 0; channel < block.dimension; ++channel)
            {
                target[channel] = static_cast<float>(code[channel]) * factor;
            }
        }
    }

    int                dimension_{0};
    std::vector<float> decoded_{};
    std::vector<int>   identity_slots_{};
};

} // namespace

std::unique_ptr<DinoSimilarityEngine> dinoMakeCpuSimilarityEngine(const int dimension)
{
    return std::make_unique<CpuSimilarityEngine>(dimension);
}

void dinoAccumulateSimilarityMaxima(const float *descriptors, const std::size_t descriptor_count, const float *tokens,
                                    const int token_count, const int dimension, const int *slot_of_token,
                                    float *maxima)
{
    if (descriptor_count == 0U || token_count <= 0 || dimension <= 0)
    {
        return;
    }
    kCpuKernel(descriptors, descriptor_count, tokens, token_count, dimension, slot_of_token, maxima);
}

const char *dinoSimilarityBackendName(const DinoSimilarityBackend backend) noexcept
{
    if (backend == DinoSimilarityBackend::Cuda)
    {
        return "cuda";
    }
    return kCpuKernel == &dinoAccumulateSimilarityMaximaFast ? "cpu:avx2" : "cpu:scalar";
}

void dinoOverrideSimilarityBackend(const bool enable, const DinoSimilarityBackend backend) noexcept
{
    gBackendOverride.store(enable ? (backend == DinoSimilarityBackend::Cuda ? 1 : 0) : -1, std::memory_order_relaxed);
}

DinoSimilarityBackend dinoSelectSimilarityBackend() noexcept
{
    const int override_value = gBackendOverride.load(std::memory_order_relaxed);
    if (override_value >= 0)
    {
        return override_value == 1 ? DinoSimilarityBackend::Cuda : DinoSimilarityBackend::Cpu;
    }
    return dinoCudaSimilarityAvailable() ? DinoSimilarityBackend::Cuda : DinoSimilarityBackend::Cpu;
}

DinoSimilarityReducer::DinoSimilarityReducer(const DinoSimilarityBackend backend, const int dimension)
{
    if (dimension <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Similarity reducer requires a positive dimension");
    }
    // 契约：请求 CUDA 但设备不可用时回退 CPU，而不是失败查询。
    backend_ = backend == DinoSimilarityBackend::Cuda && !dinoCudaSimilarityAvailable()
                   ? DinoSimilarityBackend::Cpu
                   : backend;
    engine_ = backend_ == DinoSimilarityBackend::Cuda ? dinoMakeCudaSimilarityEngine(dimension)
                                                      : dinoMakeCpuSimilarityEngine(dimension);
}

DinoSimilarityReducer::~DinoSimilarityReducer() = default;

void DinoSimilarityReducer::reduceViewGroup(const DinoCompactBlock &block, const std::size_t *view_offsets,
                                            const std::size_t *view_counts, const std::size_t view_count,
                                            const float *tokens, const int token_count, float *out)
{
    engine_->reduceViewGroup(block, view_offsets, view_counts, view_count, tokens, token_count, out);
}

void DinoSimilarityReducer::reduceRegionScores(const DinoCompactBlock &block, const float *roi_vectors,
                                               const int query_view_count, float *out)
{
    engine_->reduceRegionScores(block, roi_vectors, query_view_count, out);
}

uint64_t DinoSimilarityReducer::deviceAllocatedBytes() const noexcept
{
    return engine_->deviceAllocatedBytes();
}

uint64_t DinoSimilarityReducer::deviceReservedDeltaBytes() const noexcept
{
    return engine_->deviceReservedDeltaBytes();
}

uint64_t DinoSimilarityReducer::uploadedBytes() const noexcept
{
    return engine_->uploadedBytes();
}

} // namespace irt::features::priv
