#include "OpNMSImpl.hpp"

#include <inferrt/core/Tensor.hpp>
#include <inferrt/util/CheckError.hpp>

#include <cstdint>
#include <limits>
#include <utility>

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)
#endif
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#ifdef _MSC_VER
#    pragma warning(pop)
#endif

namespace irt::cvcuda::priv {
namespace {

constexpr int NMS_THREADS_PER_BLOCK = 64;

__device__ float nms_box_area(const float *box)
{
    const float width  = fmaxf(box[2] - box[0], 0.0f);
    const float height = fmaxf(box[3] - box[1], 0.0f);
    return width * height;
}

__device__ float nms_iou(const float *lhs, const float *rhs)
{
    const float xx1 = fmaxf(lhs[0], rhs[0]);
    const float yy1 = fmaxf(lhs[1], rhs[1]);
    const float xx2 = fminf(lhs[2], rhs[2]);
    const float yy2 = fminf(lhs[3], rhs[3]);

    const float width  = fmaxf(xx2 - xx1, 0.0f);
    const float height = fmaxf(yy2 - yy1, 0.0f);
    const float inter  = width * height;
    const float area   = nms_box_area(lhs) + nms_box_area(rhs) - inter;
    return area > 0.0f ? inter / area : 0.0f;
}

__global__ void nms_bitmask_kernel(const float *boxes, const int *order, unsigned long long *masks, int num_boxes,
                                   int col_blocks, float iou_threshold)
{
    const int col_block = blockIdx.x;
    const int row_block = blockIdx.y;
    if (col_block < row_block)
    {
        return;
    }

    const int row_start = row_block * NMS_THREADS_PER_BLOCK;
    const int col_start = col_block * NMS_THREADS_PER_BLOCK;
    const int row_size  = min(num_boxes - row_start, NMS_THREADS_PER_BLOCK);
    const int col_size  = min(num_boxes - col_start, NMS_THREADS_PER_BLOCK);

    __shared__ float block_boxes[NMS_THREADS_PER_BLOCK * 4];
    if (threadIdx.x < col_size)
    {
        const int original_index = order[col_start + threadIdx.x];
        const float *box         = boxes + original_index * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            block_boxes[threadIdx.x * 4 + i] = box[i];
        }
    }
    __syncthreads();

    if (threadIdx.x >= row_size)
    {
        return;
    }

    const int sorted_row_index   = row_start + threadIdx.x;
    const int original_row_index = order[sorted_row_index];
    const float *row_box         = boxes + original_row_index * 4;

    unsigned long long mask = 0;
    const int start         = row_block == col_block ? threadIdx.x + 1 : 0;
    for (int i = start; i < col_size; ++i)
    {
        if (nms_iou(row_box, block_boxes + i * 4) > iou_threshold)
        {
            mask |= 1ULL << i;
        }
    }

    masks[static_cast<size_t>(sorted_row_index) * col_blocks + col_block] = mask;
}

__global__ void nms_select_kernel(const unsigned long long *masks, const int *order, int64_t *keep, int *keep_count,
                                  unsigned long long *removed, int num_boxes, int col_blocks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    int out_count = 0;
    for (int i = 0; i < num_boxes; ++i)
    {
        const int block_index = i / NMS_THREADS_PER_BLOCK;
        const int bit_index   = i % NMS_THREADS_PER_BLOCK;
        if ((removed[block_index] & (1ULL << bit_index)) != 0ULL)
        {
            continue;
        }

        keep[out_count++] = static_cast<int64_t>(order[i]);
        const unsigned long long *row_mask = masks + static_cast<size_t>(i) * col_blocks;
        for (int j = block_index; j < col_blocks; ++j)
        {
            removed[j] |= row_mask[j];
        }
    }

    *keep_count = out_count;
}

static inline size_t align256(const size_t bytes)
{
    return irt::checkedSizeAdd(bytes, 255, "NMS workspace alignment") & ~static_cast<size_t>(255);
}

} // namespace

NMSImpl::Workspace::~Workspace() noexcept
{
    reset();
}

NMSImpl::Workspace::Workspace(Workspace &&other) noexcept
    : data_(other.data_)
    , bytes_(other.bytes_)
    , device_id_(other.device_id_)
    , stream_(other.stream_)
{
    other.data_      = nullptr;
    other.bytes_     = 0;
    other.device_id_ = -1;
    other.stream_    = nullptr;
}

NMSImpl::Workspace &NMSImpl::Workspace::operator=(Workspace &&other) noexcept
{
    if (this != &other)
    {
        reset();
        data_      = other.data_;
        bytes_     = other.bytes_;
        device_id_ = other.device_id_;
        stream_    = other.stream_;
        other.data_      = nullptr;
        other.bytes_     = 0;
        other.device_id_ = -1;
        other.stream_    = nullptr;
    }
    return *this;
}

void NMSImpl::Workspace::allocate(const size_t bytes, const int device_id, const cudaStream_t stream)
{
    if (bytes == 0 || device_id < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "NMS workspace requires positive bytes and a non-negative device");
    }

    int previous_device = -1;
    const auto query_status = cudaGetDevice(&previous_device);
    if (query_status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "NMS workspace device query failed: %s",
                             cudaGetErrorString(query_status));
    }
    if (previous_device != device_id)
    {
        const auto set_status = cudaSetDevice(device_id);
        if (set_status != cudaSuccess)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "NMS workspace device %d is unavailable: %s",
                                 device_id, cudaGetErrorString(set_status));
        }
    }

    void *candidate = nullptr;
    const auto allocation_status = cudaMallocAsync(&candidate, bytes, stream);
    if (allocation_status != cudaSuccess || candidate == nullptr)
    {
        if (previous_device != device_id)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                             "NMS workspace cudaMallocAsync failed for %zu bytes: %s", bytes,
                             cudaGetErrorString(allocation_status));
    }

    data_      = candidate;
    bytes_     = bytes;
    device_id_ = device_id;
    stream_    = stream;
    if (previous_device != device_id)
    {
        (void)cudaSetDevice(previous_device);
    }
}

void NMSImpl::Workspace::reset() noexcept
{
    if (data_ == nullptr)
    {
        return;
    }

    int previous_device = -1;
    const bool have_previous_device = cudaGetDevice(&previous_device) == cudaSuccess;
    if (device_id_ >= 0 && (!have_previous_device || previous_device != device_id_))
    {
        (void)cudaSetDevice(device_id_);
    }

    // Synchronizing first makes reset safe when a caller changes streams or
    // destroys the operator while work is still queued on the owning stream.
    if (stream_ != nullptr)
    {
        (void)cudaStreamSynchronize(stream_);
    }
    const auto free_status = cudaFreeAsync(data_, stream_);
    if (free_status == cudaSuccess)
    {
        (void)cudaStreamSynchronize(stream_);
    }
    else
    {
        // Keep destruction best-effort and leak-free even if the stream has
        // already been torn down by an embedding application.
        (void)cudaFree(data_);
    }

    if (have_previous_device && previous_device != device_id_)
    {
        (void)cudaSetDevice(previous_device);
    }
    data_      = nullptr;
    bytes_     = 0;
    device_id_ = -1;
    stream_    = nullptr;
}

NMSImpl::~NMSImpl()
{
    synchronizeWorkspaceStream();
    freeWorkspace();
}

void NMSImpl::freeWorkspace() noexcept
{
    if (workspace_.data() != nullptr)
    {
        ++release_count_;
    }
    workspace_.reset();
    workspace_bytes_      = 0;
    d_sorted_scores_      = nullptr;
    d_order_              = nullptr;
    d_masks_              = nullptr;
    d_removed_            = nullptr;
    allocated_boxes_      = 0;
    allocated_col_blocks_ = 0;

}

void NMSImpl::synchronizeWorkspaceStream() noexcept
{
    if (workspace_.data() == nullptr || workspace_.deviceId() < 0)
    {
        return;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess)
    {
        return;
    }
    if (cudaSetDevice(workspace_.deviceId()) != cudaSuccess)
    {
        return;
    }
    (void)cudaStreamSynchronize(workspace_.stream());
    (void)cudaSetDevice(current_device);
}

void NMSImpl::bindWorkspaceToStream(const cudaStream_t stream)
{
    int current_device = -1;
    IRT_CHECK_THROW(cudaGetDevice(&current_device), "NMS current device query failed");
    if (workspace_.data() != nullptr
        && (workspace_.deviceId() != current_device || workspace_.stream() != stream))
    {
        // The workspace is single-owner.  Complete work submitted on the old
        // stream before releasing or reusing it on another stream/device.
        synchronizeWorkspaceStream();
        freeWorkspace();
    }
    workspace_device_ = current_device;
    workspace_stream_ = stream;
}

void NMSImpl::ensureWorkspace(const int num_boxes, const int col_blocks)
{
    if (num_boxes <= allocated_boxes_ && col_blocks <= allocated_col_blocks_ && workspace_.data() != nullptr)
    {
        return;
    }

    const int target_boxes = std::max(
        num_boxes, allocated_boxes_ > std::numeric_limits<int>::max() / 2 ? num_boxes : allocated_boxes_ * 2);
    const int target_col_blocks = std::max(
        col_blocks, allocated_col_blocks_ > std::numeric_limits<int>::max() / 2 ? col_blocks : allocated_col_blocks_ * 2);

    const size_t scores_size = align256(irt::checkedSizeMul(static_cast<size_t>(target_boxes), sizeof(float), "NMS scores"));
    const size_t order_size  = align256(irt::checkedSizeMul(static_cast<size_t>(target_boxes), sizeof(int), "NMS order"));
    const size_t masks_elements
        = irt::checkedSizeMul(static_cast<size_t>(target_boxes), static_cast<size_t>(target_col_blocks), "NMS masks");
    const size_t masks_size = align256(irt::checkedSizeMul(masks_elements, sizeof(unsigned long long), "NMS masks"));
    const size_t removed_size
        = align256(irt::checkedSizeMul(static_cast<size_t>(target_col_blocks), sizeof(unsigned long long), "NMS removed"));
    const size_t total_size = irt::checkedSizeAdd(
        irt::checkedSizeAdd(irt::checkedSizeAdd(scores_size, order_size, "NMS workspace"), masks_size, "NMS workspace"), removed_size,
        "NMS workspace");

    synchronizeWorkspaceStream();
    freeWorkspace();

    workspace_.allocate(total_size, workspace_device_, workspace_stream_);
    ++allocation_count_;

    workspace_bytes_      = total_size;
    allocated_boxes_      = target_boxes;
    allocated_col_blocks_ = target_col_blocks;

    char *base = static_cast<char *>(workspace_.data());
    d_sorted_scores_ = reinterpret_cast<float *>(base);
    d_order_         = reinterpret_cast<int *>(base + scores_size);
    d_masks_         = reinterpret_cast<unsigned long long *>(base + scores_size + order_size);
    d_removed_       = reinterpret_cast<unsigned long long *>(base + scores_size + order_size + masks_size);
}

NMSWorkspaceStats NMSImpl::workspaceStats() const noexcept
{
    return {allocation_count_, release_count_, workspace_bytes_, workspace_device_,
            reinterpret_cast<std::uintptr_t>(workspace_stream_)};
}

void NMSImpl::RunNMS(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                     float iou_threshold, cudaStream_t stream)
{
    if (num_boxes == 0)
    {
        IRT_CHECK_THROW(cudaMemsetAsync(d_keep_count, 0, sizeof(int), stream), "NMS count initialization failed");
        return;
    }

    const size_t block_count
        = (static_cast<size_t>(num_boxes) + static_cast<size_t>(NMS_THREADS_PER_BLOCK) - 1)
        / static_cast<size_t>(NMS_THREADS_PER_BLOCK);
    if (block_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "NMS block count exceeds supported range");
    }
    const int col_blocks = static_cast<int>(block_count);
    bindWorkspaceToStream(stream);
    ensureWorkspace(num_boxes, col_blocks);

    const size_t score_bytes = irt::checkedSizeMul(static_cast<size_t>(num_boxes), sizeof(float), "NMS score copy");
    IRT_CHECK_THROW(cudaMemcpyAsync(d_sorted_scores_, d_scores, score_bytes, cudaMemcpyDeviceToDevice, stream),
                    "NMS score copy failed: boxes=%d", num_boxes);

    auto policy     = thrust::cuda::par.on(stream);
    auto scores_ptr = thrust::device_pointer_cast(d_sorted_scores_);
    auto order_ptr  = thrust::device_pointer_cast(d_order_);
    thrust::sequence(policy, order_ptr, order_ptr + num_boxes);
    thrust::stable_sort_by_key(policy, scores_ptr, scores_ptr + num_boxes, order_ptr, thrust::greater<float>());

    const dim3 grid(col_blocks, col_blocks);
    nms_bitmask_kernel<<<grid, NMS_THREADS_PER_BLOCK, 0, stream>>>(d_boxes, d_order_, d_masks_, num_boxes,
                                                                   col_blocks, iou_threshold);
    IRT_CHECK_THROW(cudaPeekAtLastError(), "NMS bitmask kernel launch failed: boxes=%d blocks=%d", num_boxes,
                    col_blocks);

    const size_t removed_bytes
        = irt::checkedSizeMul(static_cast<size_t>(col_blocks), sizeof(unsigned long long), "NMS removed masks");
    IRT_CHECK_THROW(cudaMemsetAsync(d_removed_, 0, removed_bytes, stream),
                    "NMS removed mask initialization failed: blocks=%d", col_blocks);
    nms_select_kernel<<<1, 1, 0, stream>>>(d_masks_, d_order_, d_keep, d_keep_count, d_removed_, num_boxes,
                                           col_blocks);
    IRT_CHECK_THROW(cudaPeekAtLastError(), "NMS select kernel launch failed: boxes=%d blocks=%d", num_boxes,
                    col_blocks);
}

} // namespace irt::cvcuda::priv
