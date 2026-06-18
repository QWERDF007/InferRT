#include "OpNMSImpl.hpp"

#include <inferrt/util/CheckError.hpp>

#include <cstdint>

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

} // namespace

void NMSImpl::RunNMS(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                     float iou_threshold, cudaStream_t stream)
{
    if (num_boxes == 0)
    {
        IRT_CHECK_THROW(cudaMemsetAsync(d_keep_count, 0, sizeof(int), stream), "NMS count initialization failed");
        return;
    }

    const int    col_blocks = (num_boxes + NMS_THREADS_PER_BLOCK - 1) / NMS_THREADS_PER_BLOCK;
    const size_t mask_count = static_cast<size_t>(num_boxes) * col_blocks;

    float              *d_sorted_scores = nullptr;
    int                *d_order         = nullptr;
    unsigned long long *d_masks         = nullptr;
    unsigned long long *d_removed       = nullptr;

    try
    {
        IRT_CHECK_THROW(cudaMalloc(&d_sorted_scores, static_cast<size_t>(num_boxes) * sizeof(float)),
                        "NMS score workspace allocation failed: boxes=%d", num_boxes);
        IRT_CHECK_THROW(cudaMalloc(&d_order, static_cast<size_t>(num_boxes) * sizeof(int)),
                        "NMS order workspace allocation failed: boxes=%d", num_boxes);
        IRT_CHECK_THROW(cudaMalloc(&d_masks, mask_count * sizeof(unsigned long long)),
                        "NMS mask workspace allocation failed: boxes=%d blocks=%d", num_boxes, col_blocks);
        IRT_CHECK_THROW(cudaMalloc(&d_removed, static_cast<size_t>(col_blocks) * sizeof(unsigned long long)),
                        "NMS removed workspace allocation failed: blocks=%d", col_blocks);

        IRT_CHECK_THROW(cudaMemcpyAsync(d_sorted_scores, d_scores, static_cast<size_t>(num_boxes) * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream),
                        "NMS score copy failed: boxes=%d", num_boxes);

        auto policy     = thrust::cuda::par.on(stream);
        auto scores_ptr = thrust::device_pointer_cast(d_sorted_scores);
        auto order_ptr  = thrust::device_pointer_cast(d_order);
        thrust::sequence(policy, order_ptr, order_ptr + num_boxes);
        thrust::stable_sort_by_key(policy, scores_ptr, scores_ptr + num_boxes, order_ptr, thrust::greater<float>());

        const dim3 grid(col_blocks, col_blocks);
        nms_bitmask_kernel<<<grid, NMS_THREADS_PER_BLOCK, 0, stream>>>(d_boxes, d_order, d_masks, num_boxes,
                                                                       col_blocks, iou_threshold);
        IRT_CHECK_THROW(cudaPeekAtLastError(), "NMS bitmask kernel launch failed: boxes=%d blocks=%d", num_boxes,
                        col_blocks);

        IRT_CHECK_THROW(cudaMemsetAsync(d_removed, 0, static_cast<size_t>(col_blocks) * sizeof(unsigned long long),
                                        stream),
                        "NMS removed mask initialization failed: blocks=%d", col_blocks);
        nms_select_kernel<<<1, 1, 0, stream>>>(d_masks, d_order, d_keep, d_keep_count, d_removed, num_boxes,
                                               col_blocks);
        IRT_CHECK_THROW(cudaPeekAtLastError(), "NMS select kernel launch failed: boxes=%d blocks=%d", num_boxes,
                        col_blocks);
    }
    catch (...)
    {
        cudaFree(d_sorted_scores);
        cudaFree(d_order);
        cudaFree(d_masks);
        cudaFree(d_removed);
        throw;
    }

    IRT_CHECK_THROW(cudaFree(d_sorted_scores), "NMS score workspace release failed");
    IRT_CHECK_THROW(cudaFree(d_order), "NMS order workspace release failed");
    IRT_CHECK_THROW(cudaFree(d_masks), "NMS mask workspace release failed");
    IRT_CHECK_THROW(cudaFree(d_removed), "NMS removed workspace release failed");
}

} // namespace irt::cvcuda::priv
