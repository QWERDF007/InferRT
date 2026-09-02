#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>
#include <inferrt/cvcuda/OpNMS.hpp>

#include <cstdint>
#include <memory>

namespace irt::cvcuda::priv {

using irt::cvcuda::NMSWorkspaceStats;

class NMSImpl final : public IOperatorImpl
{
public:
    explicit NMSImpl()  = default;
    ~NMSImpl() override;

    void operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                    float iou_threshold, cudaStream_t stream);

private:
    /**
     * Owns one asynchronous CUDA allocation together with the device and
     * stream that establish its lifetime ordering.
     *
     * The owner is intentionally private to NMS.  A workspace is only reused
     * after the previous stream has completed, so switching streams cannot
     * expose a buffer while an earlier invocation still accesses it.
     */
    class Workspace final
    {
    public:
        Workspace() = default;
        ~Workspace() noexcept;

        Workspace(const Workspace &)            = delete;
        Workspace &operator=(const Workspace &) = delete;
        Workspace(Workspace &&other) noexcept;
        Workspace &operator=(Workspace &&other) noexcept;

        void allocate(size_t bytes, int device_id, cudaStream_t stream);
        void reset() noexcept;

        [[nodiscard]] void *data() const noexcept { return data_; }
        [[nodiscard]] int deviceId() const noexcept { return device_id_; }
        [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    private:
        void *       data_{nullptr};
        size_t       bytes_{0};
        int          device_id_{-1};
        cudaStream_t stream_{nullptr};
    };

    void ensureWorkspace(int num_boxes, int col_blocks);
    void bindWorkspaceToStream(cudaStream_t stream);
    void synchronizeWorkspaceStream() noexcept;
    void freeWorkspace() noexcept;
    void RunNMS(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                float iou_threshold, cudaStream_t stream);

    Workspace           workspace_;
    size_t              workspace_bytes_{0};
    float              *d_sorted_scores_{nullptr};
    int                *d_order_{nullptr};
    unsigned long long *d_masks_{nullptr};
    unsigned long long *d_removed_{nullptr};
    int                 allocated_boxes_{0};
    int                 allocated_col_blocks_{0};
    size_t              allocation_count_{0};
    size_t              release_count_{0};
    int                 workspace_device_{-1};
    cudaStream_t        workspace_stream_{nullptr};

public:
    [[nodiscard]] NMSWorkspaceStats workspaceStats() const noexcept;
};

} // namespace irt::cvcuda::priv
