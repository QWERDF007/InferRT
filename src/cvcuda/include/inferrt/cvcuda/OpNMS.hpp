#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>

#include <cstdint>
#include <cstddef>

namespace irt::cvcuda {

/** Runtime workspace counters for one NMS operator instance. */
struct NMSWorkspaceStats
{
    size_t          allocation_count{0};
    size_t          release_count{0};
    size_t          workspace_bytes{0};
    int             device_id{-1};
    std::uintptr_t  stream{0};
};

class INFERRT_CVCUDA_API NMS final : public IOperator
{
public:
    explicit NMS();

    ~NMS();

    NMS(const NMS &)            = delete;
    NMS &operator=(const NMS &) = delete;
    NMS(NMS &&) noexcept;
    NMS &operator=(NMS &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep,
                                       int *d_keep_count, int num_boxes, float iou_threshold,
                                       cudaStream_t stream = nullptr);

    /** Returns allocation/reuse counters for the current workspace. */
    [[nodiscard]] NMSWorkspaceStats workspaceStats() const noexcept;

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
