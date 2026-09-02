#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

class INFERRT_CVCUDA_API RoIAlign final : public IOperator
{
public:
    explicit RoIAlign(cv::Size output_size, float spatial_scale = 1.0f, int sampling_ratio = -1,
                      bool aligned = false);

    ~RoIAlign();

    RoIAlign(const RoIAlign &)            = delete;
    RoIAlign &operator=(const RoIAlign &) = delete;
    RoIAlign(RoIAlign &&) noexcept;
    RoIAlign &operator=(RoIAlign &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const float *d_input, const float *d_rois, float *d_output, int batches,
                                       int channels, cv::Size input_size, int num_rois,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
    cv::Size       output_size_;
    float          spatial_scale_;
    int            sampling_ratio_;
    bool           aligned_;
};

} // namespace irt::cvcuda
