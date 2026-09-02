#include "priv/OpRoIAlignImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpRoIAlign.h>
#include <inferrt/cvcuda/OpRoIAlign.hpp>

namespace irt::cvcuda {

using irt::ProtectCall;

IRTStatus roiAlign(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                   cv::Size input_size, int num_rois, cv::Size output_size, float spatial_scale, int sampling_ratio,
                   bool aligned, cudaStream_t stream)
{
    RoIAlign roi_align_op(output_size, spatial_scale, sampling_ratio, aligned);
    return roi_align_op(d_input, d_rois, d_output, batches, channels, input_size, num_rois, stream);
}

IRTStatus roi_align(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                    cv::Size input_size, int num_rois, cv::Size output_size, float spatial_scale, int sampling_ratio,
                    bool aligned, cudaStream_t stream)
{
    return roiAlign(d_input, d_rois, d_output, batches, channels, input_size, num_rois, output_size, spatial_scale,
                    sampling_ratio, aligned, stream);
}

RoIAlign::RoIAlign(cv::Size output_size, float spatial_scale, int sampling_ratio, bool aligned)
    : output_size_(output_size)
    , spatial_scale_(spatial_scale)
    , sampling_ratio_(sampling_ratio)
    , aligned_(aligned)
{
    impl_ = std::make_unique<priv::RoIAlignImpl>();
}

RoIAlign::~RoIAlign() = default;

RoIAlign::RoIAlign(RoIAlign &&) noexcept = default;

RoIAlign &RoIAlign::operator=(RoIAlign &&) noexcept = default;

IRTStatus RoIAlign::operator()(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                               cv::Size input_size, int num_rois, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            int2 _input_size;
            _input_size.x = input_size.width;
            _input_size.y = input_size.height;

            int2 _output_size;
            _output_size.x = output_size_.width;
            _output_size.y = output_size_.height;

            auto *roiAlignImpl = static_cast<priv::RoIAlignImpl *>(impl_.get());
            (*roiAlignImpl)(d_input, d_rois, d_output, batches, channels, _input_size, num_rois, _output_size,
                            spatial_scale_, sampling_ratio_, aligned_, stream);
        });
    return status;
}

} // namespace irt::cvcuda
