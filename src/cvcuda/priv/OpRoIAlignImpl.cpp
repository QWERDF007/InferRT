#include "OpRoIAlignImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>

#include <cmath>

namespace irt::cvcuda::priv {

void RoIAlignImpl::operator()(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                              const int2 input_size, int num_rois, const int2 output_size, float spatial_scale,
                              int sampling_ratio, bool aligned, cudaStream_t stream)
{
    if (batches <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid batch count");
    }
    if (channels <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid channel count");
    }
    if (input_size.x <= 0 || input_size.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid input size");
    }
    if (output_size.x <= 0 || output_size.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid output size");
    }
    if (num_rois < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_rois must be non-negative");
    }
    if (!std::isfinite(spatial_scale))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "spatial_scale must be finite");
    }

    (void)irt::checkedSizeProduct({static_cast<size_t>(num_rois), static_cast<size_t>(channels),
                                   static_cast<size_t>(output_size.x), static_cast<size_t>(output_size.y)},
                                  "RoIAlign output");
    (void)irt::checkedSizeProduct({static_cast<size_t>(batches), static_cast<size_t>(channels),
                                   static_cast<size_t>(input_size.x), static_cast<size_t>(input_size.y)},
                                  "RoIAlign input");

    const bool has_output = num_rois > 0 && channels > 0 && output_size.x > 0 && output_size.y > 0;
    if (has_output && d_input == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Input pointer is null");
    }
    if (num_rois > 0 && d_rois == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "RoIs pointer is null");
    }
    if (has_output && d_output == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Output pointer is null");
    }

    RunRoIAlign(d_input, d_rois, d_output, batches, channels, input_size, num_rois, output_size, spatial_scale,
                sampling_ratio, aligned, stream);
}

} // namespace irt::cvcuda::priv
