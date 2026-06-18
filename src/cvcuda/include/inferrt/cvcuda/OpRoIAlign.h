#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

[[nodiscard]] INFERRT_CVCUDA_API IRTStatus roiAlign(const float *d_input, const float *d_rois, float *d_output,
                                                    int batches, int channels, cv::Size input_size, int num_rois,
                                                    cv::Size output_size, float spatial_scale = 1.0f,
                                                    int sampling_ratio = -1, bool aligned = false,
                                                    cudaStream_t stream = nullptr);

[[nodiscard]] INFERRT_CVCUDA_API IRTStatus roi_align(const float *d_input, const float *d_rois, float *d_output,
                                                     int batches, int channels, cv::Size input_size, int num_rois,
                                                     cv::Size output_size, float spatial_scale = 1.0f,
                                                     int sampling_ratio = -1, bool aligned = false,
                                                     cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
