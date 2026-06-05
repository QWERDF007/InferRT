#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

inline constexpr int ADAPTIVE_THRESH_PERCENTAGE = 1000;

template<typename T>
[[nodiscard]] INFERRT_CVCUDA_API IRTStatus adaptiveThreshold(const T *d_src, T *d_dst, cv::Size size, const int CH,
                                                             const double maxval, const int adaptive_method,
                                                             const int threshold_type, const int block_size,
                                                             const double param, const float *d_weights = nullptr,
                                                             cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
