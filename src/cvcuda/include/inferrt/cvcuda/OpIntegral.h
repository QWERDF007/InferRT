#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

template<typename T, typename CT>
[[nodiscard]] INFERRT_CVCUDA_API IRTStatus integral(const T *d_src, CT *d_dst, cv::Size ssize, const int CH,
                                                    cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
