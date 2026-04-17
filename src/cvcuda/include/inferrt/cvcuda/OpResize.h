#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>


namespace irt::cvcuda {

template<typename T>
[[nodiscard]] INFERRT_CVCUDA_API IRTStatus resize(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize,
                                                  const int CH, const int interpolation = cv::INTER_LINEAR,
                                                  cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
