#pragma once

#include <cuda_runtime.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace inferrt { namespace cvcuda {

template<typename T>
INFERRT_CVCUDA_API void resize(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                               const int interpolation = cv::INTER_LINEAR, cudaStream_t stream = nullptr);

}} // namespace inferrt::cvcuda
