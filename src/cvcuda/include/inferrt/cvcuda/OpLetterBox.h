#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

#include <cstdint>

namespace irt::cvcuda {

[[nodiscard]] INFERRT_CVCUDA_API IRTStatus letterBox(const uint8_t *d_src, float *d_dst, cv::Size ssize,
                                                     cv::Size dsize, const int CH,
                                                     cudaStream_t stream = nullptr);

/** Apply CUDA letterbox using the shared backend-neutral preprocessing contract. */
[[nodiscard]] INFERRT_CVCUDA_API IRTStatus letterBox(const uint8_t *d_src, float *d_dst, cv::Size ssize,
                                                     cv::Size dsize, const int CH,
                                                     const irt::PreprocessSpec &spec,
                                                     cudaStream_t stream = nullptr);

[[nodiscard]] INFERRT_CVCUDA_API IRTStatus letter_box(const uint8_t *d_src, float *d_dst, cv::Size ssize,
                                                      cv::Size dsize, const int CH,
                                                      cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
