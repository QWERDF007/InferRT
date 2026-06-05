#include "OpAdaptiveThresholdImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpAdaptiveThreshold.h>
#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {

template<typename T>
void AdaptiveThresholdImpl<T>::operator()(const T *d_src, T *d_dst, const int2 size, const int sstride,
                                          const int dstride, const int CH, const double maxval,
                                          const int adaptive_method, const int threshold_type, const int block_size,
                                          const double param, const float *d_weights, cudaStream_t stream)
{
    if (d_src == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Source pointer is null");
    }
    if (d_dst == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Destination pointer is null");
    }
    if (size.x <= 0 || size.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid image size");
    }
    if (CH != 1 && CH != 3)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid channel count (must be 1 or 3)");
    }
    if (block_size <= 1 || (block_size % 2) == 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Block size must be odd and greater than 1");
    }
    if (threshold_type != cv::THRESH_BINARY && threshold_type != cv::THRESH_BINARY_INV)
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Threshold type not implemented: %d", threshold_type);
    }
    if (adaptive_method != cv::ADAPTIVE_THRESH_MEAN_C && adaptive_method != cv::ADAPTIVE_THRESH_GAUSSIAN_C
        && adaptive_method != irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE)
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Adaptive threshold method not implemented: %d",
                        adaptive_method);
    }
    if (adaptive_method == cv::ADAPTIVE_THRESH_GAUSSIAN_C && d_weights == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Gaussian weights pointer is null");
    }

    RunAdaptiveThreshold(d_src, d_dst, size, sstride, dstride, CH, maxval, adaptive_method, threshold_type, block_size,
                         param, d_weights, stream);
}

template void AdaptiveThresholdImpl<uint8_t>::operator()(const uint8_t *, uint8_t *, const int2, const int, const int,
                                                         const int, const double, const int, const int, const int,
                                                         const double, const float *, cudaStream_t);

} // namespace irt::cvcuda::priv
