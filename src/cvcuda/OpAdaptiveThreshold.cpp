#include "priv/OpAdaptiveThresholdImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpAdaptiveThreshold.h>
#include <inferrt/cvcuda/OpAdaptiveThreshold.hpp>

#include <cstdint>

namespace irt::cvcuda {

using irt::ProtectCall;

template<typename T>
IRTStatus adaptiveThreshold(const T *d_src, T *d_dst, cv::Size size, const int CH, const double maxval,
                            const int adaptive_method, const int threshold_type, const int block_size,
                            const double param, const float *d_weights, cudaStream_t stream)
{
    AdaptiveThreshold<T> adaptive_threshold_op;
    return adaptive_threshold_op(d_src, d_dst, size, CH, maxval, adaptive_method, threshold_type, block_size, param,
                                 d_weights, stream);
}

template INFERRT_CVCUDA_API IRTStatus adaptiveThreshold<uint8_t>(const uint8_t *, uint8_t *, cv::Size, const int,
                                                                 const double, const int, const int, const int,
                                                                 const double, const float *, cudaStream_t);

template<typename T>
AdaptiveThreshold<T>::AdaptiveThreshold()
{
    impl_ = std::make_unique<priv::AdaptiveThresholdImpl<T>>();
}

template<typename T>
AdaptiveThreshold<T>::~AdaptiveThreshold() = default;

template<typename T>
AdaptiveThreshold<T>::AdaptiveThreshold(AdaptiveThreshold &&) noexcept = default;

template<typename T>
AdaptiveThreshold<T> &AdaptiveThreshold<T>::operator=(AdaptiveThreshold &&) noexcept = default;

template<typename T>
IRTStatus AdaptiveThreshold<T>::operator()(const T *d_src, T *d_dst, cv::Size size, const int CH, const double maxval,
                                           const int adaptive_method, const int threshold_type, const int block_size,
                                           const double param, const float *d_weights, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            int2 _size;
            _size.x = size.width;
            _size.y = size.height;

            const int sstride = size.width * CH;
            const int dstride = size.width * CH;

            auto *adaptiveThresholdImpl = static_cast<priv::AdaptiveThresholdImpl<T> *>(impl_.get());
            (*adaptiveThresholdImpl)(d_src, d_dst, _size, sstride, dstride, CH, maxval, adaptive_method, threshold_type,
                                     block_size, param, d_weights, stream);
        });
    return status;
}

template class INFERRT_CVCUDA_API AdaptiveThreshold<uint8_t>;

} // namespace irt::cvcuda
