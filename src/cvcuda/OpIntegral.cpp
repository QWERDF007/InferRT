#include "priv/OpIntegralImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpIntegral.h>
#include <inferrt/cvcuda/OpIntegral.hpp>

#include <cstdint>

namespace irt::cvcuda {

using irt::ProtectCall;

template<typename T, typename CT>
IRTStatus integral(const T *d_src, CT *d_dst, cv::Size ssize, const int CH, cudaStream_t stream)
{
    Integral<T, CT> integral_op;
    return integral_op(d_src, d_dst, ssize, CH, stream);
}

template INFERRT_CVCUDA_API IRTStatus integral<uint8_t, uint32_t>(const uint8_t *, uint32_t *, cv::Size, const int,
                                                                  cudaStream_t);
template INFERRT_CVCUDA_API IRTStatus integral<float, float>(const float *, float *, cv::Size, const int,
                                                             cudaStream_t);

template<typename T, typename CT>
Integral<T, CT>::Integral()
{
    impl_ = new priv::IntegralImpl<T, CT>();
}

template<typename T, typename CT>
Integral<T, CT>::~Integral()
{
    if (impl_)
    {
        delete impl_;
        impl_ = nullptr;
    }
}

template<typename T, typename CT>
IRTStatus Integral<T, CT>::operator()(const T *d_src, CT *d_dst, cv::Size ssize, const int CH, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            int2 _ssize;
            _ssize.x = ssize.width;
            _ssize.y = ssize.height;

            const int sstride = ssize.width * CH;

            auto *integralImpl = static_cast<priv::IntegralImpl<T, CT> *>(impl_);
            (*integralImpl)(d_src, d_dst, _ssize, sstride, CH, stream);
        });
    return status;
}

template class INFERRT_CVCUDA_API Integral<uint8_t, uint32_t>;
template class INFERRT_CVCUDA_API Integral<float, float>;

} // namespace irt::cvcuda
