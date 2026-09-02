#include "priv/OpIntegralImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/cvcuda/OpIntegral.h>
#include <inferrt/cvcuda/OpIntegral.hpp>

#include <cstdint>
#include <limits>

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
    impl_ = std::make_unique<priv::IntegralImpl<T, CT>>();
}

template<typename T, typename CT>
Integral<T, CT>::~Integral() = default;

template<typename T, typename CT>
Integral<T, CT>::Integral(Integral &&) noexcept = default;

template<typename T, typename CT>
Integral<T, CT> &Integral<T, CT>::operator=(Integral &&) noexcept = default;

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

            if (ssize.width <= 0 || ssize.height <= 0 || (CH != 1 && CH != 3))
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid Integral source shape or channel count");
            }
            if (ssize.width == std::numeric_limits<int>::max())
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "Integral output width exceeds int range");
            }
            (void)irt::checkedSizeToInt(
                irt::checkedSizeMul(static_cast<size_t>(ssize.width), static_cast<size_t>(CH),
                                    "Integral source stride"),
                "Integral source stride");

            int2 _ssize;
            _ssize.x = ssize.width;
            _ssize.y = ssize.height;

            const int sstride = ssize.width * CH;

            auto *integralImpl = static_cast<priv::IntegralImpl<T, CT> *>(impl_.get());
            (*integralImpl)(d_src, d_dst, _ssize, sstride, CH, stream);
        });
    return status;
}

template class INFERRT_CVCUDA_API Integral<uint8_t, uint32_t>;
template class INFERRT_CVCUDA_API Integral<float, float>;

} // namespace irt::cvcuda
