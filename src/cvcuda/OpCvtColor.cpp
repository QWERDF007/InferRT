#include "priv/OpCvtColorImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpCvtColor.h>
#include <inferrt/cvcuda/OpCvtColor.hpp>

#include <cstdint>

namespace irt::cvcuda {

using irt::ProtectCall;

template<typename T>
IRTStatus cvtColor(const T *d_src, T *d_dst, cv::Size size, const int code, cudaStream_t stream)
{
    CvtColor<T> cvt_color_op;
    return cvt_color_op(d_src, d_dst, size, code, stream);
}

template INFERRT_CVCUDA_API IRTStatus cvtColor<uint8_t>(const uint8_t *, uint8_t *, cv::Size, const int,
                                                        cudaStream_t);
template INFERRT_CVCUDA_API IRTStatus cvtColor<float>(const float *, float *, cv::Size, const int, cudaStream_t);

template<typename T>
CvtColor<T>::CvtColor()
{
    impl_ = std::make_unique<priv::CvtColorImpl<T>>();
}

template<typename T>
CvtColor<T>::~CvtColor() = default;

template<typename T>
CvtColor<T>::CvtColor(CvtColor &&) noexcept = default;

template<typename T>
CvtColor<T> &CvtColor<T>::operator=(CvtColor &&) noexcept = default;

template<typename T>
IRTStatus CvtColor<T>::operator()(const T *d_src, T *d_dst, cv::Size size, const int code, cudaStream_t stream)
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

            auto *cvtColorImpl = static_cast<priv::CvtColorImpl<T> *>(impl_.get());
            (*cvtColorImpl)(d_src, d_dst, _size, code, stream);
        });
    return status;
}

template class INFERRT_CVCUDA_API CvtColor<uint8_t>;
template class INFERRT_CVCUDA_API CvtColor<float>;

} // namespace irt::cvcuda
