#include "OpCvtColorImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {

template<typename T>
void CvtColorImpl<T>::operator()(const T *d_src, T *d_dst, const int2 size, const int code, cudaStream_t stream)
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

    RunCvtColor(d_src, d_dst, size, code, stream);
}

template void CvtColorImpl<uint8_t>::operator()(const uint8_t *, uint8_t *, const int2, const int, cudaStream_t);
template void CvtColorImpl<float>::operator()(const float *, float *, const int2, const int, cudaStream_t);

} // namespace irt::cvcuda::priv
