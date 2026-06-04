#include "OpIntegralImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {

template<typename T, typename CT>
void IntegralImpl<T, CT>::operator()(const T *d_src, CT *d_dst, const int2 ssize, const int sstride, const int CH,
                                     cudaStream_t stream)
{
    if (d_src == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Source pointer is null");
    }
    if (d_dst == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Destination pointer is null");
    }
    if (ssize.x <= 0 || ssize.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid source size");
    }
    if (CH != 1 && CH != 3)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid channel count (must be 1 or 3)");
    }

    RunIntegral(d_src, d_dst, ssize, sstride, CH, stream);
}

template void IntegralImpl<uint8_t, uint32_t>::operator()(const uint8_t *, uint32_t *, const int2, const int,
                                                          const int, cudaStream_t);
template void IntegralImpl<float, float>::operator()(const float *, float *, const int2, const int, const int,
                                                     cudaStream_t);

} // namespace irt::cvcuda::priv
