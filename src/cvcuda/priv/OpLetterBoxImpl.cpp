#include "OpLetterBoxImpl.hpp"

#include <inferrt/core/Exception.hpp>

namespace irt::cvcuda::priv {

void LetterBoxImpl::operator()(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride,
                               const int2 dsize, const int CH, cudaStream_t stream)
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
    if (dsize.x <= 0 || dsize.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid destination size");
    }
    if (CH != 1 && CH != 3)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid channel count (must be 1 or 3)");
    }

    RunLetterBox(d_src, d_dst, ssize, sstride, dsize, CH, stream);
}

} // namespace irt::cvcuda::priv
