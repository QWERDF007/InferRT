#include "OpNormalizeImpl.hpp"

#include <inferrt/core/Exception.hpp>

namespace irt::cvcuda::priv {

void NormalizeImpl::operator()(const uint8_t *d_src, float *d_dst, const int2 size, const int channels,
                               const float *mean, const float *stddev, cudaStream_t stream)
{
    if (d_src == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Source pointer is null");
    }
    if (d_dst == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Destination pointer is null");
    }
    if (mean == nullptr || stddev == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Normalization mean and stddev must not be null");
    }
    if (size.x <= 0 || size.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid image size");
    }
    if (channels != 1 && channels != 3)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channel count must be 1 or 3");
    }
    for (int channel = 0; channel < channels; ++channel)
    {
        if (stddev[channel] == 0.0F)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Normalization stddev must not contain zero");
        }
    }

    run(d_src, d_dst, size, channels, mean, stddev, stream);
}

} // namespace irt::cvcuda::priv
