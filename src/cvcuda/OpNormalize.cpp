#include "priv/OpNormalizeImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpNormalize.h>
#include <inferrt/cvcuda/OpNormalize.hpp>

namespace irt::cvcuda {

using irt::ProtectCall;

IRTStatus normalize(const uint8_t *d_src, float *d_dst, const cv::Size size, const int channels, const float *mean,
                    const float *stddev, cudaStream_t stream)
{
    Normalize op;
    return op(d_src, d_dst, size, channels, mean, stddev, stream);
}

Normalize::Normalize()
    : impl_(new priv::NormalizeImpl())
{
}

Normalize::~Normalize()
{
    delete impl_;
}

IRTStatus Normalize::operator()(const uint8_t *d_src, float *d_dst, const cv::Size size, const int channels,
                                const float *mean, const float *stddev, cudaStream_t stream)
{
    return ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            const int2 image_size{size.width, size.height};
            auto      *normalize_impl = static_cast<priv::NormalizeImpl *>(impl_);
            (*normalize_impl)(d_src, d_dst, image_size, channels, mean, stddev, stream);
        });
}

} // namespace irt::cvcuda
