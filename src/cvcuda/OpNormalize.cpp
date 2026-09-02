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
    return op(d_src, d_dst, size, channels, mean, stddev, 1.0F / 255.0F, stream);
}

IRTStatus normalize(const uint8_t *d_src, float *d_dst, const cv::Size size, const int channels, const float *mean,
                    const float *stddev, const float scale, cudaStream_t stream)
{
    Normalize op;
    return op(d_src, d_dst, size, channels, mean, stddev, scale, stream);
}

Normalize::Normalize()
    : impl_(std::make_unique<priv::NormalizeImpl>())
{
}

Normalize::~Normalize() = default;

Normalize::Normalize(Normalize &&) noexcept = default;

Normalize &Normalize::operator=(Normalize &&) noexcept = default;

IRTStatus Normalize::operator()(const uint8_t *d_src, float *d_dst, const cv::Size size, const int channels,
                                const float *mean, const float *stddev, cudaStream_t stream)
{
    return (*this)(d_src, d_dst, size, channels, mean, stddev, 1.0F / 255.0F, stream);
}

IRTStatus Normalize::operator()(const uint8_t *d_src, float *d_dst, const cv::Size size, const int channels,
                                const float *mean, const float *stddev, const float scale, cudaStream_t stream)
{
    return ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            const int2 image_size{size.width, size.height};
            auto      *normalize_impl = static_cast<priv::NormalizeImpl *>(impl_.get());
            (*normalize_impl)(d_src, d_dst, image_size, channels, mean, stddev, scale, stream);
        });
}

} // namespace irt::cvcuda
