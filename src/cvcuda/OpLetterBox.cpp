#include "priv/OpLetterBoxImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/cvcuda/OpLetterBox.h>
#include <inferrt/cvcuda/OpLetterBox.hpp>

#include <limits>

namespace irt::cvcuda {

using irt::ProtectCall;

namespace {

irt::PreprocessSpec defaultSpec(const cv::Size dsize, const int channels)
{
    irt::PreprocessSpec spec;
    spec.input_width      = dsize.width;
    spec.input_height     = dsize.height;
    spec.input_channels   = channels;
    spec.source_channels  = channels;
    spec.src_color        = channels == 1 ? irt::ColorFormat::GRAY
                                          : channels == 4 ? irt::ColorFormat::BGRA : irt::ColorFormat::BGR;
    spec.dst_color        = channels == 1 ? irt::ColorFormat::GRAY
                                          : channels == 4 ? irt::ColorFormat::RGBA : irt::ColorFormat::RGB;
    spec.padding_mode     = irt::PaddingMode::Letterbox;
    spec.mean.assign(static_cast<size_t>(channels), 0.0F);
    spec.stddev.assign(static_cast<size_t>(channels), 1.0F);
    return spec;
}

priv::LetterBoxImpl::Parameters makeParameters(const irt::PreprocessSpec &spec, const cv::Size ssize,
                                               const cv::Size dsize, const int channels)
{
    spec.validate();
    if (spec.input_width != dsize.width || spec.input_height != dsize.height
        || (spec.source_width != 0
            && (spec.source_width != ssize.width || spec.source_height != ssize.height)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "LetterBox PreprocessSpec dimensions do not match source/destination tensors");
    }
    if (spec.padding_mode != irt::PaddingMode::Letterbox || spec.interpolation != irt::Interpolation::Linear)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "CUDA letterbox supports only linear interpolation and Letterbox padding");
    }
    if (irt::colorChannels(spec.src_color) != channels || irt::colorChannels(spec.dst_color) != channels
        || (channels != 1 && channels != 3 && channels != 4))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "CUDA letterbox requires matching 1, 3 or 4 channel color formats");
    }

    priv::LetterBoxImpl::Parameters parameters;
    parameters.pad_value           = spec.pad_value;
    parameters.scale               = spec.scale;
    parameters.pad_after_normalize = spec.pad_after_normalize;
    parameters.padding_alignment   = spec.padding_alignment;
    for (int channel = 0; channel < channels; ++channel)
    {
        parameters.mean[static_cast<size_t>(channel)]   = spec.mean[static_cast<size_t>(channel)];
        parameters.stddev[static_cast<size_t>(channel)] = spec.stddev[static_cast<size_t>(channel)];
        parameters.channel_map[static_cast<size_t>(channel)] = channel;
    }
    const bool swap_rb = (spec.src_color == irt::ColorFormat::BGR && spec.dst_color == irt::ColorFormat::RGB)
                      || (spec.src_color == irt::ColorFormat::RGB && spec.dst_color == irt::ColorFormat::BGR)
                      || (spec.src_color == irt::ColorFormat::BGRA && spec.dst_color == irt::ColorFormat::RGBA)
                      || (spec.src_color == irt::ColorFormat::RGBA && spec.dst_color == irt::ColorFormat::BGRA);
    if (swap_rb)
    {
        parameters.channel_map[0] = 2;
        parameters.channel_map[2] = 0;
    }
    return parameters;
}

} // namespace

IRTStatus letterBox(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                    cudaStream_t stream)
{
    LetterBox letter_box_op;
    return letter_box_op(d_src, d_dst, ssize, dsize, CH, defaultSpec(dsize, CH), stream);
}

IRTStatus letterBox(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                    const irt::PreprocessSpec &spec, cudaStream_t stream)
{
    LetterBox letter_box_op;
    return letter_box_op(d_src, d_dst, ssize, dsize, CH, spec, stream);
}

IRTStatus letter_box(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                     cudaStream_t stream)
{
    return letterBox(d_src, d_dst, ssize, dsize, CH, stream);
}

LetterBox::LetterBox()
{
    impl_ = std::make_unique<priv::LetterBoxImpl>();
}

LetterBox::~LetterBox() = default;

LetterBox::LetterBox(LetterBox &&) noexcept = default;

LetterBox &LetterBox::operator=(LetterBox &&) noexcept = default;

IRTStatus LetterBox::operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                cudaStream_t stream)
{
    return (*this)(d_src, d_dst, ssize, dsize, CH, defaultSpec(dsize, CH), stream);
}

IRTStatus LetterBox::operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                const irt::PreprocessSpec &spec, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            if (d_src == nullptr || d_dst == nullptr || ssize.width <= 0 || ssize.height <= 0 || dsize.width <= 0
                || dsize.height <= 0)
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "LetterBox source/destination is invalid");
            }
            const size_t stride = irt::checkedSizeMul(static_cast<size_t>(ssize.width), static_cast<size_t>(CH),
                                                      "LetterBox source stride");
            if (stride > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "LetterBox source stride is too large");
            }

            int2 _ssize{ssize.width, ssize.height};
            int2 _dsize{dsize.width, dsize.height};
            const auto geometry = irt::resolvePreprocessGeometry(spec, ssize.width, ssize.height);
            const auto parameters = makeParameters(spec, ssize, dsize, CH);

            auto *letterBoxImpl = static_cast<priv::LetterBoxImpl *>(impl_.get());
            (*letterBoxImpl)(d_src, d_dst, _ssize, static_cast<int>(stride), _dsize, CH, parameters, geometry, stream);
        });
    return status;
}

} // namespace irt::cvcuda
