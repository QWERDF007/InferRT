#include "priv/OpLetterBoxImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpLetterBox.h>
#include <inferrt/cvcuda/OpLetterBox.hpp>

namespace irt::cvcuda {

using irt::ProtectCall;

IRTStatus letterBox(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                    cudaStream_t stream)
{
    LetterBox letter_box_op;
    return letter_box_op(d_src, d_dst, ssize, dsize, CH, stream);
}

IRTStatus letter_box(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                     cudaStream_t stream)
{
    return letterBox(d_src, d_dst, ssize, dsize, CH, stream);
}

LetterBox::LetterBox()
{
    impl_ = new priv::LetterBoxImpl();
}

LetterBox::~LetterBox()
{
    if (impl_)
    {
        delete impl_;
        impl_ = nullptr;
    }
}

IRTStatus LetterBox::operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                cudaStream_t stream)
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

            int2 _dsize;
            _dsize.x = dsize.width;
            _dsize.y = dsize.height;

            const int sstride = ssize.width * CH;

            auto *letterBoxImpl = static_cast<priv::LetterBoxImpl *>(impl_);
            (*letterBoxImpl)(d_src, d_dst, _ssize, sstride, _dsize, CH, stream);
        });
    return status;
}

} // namespace irt::cvcuda
