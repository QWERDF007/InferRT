#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>

namespace irt::cvcuda {

class INFERRT_CVCUDA_API LetterBox final : public IOperator
{
public:
    explicit LetterBox();

    ~LetterBox();

    LetterBox(const LetterBox &)            = delete;
    LetterBox &operator=(const LetterBox &) = delete;
    LetterBox(LetterBox &&) noexcept;
    LetterBox &operator=(LetterBox &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize,
                                       const int CH, cudaStream_t stream = nullptr);

    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize,
                                       const int CH, const irt::PreprocessSpec &spec,
                                       cudaStream_t stream = nullptr);

    /** Apply letterbox to a pitched HWC source buffer. Zero selects packed stride. */
    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize,
                                       const int CH, const irt::PreprocessSpec &spec, std::size_t source_stride_bytes,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
