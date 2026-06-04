#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

#include <cstdint>

namespace irt::cvcuda {

class INFERRT_CVCUDA_API LetterBox final : public IOperator
{
public:
    explicit LetterBox();

    ~LetterBox();

    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size ssize, cv::Size dsize,
                                       const int CH, cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_;
    }

private:
    OperatorHandle impl_;
};

} // namespace irt::cvcuda
