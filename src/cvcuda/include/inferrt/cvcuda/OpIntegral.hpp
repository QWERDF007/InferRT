#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

template<typename T, typename CT>
class INFERRT_CVCUDA_API Integral final : public IOperator
{
public:
    explicit Integral();

    ~Integral();

    [[nodiscard]] IRTStatus operator()(const T *d_src, CT *d_dst, cv::Size ssize, const int CH,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_;
    }

private:
    OperatorHandle impl_;
};

} // namespace irt::cvcuda
