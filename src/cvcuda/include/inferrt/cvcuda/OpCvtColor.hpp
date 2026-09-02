#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

template<typename T>
class INFERRT_CVCUDA_API CvtColor final : public IOperator
{
public:
    explicit CvtColor();

    ~CvtColor();

    CvtColor(const CvtColor &)            = delete;
    CvtColor &operator=(const CvtColor &) = delete;
    CvtColor(CvtColor &&) noexcept;
    CvtColor &operator=(CvtColor &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const T *d_src, T *d_dst, cv::Size size, const int code,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
