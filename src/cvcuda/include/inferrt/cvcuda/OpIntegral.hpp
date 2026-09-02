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

    Integral(const Integral &)            = delete;
    Integral &operator=(const Integral &) = delete;
    Integral(Integral &&) noexcept;
    Integral &operator=(Integral &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const T *d_src, CT *d_dst, cv::Size ssize, const int CH,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
