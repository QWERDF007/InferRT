#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

template<typename T>
class INFERRT_CVCUDA_API Resize final : public IOperator
{
public:
    explicit Resize();

    ~Resize();

    Resize(const Resize &)            = delete;
    Resize &operator=(const Resize &) = delete;
    Resize(Resize &&) noexcept;
    Resize &operator=(Resize &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                       const int interpolation = cv::INTER_LINEAR, cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
