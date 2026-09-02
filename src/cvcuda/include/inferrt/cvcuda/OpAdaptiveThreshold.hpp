#pragma once

#include "IOperator.hpp"
#include "OpAdaptiveThreshold.h"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

template<typename T>
class INFERRT_CVCUDA_API AdaptiveThreshold final : public IOperator
{
public:
    explicit AdaptiveThreshold();

    ~AdaptiveThreshold();

    AdaptiveThreshold(const AdaptiveThreshold &)            = delete;
    AdaptiveThreshold &operator=(const AdaptiveThreshold &) = delete;
    AdaptiveThreshold(AdaptiveThreshold &&) noexcept;
    AdaptiveThreshold &operator=(AdaptiveThreshold &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const T *d_src, T *d_dst, cv::Size size, const int CH, const double maxval,
                                       const int adaptive_method, const int threshold_type, const int block_size,
                                       const double param, const float *d_weights = nullptr,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
