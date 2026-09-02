/**
 * @file OpNormalize.hpp
 * @brief Normalize 算子的 C++ 包装接口。
 */

#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/core.hpp>

#include <cstdint>

namespace irt::cvcuda {

class INFERRT_CVCUDA_API Normalize final : public IOperator
{
public:
    Normalize();
    ~Normalize() override;

    Normalize(const Normalize &)            = delete;
    Normalize &operator=(const Normalize &) = delete;
    Normalize(Normalize &&) noexcept;
    Normalize &operator=(Normalize &&) noexcept;

    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size size, int channels,
                                       const float *mean, const float *stddev, cudaStream_t stream = nullptr);

    [[nodiscard]] IRTStatus operator()(const uint8_t *d_src, float *d_dst, cv::Size size, int channels,
                                       const float *mean, const float *stddev, float scale,
                                       cudaStream_t stream = nullptr);

    OperatorHandle handle() const noexcept override
    {
        return impl_.get();
    }

private:
    OperatorImplPtr impl_;
};

} // namespace irt::cvcuda
