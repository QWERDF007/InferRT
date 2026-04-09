#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda {

class Resize final : public IOperator
{
public:
    explicit Resize();

    ~Resize() = default;

    template<typename T>
    [[nodiscard]] int operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                 const int interpolation = cv::INTER_LINEAR, cudaStream_t stream = nullptr);

private:
    std::unique_ptr<priv::IOperatorImpl> impl_;
};

} // namespace irt::cvcuda