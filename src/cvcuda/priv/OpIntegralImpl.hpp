#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

namespace irt::cvcuda::priv {

template<typename T, typename CT>
class IntegralImpl final : public IOperatorImpl
{
public:
    explicit IntegralImpl() = default;
    ~IntegralImpl() override = default;

    void operator()(const T *d_src, CT *d_dst, const int2 ssize, const int sstride, const int CH,
                    cudaStream_t stream);

private:
    void RunIntegral(const T *d_src, CT *d_dst, const int2 ssize, const int sstride, const int CH,
                     cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
