#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

namespace irt::cvcuda::priv {

template<typename T>
class CvtColorImpl final : public IOperatorImpl
{
public:
    explicit CvtColorImpl() = default;
    ~CvtColorImpl() override = default;

    void operator()(const T *d_src, T *d_dst, const int2 size, const int code, cudaStream_t stream);

private:
    void RunCvtColor(const T *d_src, T *d_dst, const int2 size, const int code, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
