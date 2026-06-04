#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace irt::cvcuda::priv {

class LetterBoxImpl final : public IOperatorImpl
{
public:
    explicit LetterBoxImpl()  = default;
    ~LetterBoxImpl() override = default;

    void operator()(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                    const int CH, cudaStream_t stream);

private:
    void RunLetterBox(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                      const int CH, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
