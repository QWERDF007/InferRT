#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/PreprocessSpec.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {

class LetterBoxImpl final : public IOperatorImpl
{
public:
    struct Parameters
    {
        float pad_value{114.0F};
        float scale{1.0F / 255.0F};
        bool  pad_after_normalize{false};
        irt::PaddingAlignment padding_alignment{irt::PaddingAlignment::Center};
        float mean[4]{0.0F, 0.0F, 0.0F, 0.0F};
        float stddev[4]{1.0F, 1.0F, 1.0F, 1.0F};
        int   channel_map[4]{0, 1, 2, 3};
    };

    explicit LetterBoxImpl()  = default;
    ~LetterBoxImpl() override = default;

    void operator()(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                    const int CH, const Parameters &parameters, const irt::PreprocessGeometry &geometry,
                    cudaStream_t stream);

private:
    void RunLetterBox(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                      const int CH, const Parameters &parameters, const irt::PreprocessGeometry &geometry,
                      cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
