#pragma once

#include <inferrt/core/Status.h>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace irt::ops {

class INFERRT_OPS_API RoIAlign final
{
public:
    RoIAlign(int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned);

    explicit RoIAlign(std::pair<int, int> output_size, float spatial_scale = 1.0f, int sampling_ratio = -1,
                      bool aligned = false);

    [[nodiscard]] int   pooledHeight() const noexcept;
    [[nodiscard]] int   pooledWidth() const noexcept;
    [[nodiscard]] float spatialScale() const noexcept;
    [[nodiscard]] int   samplingRatio() const noexcept;
    [[nodiscard]] bool  aligned() const noexcept;

    [[nodiscard]] std::pair<int, int> outputSize() const noexcept;

    void forward(const float *input, const int64_t input_shape[4], const float *rois, int64_t num_rois,
                 float *output) const;

    [[nodiscard]] std::vector<float> forward(const float *input, const int64_t input_shape[4], const float *rois,
                                             int64_t num_rois) const;

private:
    int   pooled_height_{0};
    int   pooled_width_{0};
    float spatial_scale_{1.0f};
    int   sampling_ratio_{-1};
    bool  aligned_{false};
};

} // namespace irt::ops
