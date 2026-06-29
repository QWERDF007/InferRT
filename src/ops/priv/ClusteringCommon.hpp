#pragma once

#include <cstdint>

namespace irt::ops::detail {

void validateSampleMatrix(const float *samples, int64_t num_samples, int64_t num_features);

[[nodiscard]] double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

[[nodiscard]] double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

} // namespace irt::ops::detail
