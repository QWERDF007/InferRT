#pragma once
#include <inferrt/features/RoiFeature.hpp>
#include <memory>

namespace irt::features {
// Reusable embedding API: lets callers extract once, then use the matrix for
// evaluation or existing irt::ops::hdbscan without constructing a search index.
class INFERRT_FEATURES_API RoiFeatureEncoder {
public:
    RoiFeatureEncoder(const RoiFeatureConfig&, const std::filesystem::path& weights);
    ~RoiFeatureEncoder();
    RoiFeatureEncoder(const RoiFeatureEncoder&) = delete;
    RoiFeatureEncoder& operator=(const RoiFeatureEncoder&) = delete;
    int featureDim() const;
    std::vector<float> extract(const std::vector<RoiFeatureItem>& items);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
