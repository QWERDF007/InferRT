#pragma once

/**
 * @file LegacyRoiFeatureExtractor.hpp
 * @brief ROI 特征抽取器的稳定私有接口。
 *
 * 模型特征图转换、PCA、ROIAlign 和批处理实现位于同名 `.cpp`，调用方只
 * 依赖这个深模块的输入输出契约。
 */

#include <inferrt/features/RoiFeature.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace irt::features::priv {

class LegacyRoiFeatureExtractor
{
public:
    LegacyRoiFeatureExtractor(const RoiFeatureConfig &config, const std::filesystem::path &weights_file);
    ~LegacyRoiFeatureExtractor();

    LegacyRoiFeatureExtractor(const LegacyRoiFeatureExtractor &) = delete;
    LegacyRoiFeatureExtractor &operator=(const LegacyRoiFeatureExtractor &) = delete;
    LegacyRoiFeatureExtractor(LegacyRoiFeatureExtractor &&) noexcept;
    LegacyRoiFeatureExtractor &operator=(LegacyRoiFeatureExtractor &&) noexcept;

    int featureDim() const;
    size_t maxBatchSize() const noexcept;
    std::vector<float> extract(const RoiFeatureItem &item);
    std::vector<float> extractBatch(const std::vector<RoiFeatureItem> &items, size_t begin, size_t count);
    std::vector<float> extractItems(const std::vector<RoiFeatureItem> &items);
    std::vector<float> extractAll(const std::vector<RoiFeatureItem> &items,
                                  const std::function<void(size_t, size_t, size_t, size_t)> &progress = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features::priv
