#pragma once

/**
 * @file RoiFeatureExtractor.hpp
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

class RoiFeatureExtractor
{
public:
    RoiFeatureExtractor(const RoiFeatureConfig &config, const std::filesystem::path &weights_file);
    ~RoiFeatureExtractor();

    RoiFeatureExtractor(const RoiFeatureExtractor &) = delete;
    RoiFeatureExtractor &operator=(const RoiFeatureExtractor &) = delete;
    RoiFeatureExtractor(RoiFeatureExtractor &&) noexcept;
    RoiFeatureExtractor &operator=(RoiFeatureExtractor &&) noexcept;

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
