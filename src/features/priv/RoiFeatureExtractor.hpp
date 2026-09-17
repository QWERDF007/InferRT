#pragma once

/**
 * @file RoiFeatureExtractor.hpp
 * @brief ROI 特征抽取器的稳定私有接口。
 *
 * 共享语义特征和流式批处理实现位于同名 `.cpp`。
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
    RoiFeatureWorkStats workStats() const noexcept;
    size_t maxBatchSize() const noexcept;
    std::vector<float> extract(const RoiFeatureItem &item);
    std::vector<float> extractBatch(const std::vector<RoiFeatureItem> &items, size_t begin, size_t count);
    std::vector<float> extractItems(const std::vector<RoiFeatureItem> &items);
    std::vector<float> extractAll(const std::vector<RoiFeatureItem> &items,
                                  const std::function<void(size_t, size_t, size_t, size_t)> &progress = {});

    // Completed ROI rows in arbitrary original-index order, at most one crop batch per callback.
    using BatchConsumer = std::function<void(const std::vector<size_t>&, const std::vector<float>&)>;
    void extractTo(const std::vector<RoiFeatureItem>& items, const BatchConsumer& consume,
                   const std::function<void(size_t,size_t,size_t,size_t)>& progress = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features::priv
