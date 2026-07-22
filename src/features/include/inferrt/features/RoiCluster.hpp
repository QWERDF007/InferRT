#pragma once

/**
 * @file RoiCluster.hpp
 * @brief ROI 特征提取、ROIAlign 与 HDBSCAN 聚类公共 API。
 */

#include <inferrt/features/Export.h>
#include <inferrt/features/RoiFeature.hpp>
#include <inferrt/ops/HDBSCAN.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace irt::features {

/// 默认 ROI 聚类模型名称。
inline constexpr const char *kDefaultRoiClusterModelName = kDefaultRoiFeatureModelName;

/// 默认 ROI 聚类特征图张量名称。
inline constexpr const char *kDefaultRoiClusterFeatureName = kDefaultRoiFeatureName;

/// ROI 聚类输入条目；与 ROI 搜索共用 ROI 坐标和图像字段。
using RoiClusterItem = RoiFeatureItem;

/// ROI 聚类框类型。
using RoiClusterBox = RoiFeatureBox;

/** @brief ROI 聚类单条结果。 */
struct RoiClusterAssignment
{
    int64_t roi_id{0};       ///< 输入 ROI ID。
    int64_t cluster_id{-1};  ///< HDBSCAN 聚类标签；-1 表示噪声。
    double  probability{0.0}; ///< HDBSCAN 分配置信度。
};

/** @brief ROI 聚类进度阶段。 */
enum class RoiClusterStage
{
    Unknown,            ///< 未知阶段。
    LoadingModel,       ///< 正在加载模型。
    ExtractingFeatures, ///< 正在提取图像特征图并执行 ROIAlign。
    Clustering,         ///< 正在执行 HDBSCAN。
};

/** @brief 将 ROI 聚类阶段转换为日志字符串。 */
inline const char *roiClusterStageName(RoiClusterStage stage) noexcept
{
    switch (stage)
    {
    case RoiClusterStage::Unknown:
        return "unknown";
    case RoiClusterStage::LoadingModel:
        return "loading_model";
    case RoiClusterStage::ExtractingFeatures:
        return "extracting_features";
    case RoiClusterStage::Clustering:
        return "clustering";
    }
    return "unknown";
}

/** @brief ROI 聚类进度信息。 */
struct RoiClusterProgress
{
    RoiClusterStage stage{RoiClusterStage::Unknown}; ///< 当前阶段。
    size_t          batch_index{0};                  ///< 当前 batch 编号。
    size_t          batch_begin{0};                  ///< 当前 batch 起始 ROI 下标。
    size_t          batch_count{0};                  ///< 当前 batch ROI 数量。
    size_t          processed_count{0};              ///< 当前阶段已处理数量。
    size_t          total_count{0};                  ///< 当前阶段总数量。
};

using RoiClusterProgressCallback = std::function<void(const RoiClusterProgress &)>;

/** @brief ROI 聚类配置。 */
struct RoiClusterConfig : public RoiFeatureConfig
{
    RoiClusterConfig()
    {
        model_name   = kDefaultRoiClusterModelName;
        feature_name = kDefaultRoiClusterFeatureName;
    }

    irt::ops::HDBSCANConfig hdbscan{}; ///< HDBSCAN 参数。
};

/** @brief ROI 聚类结果。 */
struct RoiClusterResult
{
    std::vector<RoiClusterAssignment> assignments; ///< 与输入 ROI 顺序一致的聚类结果。
    int                              feature_dim{0}; ///< HDBSCAN 输入特征维度。
    int64_t                          cluster_count{0}; ///< 非噪声簇数量。
    int64_t                          noise_count{0}; ///< 噪声 ROI 数量。
};

/**
 * @brief 使用模型特征图、ROIAlign 和 HDBSCAN 对 ROI 进行聚类。
 *
 * 构建特征时会按图像路径分组，同一张图像只执行一次模型前向；该图像的多个 ROI
 * 在共享特征图上批量执行 ROIAlign。
 */
class INFERRT_FEATURES_API RoiCluster
{
public:
    static constexpr const char *kDefaultModelName   = kDefaultRoiClusterModelName;
    static constexpr const char *kDefaultFeatureName = kDefaultRoiClusterFeatureName;

    explicit RoiCluster(RoiClusterConfig config = {});
    ~RoiCluster();

    RoiCluster(const RoiCluster &)            = delete;
    RoiCluster &operator=(const RoiCluster &) = delete;

    RoiCluster(RoiCluster &&other) noexcept;
    RoiCluster &operator=(RoiCluster &&other) noexcept;

    /** @brief 对 ROI 条目执行特征提取和 HDBSCAN 聚类。 */
    RoiClusterResult cluster(const std::filesystem::path &weights_file,
                             const std::vector<RoiClusterItem> &items,
                             RoiClusterProgressCallback progress_callback = {});

    const RoiClusterConfig &config() const noexcept;
    int                     featureDim() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features
