#pragma once

/**
 * @file ImageCluster.hpp
 * @brief 图像聚类公共 API、配置类型与 ``ImageCluster`` 类声明。
 */

#include <inferrt/features/Export.h>
#include <inferrt/features/ImageSearch.hpp>
#include <inferrt/ops/HDBSCAN.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/// 默认图像聚类模型名称。
inline constexpr const char *kDefaultImageClusterModelName = kDefaultImageSearchModelName;

/// 默认图像聚类特征层名称。
inline constexpr const char *kDefaultImageClusterFeatureName = kDefaultImageSearchFeatureName;

/**
 * @brief 图像聚类输入条目。
 */
struct ImageClusterItem
{
    int64_t               image_id{0}; ///< 调用方提供的图像唯一 ID。
    std::filesystem::path image_path;  ///< 图像文件路径。
};

/**
 * @brief 单张图像的聚类分配结果。
 */
struct ImageClusterAssignment
{
    int64_t image_id{0};     ///< 调用方提供的图像 ID。
    int64_t cluster_id{-1};  ///< HDBSCAN 聚类标签；-1 表示噪声。
    double  probability{0.0}; ///< HDBSCAN 分配置信度。
};

/**
 * @brief 图像聚类进度阶段。
 */
enum class ImageClusterStage
{
    Unknown,           ///< 未知或未初始化阶段。
    Started,           ///< 聚类流程已开始。
    LoadingModel,      ///< 正在加载特征提取模型。
    ExtractingFeatures, ///< 正在抽取图像特征。
    Clustering,        ///< 正在执行 HDBSCAN 聚类。
    Finished,          ///< 聚类流程完成。
};

/**
 * @brief 将聚类阶段枚举转换为稳定日志字符串。
 */
inline const char *imageClusterStageName(ImageClusterStage stage) noexcept
{
    switch (stage)
    {
    case ImageClusterStage::Unknown:
        return "unknown";
    case ImageClusterStage::Started:
        return "started";
    case ImageClusterStage::LoadingModel:
        return "loading_model";
    case ImageClusterStage::ExtractingFeatures:
        return "extracting_features";
    case ImageClusterStage::Clustering:
        return "clustering";
    case ImageClusterStage::Finished:
        return "finished";
    }
    return "unknown";
}

/**
 * @brief 图像聚类进度信息。
 */
struct ImageClusterProgress
{
    ImageClusterStage stage{ImageClusterStage::Unknown}; ///< 当前阶段。
    size_t            batch_index{0};                    ///< 当前批次编号。
    size_t            batch_begin{0};                    ///< 当前批次起始图像下标。
    size_t            batch_count{0};                    ///< 当前批次图像数量。
    size_t            processed_count{0};                ///< 当前阶段已处理数量。
    size_t            total_count{0};                    ///< 当前阶段总数量。
};

using ImageClusterProgressCallback = std::function<void(const ImageClusterProgress &)>;

/**
 * @brief 图像聚类配置。
 *
 * 继承 ``ImageSearchConfig`` 中的模型、预处理、归一化和推理批量配置；聚类不使用 Faiss，
 * 但保留基类字段以便复用既有设置读取逻辑。
 */
struct ImageClusterConfig : public ImageSearchConfig
{
    ImageClusterConfig()
    {
        model_name   = kDefaultImageClusterModelName;
        feature_name = kDefaultImageClusterFeatureName;
    }

    bool use_pca{false}; ///< 是否对每张图自己的特征图/patch token 通道维训练本地 PCA 并降维。
    int  pca_dim{0};     ///< PCA 输出通道数；启用 PCA 时必须为正数。

    irt::ops::HDBSCANConfig hdbscan{}; ///< HDBSCAN 聚类参数。
};

/**
 * @brief 图像聚类结果。
 */
struct ImageClusterResult
{
    std::vector<ImageClusterAssignment> assignments; ///< 与输入顺序一致的图像聚类结果。
    int                                 feature_dim{0};    ///< 实际送入 HDBSCAN 的特征维度。
    int64_t                             cluster_count{0};  ///< 非噪声簇数量。
    int64_t                             noise_count{0};    ///< 噪声图像数量。
};

/**
 * @brief 基于 InferRT 图像特征和 HDBSCAN 的图像聚类器。
 */
class INFERRT_FEATURES_API ImageCluster
{
public:
    static constexpr const char *kDefaultModelName   = kDefaultImageClusterModelName;
    static constexpr const char *kDefaultFeatureName = kDefaultImageClusterFeatureName;

    explicit ImageCluster(ImageClusterConfig config = {});
    ~ImageCluster();

    ImageCluster(const ImageCluster &)            = delete;
    ImageCluster &operator=(const ImageCluster &) = delete;

    ImageCluster(ImageCluster &&other) noexcept;
    ImageCluster &operator=(ImageCluster &&other) noexcept;

    /**
     * @brief 对显式图像条目列表执行聚类。
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param items 待聚类图像条目；调用方负责提供稳定且唯一的图像 ID。
     * @param progress_callback 可选进度回调。
     * @return 聚类标签和统计信息。
     */
    ImageClusterResult cluster(const std::filesystem::path &weights_file, const std::vector<ImageClusterItem> &items,
                               ImageClusterProgressCallback progress_callback = {});

    const ImageClusterConfig &config() const noexcept;
    int                       featureDim() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features
