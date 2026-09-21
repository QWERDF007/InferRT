#pragma once

/**
 * @file RoiSearch.hpp
 * @brief ROI 搜索匹配公共 API、配置类型与 ``RoiSearch`` 类声明。
 */

#include <inferrt/features/Export.h>
#include <inferrt/features/RoiFeature.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/// 默认 ROI 检索模型名称。
inline constexpr const char *kDefaultRoiSearchModelName = kDefaultRoiFeatureModelName;

/// 默认 ROI 检索特征图张量名称。
inline constexpr const char *kDefaultRoiSearchFeatureName = kDefaultRoiFeatureName;

/// 默认 ROIAlign 输出高度。
inline constexpr int kDefaultRoiSearchPooledHeight = kDefaultRoiFeaturePooledHeight;

/// 默认 ROIAlign 输出宽度。
inline constexpr int kDefaultRoiSearchPooledWidth = kDefaultRoiFeaturePooledWidth;

/**
 * @brief ROI 框，坐标基于原始输入图像像素。
 *
 * ``x2`` 与 ``y2`` 是半开区间右下边界。默认在原图裁剪后重编码，并按 ROI 覆盖权重汇聚。
 */
using RoiSearchBox = RoiFeatureBox;

/**
 * @brief ROI 特征库条目。
 */
using RoiSearchItem = RoiFeatureItem;

/**
 * @brief ROI 搜索结果。
 */
struct RoiSearchResult
{
    float   score{0.0f}; ///< 与查询 ROI 的相似度分数。
    int64_t roi_id{0};   ///< 命中的图库 ROI ID。
};

/**
 * @brief ROI 检索配置。
 *
 * 继承 ``RoiFeatureConfig`` 中的模型、预处理、归一化、ROIAlign 和 PCA 配置；搜索额外使用
 * ``ImageSearchConfig`` 提供的 Faiss 索引字段。
 */
struct RoiSearchConfig : public RoiFeatureConfig
{
    RoiSearchConfig()
    {
        model_name   = kDefaultRoiSearchModelName;
        feature_name = kDefaultRoiSearchFeatureName;
    }

    // ROIAlign/PCA 字段继承自 RoiFeatureConfig，搜索与聚类共享同一组设置。
    bool exact_search{true}; ///< 精确内积搜索（IndexFlatIP / GpuIndexFlatIP）；无量化近似损失，遵循 faiss_backend 与 model_precision。

};

using RoiSearchBuildProgress         = ImageSearchBuildProgress;
using RoiSearchBuildProgressCallback = ImageSearchBuildProgressCallback;

/**
 * @brief 默认使用 DINO 裁剪区域特征与精确 Faiss 内积的 ROI 搜索器。
 *
 * 搜索与聚类调用同一提取器。多边形通过 RoiSearchItem 重载传入；旧矩形 API 保留。
 * LegacyRoiAlign 是旧描述子对照模式。
 */
class INFERRT_FEATURES_API RoiSearch
{
public:
    /// 默认模型名称。
    static constexpr const char *kDefaultModelName = kDefaultRoiSearchModelName;

    /// 默认特征图张量名称。
    static constexpr const char *kDefaultFeatureName = kDefaultRoiSearchFeatureName;

    /// 默认返回结果数量。
    static constexpr int kDefaultTopK = ImageSearch::kDefaultTopK;

    /**
     * @brief 构造 ROI 搜索器。
     * @param config ROI 检索配置。
     */
    explicit RoiSearch(RoiSearchConfig config = {});

    /** @brief 析构 ROI 搜索器。 */
    ~RoiSearch();

    RoiSearch(const RoiSearch &)            = delete;
    RoiSearch &operator=(const RoiSearch &) = delete;

    RoiSearch(RoiSearch &&other) noexcept;
    RoiSearch &operator=(RoiSearch &&other) noexcept;

    /**
     * @brief 构建或加载 ROI 特征库索引。
     *
     * 当 ``rebuild_index`` 为 false 且索引和 manifest 均匹配当前配置及 ROI ID 序列时直接加载；否则重建。
     *
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param gallery_items 待写入特征库的 ROI 条目列表；调用方负责保证 ID 顺序稳定且唯一。
     * @param index_file Faiss 索引路径；为空时使用当前工作目录下的时间戳 ``.faiss`` 文件。
     * @param rebuild_index 是否强制重建索引。
     * @param progress_callback 可选进度回调。
     */
    void buildOrLoad(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                     const std::filesystem::path &index_file = {}, bool rebuild_index = false,
                     RoiSearchBuildProgressCallback progress_callback = {});

    /**
     * @brief 从 ROI 条目列表构建特征库索引。
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param gallery_items 待写入特征库的 ROI 条目列表。
     * @param index_file Faiss 索引路径；为空时使用当前工作目录下的时间戳 ``.faiss`` 文件。
     * @param progress_callback 可选进度回调。
     */
    void build(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
               const std::filesystem::path &index_file = {}, RoiSearchBuildProgressCallback progress_callback = {});

    /**
     * @brief 加载已构建的 ROI 特征库索引。
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param index_file Faiss 索引路径。
     */
    void load(const std::filesystem::path &weights_file, const std::filesystem::path &index_file);

    /**
     * @brief 查询单张图像中的一个 ROI。
     * @param query_image 查询图像路径。
     * @param roi 查询 ROI，坐标基于原图像素。
     * @param top_k 返回结果数量。
     * @return 按相似度从高到低排列的命中列表。
     */
    std::vector<RoiSearchResult> search(const std::filesystem::path &query_image, const RoiSearchBox &roi,
                                        int top_k = kDefaultTopK);

    /** @brief 判断索引是否已经可搜索。 */
    // Polygon query: pass polygon vertices in original image coordinates.
    std::vector<RoiSearchResult> search(const RoiSearchItem &query, int top_k = kDefaultTopK);

    bool isReady() const noexcept;

    /** @brief 获取当前配置。 */
    const RoiSearchConfig &config() const noexcept;

    /** @brief 获取当前索引路径。 */
    const std::filesystem::path &indexPath() const noexcept;

    /** @brief 获取当前特征库 ROI ID 列表。 */
    std::vector<int64_t> galleryIds() const;

    /** @brief 获取 ROI 特征向量维度。 */
    int featureDim() const noexcept;

    /**
     * @brief 生成默认 ROI 索引路径。
     * @param output_dir 索引输出目录。
     * @param model_name 模型名称。
     * @param feature_name 特征图张量名称。
     * @return ``<output_dir>/<timestamp>.faiss``。
     */
    static std::filesystem::path defaultIndexPath(const std::filesystem::path &output_dir,
                                                  const std::string &model_name, const std::string &feature_name);

    RoiFeatureWorkStats featureWorkStats() const noexcept;
    std::vector<RoiSearchResult> searchByRoiId(int64_t roi_id, int top_k = kDefaultTopK);
    std::vector<RoiSearchResult> repeatSearch(int top_k = kDefaultTopK);

private:
    class Impl;
    std::unique_ptr<Impl> impl_; ///< 私有实现，隐藏 Faiss、ROIAlign 与模型细节。
};

} // namespace irt::features
