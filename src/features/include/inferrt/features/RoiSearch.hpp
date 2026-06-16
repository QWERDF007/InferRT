#pragma once

/**
 * @file RoiSearch.hpp
 * @brief ROI 搜索匹配公共 API、配置类型与 ``RoiSearch`` 类声明。
 */

#include <inferrt/features/Export.h>
#include <inferrt/features/ImageSearch.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/// 默认 ROI 检索模型名称。
inline constexpr const char *kDefaultRoiSearchModelName = kDefaultImageSearchModelName;

/// 默认 ROI 检索特征图张量名称。
inline constexpr const char *kDefaultRoiSearchFeatureName = kDefaultImageSearchFeatureName;

/// 默认 ROIAlign 输出高度。
inline constexpr int kDefaultRoiSearchPooledHeight = 7;

/// 默认 ROIAlign 输出宽度。
inline constexpr int kDefaultRoiSearchPooledWidth = 7;

/**
 * @brief ROI 框，坐标基于原始输入图像像素。
 *
 * ``x2`` 与 ``y2`` 表示右下边界，要求分别大于 ``x1`` 与 ``y1``。实现会按原图尺寸把该 ROI
 * 映射到模型输出特征图坐标，再执行 ROIAlign。
 */
struct RoiSearchBox
{
    float x1{0.0f}; ///< 左上角 x 坐标。
    float y1{0.0f}; ///< 左上角 y 坐标。
    float x2{0.0f}; ///< 右下边界 x 坐标。
    float y2{0.0f}; ///< 右下边界 y 坐标。
};

/**
 * @brief ROI 特征库条目。
 */
struct RoiSearchItem
{
    std::filesystem::path image_path; ///< ROI 所属图像路径。
    RoiSearchBox          roi;        ///< 原图坐标系下的 ROI。
};

/**
 * @brief ROI 搜索结果。
 */
struct RoiSearchResult
{
    float                 score{0.0f};   ///< 与查询 ROI 的相似度分数。
    std::filesystem::path image_path;    ///< 命中的图库图像路径。
    RoiSearchBox          roi;           ///< 命中的图库 ROI。
    size_t                item_index{0}; ///< 命中条目在特征库中的下标。
};

/**
 * @brief ROI 检索配置。
 *
 * 继承 ``ImageSearchConfig`` 中的模型、预处理、归一化和 Faiss 配置；新增字段控制 ROIAlign
 * 的统一输出空间大小。
 */
struct RoiSearchConfig : public ImageSearchConfig
{
    RoiSearchConfig()
    {
        model_name   = kDefaultRoiSearchModelName;
        feature_name = kDefaultRoiSearchFeatureName;
    }

    int  pooled_height{kDefaultRoiSearchPooledHeight}; ///< ROIAlign 输出高度。
    int  pooled_width{kDefaultRoiSearchPooledWidth};   ///< ROIAlign 输出宽度。
    int  sampling_ratio{-1};                           ///< ROIAlign 采样率，-1 表示自适应。
    bool aligned{false};                               ///< 是否使用 aligned ROIAlign 坐标规则。
};

using RoiSearchBuildProgress         = ImageSearchBuildProgress;
using RoiSearchBuildProgressCallback = ImageSearchBuildProgressCallback;

/**
 * @brief 基于特征图 ROIAlign 与 Faiss 的 ROI 搜索匹配器。
 *
 * 流程为：模型抽取空间特征图，将原图 ROI 映射到特征图坐标，使用 ROIAlign 统一 ROI 特征大小，
 * 展平并归一化后写入 Faiss 特征库；查询时对输入 ROI 执行相同处理并返回 Top-K 命中。
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
     * 当 ``rebuild_index`` 为 false 且索引、ROI 映射和元数据均匹配当前配置时直接加载；否则重建。
     *
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param gallery_items 待写入特征库的 ROI 条目列表。
     * @param index_file Faiss 索引路径，不可为空。
     * @param rebuild_index 是否强制重建索引。
     * @param progress_callback 可选进度回调。
     */
    void buildOrLoad(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                     const std::filesystem::path &index_file, bool rebuild_index = false,
                     RoiSearchBuildProgressCallback progress_callback = {});

    /**
     * @brief 从 ROI 条目列表构建特征库索引。
     * @param weights_file 模型权重、engine 或图模型文件。
     * @param gallery_items 待写入特征库的 ROI 条目列表。
     * @param index_file Faiss 索引路径，不可为空。
     * @param progress_callback 可选进度回调。
     */
    void build(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
               const std::filesystem::path &index_file, RoiSearchBuildProgressCallback progress_callback = {});

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
    bool isReady() const noexcept;

    /** @brief 获取当前配置。 */
    const RoiSearchConfig &config() const noexcept;

    /** @brief 获取当前索引路径。 */
    const std::filesystem::path &indexPath() const noexcept;

    /** @brief 获取当前特征库 ROI 条目列表。 */
    std::vector<RoiSearchItem> galleryItems() const;

    /** @brief 获取 ROI 特征向量维度。 */
    int featureDim() const noexcept;

    /**
     * @brief 生成默认 ROI 索引路径。
     * @param output_dir 索引输出目录。
     * @param model_name 模型名称。
     * @param feature_name 特征图张量名称。
     * @return ``<output_dir>/<model>_<feature>.roi.faiss``。
     */
    static std::filesystem::path defaultIndexPath(const std::filesystem::path &output_dir,
                                                  const std::string &model_name, const std::string &feature_name);

private:
    class Impl;
    std::unique_ptr<Impl> impl_; ///< 私有实现，隐藏 Faiss、ROIAlign 与模型细节。
};

} // namespace irt::features
