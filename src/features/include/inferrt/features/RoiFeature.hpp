#pragma once

/**
 * @file RoiFeature.hpp
 * @brief ROI 特征提取共用的数据结构与配置。
 */

#include <inferrt/features/ImageSearch.hpp>

#include <cstdint>
#include <filesystem>

namespace irt::features {

/// 默认 ROI 特征模型名称。
inline constexpr const char *kDefaultRoiFeatureModelName = kDefaultImageSearchModelName;

/// 默认 ROI 特征张量名称。
inline constexpr const char *kDefaultRoiFeatureName = kDefaultImageSearchFeatureName;

/// 默认 ROIAlign 输出高度。
inline constexpr int kDefaultRoiFeaturePooledHeight = 7;

/// 默认 ROIAlign 输出宽度。
inline constexpr int kDefaultRoiFeaturePooledWidth = 7;

/**
 * @brief 原图像素坐标系下的 ROI 框。
 */
struct RoiFeatureBox
{
    float x1{0.0f}; ///< 左上角 x 坐标。
    float y1{0.0f}; ///< 左上角 y 坐标。
    float x2{0.0f}; ///< 右下角 x 坐标。
    float y2{0.0f}; ///< 右下角 y 坐标。
};

/**
 * @brief ROI 特征提取输入条目。
 */
struct RoiFeatureItem
{
    int64_t               roi_id{0};  ///< 调用方提供的 ROI 唯一 ID。
    std::filesystem::path image_path; ///< ROI 所属图像路径。
    RoiFeatureBox         roi;        ///< 原图坐标系下的 ROI。
};

/**
 * @brief ROIAlign 与模型特征提取共用配置。
 *
 * Faiss 字段由 ``ImageSearchConfig`` 继承而来，供 ROI 搜索和其他特征库流程复用；
 * ROI 聚类不会使用 Faiss 字段。
 */
struct RoiFeatureConfig : public ImageSearchConfig
{
    RoiFeatureConfig()
    {
        model_name   = kDefaultRoiFeatureModelName;
        feature_name = kDefaultRoiFeatureName;
    }

    int  pooled_height{kDefaultRoiFeaturePooledHeight}; ///< ROIAlign 输出高度。
    int  pooled_width{kDefaultRoiFeaturePooledWidth};   ///< ROIAlign 输出宽度。
    int  sampling_ratio{-1};                             ///< ROIAlign 采样率，-1 表示自适应。
    bool aligned{false};                                 ///< 是否使用 aligned ROIAlign 坐标规则。
    bool use_pca{false};                                 ///< 是否按图像特征图执行局部 PCA。
    int  pca_dim{0};                                     ///< 局部 PCA 输出通道数。
};

} // namespace irt::features
