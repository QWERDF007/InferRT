#pragma once

/**
 * @file ShapeTemplateMatcher.hpp
 * @brief 形状模板匹配多版本公共入口。
 *
 * 该头文件只提供版本无关的接口、工厂和工具函数，不在 ``irt::features`` 根命名空间
 * 提供具体实现。请显式包含并使用
 * ``irt::features::v0::ShapeTemplateMatcher`` 或
 * ``irt::features::v1::ShapeTemplateMatcherFast``；也可通过工厂获取版本无关接口。
 */

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features {

/** @brief 根据版本创建形状模板匹配器。 */
INFERRT_FEATURES_API std::unique_ptr<IShapeTemplateMatcher>
createShapeTemplateMatcher(ShapeTemplateMatcherVersion version, ShapeTemplateMatcherConfig config = {});

/** @brief 获取版本的稳定命令行/日志名称（``v0`` 或 ``v1``）。 */
INFERRT_FEATURES_API const char *shapeTemplateMatcherVersionName(ShapeTemplateMatcherVersion version) noexcept;

/** @brief 根据闭区间角度/尺度范围生成模板变体列表。 */
INFERRT_FEATURES_API std::vector<ShapeTemplateVariant>
makeShapeTemplateAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees, float angle_step_degrees,
                                    float scale_begin = 1.0f, float scale_end = 1.0f,
                                    float scale_step = 1.0f);

/** @brief 对图像应用与模板变体训练一致的中心旋转/缩放变换。 */
INFERRT_FEATURES_API cv::Mat transformShapeTemplateImage(const cv::Mat &image, ShapeTemplateVariant variant,
                                                          const cv::Scalar &border_value = cv::Scalar());

} // namespace irt::features
