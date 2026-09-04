#pragma once

/**
 * @file ShapeTemplateMatcher.hpp
 * @brief 形状模板匹配公共入口。
 *
 * 该头文件只提供版本无关的接口和自动选择工厂。具体算法和 ISA 选择
 * 保留在 features 模块内部，不属于安装包的公开接口。
 */

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features {

/** @brief 根据当前 CPU 能力自动创建最合适的形状模板匹配器。 */
INFERRT_FEATURES_API std::unique_ptr<IShapeTemplateMatcher>
createShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});

/** @brief 根据闭区间角度/尺度范围生成模板变体列表。 */
INFERRT_FEATURES_API std::vector<ShapeTemplateVariant>
makeShapeTemplateAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees, float angle_step_degrees,
                                    float scale_begin = 1.0f, float scale_end = 1.0f,
                                    float scale_step = 1.0f);

/** @brief 对图像应用与模板变体训练一致的中心旋转/缩放变换。 */
INFERRT_FEATURES_API cv::Mat transformShapeTemplateImage(const cv::Mat &image, ShapeTemplateVariant variant,
                                                          const cv::Scalar &border_value = cv::Scalar());

} // namespace irt::features
