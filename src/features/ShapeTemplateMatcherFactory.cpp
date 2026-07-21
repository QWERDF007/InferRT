/**
 * @file ShapeTemplateMatcherFactory.cpp
 * @brief 形状模板匹配的版本无关工厂和公共工具函数。
 */

#include "priv/ShapeTemplateMatcherEngine.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ShapeTemplateMatcher.hpp>
#include <inferrt/features/v0/ShapeTemplateMatcher.hpp>
#include <inferrt/features/v1/ShapeTemplateMatcherFast.hpp>

#include <memory>
#include <utility>

namespace irt::features {

std::unique_ptr<IShapeTemplateMatcher>
createShapeTemplateMatcher(ShapeTemplateMatcherVersion version, ShapeTemplateMatcherConfig config)
{
    switch (version)
    {
    case ShapeTemplateMatcherVersion::V0:
        return std::make_unique<v0::ShapeTemplateMatcher>(std::move(config));
    case ShapeTemplateMatcherVersion::V1:
        return std::make_unique<v1::ShapeTemplateMatcherFast>(std::move(config));
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported shape template matcher version");
}

const char *shapeTemplateMatcherVersionName(ShapeTemplateMatcherVersion version) noexcept
{
    switch (version)
    {
    case ShapeTemplateMatcherVersion::V0:
        return "v0";
    case ShapeTemplateMatcherVersion::V1:
        return "v1";
    }
    return "unknown";
}

std::vector<ShapeTemplateVariant>
makeShapeTemplateAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees, float angle_step_degrees,
                                    float scale_begin, float scale_end, float scale_step)
{
    return detail::makeShapeTemplateAngleScaleVariants(angle_begin_degrees, angle_end_degrees, angle_step_degrees,
                                                        scale_begin, scale_end, scale_step);
}

cv::Mat transformShapeTemplateImage(const cv::Mat &image, ShapeTemplateVariant variant,
                                    const cv::Scalar &border_value)
{
    return detail::transformShapeTemplateImage(image, variant, border_value);
}

} // namespace irt::features
