/**
 * @file ShapeTemplateMatcherFactory.cpp
 * @brief 形状模板匹配的版本无关工厂和公共工具函数。
 */

#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include "priv/ShapeTemplateMatcherFactory.hpp"
#include "priv/ShapeTemplateMatcherEngine.hpp"
#include "priv/ShapeTemplateMatcherVariants.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <utility>

namespace irt::features {
namespace {

bool supportsAvx2() noexcept
{
    return cv::checkHardwareSupport(CV_CPU_AVX2);
}

bool supportsAvx512() noexcept
{
    return cv::checkHardwareSupport(CV_CPU_AVX_512F) && cv::checkHardwareSupport(CV_CPU_AVX_512BW);
}

} // namespace

namespace priv {

std::unique_ptr<IShapeTemplateMatcher> createShapeTemplateMatcherForImplementation(
    const ShapeTemplateMatcherImplementation implementation, ShapeTemplateMatcherConfig config)
{
    switch (implementation)
    {
    case ShapeTemplateMatcherImplementation::Scalar:
        return std::make_unique<ScalarShapeTemplateMatcher>(std::move(config));
    case ShapeTemplateMatcherImplementation::Avx2:
        if (!supportsAvx2())
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION,
                                 "The AVX2 shape template matcher requires an AVX2-capable CPU");
        }
        return std::make_unique<Avx2ShapeTemplateMatcher>(std::move(config));
    case ShapeTemplateMatcherImplementation::Avx512:
        if (!supportsAvx512())
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION,
                                 "The AVX512 shape template matcher requires AVX512F and AVX512BW");
        }
        return std::make_unique<Avx512ShapeTemplateMatcher>(std::move(config));
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported shape template matcher implementation");
}

const char *shapeTemplateMatcherImplementationName(const ShapeTemplateMatcherImplementation implementation) noexcept
{
    switch (implementation)
    {
    case ShapeTemplateMatcherImplementation::Scalar:
        return "scalar";
    case ShapeTemplateMatcherImplementation::Avx2:
        return "avx2";
    case ShapeTemplateMatcherImplementation::Avx512:
        return "avx512";
    }
    return "unknown";
}

bool shapeTemplateMatcherImplementationSupported(const ShapeTemplateMatcherImplementation implementation) noexcept
{
    switch (implementation)
    {
    case ShapeTemplateMatcherImplementation::Scalar:
        return true;
    case ShapeTemplateMatcherImplementation::Avx2:
        return supportsAvx2();
    case ShapeTemplateMatcherImplementation::Avx512:
        return supportsAvx512();
    }
    return false;
}

} // namespace priv

std::unique_ptr<IShapeTemplateMatcher> createShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
{
    const auto implementation = priv::shapeTemplateMatcherImplementationSupported(
                                    priv::ShapeTemplateMatcherImplementation::Avx512)
                              ? priv::ShapeTemplateMatcherImplementation::Avx512
                              : priv::shapeTemplateMatcherImplementationSupported(
                                    priv::ShapeTemplateMatcherImplementation::Avx2)
                                    ? priv::ShapeTemplateMatcherImplementation::Avx2
                                    : priv::ShapeTemplateMatcherImplementation::Scalar;
    return priv::createShapeTemplateMatcherForImplementation(implementation, std::move(config));
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
