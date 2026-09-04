/**
 * @file ShapeTemplateMatcher.cpp
 * @brief 标量形状模板匹配内部适配器。
 */

#include "priv/ShapeTemplateMatcherImpl.hpp"

#include "priv/ShapeTemplateMatcherVariants.hpp"

#include <memory>
#include <utility>

namespace irt::features::priv {

ScalarShapeTemplateMatcher::ScalarShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : ShapeTemplateMatcherAdapter(
          std::make_unique<::irt::features::v0::detail::ShapeTemplateMatcherImpl>(std::move(config)))
{
}

ScalarShapeTemplateMatcher::~ScalarShapeTemplateMatcher() = default;
ScalarShapeTemplateMatcher::ScalarShapeTemplateMatcher(ScalarShapeTemplateMatcher &&) noexcept = default;
ScalarShapeTemplateMatcher &ScalarShapeTemplateMatcher::operator=(ScalarShapeTemplateMatcher &&) noexcept = default;

} // namespace irt::features::priv
