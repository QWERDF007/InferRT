/**
 * @file ShapeTemplateMatcher.cpp
 * @brief v0 原始形状模板匹配器的公开入口。
 */

#include "priv/ShapeTemplateMatcherImpl.hpp"

#include <inferrt/features/v0/ShapeTemplateMatcher.hpp>

#include <memory>
#include <utility>

namespace irt::features::v0 {

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : ::irt::features::detail::ShapeTemplateMatcherBase(
          std::make_unique<detail::ShapeTemplateMatcherImpl>(std::move(config)))
{
}

ShapeTemplateMatcher::~ShapeTemplateMatcher() = default;
ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept = default;
ShapeTemplateMatcher &ShapeTemplateMatcher::operator=(ShapeTemplateMatcher &&) noexcept = default;

} // namespace irt::features::v0
