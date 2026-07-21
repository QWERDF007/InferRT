/**
 * @file ShapeTemplateMatcherFast.cpp
 * @brief v1 快速形状模板匹配器的公开入口。
 */

#include "priv/ShapeTemplateMatcherFastImpl.hpp"

#include <inferrt/features/v1/ShapeTemplateMatcherFast.hpp>

#include <memory>
#include <utility>

namespace irt::features::v1 {

ShapeTemplateMatcherFast::ShapeTemplateMatcherFast(ShapeTemplateMatcherConfig config)
    : ::irt::features::detail::ShapeTemplateMatcherBase(
          std::make_unique<detail::ShapeTemplateMatcherFastImpl>(std::move(config)))
{
}

ShapeTemplateMatcherFast::~ShapeTemplateMatcherFast() = default;
ShapeTemplateMatcherFast::ShapeTemplateMatcherFast(ShapeTemplateMatcherFast &&) noexcept = default;
ShapeTemplateMatcherFast &ShapeTemplateMatcherFast::operator=(ShapeTemplateMatcherFast &&) noexcept = default;

} // namespace irt::features::v1
