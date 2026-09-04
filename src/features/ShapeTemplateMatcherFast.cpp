/**
 * @file ShapeTemplateMatcherFast.cpp
 * @brief AVX2 形状模板匹配内部适配器。
 */

#include "priv/ShapeTemplateMatcherFastImpl.hpp"

#include "priv/ShapeTemplateMatcherVariants.hpp"

#include <memory>
#include <utility>

namespace irt::features::priv {

Avx2ShapeTemplateMatcher::Avx2ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : ShapeTemplateMatcherAdapter(
          std::make_unique<::irt::features::v1::detail::ShapeTemplateMatcherFastImpl>(std::move(config)))
{
}

Avx2ShapeTemplateMatcher::~Avx2ShapeTemplateMatcher() = default;
Avx2ShapeTemplateMatcher::Avx2ShapeTemplateMatcher(Avx2ShapeTemplateMatcher &&) noexcept = default;
Avx2ShapeTemplateMatcher &Avx2ShapeTemplateMatcher::operator=(Avx2ShapeTemplateMatcher &&) noexcept = default;

} // namespace irt::features::priv
