/**
 * @file ShapeTemplateMatcherAvx512.cpp
 * @brief AVX512 形状模板匹配内部适配器。
 */

#include "priv/ShapeTemplateMatcherAvx512Impl.hpp"

#include "priv/ShapeTemplateMatcherVariants.hpp"

#include <memory>
#include <utility>

namespace irt::features::priv {

Avx512ShapeTemplateMatcher::Avx512ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : ShapeTemplateMatcherAdapter(
          std::make_unique<::irt::features::v2::detail::ShapeTemplateMatcherAvx512Impl>(std::move(config)))
{
}

Avx512ShapeTemplateMatcher::~Avx512ShapeTemplateMatcher() = default;
Avx512ShapeTemplateMatcher::Avx512ShapeTemplateMatcher(Avx512ShapeTemplateMatcher &&) noexcept = default;
Avx512ShapeTemplateMatcher &Avx512ShapeTemplateMatcher::operator=(Avx512ShapeTemplateMatcher &&) noexcept = default;

} // namespace irt::features::priv
