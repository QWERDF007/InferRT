/**
 * @file ShapeTemplateMatcherAvx512.cpp
 * @brief v2 AVX512 形状模板匹配器的公开入口。
 */

#include "priv/ShapeTemplateMatcherAvx512Impl.hpp"

#include <inferrt/features/v2/ShapeTemplateMatcherAvx512.hpp>

#include <memory>
#include <utility>

namespace irt::features::v2 {

ShapeTemplateMatcherAvx512::ShapeTemplateMatcherAvx512(ShapeTemplateMatcherConfig config)
    : ::irt::features::detail::ShapeTemplateMatcherBase(
          std::make_unique<detail::ShapeTemplateMatcherAvx512Impl>(std::move(config)))
{
}

ShapeTemplateMatcherAvx512::~ShapeTemplateMatcherAvx512() = default;
ShapeTemplateMatcherAvx512::ShapeTemplateMatcherAvx512(ShapeTemplateMatcherAvx512 &&) noexcept = default;
ShapeTemplateMatcherAvx512 &ShapeTemplateMatcherAvx512::operator=(ShapeTemplateMatcherAvx512 &&) noexcept = default;

} // namespace irt::features::v2
