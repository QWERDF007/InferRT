#pragma once

/**
 * @file ShapeTemplateMatcherAvx512Impl.hpp
 * @brief v2 AVX512 形状模板匹配器的私有具体实现。
 */

#include "ShapeTemplateMatcherEngine.hpp"

namespace irt::features::v2::detail {

/** @brief v2 的 AVX512F/BW 快速实现，仅注入 v2::detail 内的热点内核。 */
class ShapeTemplateMatcherAvx512Impl final : public ::irt::features::detail::ShapeTemplateMatcherEngine
{
public:
    explicit ShapeTemplateMatcherAvx512Impl(ShapeTemplateMatcherConfig config);
};

} // namespace irt::features::v2::detail
