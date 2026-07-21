#pragma once

/**
 * @file ShapeTemplateMatcherFastImpl.hpp
 * @brief v1 快速形状模板匹配器的私有具体实现。
 */

#include "ShapeTemplateMatcherEngine.hpp"

namespace irt::features::v1::detail {

/** @brief v1 的 AVX2 快速实现，仅注入 v1::detail 内的快速热点内核。 */
class ShapeTemplateMatcherFastImpl final : public ::irt::features::detail::ShapeTemplateMatcherEngine
{
public:
    explicit ShapeTemplateMatcherFastImpl(ShapeTemplateMatcherConfig config);
};

} // namespace irt::features::v1::detail
