#pragma once

/**
 * @file ShapeTemplateMatcherImpl.hpp
 * @brief v0 形状模板匹配器的私有具体实现。
 */

#include "ShapeTemplateMatcherEngine.hpp"

namespace irt::features::v0::detail {

/** @brief v0 的原始标量实现，仅注入 v0::detail 内的基准热点内核。 */
class ShapeTemplateMatcherImpl final : public ::irt::features::detail::ShapeTemplateMatcherEngine
{
public:
    explicit ShapeTemplateMatcherImpl(ShapeTemplateMatcherConfig config);
};

} // namespace irt::features::v0::detail
