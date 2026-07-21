#pragma once

/**
 * @file v1/ShapeTemplateMatcherFast.hpp
 * @brief v1 快速形状模板匹配器。
 */

#include <inferrt/features/ShapeTemplateMatcherBase.hpp>

namespace irt::features::v1 {

/**
 * @brief v1 快速形状模板匹配器。
 *
 * @details 该类使用 AVX2 热点内核和优化训练流程；构造时要求运行 CPU 支持 AVX2。
 * 公共 API 由共用基类实现，保证与 v0 的调用顺序、持久化和结果访问流程一致。
 */
class INFERRT_FEATURES_API ShapeTemplateMatcherFast final : public ::irt::features::detail::ShapeTemplateMatcherBase
{
public:
    /** @brief 使用给定配置构造 v1 快速匹配器。 */
    explicit ShapeTemplateMatcherFast(ShapeTemplateMatcherConfig config = {});
    ~ShapeTemplateMatcherFast() override;
    ShapeTemplateMatcherFast(const ShapeTemplateMatcherFast &)            = delete;
    ShapeTemplateMatcherFast &operator=(const ShapeTemplateMatcherFast &) = delete;
    ShapeTemplateMatcherFast(ShapeTemplateMatcherFast &&) noexcept;
    ShapeTemplateMatcherFast &operator=(ShapeTemplateMatcherFast &&) noexcept;
};

} // namespace irt::features::v1
