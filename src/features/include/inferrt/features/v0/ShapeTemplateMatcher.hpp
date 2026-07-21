#pragma once

/**
 * @file v0/ShapeTemplateMatcher.hpp
 * @brief v0 原始形状模板匹配器。
 */

#include <inferrt/features/ShapeTemplateMatcherBase.hpp>

namespace irt::features::v0 {

/**
 * @brief v0 原始形状模板匹配器。
 *
 * @details 该类固定使用参考实现，训练保持原始串行流程，匹配不使用 v1 的快速热点。
 * 公共 API 由共用基类实现，保证与 v1 的调用顺序、持久化和结果访问流程一致。
 */
class INFERRT_FEATURES_API ShapeTemplateMatcher final : public ::irt::features::detail::ShapeTemplateMatcherBase
{
public:
    /** @brief 使用给定配置构造 v0 匹配器。 */
    explicit ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~ShapeTemplateMatcher() override;
    ShapeTemplateMatcher(const ShapeTemplateMatcher &)            = delete;
    ShapeTemplateMatcher &operator=(const ShapeTemplateMatcher &) = delete;
    ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept;
    ShapeTemplateMatcher &operator=(ShapeTemplateMatcher &&) noexcept;
};

} // namespace irt::features::v0
