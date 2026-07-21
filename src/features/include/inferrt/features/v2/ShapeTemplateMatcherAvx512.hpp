#pragma once

/**
 * @file v2/ShapeTemplateMatcherAvx512.hpp
 * @brief v2 AVX512 形状模板匹配器。
 */

#include <inferrt/features/ShapeTemplateMatcherBase.hpp>

namespace irt::features::v2 {

/**
 * @brief 使用 AVX512F/BW 热点内核的 v2 形状模板匹配器。
 *
 * @details 构造时会检查运行 CPU 的 AVX512F 和 AVX512BW 支持。训练和匹配均使用 v2::detail
 * 内的独立实现；公共 API 继续由共享转发基类提供，模板格式和结果与 v0/v1 严格兼容。
 */
class INFERRT_FEATURES_API ShapeTemplateMatcherAvx512 final
    : public ::irt::features::detail::ShapeTemplateMatcherBase
{
public:
    /** @brief 使用给定配置构造 v2 AVX512 匹配器。 */
    explicit ShapeTemplateMatcherAvx512(ShapeTemplateMatcherConfig config = {});
    ~ShapeTemplateMatcherAvx512() override;
    ShapeTemplateMatcherAvx512(const ShapeTemplateMatcherAvx512 &)            = delete;
    ShapeTemplateMatcherAvx512 &operator=(const ShapeTemplateMatcherAvx512 &) = delete;
    ShapeTemplateMatcherAvx512(ShapeTemplateMatcherAvx512 &&) noexcept;
    ShapeTemplateMatcherAvx512 &operator=(ShapeTemplateMatcherAvx512 &&) noexcept;
};

} // namespace irt::features::v2
