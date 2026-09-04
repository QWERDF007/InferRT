#pragma once

#include "ShapeTemplateMatcherAdapter.hpp"

namespace irt::features::priv {

class ScalarShapeTemplateMatcher final : public ShapeTemplateMatcherAdapter
{
public:
    explicit ScalarShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~ScalarShapeTemplateMatcher() override;
    ScalarShapeTemplateMatcher(const ScalarShapeTemplateMatcher &)            = delete;
    ScalarShapeTemplateMatcher &operator=(const ScalarShapeTemplateMatcher &) = delete;
    ScalarShapeTemplateMatcher(ScalarShapeTemplateMatcher &&) noexcept;
    ScalarShapeTemplateMatcher &operator=(ScalarShapeTemplateMatcher &&) noexcept;
};

class Avx2ShapeTemplateMatcher final : public ShapeTemplateMatcherAdapter
{
public:
    explicit Avx2ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~Avx2ShapeTemplateMatcher() override;
    Avx2ShapeTemplateMatcher(const Avx2ShapeTemplateMatcher &)            = delete;
    Avx2ShapeTemplateMatcher &operator=(const Avx2ShapeTemplateMatcher &) = delete;
    Avx2ShapeTemplateMatcher(Avx2ShapeTemplateMatcher &&) noexcept;
    Avx2ShapeTemplateMatcher &operator=(Avx2ShapeTemplateMatcher &&) noexcept;
};

class Avx512ShapeTemplateMatcher final : public ShapeTemplateMatcherAdapter
{
public:
    explicit Avx512ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~Avx512ShapeTemplateMatcher() override;
    Avx512ShapeTemplateMatcher(const Avx512ShapeTemplateMatcher &)            = delete;
    Avx512ShapeTemplateMatcher &operator=(const Avx512ShapeTemplateMatcher &) = delete;
    Avx512ShapeTemplateMatcher(Avx512ShapeTemplateMatcher &&) noexcept;
    Avx512ShapeTemplateMatcher &operator=(Avx512ShapeTemplateMatcher &&) noexcept;
};

} // namespace irt::features::priv
