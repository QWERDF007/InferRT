#pragma once

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features::priv {

/** Algorithm choices are private to parity tests and benchmark tooling. */
enum class ShapeTemplateMatcherImplementation
{
    Scalar,
    Avx2,
    Avx512,
};

INFERRT_FEATURES_API std::unique_ptr<IShapeTemplateMatcher>
createShapeTemplateMatcherForImplementation(ShapeTemplateMatcherImplementation implementation,
                                            ShapeTemplateMatcherConfig config = {});

INFERRT_FEATURES_API const char *
shapeTemplateMatcherImplementationName(ShapeTemplateMatcherImplementation implementation) noexcept;

INFERRT_FEATURES_API bool
shapeTemplateMatcherImplementationSupported(ShapeTemplateMatcherImplementation implementation) noexcept;

} // namespace irt::features::priv
