#pragma once

#include "ShapeTemplateMatcherEngine.hpp"

#include <filesystem>
#include <vector>

namespace irt::features::detail {

void validateShapeTemplateConfig(const ShapeTemplateMatcherConfig &config);

void validateShapeTemplateInfo(const ShapeTemplateInfo &info, int min_features);

void saveShapeTemplateYaml(const std::filesystem::path &template_file,
                            const ShapeTemplateMatcherConfig &config,
                            const std::vector<ShapeTemplateInfo> &templates);

void loadShapeTemplateYaml(const std::filesystem::path &template_file,
                            ShapeTemplateMatcherConfig &config,
                            std::vector<ShapeTemplateInfo> &templates,
                            const ShapeTemplateMatcherKernel *kernel);

} // namespace irt::features::detail
