#pragma once

/**
 * @file DinoProfile.hpp
 * @brief YAML profile validation and serialization.
 */

#include <yaml-cpp/yaml.h>

#include <inferrt/features/DinoRegionSearch.hpp>

#include <string>

namespace irt::features::priv {

/**
 * @brief 每候选空间峰值的硬上限。
 *
 * 峰值在滑动过程中按行独立收集，行内用定长数组承载，因此上限必须是编译期常量；profile 超出该值
 * 会被拒绝，而不是被静默截断。
 */
constexpr int kDinoMaxPeaksPerCandidate = 8;


/** @brief Validate semantic configuration; throws irt::Exception on invalid values. */
void dinoValidateConfig(const DinoRegionSearchConfig &config);

/** @brief Serialize configuration to a single canonical YAML node. */
YAML::Node dinoConfigToYamlNode(const DinoRegionSearchConfig &config);

/** @brief Parse configuration from a YAML node. */
DinoRegionSearchConfig dinoConfigFromYamlNode(const YAML::Node &node);

/** @brief Resolve encoder raster edge from patch size. */
int dinoResolveEncoderEdge(const DinoRegionSearchConfig &config, int patch_size);

} // namespace irt::features::priv
