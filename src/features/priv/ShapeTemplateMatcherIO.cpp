/**
 * @file ShapeTemplateMatcherIO.cpp
 * @brief 形状模板匹配配置校验与 YAML 文件序列化/反序列化实现。
 */

#include "ShapeTemplateMatcherIO.hpp"

#include <inferrt/core/Exception.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features::detail {
namespace {

constexpr int kShapeTemplateFileFormatVersion = 4;

void validateScoreRange(float value, const char *name)
{
    if (!std::isfinite(value) || value < 0.0f || value > 100.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be finite and in [0, 100]", name);
    }
}

template<typename T>
void readYamlIfPresent(const YAML::Node &node, const char *key, T &value)
{
    const YAML::Node child = node[key];
    if (child && !child.IsNull())
    {
        value = child.as<T>();
    }
}

template<typename T>
T readRequiredYamlValue(const YAML::Node &node, const char *key)
{
    const YAML::Node child = node[key];
    if (!child || child.IsNull())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template YAML is missing required field: %s",
                             key);
    }
    return child.as<T>();
}

} // namespace

void validateShapeTemplateConfig(const ShapeTemplateMatcherConfig &config)
{
    if (config.num_features <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ShapeTemplateMatcher num_features must be positive");
    }
    if (config.min_features <= 0 || config.min_features > config.num_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher min_features must be positive and <= num_features");
    }
    if (!std::isfinite(config.weak_threshold) || config.weak_threshold < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher weak_threshold must be finite and non-negative");
    }
    if (!std::isfinite(config.strong_threshold) || config.strong_threshold < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher strong_threshold must be finite and non-negative");
    }
    if (config.max_label_difference < 0 || config.max_label_difference > 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_label_difference must be in [0, 4]");
    }
    validateScoreRange(config.match_threshold, "ShapeTemplateMatcher match_threshold");
    if (!std::isfinite(config.nms_threshold) || config.nms_threshold > 1.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher nms_threshold must be finite and <= 1");
    }
    if (config.max_results < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_results must be non-negative");
    }
    if (config.scan_step <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ShapeTemplateMatcher scan_step must be positive");
    }
    if (config.max_parallelism < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_parallelism must be non-negative");
    }
    if (config.max_training_parallelism < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_training_parallelism must be non-negative");
    }
    if (!std::isfinite(config.min_feature_distance) || config.min_feature_distance < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher min_feature_distance must be finite and non-negative");
    }
}

void validateShapeTemplateInfo(const ShapeTemplateInfo &info, int min_features)
{
    if (info.width <= 0 || info.height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid size");
    }
    if (static_cast<int>(info.features.size()) < min_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has too few features");
    }
    if (info.template_width <= 0 || info.template_height <= 0 || info.template_width < info.width
        || info.template_height < info.height)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid canvas size");
    }
    if (!std::isfinite(info.angle_degrees) || !std::isfinite(info.scale) || info.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid variant metadata");
    }
    for (const auto &feature : info.features)
    {
        if (feature.x < 0 || feature.x >= info.width || feature.y < 0 || feature.y >= info.height)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template feature is out of bounds");
        }
        if (feature.label < 0 || feature.label > 7 || !std::isfinite(feature.angle_degrees))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template feature label is invalid");
        }
    }
}

void saveShapeTemplateYaml(const fs::path &template_file,
                            const ShapeTemplateMatcherConfig &config,
                            const std::vector<ShapeTemplateInfo> &templates)
{
    if (template_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file path must not be empty");
    }
    if (!template_file.parent_path().empty())
    {
        fs::create_directories(template_file.parent_path());
    }

    std::ofstream output(template_file);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open shape template file for write: %s",
                             template_file.string().c_str());
    }

    YAML::Emitter emitter;
    emitter.SetFloatPrecision(std::numeric_limits<float>::max_digits10);
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "version" << YAML::Value << kShapeTemplateFileFormatVersion;
    emitter << YAML::Key << "config" << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "num_features" << YAML::Value << config.num_features;
    emitter << YAML::Key << "min_features" << YAML::Value << config.min_features;
    emitter << YAML::Key << "weak_threshold" << YAML::Value << config.weak_threshold;
    emitter << YAML::Key << "strong_threshold" << YAML::Value << config.strong_threshold;
    emitter << YAML::Key << "max_label_difference" << YAML::Value << config.max_label_difference;
    emitter << YAML::Key << "match_threshold" << YAML::Value << config.match_threshold;
    emitter << YAML::Key << "nms_threshold" << YAML::Value << config.nms_threshold;
    emitter << YAML::Key << "max_results" << YAML::Value << config.max_results;
    emitter << YAML::Key << "scan_step" << YAML::Value << config.scan_step;
    emitter << YAML::Key << "max_parallelism" << YAML::Value << config.max_parallelism;
    emitter << YAML::Key << "min_feature_distance" << YAML::Value << config.min_feature_distance;
    emitter << YAML::Key << "use_gaussian_gradient" << YAML::Value << config.use_gaussian_gradient;
    emitter << YAML::Key << "use_orientation_histogram" << YAML::Value << config.use_orientation_histogram;
    emitter << YAML::Key << "use_edge_nms" << YAML::Value << config.use_edge_nms;
    emitter << YAML::Key << "use_edge_connectivity" << YAML::Value << config.use_edge_connectivity;
    emitter << YAML::Key << "use_polarity_invariant" << YAML::Value << config.use_polarity_invariant;
    emitter << YAML::Key << "use_spatial_spread" << YAML::Value << config.use_spatial_spread;
    emitter << YAML::Key << "reuse_base_features_for_variants" << YAML::Value
            << config.reuse_base_features_for_variants;
    emitter << YAML::EndMap;

    emitter << YAML::Key << "templates" << YAML::Value << YAML::BeginSeq;
    for (const auto &templ : templates)
    {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "template_id" << YAML::Value << templ.template_id;
        emitter << YAML::Key << "width" << YAML::Value << templ.width;
        emitter << YAML::Key << "height" << YAML::Value << templ.height;
        emitter << YAML::Key << "template_width" << YAML::Value << templ.template_width;
        emitter << YAML::Key << "template_height" << YAML::Value << templ.template_height;
        emitter << YAML::Key << "tl_x" << YAML::Value << templ.tl_x;
        emitter << YAML::Key << "tl_y" << YAML::Value << templ.tl_y;
        emitter << YAML::Key << "angle_degrees" << YAML::Value << templ.angle_degrees;
        emitter << YAML::Key << "scale" << YAML::Value << templ.scale;
        emitter << YAML::Key << "features" << YAML::Value << YAML::BeginSeq;
        for (const auto &feature : templ.features)
        {
            emitter << YAML::Flow << YAML::BeginSeq << feature.x << feature.y << feature.label
                    << feature.angle_degrees << YAML::EndSeq;
        }
        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;

    if (!emitter.good())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to serialize shape template YAML: %s",
                             emitter.GetLastError().c_str());
    }
    output << emitter.c_str() << '\n';
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write shape template file: %s",
                             template_file.string().c_str());
    }
}

void loadShapeTemplateYaml(const fs::path &template_file,
                            ShapeTemplateMatcherConfig &config,
                            std::vector<ShapeTemplateInfo> &templates,
                            const ShapeTemplateMatcherKernel *kernel)
{
    if (template_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file path must not be empty");
    }
    std::error_code ec;
    if (!fs::is_regular_file(template_file, ec))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file does not exist: %s",
                             template_file.string().c_str());
    }

    try
    {
        const YAML::Node root = YAML::LoadFile(template_file.string());
        if (!root || root.IsNull() || !root.IsMap())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Shape template YAML root must be a mapping");
        }

        const int version = readRequiredYamlValue<int>(root, "version");
        if (version != kShapeTemplateFileFormatVersion)
        {
            throw irt::Exception(
                irt::Status::ERROR_INVALID_ARGUMENT,
                "Unsupported shape template file version: %d; only v%d compact feature arrays are supported", version,
                kShapeTemplateFileFormatVersion);
        }

        ShapeTemplateMatcherConfig loaded_config = config;
        const YAML::Node           config_node   = root["config"];
        if (config_node && !config_node.IsNull())
        {
            if (!config_node.IsMap())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Shape template YAML config node must be a mapping");
            }
            readYamlIfPresent(config_node, "num_features", loaded_config.num_features);
            readYamlIfPresent(config_node, "min_features", loaded_config.min_features);
            readYamlIfPresent(config_node, "weak_threshold", loaded_config.weak_threshold);
            readYamlIfPresent(config_node, "strong_threshold", loaded_config.strong_threshold);
            readYamlIfPresent(config_node, "max_label_difference", loaded_config.max_label_difference);
            readYamlIfPresent(config_node, "match_threshold", loaded_config.match_threshold);
            readYamlIfPresent(config_node, "nms_threshold", loaded_config.nms_threshold);
            readYamlIfPresent(config_node, "max_results", loaded_config.max_results);
            readYamlIfPresent(config_node, "scan_step", loaded_config.scan_step);
            readYamlIfPresent(config_node, "max_parallelism", loaded_config.max_parallelism);
            readYamlIfPresent(config_node, "min_feature_distance", loaded_config.min_feature_distance);
            readYamlIfPresent(config_node, "use_gaussian_gradient", loaded_config.use_gaussian_gradient);
            readYamlIfPresent(config_node, "use_orientation_histogram", loaded_config.use_orientation_histogram);
            readYamlIfPresent(config_node, "use_edge_nms", loaded_config.use_edge_nms);
            readYamlIfPresent(config_node, "use_edge_connectivity", loaded_config.use_edge_connectivity);
            readYamlIfPresent(config_node, "use_polarity_invariant", loaded_config.use_polarity_invariant);
            readYamlIfPresent(config_node, "use_spatial_spread", loaded_config.use_spatial_spread);
            readYamlIfPresent(config_node, "reuse_base_features_for_variants",
                              loaded_config.reuse_base_features_for_variants);
        }
        validateShapeTemplateConfig(loaded_config);
        if (kernel
            && (loaded_config.use_gaussian_gradient || loaded_config.use_orientation_histogram
                || loaded_config.use_edge_nms || loaded_config.use_edge_connectivity
                || loaded_config.use_polarity_invariant || loaded_config.use_spatial_spread
                || loaded_config.reuse_base_features_for_variants)
            && !kernel->supportsApproximatePreprocessing())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Approximate v1 preprocessing/variant-reuse options are supported by v1 only");
        }

        std::vector<ShapeTemplateInfo> loaded_templates;
        const YAML::Node templates_node = root["templates"];
        if (templates_node && !templates_node.IsNull() && !templates_node.IsSequence())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Shape template YAML templates node must be a list");
        }

        if (templates_node && !templates_node.IsNull())
        {
            for (const auto &node : templates_node)
            {
                if (!node.IsMap())
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Each shape template YAML entry must be a mapping");
                }

                ShapeTemplateInfo info;
                readYamlIfPresent(node, "template_id", info.template_id);
                info.width         = readRequiredYamlValue<int>(node, "width");
                info.height        = readRequiredYamlValue<int>(node, "height");
                info.template_width  = readRequiredYamlValue<int>(node, "template_width");
                info.template_height = readRequiredYamlValue<int>(node, "template_height");
                info.tl_x          = readRequiredYamlValue<int>(node, "tl_x");
                info.tl_y          = readRequiredYamlValue<int>(node, "tl_y");
                info.angle_degrees = readRequiredYamlValue<float>(node, "angle_degrees");
                info.scale         = readRequiredYamlValue<float>(node, "scale");

                const YAML::Node features_node = node["features"];
                if (!features_node || features_node.IsNull() || !features_node.IsSequence())
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Shape template features must be a list of [x, y, label, angle_degrees] arrays");
                }

                for (const auto &feature_node : features_node)
                {
                    if (!feature_node.IsSequence() || feature_node.size() != 4U)
                    {
                        throw irt::Exception(
                            irt::Status::ERROR_INVALID_ARGUMENT,
                            "Each shape template feature must contain exactly [x, y, label, angle_degrees]");
                    }
                    ShapeTemplateFeature feature;
                    feature.x             = feature_node[0].as<int>();
                    feature.y             = feature_node[1].as<int>();
                    feature.label         = feature_node[2].as<int>();
                    feature.angle_degrees = feature_node[3].as<float>();
                    info.features.push_back(feature);
                }

                info.template_id = static_cast<int>(loaded_templates.size());
                validateShapeTemplateInfo(info, loaded_config.min_features);
                loaded_templates.push_back(std::move(info));
            }
        }

        config    = loaded_config;
        templates = std::move(loaded_templates);
    }
    catch (const YAML::Exception &exception)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to parse shape template YAML '%s': %s",
                             template_file.string().c_str(), exception.what());
    }
}

} // namespace irt::features::detail
