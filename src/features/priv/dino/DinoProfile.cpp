/**
 * @file DinoProfile.cpp
 * @brief YAML profile validation and serialization.
 */

#include "DinoProfile.hpp"
#include "DinoPaths.hpp"

#include <algorithm>
#include <cmath>

namespace irt::features::priv {

namespace {

void requireFinite(const double value, const char *name)
{
    if (!std::isfinite(value))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile field %s must be finite", name);
    }
}

void requirePositive(const double value, const char *name)
{
    requireFinite(value, name);
    if (!(value > 0.0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile field %s must be positive", name);
    }
}

void requireRange(const double value, const double low, const double high, const char *name)
{
    requireFinite(value, name);
    if (value < low || value > high)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile field %s must be within [%g, %g]", name, low,
                             high);
    }
}

} // namespace

void dinoValidateConfig(const DinoRegionSearchConfig &config)
{
    if (config.profile_id.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile id must not be empty");
    }
    if (config.model_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile model name must not be empty");
    }
    if (config.weights_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile requires a backbone weights file");
    }
    config.model_runtime.validate();

    if (config.encoder_edge != 0 && config.encoder_edge < 32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile encoder edge must be 0 or >= 32");
    }
    if (config.model_batch_size == 0U || config.model_batch_size > 64U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile model batch size must be within 1..64");
    }

    if (config.gallery_tile_edges.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile requires at least one gallery tile edge");
    }
    for (const auto edge : config.gallery_tile_edges)
    {
        if (edge < 32)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery tile edge must be >= 32 pixels");
        }
    }
    requireRange(config.view_overlap, 0.0, 0.75, "view_overlap");

    if (config.coarse_dimension < 1 || config.coarse_dimension > 384)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Coarse dimension must be in 1..384");
    if (config.local_representatives != 0 && config.local_representatives != 64 && config.local_representatives != 128)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local representatives must be 0, 64 or 128");
    if (config.fine_verify_k > 256U)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Verification budget must be at most 256");
    if (config.region_window_ratios.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile requires at least one window ratio");
    }
    for (const auto ratio : config.region_window_ratios)
    {
        requireRange(ratio, 0.05, 1.0, "region_window_ratios");
    }
    requireRange(config.region_window_stride_ratio, 0.05, 1.0, "region_window_stride_ratio");
    requireRange(config.region_min_valid_fraction, 0.0, 1.0, "region_min_valid_fraction");
    requireRange(config.merge_epsilon, 0.0, 1.0, "merge_epsilon");
    if (config.validated_min_image_edge < 1 || config.validated_max_image_edge <= config.validated_min_image_edge)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile validated image edge range is invalid");
    }
    requirePositive(config.validated_min_target_short_px, "validated_min_target_short_px");
    requirePositive(config.validated_max_target_aspect, "validated_max_target_aspect");
    requireRange(config.coarse_dedup_iou, 0.0, 1.0, "coarse_dedup_iou");
    requireRange(config.coarse_dedup_area_ratio, 1.0, 64.0, "coarse_dedup_area_ratio");
    requireRange(config.fine_template_scale_step, 1.01, 4.0, "fine_template_scale_step");
    if (config.fine_refinement_rounds < 0 || config.fine_refinement_rounds > 8)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile refinement rounds must be within 0..8");
    }
    if (config.max_leaf_side_patches < 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile max leaf side must be >= 1 patch");
    }

    if (config.query_roi_target_lengths.size() > 3)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "At most three query contexts are supported");
    if (config.query_roi_target_lengths.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile requires at least one ROI target length");
    }
    for (const auto length : config.query_roi_target_lengths)
    {
        requirePositive(length, "query_roi_target_lengths");
    }
    if (config.query_local_cells < 1 || config.query_local_cells > 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile query local cells must be within 1..4");
    }
    if (config.query_local_max_per_cell < 1 || config.query_local_max_per_cell > 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile query local max per cell must be within 1..2");
    }
    if (config.query_min_local_evidence < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile query min local evidence must be >= 0");
    }

    if (config.region_topk == 0U || config.local_view_topk == 0U || config.channel_candidate_limit == 0U
        || config.coarse_k == 0U || config.final_k == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile candidate quotas must be positive");
    }
    if (config.coarse_k > config.channel_candidate_limit)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Profile coarse_K must not exceed the per-channel candidate limit");
    }
    if (config.final_k > config.coarse_k)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile final_K must not exceed coarse_K");
    }
    if (config.region_scan_block == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile scan block must be positive");
    }

    requireRange(config.fine_candidate_expand, 1.0, 4.0, "fine_candidate_expand");
    if (config.fine_template_min_short_patches < 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile template min short side must be >= 1 patch");
    }
    if (config.fine_template_max_sizes < 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile template max sizes must be >= 1");
    }
    if (config.fine_peaks_per_candidate < 1 || config.fine_peaks_per_candidate > kDinoMaxPeaksPerCandidate)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Profile peaks per candidate must be within 1..%d", kDinoMaxPeaksPerCandidate);
    }
    requireRange(config.fine_match_cosine_threshold, -1.0, 1.0, "fine_match_cosine_threshold");
    requireRange(config.fine_position_tolerance, 0.0, 2.0, "fine_position_tolerance");
    requireRange(config.fine_nms_iou, 0.0, 1.0, "fine_nms_iou");

    const double weight_sum = config.score_weight_template + config.score_weight_coverage + config.score_weight_consistency;
    if (std::abs(weight_sum - 1.0) > 1e-6)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Profile score weights must sum to 1.0, got %g", weight_sum);
    }
    for (const auto weight : {config.score_weight_template, config.score_weight_coverage,
                              config.score_weight_consistency})
    {
        requireRange(weight, 0.0, 1.0, "score weights");
    }

    if (config.enable_decision_threshold)
    {
        requireRange(config.decision_threshold, 0.0, 1.0, "decision_threshold");
    }
    if (config.query_deadline_ms <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile query deadline must be positive");
    }
}

YAML::Node dinoConfigToYamlNode(const DinoRegionSearchConfig &config)
{
    YAML::Node node;
    node["profile_id"] = config.profile_id;
    node["mode"] = config.consistency_mode == DinoConsistencyMode::Appearance ? "appearance" : "instance";

    YAML::Node model;
    model["name"] = config.model_name;
    model["weights_path"] = dinoPathToUtf8(config.weights_file);
    model["encoder_edge"] = config.encoder_edge;
    model["precision"] = config.model_precision == irt::model::ModelPrecision::FP16 ? "fp16" : "fp32";
    model["batch_size"] = config.model_batch_size;
    model["runtime"] = config.model_runtime.toString();
    node["model"] = model;

    YAML::Node input;
    input["min_image_edge"] = config.validated_min_image_edge;
    input["max_image_edge"] = config.validated_max_image_edge;
    input["min_target_short_px"] = config.validated_min_target_short_px;
    input["max_target_aspect"] = config.validated_max_target_aspect;
    node["input"] = input;

    YAML::Node views;
    views["source_tile_edges"] = config.gallery_tile_edges;
    views["overlap"] = config.view_overlap;
    node["views"] = views;

    YAML::Node query;
    query["roi_long_edges"] = config.query_roi_target_lengths;
    query["grid_bins"] = config.query_local_cells;
    query["max_tokens_per_bin"] = config.query_local_max_per_cell;
    query["min_local_evidence"] = config.query_min_local_evidence;
    node["query"] = query;

    YAML::Node regions;
    regions["coarse_dimension"] = config.coarse_dimension;
    regions["local_representatives"] = config.local_representatives;
    regions["window_ratios"] = config.region_window_ratios;
    regions["stride_ratio"] = config.region_window_stride_ratio;
    regions["min_valid_fraction"] = config.region_min_valid_fraction;
    node["regions"] = regions;

    YAML::Node merge;
    merge["enabled"] = config.merge_enabled;
    merge["epsilon"] = config.merge_epsilon;
    merge["max_leaf_side_patches"] = config.max_leaf_side_patches;
    node["merge"] = merge;

    YAML::Node quantization;
    quantization["format"] = config.quantize_int8 ? "int8" : "fp32";
    node["quantization"] = quantization;

    YAML::Node search;
    search["block_descriptors"] = config.region_scan_block;
    search["region_pool"] = config.region_topk;
    search["local_view_pool"] = config.local_view_topk; // retained only for old profile compatibility
    search["verify_k"] = config.fine_verify_k;
    search["local_region_pool"] = config.channel_candidate_limit;
    search["coarse_k"] = config.coarse_k;
    search["coarse_dedup_iou"] = config.coarse_dedup_iou;
    search["coarse_dedup_area_ratio"] = config.coarse_dedup_area_ratio;
    node["search"] = search;

    YAML::Node fine;
    fine["candidate_expand"] = config.fine_candidate_expand;
    fine["min_template_short_patches"] = config.fine_template_min_short_patches;
    fine["scale_step"] = config.fine_template_scale_step;
    fine["max_scales"] = config.fine_template_max_sizes;
    fine["peaks_per_candidate"] = config.fine_peaks_per_candidate;
    fine["refinement_rounds"] = config.fine_refinement_rounds;
    fine["match_cosine_threshold"] = config.fine_match_cosine_threshold;
    fine["normalized_position_tolerance"] = config.fine_position_tolerance;
    fine["score_weights"] = std::vector<double>{config.score_weight_template, config.score_weight_coverage,
                                               config.score_weight_consistency};
    fine["nms_iou"] = config.fine_nms_iou;
    fine["final_k"] = config.final_k;
    node["fine"] = fine;

    YAML::Node decision;
    if (config.enable_decision_threshold)
    {
        decision["threshold"] = config.decision_threshold;
    }
    else
    {
        decision["threshold"] = YAML::Null;
    }
    node["decision"] = decision;

    node["deadline_ms"] = config.query_deadline_ms;

    return node;
}

DinoRegionSearchConfig dinoConfigFromYamlNode(const YAML::Node &node)
{
    if (!node || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile must be a YAML mapping");
    }

    DinoRegionSearchConfig config;

    if (node["profile_id"])
    {
        config.profile_id = node["profile_id"].as<std::string>();
    }
    if (node["mode"])
    {
        const auto mode = node["mode"].as<std::string>();
        if (mode == "appearance")
        {
            config.consistency_mode = DinoConsistencyMode::Appearance;
        }
        else if (mode == "instance")
        {
            config.consistency_mode = DinoConsistencyMode::Instance;
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Profile mode must be appearance or instance, got '%s'", mode.c_str());
        }
    }


    if (const auto model = node["model"])
    {
        if (model["name"])
        {
            config.model_name = model["name"].as<std::string>();
        }
        if (model["weights_path"])
        {
            config.weights_file = dinoPathFromUtf8(model["weights_path"].as<std::string>());
        }
        if (model["encoder_edge"])
        {
            config.encoder_edge = model["encoder_edge"].as<int>();
        }
        if (model["precision"])
        {
            const auto prec = model["precision"].as<std::string>();
            if (prec == "fp16")
            {
                config.model_precision = irt::model::ModelPrecision::FP16;
            }
            else if (prec == "fp32")
            {
                config.model_precision = irt::model::ModelPrecision::FP32;
            }
            else
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Profile model precision must be fp16 or fp32, got '%s'", prec.c_str());
            }
        }
        if (model["batch_size"])
        {
            config.model_batch_size = model["batch_size"].as<size_t>();
        }
        if (model["runtime"])
        {
            config.model_runtime = irt::model::ModelRuntime(model["runtime"].as<std::string>());
        }
    }

    if (const auto input = node["input"])
    {
        if (input["min_image_edge"]) config.validated_min_image_edge = input["min_image_edge"].as<int>();
        if (input["max_image_edge"]) config.validated_max_image_edge = input["max_image_edge"].as<int>();
        if (input["min_target_short_px"]) config.validated_min_target_short_px = input["min_target_short_px"].as<double>();
        if (input["max_target_aspect"]) config.validated_max_target_aspect = input["max_target_aspect"].as<double>();
    }

    if (const auto views = node["views"])
    {
        if (views["source_tile_edges"])
        {
            config.gallery_tile_edges = views["source_tile_edges"].as<std::vector<int>>();
        }
        if (views["overlap"])
        {
            config.view_overlap = views["overlap"].as<double>();
        }
    }

    if (const auto query = node["query"])
    {
        if (query["roi_long_edges"])
        {
            config.query_roi_target_lengths = query["roi_long_edges"].as<std::vector<double>>();
        }
        if (query["grid_bins"]) config.query_local_cells = query["grid_bins"].as<int>();
        if (query["max_tokens_per_bin"]) config.query_local_max_per_cell = query["max_tokens_per_bin"].as<int>();
        if (query["min_local_evidence"]) config.query_min_local_evidence = query["min_local_evidence"].as<int>();
    }

    if (const auto regions = node["regions"])
    {
        if (regions["coarse_dimension"]) config.coarse_dimension = regions["coarse_dimension"].as<int>();
        if (regions["local_representatives"]) config.local_representatives = regions["local_representatives"].as<int>();
        if (regions["window_ratios"])
        {
            config.region_window_ratios = regions["window_ratios"].as<std::vector<double>>();
        }
        if (regions["stride_ratio"]) config.region_window_stride_ratio = regions["stride_ratio"].as<double>();
        if (regions["min_valid_fraction"]) config.region_min_valid_fraction = regions["min_valid_fraction"].as<double>();
    }

    if (const auto merge = node["merge"])
    {
        if (merge["enabled"]) config.merge_enabled = merge["enabled"].as<bool>();
        if (merge["epsilon"]) config.merge_epsilon = merge["epsilon"].as<double>();
        if (merge["max_leaf_side_patches"]) config.max_leaf_side_patches = merge["max_leaf_side_patches"].as<int>();
    }

    if (const auto quant = node["quantization"])
    {
        if (quant["format"])
        {
            const auto fmt = quant["format"].as<std::string>();
            if (fmt == "int8")
            {
                config.quantize_int8 = true;
            }
            else if (fmt == "fp32")
            {
                config.quantize_int8 = false;
            }
            else
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Profile quantization format must be int8 or fp32, got '%s'", fmt.c_str());
            }
        }
    }

    if (const auto search = node["search"])
    {
        if (search["block_descriptors"]) config.region_scan_block = search["block_descriptors"].as<size_t>();
        if (search["region_pool"]) config.region_topk = search["region_pool"].as<size_t>();
        if (search["verify_k"]) config.fine_verify_k = search["verify_k"].as<size_t>();
        if (search["local_view_pool"]) config.local_view_topk = search["local_view_pool"].as<size_t>();
        if (search["local_region_pool"]) config.channel_candidate_limit = search["local_region_pool"].as<size_t>();
        if (search["coarse_k"]) config.coarse_k = search["coarse_k"].as<size_t>();
        if (search["coarse_dedup_iou"]) config.coarse_dedup_iou = search["coarse_dedup_iou"].as<double>();
        if (search["coarse_dedup_area_ratio"]) config.coarse_dedup_area_ratio = search["coarse_dedup_area_ratio"].as<double>();
    }

    if (const auto fine = node["fine"])
    {
        if (fine["candidate_expand"]) config.fine_candidate_expand = fine["candidate_expand"].as<double>();
        if (fine["min_template_short_patches"]) config.fine_template_min_short_patches = fine["min_template_short_patches"].as<int>();
        if (fine["scale_step"]) config.fine_template_scale_step = fine["scale_step"].as<double>();
        if (fine["max_scales"]) config.fine_template_max_sizes = fine["max_scales"].as<int>();
        if (fine["peaks_per_candidate"]) config.fine_peaks_per_candidate = fine["peaks_per_candidate"].as<int>();
        if (fine["refinement_rounds"]) config.fine_refinement_rounds = fine["refinement_rounds"].as<int>();
        if (fine["match_cosine_threshold"]) config.fine_match_cosine_threshold = fine["match_cosine_threshold"].as<double>();
        if (fine["normalized_position_tolerance"]) config.fine_position_tolerance = fine["normalized_position_tolerance"].as<double>();
        if (fine["nms_iou"]) config.fine_nms_iou = fine["nms_iou"].as<double>();
        if (fine["final_k"]) config.final_k = fine["final_k"].as<size_t>();
        if (fine["score_weights"] && fine["score_weights"].IsSequence() && fine["score_weights"].size() == 3U)
        {
            config.score_weight_template = fine["score_weights"][0].as<double>();
            config.score_weight_coverage = fine["score_weights"][1].as<double>();
            config.score_weight_consistency = fine["score_weights"][2].as<double>();
        }
    }

    if (const auto decision = node["decision"])
    {
        if (decision["threshold"] && !decision["threshold"].IsNull())
        {
            config.enable_decision_threshold = true;
            config.decision_threshold = decision["threshold"].as<double>();
        }
        else
        {
            config.enable_decision_threshold = false;
            config.decision_threshold = 0.0;
        }
    }

    if (node["deadline_ms"])
    {
        config.query_deadline_ms = node["deadline_ms"].as<int64_t>();
    }

    return config;
}

int dinoResolveEncoderEdge(const DinoRegionSearchConfig &config, const int patch_size)
{
    if (patch_size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Backbone patch size must be positive");
    }
    if (config.encoder_edge != 0)
    {
        if (config.encoder_edge % patch_size != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Encoder edge %d must be divisible by backbone patch size %d", config.encoder_edge,
                                 patch_size);
        }
        return config.encoder_edge;
    }

    // spec DEFAULT：输入长边 512/518，即 patch 的整数倍中最接近 512 的取值。
    const int multiplier = static_cast<int>(std::lround(512.0 / static_cast<double>(patch_size)));
    const int edge       = std::max(1, multiplier) * patch_size;
    if (edge < 32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Resolved encoder edge is too small: %d", edge);
    }
    return edge;
}

} // namespace irt::features::priv

namespace irt::features {

void DinoRegionSearchConfig::validate() const
{
    priv::dinoValidateConfig(*this);
}

int DinoRegionSearchConfig::resolvedEncoderEdge(const int patch_size) const
{
    return priv::dinoResolveEncoderEdge(*this, patch_size);
}

} // namespace irt::features
