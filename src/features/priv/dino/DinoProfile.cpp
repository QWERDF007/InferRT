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
    if (config.preset_id.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Preset id must not be empty");
    }
    if (config.model.model_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model name must not be empty");
    }
    if (config.model.weights_id.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model weights_id must not be empty");
    }
    config.runtime.model_runtime.validate();

    if (config.model.encoder_edge != 0 && config.model.encoder_edge < 32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model encoder edge must be 0 or >= 32");
    }
    if (config.runtime.model_batch_size == 0U || config.runtime.model_batch_size > 64U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime model batch size must be within 1..64");
    }

    if (config.gallery_views.gallery_tile_edges.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery views requires at least one tile edge");
    }
    for (const auto edge : config.gallery_views.gallery_tile_edges)
    {
        if (edge < 32)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery tile edge must be >= 32 pixels");
        }
    }
    requireRange(config.gallery_views.view_overlap, 0.0, 0.75, "view_overlap");

    if (config.descriptors.coarse_dimension < 1 || config.descriptors.coarse_dimension > 384)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Coarse dimension must be in 1..384");
    if (config.descriptors.local_representatives != 0 && config.descriptors.local_representatives != 64
        && config.descriptors.local_representatives != 128)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local representatives must be 0, 64 or 128");
    if (config.fine_match.fine_verify_k > 256U)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Verification budget must be at most 256");
    if (config.descriptors.region_window_ratios.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Descriptors requires at least one window ratio");
    }
    for (const auto ratio : config.descriptors.region_window_ratios)
    {
        requireRange(ratio, 0.05, 1.0, "region_window_ratios");
    }
    requireRange(config.descriptors.region_window_stride_ratio, 0.05, 1.0, "region_window_stride_ratio");
    requireRange(config.descriptors.region_min_valid_fraction, 0.0, 1.0, "region_min_valid_fraction");
    requireRange(config.descriptors.merge_epsilon, 0.0, 1.0, "merge_epsilon");
    if (config.descriptors.max_leaf_side_patches < 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Descriptors max leaf side must be >= 1 patch");
    }

    if (config.diagnostics.validated_min_image_edge < 1
        || config.diagnostics.validated_max_image_edge <= config.diagnostics.validated_min_image_edge)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Diagnostics validated image edge range is invalid");
    }
    requirePositive(config.diagnostics.validated_min_target_short_px, "validated_min_target_short_px");
    requirePositive(config.diagnostics.validated_max_target_aspect, "validated_max_target_aspect");

    requireRange(config.coarse_scan.coarse_dedup_iou, 0.0, 1.0, "coarse_dedup_iou");
    requireRange(config.coarse_scan.coarse_dedup_area_ratio, 1.0, 64.0, "coarse_dedup_area_ratio");
    requireRange(config.fine_match.fine_template_scale_step, 1.01, 4.0, "fine_template_scale_step");
    if (config.fine_match.fine_refinement_rounds < 0 || config.fine_match.fine_refinement_rounds > 8)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Fine refinement rounds must be within 0..8");
    }

    if (config.query_features.query_roi_target_lengths.size() > 3)
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "At most three query contexts are supported");
    if (config.query_features.query_roi_target_lengths.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query features requires at least one ROI target length");
    }
    for (const auto length : config.query_features.query_roi_target_lengths)
    {
        requirePositive(length, "query_roi_target_lengths");
    }
    if (config.query_features.query_local_cells < 1 || config.query_features.query_local_cells > 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query local cells must be within 1..4");
    }
    if (config.query_features.query_local_max_per_cell < 1 || config.query_features.query_local_max_per_cell > 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query local max per cell must be within 1..2");
    }
    if (config.query_features.query_min_local_evidence < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query min local evidence must be >= 0");
    }

    if (config.coarse_scan.region_topk == 0U || config.coarse_scan.channel_candidate_limit == 0U
        || config.coarse_scan.coarse_k == 0U || config.coarse_scan.final_k == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Coarse scan candidate quotas must be positive");
    }
    if (config.coarse_scan.coarse_k > config.coarse_scan.channel_candidate_limit)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Coarse scan coarse_k must not exceed channel_candidate_limit");
    }
    if (config.coarse_scan.final_k > config.coarse_scan.coarse_k)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Coarse scan final_k must not exceed coarse_k");
    }
    if (config.runtime.region_scan_block == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime scan block must be positive");
    }

    requireRange(config.fine_match.fine_candidate_expand, 1.0, 4.0, "fine_candidate_expand");
    if (config.fine_match.fine_template_max_sizes < 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Fine template max sizes must be >= 1");
    }
    if (config.fine_match.fine_peaks_per_candidate < 1
        || config.fine_match.fine_peaks_per_candidate > kDinoMaxPeaksPerCandidate)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Fine peaks per candidate must be within 1..%d", kDinoMaxPeaksPerCandidate);
    }
    requireRange(config.fine_match.fine_match_cosine_threshold, -1.0, 1.0, "fine_match_cosine_threshold");
    requireRange(config.fine_match.fine_nms_iou, 0.0, 1.0, "fine_nms_iou");

    const double weight_sum = config.fine_match.score_weight_template + config.fine_match.score_weight_coverage
                            + config.fine_match.score_weight_consistency;
    if (std::abs(weight_sum - 1.0) > 1e-6)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Fine match score weights must sum to 1.0, got %g", weight_sum);
    }
    for (const auto weight : {config.fine_match.score_weight_template, config.fine_match.score_weight_coverage,
                              config.fine_match.score_weight_consistency})
    {
        requireRange(weight, 0.0, 1.0, "score weights");
    }

    if (config.decision.enable_decision_threshold)
    {
        requireRange(config.decision.decision_threshold, 0.0, 1.0, "decision_threshold");
    }
    if (config.runtime.query_deadline_ms <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime query deadline must be positive");
    }
}

YAML::Node dinoConfigToYamlNode(const DinoRegionSearchConfig &config)
{
    YAML::Node node;
    node["preset_id"] = config.preset_id;

    YAML::Node model;
    model["model_name"] = config.model.model_name;
    model["weights_file"] = dinoPathToUtf8(config.model.weights_file);
    model["weights_id"] = config.model.weights_id;
    model["encoder_edge"] = config.model.encoder_edge;
    node["model"] = model;

    YAML::Node gallery_views;
    gallery_views["gallery_tile_edges"] = config.gallery_views.gallery_tile_edges;
    gallery_views["view_overlap"] = config.gallery_views.view_overlap;
    node["gallery_views"] = gallery_views;

    YAML::Node descriptors;
    descriptors["region_window_ratios"] = config.descriptors.region_window_ratios;
    descriptors["region_window_stride_ratio"] = config.descriptors.region_window_stride_ratio;
    descriptors["region_min_valid_fraction"] = config.descriptors.region_min_valid_fraction;
    descriptors["coarse_dimension"] = config.descriptors.coarse_dimension;
    descriptors["local_representatives"] = config.descriptors.local_representatives;
    descriptors["merge_enabled"] = config.descriptors.merge_enabled;
    descriptors["merge_epsilon"] = config.descriptors.merge_epsilon;
    descriptors["max_leaf_side_patches"] = config.descriptors.max_leaf_side_patches;
    descriptors["quantize_int8"] = config.descriptors.quantize_int8;
    node["descriptors"] = descriptors;

    YAML::Node query_features;
    query_features["query_roi_target_lengths"] = config.query_features.query_roi_target_lengths;
    query_features["query_local_cells"] = config.query_features.query_local_cells;
    query_features["query_local_max_per_cell"] = config.query_features.query_local_max_per_cell;
    query_features["query_min_local_evidence"] = config.query_features.query_min_local_evidence;
    node["query_features"] = query_features;

    YAML::Node coarse_scan;
    coarse_scan["region_topk"] = config.coarse_scan.region_topk;
    coarse_scan["channel_candidate_limit"] = config.coarse_scan.channel_candidate_limit;
    coarse_scan["coarse_k"] = config.coarse_scan.coarse_k;
    coarse_scan["coarse_dedup_iou"] = config.coarse_scan.coarse_dedup_iou;
    coarse_scan["coarse_dedup_area_ratio"] = config.coarse_scan.coarse_dedup_area_ratio;
    coarse_scan["final_k"] = config.coarse_scan.final_k;
    node["coarse_scan"] = coarse_scan;

    YAML::Node fine_match;
    fine_match["fine_verify_k"] = config.fine_match.fine_verify_k;
    fine_match["fine_candidate_expand"] = config.fine_match.fine_candidate_expand;
    fine_match["fine_template_scale_step"] = config.fine_match.fine_template_scale_step;
    fine_match["fine_template_max_sizes"] = config.fine_match.fine_template_max_sizes;
    fine_match["fine_peaks_per_candidate"] = config.fine_match.fine_peaks_per_candidate;
    fine_match["fine_refinement_rounds"] = config.fine_match.fine_refinement_rounds;
    fine_match["fine_match_cosine_threshold"] = config.fine_match.fine_match_cosine_threshold;
    fine_match["fine_nms_iou"] = config.fine_match.fine_nms_iou;
    fine_match["consistency_mode"]
        = config.fine_match.consistency_mode == DinoConsistencyMode::Appearance ? "appearance" : "instance";
    fine_match["score_weight_template"] = config.fine_match.score_weight_template;
    fine_match["score_weight_coverage"] = config.fine_match.score_weight_coverage;
    fine_match["score_weight_consistency"] = config.fine_match.score_weight_consistency;
    node["fine_match"] = fine_match;

    YAML::Node decision;
    decision["enable_decision_threshold"] = config.decision.enable_decision_threshold;
    decision["decision_threshold"] = config.decision.decision_threshold;
    node["decision"] = decision;

    YAML::Node runtime;
    runtime["model_runtime"] = config.runtime.model_runtime.toString();
    runtime["model_precision"]
        = config.runtime.model_precision == irt::model::ModelPrecision::FP16 ? "fp16" : "fp32";
    runtime["model_batch_size"] = config.runtime.model_batch_size;
    runtime["query_deadline_ms"] = config.runtime.query_deadline_ms;
    runtime["region_scan_block"] = config.runtime.region_scan_block;
    runtime["scan_backend"] = config.runtime.scan_backend == DinoScanBackend::Cpu    ? "cpu"
                            : config.runtime.scan_backend == DinoScanBackend::Cuda   ? "cuda"
                                                                                     : "auto";
    node["runtime"] = runtime;

    YAML::Node diagnostics;
    diagnostics["validated_min_image_edge"] = config.diagnostics.validated_min_image_edge;
    diagnostics["validated_max_image_edge"] = config.diagnostics.validated_max_image_edge;
    diagnostics["validated_min_target_short_px"] = config.diagnostics.validated_min_target_short_px;
    diagnostics["validated_max_target_aspect"] = config.diagnostics.validated_max_target_aspect;
    node["diagnostics"] = diagnostics;

    return node;
}

DinoRegionSearchConfig dinoConfigFromYamlNode(const YAML::Node &node)
{
    if (!node || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Profile/Config must be a YAML mapping");
    }

    DinoRegionSearchConfig config;

    // Top-level / legacy identifiers
    if (node["preset_id"])
    {
        config.preset_id = node["preset_id"].as<std::string>();
    }
    else if (node["profile_id"])
    {
        config.preset_id = node["profile_id"].as<std::string>();
    }

    if (node["mode"])
    {
        const auto mode = node["mode"].as<std::string>();
        if (mode == "appearance")
        {
            config.fine_match.consistency_mode = DinoConsistencyMode::Appearance;
        }
        else if (mode == "instance")
        {
            config.fine_match.consistency_mode = DinoConsistencyMode::Instance;
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Profile mode must be appearance or instance, got '%s'", mode.c_str());
        }
    }
    if (node["deadline_ms"])
    {
        config.runtime.query_deadline_ms = node["deadline_ms"].as<int64_t>();
    }

    // model
    if (const auto model = node["model"])
    {
        if (model["model_name"]) config.model.model_name = model["model_name"].as<std::string>();
        else if (model["name"])  config.model.model_name = model["name"].as<std::string>();

        if (model["weights_file"])      config.model.weights_file = dinoPathFromUtf8(model["weights_file"].as<std::string>());
        else if (model["weights_path"]) config.model.weights_file = dinoPathFromUtf8(model["weights_path"].as<std::string>());

        if (model["weights_id"]) config.model.weights_id = model["weights_id"].as<std::string>();
        if (model["encoder_edge"]) config.model.encoder_edge = model["encoder_edge"].as<int>();

        // legacy model fields
        if (model["precision"])
        {
            const auto prec = model["precision"].as<std::string>();
            if (prec == "fp16") config.runtime.model_precision = irt::model::ModelPrecision::FP16;
            else if (prec == "fp32") config.runtime.model_precision = irt::model::ModelPrecision::FP32;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                      "Model precision must be fp16 or fp32, got '%s'", prec.c_str());
        }
        if (model["batch_size"]) config.runtime.model_batch_size = model["batch_size"].as<size_t>();
        if (model["runtime"]) config.runtime.model_runtime = irt::model::ModelRuntime(model["runtime"].as<std::string>());
    }

    // gallery_views / legacy views
    const auto views_node = node["gallery_views"] ? node["gallery_views"] : node["views"];
    if (views_node)
    {
        if (views_node["gallery_tile_edges"])
            config.gallery_views.gallery_tile_edges = views_node["gallery_tile_edges"].as<std::vector<int>>();
        else if (views_node["source_tile_edges"])
            config.gallery_views.gallery_tile_edges = views_node["source_tile_edges"].as<std::vector<int>>();

        if (views_node["view_overlap"])
            config.gallery_views.view_overlap = views_node["view_overlap"].as<double>();
        else if (views_node["overlap"])
            config.gallery_views.view_overlap = views_node["overlap"].as<double>();
    }

    // descriptors / legacy regions, merge, quantization
    if (const auto descriptors = node["descriptors"])
    {
        if (descriptors["region_window_ratios"])
            config.descriptors.region_window_ratios = descriptors["region_window_ratios"].as<std::vector<double>>();
        if (descriptors["region_window_stride_ratio"])
            config.descriptors.region_window_stride_ratio = descriptors["region_window_stride_ratio"].as<double>();
        if (descriptors["region_min_valid_fraction"])
            config.descriptors.region_min_valid_fraction = descriptors["region_min_valid_fraction"].as<double>();
        if (descriptors["coarse_dimension"])
            config.descriptors.coarse_dimension = descriptors["coarse_dimension"].as<int>();
        if (descriptors["local_representatives"])
            config.descriptors.local_representatives = descriptors["local_representatives"].as<int>();
        if (descriptors["merge_enabled"])
            config.descriptors.merge_enabled = descriptors["merge_enabled"].as<bool>();
        if (descriptors["merge_epsilon"])
            config.descriptors.merge_epsilon = descriptors["merge_epsilon"].as<double>();
        if (descriptors["max_leaf_side_patches"])
            config.descriptors.max_leaf_side_patches = descriptors["max_leaf_side_patches"].as<int>();
        if (descriptors["quantize_int8"])
            config.descriptors.quantize_int8 = descriptors["quantize_int8"].as<bool>();
    }
    if (const auto regions = node["regions"])
    {
        if (regions["coarse_dimension"]) config.descriptors.coarse_dimension = regions["coarse_dimension"].as<int>();
        if (regions["local_representatives"]) config.descriptors.local_representatives = regions["local_representatives"].as<int>();
        if (regions["window_ratios"])
            config.descriptors.region_window_ratios = regions["window_ratios"].as<std::vector<double>>();
        if (regions["stride_ratio"]) config.descriptors.region_window_stride_ratio = regions["stride_ratio"].as<double>();
        if (regions["min_valid_fraction"]) config.descriptors.region_min_valid_fraction = regions["min_valid_fraction"].as<double>();
    }
    if (const auto merge = node["merge"])
    {
        if (merge["enabled"]) config.descriptors.merge_enabled = merge["enabled"].as<bool>();
        if (merge["epsilon"]) config.descriptors.merge_epsilon = merge["epsilon"].as<double>();
        if (merge["max_leaf_side_patches"]) config.descriptors.max_leaf_side_patches = merge["max_leaf_side_patches"].as<int>();
    }
    if (const auto quant = node["quantization"])
    {
        if (quant["format"])
        {
            const auto fmt = quant["format"].as<std::string>();
            if (fmt == "int8") config.descriptors.quantize_int8 = true;
            else if (fmt == "fp32") config.descriptors.quantize_int8 = false;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                      "Profile quantization format must be int8 or fp32, got '%s'", fmt.c_str());
        }
    }

    // query_features / legacy query
    const auto query_node = node["query_features"] ? node["query_features"] : node["query"];
    if (query_node)
    {
        if (query_node["query_roi_target_lengths"])
            config.query_features.query_roi_target_lengths = query_node["query_roi_target_lengths"].as<std::vector<double>>();
        else if (query_node["roi_long_edges"])
            config.query_features.query_roi_target_lengths = query_node["roi_long_edges"].as<std::vector<double>>();

        if (query_node["query_local_cells"]) config.query_features.query_local_cells = query_node["query_local_cells"].as<int>();
        else if (query_node["grid_bins"]) config.query_features.query_local_cells = query_node["grid_bins"].as<int>();

        if (query_node["query_local_max_per_cell"]) config.query_features.query_local_max_per_cell = query_node["query_local_max_per_cell"].as<int>();
        else if (query_node["max_tokens_per_bin"]) config.query_features.query_local_max_per_cell = query_node["max_tokens_per_bin"].as<int>();

        if (query_node["query_min_local_evidence"]) config.query_features.query_min_local_evidence = query_node["query_min_local_evidence"].as<int>();
        else if (query_node["min_local_evidence"]) config.query_features.query_min_local_evidence = query_node["min_local_evidence"].as<int>();
    }

    // coarse_scan / legacy search
    const auto scan_node = node["coarse_scan"] ? node["coarse_scan"] : node["search"];
    if (scan_node)
    {
        if (scan_node["region_topk"]) config.coarse_scan.region_topk = scan_node["region_topk"].as<size_t>();
        else if (scan_node["region_pool"]) config.coarse_scan.region_topk = scan_node["region_pool"].as<size_t>();

        if (scan_node["channel_candidate_limit"]) config.coarse_scan.channel_candidate_limit = scan_node["channel_candidate_limit"].as<size_t>();
        else if (scan_node["local_region_pool"]) config.coarse_scan.channel_candidate_limit = scan_node["local_region_pool"].as<size_t>();

        if (scan_node["coarse_k"]) config.coarse_scan.coarse_k = scan_node["coarse_k"].as<size_t>();
        if (scan_node["coarse_dedup_iou"]) config.coarse_scan.coarse_dedup_iou = scan_node["coarse_dedup_iou"].as<double>();
        if (scan_node["coarse_dedup_area_ratio"]) config.coarse_scan.coarse_dedup_area_ratio = scan_node["coarse_dedup_area_ratio"].as<double>();
        if (scan_node["final_k"]) config.coarse_scan.final_k = scan_node["final_k"].as<size_t>();

        // legacy search runtime fields
        if (scan_node["block_descriptors"]) config.runtime.region_scan_block = scan_node["block_descriptors"].as<size_t>();
        if (scan_node["verify_k"]) config.fine_match.fine_verify_k = scan_node["verify_k"].as<size_t>();
    }

    // fine_match / legacy fine
    const auto fine_node = node["fine_match"] ? node["fine_match"] : node["fine"];
    if (fine_node)
    {
        if (fine_node["fine_verify_k"]) config.fine_match.fine_verify_k = fine_node["fine_verify_k"].as<size_t>();
        else if (fine_node["verify_k"]) config.fine_match.fine_verify_k = fine_node["verify_k"].as<size_t>();

        if (fine_node["fine_candidate_expand"]) config.fine_match.fine_candidate_expand = fine_node["fine_candidate_expand"].as<double>();
        else if (fine_node["candidate_expand"]) config.fine_match.fine_candidate_expand = fine_node["candidate_expand"].as<double>();

        if (fine_node["fine_template_scale_step"]) config.fine_match.fine_template_scale_step = fine_node["fine_template_scale_step"].as<double>();
        else if (fine_node["scale_step"]) config.fine_match.fine_template_scale_step = fine_node["scale_step"].as<double>();

        if (fine_node["fine_template_max_sizes"]) config.fine_match.fine_template_max_sizes = fine_node["fine_template_max_sizes"].as<int>();
        else if (fine_node["max_scales"]) config.fine_match.fine_template_max_sizes = fine_node["max_scales"].as<int>();

        if (fine_node["fine_peaks_per_candidate"]) config.fine_match.fine_peaks_per_candidate = fine_node["fine_peaks_per_candidate"].as<int>();
        else if (fine_node["peaks_per_candidate"]) config.fine_match.fine_peaks_per_candidate = fine_node["peaks_per_candidate"].as<int>();

        if (fine_node["fine_refinement_rounds"]) config.fine_match.fine_refinement_rounds = fine_node["fine_refinement_rounds"].as<int>();
        else if (fine_node["refinement_rounds"]) config.fine_match.fine_refinement_rounds = fine_node["refinement_rounds"].as<int>();

        if (fine_node["fine_match_cosine_threshold"]) config.fine_match.fine_match_cosine_threshold = fine_node["fine_match_cosine_threshold"].as<double>();
        else if (fine_node["match_cosine_threshold"]) config.fine_match.fine_match_cosine_threshold = fine_node["match_cosine_threshold"].as<double>();

        if (fine_node["fine_nms_iou"]) config.fine_match.fine_nms_iou = fine_node["fine_nms_iou"].as<double>();
        else if (fine_node["nms_iou"]) config.fine_match.fine_nms_iou = fine_node["nms_iou"].as<double>();

        if (fine_node["final_k"]) config.coarse_scan.final_k = fine_node["final_k"].as<size_t>();

        if (fine_node["consistency_mode"])
        {
            const auto mode = fine_node["consistency_mode"].as<std::string>();
            if (mode == "appearance") config.fine_match.consistency_mode = DinoConsistencyMode::Appearance;
            else if (mode == "instance") config.fine_match.consistency_mode = DinoConsistencyMode::Instance;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid consistency_mode '%s'", mode.c_str());
        }

        if (fine_node["score_weight_template"]) config.fine_match.score_weight_template = fine_node["score_weight_template"].as<double>();
        if (fine_node["score_weight_coverage"]) config.fine_match.score_weight_coverage = fine_node["score_weight_coverage"].as<double>();
        if (fine_node["score_weight_consistency"]) config.fine_match.score_weight_consistency = fine_node["score_weight_consistency"].as<double>();

        if (fine_node["score_weights"] && fine_node["score_weights"].IsSequence() && fine_node["score_weights"].size() == 3U)
        {
            config.fine_match.score_weight_template = fine_node["score_weights"][0].as<double>();
            config.fine_match.score_weight_coverage = fine_node["score_weights"][1].as<double>();
            config.fine_match.score_weight_consistency = fine_node["score_weights"][2].as<double>();
        }
    }

    // decision
    if (const auto decision = node["decision"])
    {
        if (decision["enable_decision_threshold"])
        {
            config.decision.enable_decision_threshold = decision["enable_decision_threshold"].as<bool>();
            if (decision["decision_threshold"]) config.decision.decision_threshold = decision["decision_threshold"].as<double>();
        }
        else if (decision["threshold"])
        {
            if (!decision["threshold"].IsNull())
            {
                config.decision.enable_decision_threshold = true;
                config.decision.decision_threshold = decision["threshold"].as<double>();
            }
            else
            {
                config.decision.enable_decision_threshold = false;
                config.decision.decision_threshold = 0.0;
            }
        }
    }

    // runtime
    if (const auto runtime = node["runtime"])
    {
        if (runtime["model_runtime"]) config.runtime.model_runtime = irt::model::ModelRuntime(runtime["model_runtime"].as<std::string>());
        else if (runtime["runtime"]) config.runtime.model_runtime = irt::model::ModelRuntime(runtime["runtime"].as<std::string>());

        if (runtime["model_precision"])
        {
            const auto prec = runtime["model_precision"].as<std::string>();
            if (prec == "fp16") config.runtime.model_precision = irt::model::ModelPrecision::FP16;
            else if (prec == "fp32") config.runtime.model_precision = irt::model::ModelPrecision::FP32;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime model precision must be fp16 or fp32");
        }
        else if (runtime["precision"])
        {
            const auto prec = runtime["precision"].as<std::string>();
            if (prec == "fp16") config.runtime.model_precision = irt::model::ModelPrecision::FP16;
            else if (prec == "fp32") config.runtime.model_precision = irt::model::ModelPrecision::FP32;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime model precision must be fp16 or fp32");
        }

        if (runtime["model_batch_size"]) config.runtime.model_batch_size = runtime["model_batch_size"].as<size_t>();
        else if (runtime["batch_size"]) config.runtime.model_batch_size = runtime["batch_size"].as<size_t>();

        if (runtime["query_deadline_ms"]) config.runtime.query_deadline_ms = runtime["query_deadline_ms"].as<int64_t>();
        else if (runtime["deadline_ms"]) config.runtime.query_deadline_ms = runtime["deadline_ms"].as<int64_t>();

        if (runtime["region_scan_block"]) config.runtime.region_scan_block = runtime["region_scan_block"].as<size_t>();
        else if (runtime["block_descriptors"]) config.runtime.region_scan_block = runtime["block_descriptors"].as<size_t>();

        if (runtime["scan_backend"])
        {
            const auto sb = runtime["scan_backend"].as<std::string>();
            if (sb == "cpu") config.runtime.scan_backend = DinoScanBackend::Cpu;
            else if (sb == "cuda") config.runtime.scan_backend = DinoScanBackend::Cuda;
            else if (sb == "auto") config.runtime.scan_backend = DinoScanBackend::Auto;
            else throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Runtime scan_backend must be auto, cpu or cuda");
        }
    }

    // diagnostics / legacy input
    const auto diag_node = node["diagnostics"] ? node["diagnostics"] : node["input"];
    if (diag_node)
    {
        if (diag_node["validated_min_image_edge"]) config.diagnostics.validated_min_image_edge = diag_node["validated_min_image_edge"].as<int>();
        else if (diag_node["min_image_edge"]) config.diagnostics.validated_min_image_edge = diag_node["min_image_edge"].as<int>();

        if (diag_node["validated_max_image_edge"]) config.diagnostics.validated_max_image_edge = diag_node["validated_max_image_edge"].as<int>();
        else if (diag_node["max_image_edge"]) config.diagnostics.validated_max_image_edge = diag_node["max_image_edge"].as<int>();

        if (diag_node["validated_min_target_short_px"]) config.diagnostics.validated_min_target_short_px = diag_node["validated_min_target_short_px"].as<double>();
        else if (diag_node["min_target_short_px"]) config.diagnostics.validated_min_target_short_px = diag_node["min_target_short_px"].as<double>();

        if (diag_node["validated_max_target_aspect"]) config.diagnostics.validated_max_target_aspect = diag_node["validated_max_target_aspect"].as<double>();
        else if (diag_node["max_target_aspect"]) config.diagnostics.validated_max_target_aspect = diag_node["max_target_aspect"].as<double>();
    }

    return config;
}

int dinoResolveEncoderEdge(const DinoRegionSearchConfig &config, const int patch_size)
{
    if (patch_size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Backbone patch size must be positive");
    }
    if (config.model.encoder_edge != 0)
    {
        if (config.model.encoder_edge % patch_size != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Encoder edge %d must be divisible by backbone patch size %d", config.model.encoder_edge,
                                 patch_size);
        }
        return config.model.encoder_edge;
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
