/**
 * @file DinoContracts.cpp
 * @brief YAML contracts for profile, request, response, and reports.
 */

#include "DinoContracts.hpp"
#include "DinoPaths.hpp"
#include "DinoProfile.hpp"

#include <yaml-cpp/yaml.h>
#include <inferrt/core/Exception.hpp>

#include <cmath>
#include <sstream>
#include <utility>

namespace irt::features::priv {

const char *dinoStatusName(const DinoSearchStatus status) noexcept
{
    switch (status)
    {
    case DinoSearchStatus::Completed:  return "completed";
    case DinoSearchStatus::Incomplete: return "incomplete";
    case DinoSearchStatus::Failed:     return "error";
    default:                           return "unknown";
    }
}

const char *dinoDecisionName(const DinoSearchDecision decision) noexcept
{
    switch (decision)
    {
    case DinoSearchDecision::RankedOnly: return "ranked_only";
    case DinoSearchDecision::Matches:    return "matches";
    case DinoSearchDecision::NoMatch:    return "no_match";
    case DinoSearchDecision::Incomplete: return "incomplete";
    case DinoSearchDecision::Error:      return "error";
    default:                             return "unknown";
    }
}

} // namespace irt::features::priv

namespace irt::features {

DinoRegionSearchConfig dinoConfigFromYaml(const std::string &text)
{
    try
    {
        const auto node = YAML::Load(text);
        auto config = priv::dinoConfigFromYamlNode(node);
        config.validate();
        return config;
    }
    catch (const YAML::Exception &error)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "YAML parse error in profile: %s", error.what());
    }
}

DinoSearchRequest dinoSearchRequestFromYaml(const std::string &text)
{
    YAML::Node node;
    try
    {
        node = YAML::Load(text);
    }
    catch (const YAML::Exception &error)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "YAML parse error in query request: %s", error.what());
    }

    try
    {
        if (!node || !node.IsMap())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request must be a YAML mapping");
        }

        DinoSearchRequest request;
        request.request_id = node["request_id"] ? node["request_id"].as<std::string>() : "yaml-query";

        if (!node["query_path"])
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Query request missing required field 'query_path'");
        }
        request.query_path = priv::dinoPathFromUtf8(node["query_path"].as<std::string>());
        if (request.query_path.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request 'query_path' must not be empty");
        }

        const bool has_bbox = node["bbox"] && !node["bbox"].IsNull();
        const bool has_polygon = node["polygon"] && !node["polygon"].IsNull();

        if (has_bbox && has_polygon)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "bbox and polygon are mutually exclusive");
        }
        if (!has_bbox && !has_polygon)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Query request requires either 'bbox' or 'polygon'");
        }

        if (has_bbox)
        {
            const auto bbox_node = node["bbox"];
            if (!bbox_node.IsSequence() || bbox_node.size() != 4U)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "bbox must be an array of 4 coordinates [x0, y0, x1, y1]");
            }
            for (size_t i = 0; i < 4U; ++i)
            {
                double val = 0.0;
                try
                {
                    val = bbox_node[i].as<double>();
                }
                catch (const std::exception &)
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "bbox coordinate must be a valid number");
                }
                if (!std::isfinite(val))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "bbox coordinates must be finite numbers");
                }
            }
            const auto x0 = bbox_node[0].as<float>();
            const auto y0 = bbox_node[1].as<float>();
            const auto x1 = bbox_node[2].as<float>();
            const auto y1 = bbox_node[3].as<float>();
            if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "bbox coordinates must fit finite float values");
            }
            if (x1 <= x0 || y1 <= y0)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "bbox dimensions must be positive (x1 > x0, y1 > y0)");
            }
            request.roi.has_bbox = true;
            request.roi.has_polygon = false;
            request.roi.bbox = DinoSearchRect{x0, y0, x1, y1};
        }
        else
        {
            const auto poly_node = node["polygon"];
            if (!poly_node.IsSequence() || poly_node.size() < 3U)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "polygon must be an array of at least 3 vertices");
            }
            request.roi.has_polygon = true;
            request.roi.has_bbox = false;
            for (size_t i = 0; i < poly_node.size(); ++i)
            {
                const auto pt = poly_node[i];
                if (!pt.IsSequence() || pt.size() != 2U)
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "polygon vertex must be [x, y]");
                }
                double x = 0.0;
                double y = 0.0;
                try
                {
                    x = pt[0].as<double>();
                    y = pt[1].as<double>();
                }
                catch (const std::exception &)
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "polygon coordinate must be a valid number");
                }
                if (!std::isfinite(x) || !std::isfinite(y))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "polygon coordinates must be finite numbers");
                }
                const float xf = static_cast<float>(x);
                const float yf = static_cast<float>(y);
                if (!std::isfinite(xf) || !std::isfinite(yf))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "polygon coordinates must fit finite float values");
                }
                request.roi.polygon.push_back(DinoSearchPoint{xf, yf});
            }
        }

        if (node["top_k"])
        {
            int64_t top_k = 0;
            try
            {
                top_k = node["top_k"].as<int64_t>();
            }
            catch (const std::exception &)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be an integer");
            }
            if (top_k < 0)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be >= 0");
            }
            request.top_k = static_cast<size_t>(top_k);
        }

        if (node["include_self"])
        {
            request.include_self = node["include_self"].as<bool>();
        }

        if (node["profile_id"])
        {
            request.profile_id = node["profile_id"].as<std::string>();
        }
        if (node["deadline_ms"])
        {
            int64_t deadline_ms = 0;
            try
            {
                deadline_ms = node["deadline_ms"].as<int64_t>();
            }
            catch (const std::exception &)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "deadline_ms must be a positive integer");
            }
            if (deadline_ms <= 0)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "deadline_ms must be a positive integer");
            }
            request.deadline_ms = deadline_ms;
        }

        return request;
    }
    catch (const YAML::Exception &error)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Invalid value in query request: %s", error.what());
    }
}

std::string dinoSearchResponseToYaml(const DinoSearchResponse &response)
{
    const auto rectToYaml = [](const DinoSearchRect &rect) {
        YAML::Node value(YAML::NodeType::Sequence);
        value.push_back(rect.x0);
        value.push_back(rect.y0);
        value.push_back(rect.x1);
        value.push_back(rect.y1);
        return value;
    };
    const auto candidatesToYaml = [](const std::vector<DinoCoarseCandidate> &candidates) {
        YAML::Node values(YAML::NodeType::Sequence);
        for (const auto &candidate : candidates)
        {
            YAML::Node value;
            value["source_path"] = priv::dinoPathToUtf8(candidate.source_path);
            value["bbox"] = YAML::Node(YAML::NodeType::Sequence);
            value["bbox"].push_back(candidate.bbox.x0);
            value["bbox"].push_back(candidate.bbox.y0);
            value["bbox"].push_back(candidate.bbox.x1);
            value["bbox"].push_back(candidate.bbox.y1);
            values.push_back(value);
        }
        return values;
    };

    YAML::Node node;
    node["request_id"] = response.request_id;
    node["status"] = priv::dinoStatusName(response.status);
    node["decision"] = priv::dinoDecisionName(response.decision);
    node["completed_candidates"] = response.completed_candidates;
    node["total_candidates"] = response.total_candidates;

    YAML::Node results(YAML::NodeType::Sequence);
    for (const auto &item : response.results)
    {
        YAML::Node result;
        result["image_id"] = item.image_id;
        result["source_path"] = item.source_path;
        result["bbox"] = rectToYaml(item.bbox);
        result["score"] = item.score;
        result["template_similarity"] = item.template_similarity;
        result["query_coverage"] = item.query_coverage;
        result["spatial_consistency"] = item.spatial_consistency;
        YAML::Node coarse_sources(YAML::NodeType::Sequence);
        for (const auto &source : item.coarse_sources)
        {
            coarse_sources.push_back(source);
        }
        result["coarse_sources"] = coarse_sources;
        results.push_back(result);
    }
    node["results"] = results;
    node["region_candidates"] = candidatesToYaml(response.region_candidates);
    node["local_candidates"] = candidatesToYaml(response.local_candidates);
    node["coarse_candidates"] = candidatesToYaml(response.coarse_candidates);
    node["localized_candidates"] = candidatesToYaml(response.localized_candidates);
    node["verification_candidates"] = candidatesToYaml(response.verification_candidates);
    node["score_kind"] = response.score_kind;
    node["verified_candidates"] = response.verified_candidates;

    YAML::Node timings;
    timings["decode_ms"] = response.timings.decode_ms;
    timings["query_extract_ms"] = response.timings.query_extract_ms;
    timings["region_scan_ms"] = response.timings.region_scan_ms;
    timings["local_scan_ms"] = response.timings.local_scan_ms;
    timings["local_window_rescore_ms"] = response.timings.local_window_rescore_ms;
    timings["fusion_ms"] = response.timings.fusion_ms;
    timings["fine_extract_ms"] = response.timings.fine_extract_ms;
    timings["fine_match_ms"] = response.timings.fine_match_ms;
    timings["output_ms"] = response.timings.output_ms;
    timings["queue_ms"] = response.timings.queue_ms;
    timings["wall_ms"] = response.timings.wall_ms;
    node["timings"] = timings;

    YAML::Node diagnostics;
    diagnostics["scanned_region_descriptors"] = response.diagnostics.scanned_region_descriptors;
    diagnostics["scanned_local_descriptors"] = response.diagnostics.scanned_local_descriptors;
    diagnostics["retained_local_views"] = response.diagnostics.retained_local_views;
    diagnostics["region_candidates"] = response.diagnostics.region_candidates;
    diagnostics["local_candidates"] = response.diagnostics.local_candidates;
    diagnostics["fused_candidates"] = response.diagnostics.fused_candidates;
    diagnostics["view_count"] = response.diagnostics.view_count;
    diagnostics["model_forwards"] = response.diagnostics.model_forwards;
    diagnostics["cache_hits"] = response.diagnostics.cache_hits;
    diagnostics["query_local_descriptors"] = response.diagnostics.query_local_descriptors;
    diagnostics["query_valid_cells"] = response.diagnostics.query_valid_cells;
    diagnostics["low_local_evidence"] = response.diagnostics.low_local_evidence;
    diagnostics["outside_validated_profile"] = response.diagnostics.outside_validated_profile;
    diagnostics["peak_rss_bytes"] = response.diagnostics.peak_rss_bytes;
    diagnostics["gpu_allocated_peak_bytes"] = response.diagnostics.gpu_allocated_peak_bytes;
    diagnostics["gpu_reserved_delta_peak_bytes"] = response.diagnostics.gpu_reserved_delta_peak_bytes;
    diagnostics["similarity_backend"] = response.diagnostics.similarity_backend;
    diagnostics["local_scan_compact"] = response.diagnostics.local_scan_compact;
    diagnostics["query_local_selection"] = response.diagnostics.query_local_selection;
    node["diagnostics"] = diagnostics;

    if (!response.message.empty())
    {
        node["message"] = response.message;
    }

    YAML::Emitter emitter;
    emitter << node;
    return std::string(emitter.c_str()) + "\n";
}

std::string dinoBuildReportToYaml(const DinoBuildReport &report)
{
    YAML::Node node;
    node["image_count"] = report.image_count;
    node["failed_image_count"] = report.failed_image_count;
    node["ready_with_errors"] = report.ready_with_errors;
    node["view_count"] = report.view_count;
    node["region_descriptor_count"] = report.region_descriptor_count;
    node["local_descriptor_count"] = report.local_descriptor_count;
    node["original_patch_count"] = report.original_patch_count;
    node["index_bytes"] = report.index_bytes;
    node["duration_ms"] = report.duration_ms;

    YAML::Node failed_files(YAML::NodeType::Sequence);
    for (const auto &f : report.failed_files)
    {
        failed_files.push_back(f);
    }
    node["failed_files"] = failed_files;

    YAML::Node messages(YAML::NodeType::Sequence);
    for (const auto &m : report.messages)
    {
        messages.push_back(m);
    }
    node["messages"] = messages;

    YAML::Emitter emitter;
    emitter << node;
    return std::string(emitter.c_str()) + "\n";
}

std::vector<DinoSearchRequest> dinoSearchRequestsFromYaml(const std::string &text)
{
    try {
        auto node = YAML::Load(text);
        if (!node.IsSequence()) throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Batch requests must be a YAML sequence");
        std::vector<DinoSearchRequest> requests;
        requests.reserve(node.size());
        for (size_t i = 0; i < node.size(); ++i) {
            YAML::Emitter emitter;emitter << node[i];
            auto request = dinoSearchRequestFromYaml(emitter.c_str());
            if (!node[i]["request_id"] || request.request_id.empty()) request.request_id = "batch-" + std::to_string(i + 1);
            requests.push_back(std::move(request));
        }
        return requests;
    } catch (const YAML::Exception &error) {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid batch request YAML: %s", error.what());
    }
}

} // namespace irt::features
