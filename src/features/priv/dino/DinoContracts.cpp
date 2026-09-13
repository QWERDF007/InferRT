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

    if (!node || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request must be a YAML mapping");
    }

    DinoSearchRequest request;
    request.request_id = node["request_id"] ? node["request_id"].as<std::string>() : "yaml-query";

    if (!node["query_path"])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request missing required field 'query_path'");
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
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request requires either 'bbox' or 'polygon'");
    }

    if (has_bbox)
    {
        const auto bbox_node = node["bbox"];
        if (!bbox_node.IsSequence() || bbox_node.size() != 4U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "bbox must be an array of 4 coordinates [x0, y0, x1, y1]");
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
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "bbox coordinates must be finite numbers");
            }
        }
        const auto x0 = bbox_node[0].as<float>();
        const auto y0 = bbox_node[1].as<float>();
        const auto x1 = bbox_node[2].as<float>();
        const auto y1 = bbox_node[3].as<float>();
        if (x1 <= x0 || y1 <= y0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "bbox dimensions must be positive (x1 > x0, y1 > y0)");
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
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "polygon must be an array of at least 3 vertices");
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
            double x = 0.0, y = 0.0;
            try
            {
                x = pt[0].as<double>();
                y = pt[1].as<double>();
            }
            catch (const std::exception &)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "polygon coordinate must be a valid number");
            }
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "polygon coordinates must be finite numbers");
            }
            request.roi.polygon.push_back(DinoSearchPoint{static_cast<float>(x), static_cast<float>(y)});
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

    return request;
}

std::string dinoSearchResponseToYaml(const DinoSearchResponse &response)
{
    YAML::Node node;
    node["status"] = priv::dinoStatusName(response.status);
    node["decision"] = priv::dinoDecisionName(response.decision);

    YAML::Node results(YAML::NodeType::Sequence);
    for (const auto &item : response.results)
    {
        YAML::Node result;
        result["source_path"] = item.source_path;
        YAML::Node bbox(YAML::NodeType::Sequence);
        bbox.push_back(item.bbox.x0);
        bbox.push_back(item.bbox.y0);
        bbox.push_back(item.bbox.x1);
        bbox.push_back(item.bbox.y1);
        result["bbox"] = bbox;
        result["score"] = item.score;
        results.push_back(result);
    }
    node["results"] = results;

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

} // namespace irt::features
