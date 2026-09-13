/**
 * @file DinoQuery.cpp
 * @brief 查询表示实现。
 */

#include "DinoQuery.hpp"
#include "DinoDescriptors.hpp"
#include "DinoGeometry.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>

namespace irt::features::priv {

namespace {

struct CellSelection
{
    int   first_patch{-1};
    int   second_patch{-1};
    float weight{0.0F};
};

CellSelection selectCellPatches(const DinoFeatureGrid &grid, const std::vector<float> &roi_weights,
                                const DinoRect &roi_bbox, const int cell_row, const int cell_col, const int cells)
{
    const double cell_width  = roi_bbox.width() / static_cast<double>(cells);
    const double cell_height = roi_bbox.height() / static_cast<double>(cells);

    const double cell_x0 = roi_bbox.x0 + cell_width * static_cast<double>(cell_col);
    const double cell_y0 = roi_bbox.y0 + cell_height * static_cast<double>(cell_row);
    const DinoRect cell_rect{cell_x0, cell_y0, cell_x0 + cell_width, cell_y0 + cell_height};
    const double   center_x = cell_x0 + cell_width * 0.5;
    const double   center_y = cell_y0 + cell_height * 0.5;

    struct Candidate
    {
        int    patch{-1};
        double distance{0.0};
    };

    std::vector<Candidate> candidates;
    std::vector<float>     weights;
    for (int row = 0; row < grid.plan.grid_height; ++row)
    {
        for (int col = 0; col < grid.plan.grid_width; ++col)
        {
            const auto index = static_cast<size_t>(row) * static_cast<size_t>(grid.plan.grid_width)
                             + static_cast<size_t>(col);
            if (!(roi_weights[index] > 0.0F) || !grid.patchValid(row, col))
            {
                continue;
            }
            const auto   patch_rect = grid.plan.patchRect(row, col);
            const double patch_cx   = patch_rect.x0 + patch_rect.width() * 0.5;
            const double patch_cy   = patch_rect.y0 + patch_rect.height() * 0.5;
            if (!cell_rect.contains(patch_cx, patch_cy))
            {
                continue;
            }
            const double dx = patch_cx - center_x;
            const double dy = patch_cy - center_y;
            candidates.push_back(Candidate{static_cast<int>(index), dx * dx + dy * dy});
            weights.push_back(roi_weights[index]);
        }
    }

    CellSelection selection;
    if (candidates.empty())
    {
        return selection;
    }

    // 每个格子先取空间上最接近格子中心的 patch，再取与首个描述差异最大的另一 patch。
    auto nearest = std::min_element(candidates.begin(), candidates.end(),
                                    [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
    selection.first_patch = nearest->patch;

    const float *first = grid.tokens.data()
                       + static_cast<size_t>(selection.first_patch) * static_cast<size_t>(grid.channels);
    double best_distance = -1.0;
    for (const auto &candidate : candidates)
    {
        if (candidate.patch == selection.first_patch)
        {
            continue;
        }
        const float *other = grid.tokens.data()
                           + static_cast<size_t>(candidate.patch) * static_cast<size_t>(grid.channels);
        double sum = 0.0;
        for (int channel = 0; channel < grid.channels; ++channel)
        {
            const double difference = static_cast<double>(first[channel]) - static_cast<double>(other[channel]);
            sum += difference * difference;
        }
        if (sum > best_distance)
        {
            best_distance     = sum;
            selection.second_patch = candidate.patch;
        }
    }

    double weight_sum = 0.0;
    for (const auto weight : weights)
    {
        weight_sum += weight;
    }
    selection.weight = static_cast<float>(weight_sum);
    return selection;
}

} // namespace

DinoQuery dinoBuildQuery(const DinoCanonicalImage &image, const DinoRoi &roi, DinoBackbone &backbone,
                         const DinoViewPlanner &planner, const DinoRegionSearchConfig &config,
                         size_t &model_forwards)
{
    DinoQuery query;
    query.cells = config.query_local_cells;

    const auto plans = planner.planQueryViews(roi, image.record.width, image.record.height);
    const auto spec  = dinoViewPreprocessSpec(planner.encoderEdge(), planner.patchSize());

    std::vector<DinoViewRaster> rasters;
    rasters.reserve(plans.size());
    for (const auto &plan : plans)
    {
        rasters.push_back(dinoRenderView(image.image, plan, spec));
    }
    const auto forward_start = backbone.forwardCount();
    const auto grids = backbone.extract(rasters);
    model_forwards += backbone.forwardCount() - forward_start;
    const int  cells = std::max(1, config.query_local_cells);
    const auto roi_bbox = roi.boundingBox();

    for (const auto &grid : grids)
    {
        DinoQueryView view;
        view.plan     = grid.plan;
        view.grid     = grid;
        view.roi_bbox = roi_bbox;

        const auto weights = dinoPatchRoiWeights(grid.plan, roi);
        view.roi_weights   = weights;

        double weight_sum = 0.0;
        std::vector<float> pooled(static_cast<size_t>(grid.channels), 0.0F);
        for (int row = 0; row < grid.plan.grid_height; ++row)
        {
            for (int col = 0; col < grid.plan.grid_width; ++col)
            {
                const auto   index  = static_cast<size_t>(row) * static_cast<size_t>(grid.plan.grid_width)
                                    + static_cast<size_t>(col);
                const float  weight = weights[index];
                if (!(weight > 0.0F) || !grid.patchValid(row, col))
                {
                    continue;
                }
                const float *token = grid.token(row, col);
                for (int channel = 0; channel < grid.channels; ++channel)
                {
                    pooled[static_cast<size_t>(channel)] += token[channel] * weight;
                }
                weight_sum += weight;
            }
        }
        if (!(weight_sum > 0.0))
        {
            continue;
        }
        for (auto &value : pooled)
        {
            value = static_cast<float>(value / weight_sum);
        }
        if (!dinoNormalizeVector(pooled.data(), pooled.size()))
        {
            continue;
        }
        view.roi_vector = std::move(pooled);

        for (int cell_row = 0; cell_row < cells; ++cell_row)
        {
            for (int cell_col = 0; cell_col < cells; ++cell_col)
            {
                const int cell_id = cell_row * cells + cell_col;
                const auto selection = selectCellPatches(grid, weights, roi_bbox, cell_row, cell_col, cells);
                if (selection.first_patch < 0)
                {
                    continue;
                }
                ++view.valid_cells;

                const auto append_token = [&](const int patch, const int slot, const float weight)
                {
                    if (patch < 0)
                    {
                        return;
                    }
                    const int    row   = patch / grid.plan.grid_width;
                    const int    col   = patch % grid.plan.grid_width;
                    const auto   rect  = grid.plan.patchRect(row, col);
                    DinoQueryToken token;
                    token.cell   = cell_id;
                    token.slot   = slot;
                    token.weight = weight;
                    token.source = DinoPoint{rect.x0 + rect.width() * 0.5, rect.y0 + rect.height() * 0.5};
                    const float *source = grid.token(row, col);
                    token.vector.assign(source, source + grid.channels);
                    view.tokens.push_back(std::move(token));
                };

                // 一个格子最多贡献一个格子总权重，两条描述各自占一半，避免重复加倍计票。
                const float half_weight = selection.weight * 0.5F;
                append_token(selection.first_patch, 0, half_weight);
                append_token(selection.second_patch, 1, half_weight);
            }
        }

        query.valid_cell_count = std::max(query.valid_cell_count, view.valid_cells);
        query.views.push_back(std::move(view));
    }

    if (query.views.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Query ROI produced no usable description; check ROI size and image validity");
    }

    const auto total_tokens = query.views.front().tokens.size();
    query.low_local_evidence
        = query.views.front().valid_cells < config.query_min_local_evidence || total_tokens < 2U;

    std::string         note;
    DinoValidatedRange  range;
    range.min_image_edge      = config.validated_min_image_edge;
    range.max_image_edge      = config.validated_max_image_edge;
    range.min_target_short_px = config.validated_min_target_short_px;
    range.max_target_aspect   = config.validated_max_target_aspect;
    query.outside_validated_profile
        = !dinoWithinValidatedProfile(roi, image.record.width, image.record.height, range, note);
    query.profile_note = note;

    query.selection_note = "cells=" + std::to_string(query.cells) + ";views=" + std::to_string(query.views.size())
                         + ";tokens=" + std::to_string(total_tokens)
                         + ";valid_cells=" + std::to_string(query.views.front().valid_cells);
    return query;
}

} // namespace irt::features::priv
