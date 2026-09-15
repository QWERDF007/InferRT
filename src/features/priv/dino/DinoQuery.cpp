/**
 * @file DinoQuery.cpp
 * @brief 查询表示实现。
 */

#include "DinoQuery.hpp"
#include "DinoDescriptors.hpp"
#include "DinoGeometry.hpp"
#include "DinoRetrievalCore.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>

namespace irt::features::priv {

DinoQuery dinoBuildQuery(const DinoCanonicalImage &image, const DinoRoi &roi, DinoBackbone &backbone,
                         const DinoViewPlanner &planner, const DinoRegionSearchConfig &config,
                         size_t &model_forwards)
{
    DinoQuery query;
    query.roi = roi;
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
    const retrieval::Projection projection(backbone.channels(), config.coarse_dimension);
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

        view.coarse_roi_vector = projection.apply(view.roi_vector.data());
        const auto samples = retrieval::select(grid, roi_bbox, weights, cells, config.query_local_max_per_cell);
        int previous_cell = -1;
        int slot = 0;
        for (const auto &sample : samples)
        {
            if (sample.cell != previous_cell) { ++view.valid_cells; previous_cell = sample.cell; slot = 0; }
            DinoQueryToken token;
            token.cell = sample.cell;
            token.slot = slot++;
            token.weight = sample.weight;
            token.source = sample.point;
            const float *source = grid.tokens.data() + static_cast<size_t>(sample.patch) * grid.channels;
            token.vector.assign(source, source + grid.channels);
            token.coarse_vector = projection.apply(source);
            view.tokens.push_back(std::move(token));
        }

        query.valid_cell_count = std::max(query.valid_cell_count, view.valid_cells);
        query.views.push_back(std::move(view));
    }

    if (query.views.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Query ROI produced no usable description; check ROI size and image validity");
    }

    size_t total_tokens = 0;
    for (const auto &view : query.views) total_tokens += view.tokens.size();
    query.low_local_evidence
        = query.valid_cell_count < config.query_min_local_evidence || total_tokens < 2U;

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
                         + ";valid_cells=" + std::to_string(query.valid_cell_count);
    return query;
}

} // namespace irt::features::priv
