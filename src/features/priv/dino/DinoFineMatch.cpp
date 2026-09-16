/** Bounded original-D localization followed by tight-crop verification. */
#include "DinoFineMatch.hpp"
#include "DinoGeometry.hpp"
#include "DinoPaths.hpp"
#include "DinoRetrievalCore.hpp"
#include "DinoTime.hpp"
#include <inferrt/core/Exception.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <memory>
#include <utility>

namespace irt::features::priv {
namespace {
struct FineQuery {
    DinoRect roi{};
    std::vector<retrieval::Evidence> evidence;
    std::vector<float> tokens;
};
struct TightDescription {
    std::vector<float> global;
    std::vector<std::vector<float>> cells;
    std::vector<float> weights;
};
TightDescription describeTight(const DinoFeatureGrid& grid, const DinoRoi& roi) {
    TightDescription out;
    out.global.assign(static_cast<size_t>(grid.channels), 0.f);
    out.cells.assign(16, std::vector<float>(static_cast<size_t>(grid.channels), 0.f));
    out.weights.assign(16, 0.f);
    const auto weights = dinoPatchRoiWeights(grid.plan, roi);
    const auto bbox = roi.boundingBox();
    for (int r = 0; r < grid.plan.grid_height; ++r) for (int c = 0; c < grid.plan.grid_width; ++c) {
        const size_t p = static_cast<size_t>(r * grid.plan.grid_width + c);
        if (!(weights[p] > 0) || !grid.patchValid(r, c)) continue;
        auto patch = DinoRect::intersect(grid.plan.patchRect(r, c), bbox);
        if (patch.empty()) continue;
        const float* token = grid.token(r, c);
        for (int d = 0; d < grid.channels; ++d) out.global[d] += weights[p] * token[d];
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
            DinoRect cell{bbox.x0 + bbox.width() * x / 4, bbox.y0 + bbox.height() * y / 4,
                          bbox.x0 + bbox.width() * (x + 1) / 4, bbox.y0 + bbox.height() * (y + 1) / 4};
            const auto cell_patch = DinoRect::intersect(cell, patch);
            float w = roi.is_polygon
                ? float(dinoPolygonRectArea(roi.polygon, cell_patch) / grid.plan.patchRect(r, c).area())
                : float(cell_patch.area() / patch.area()) * weights[p];
            auto index = static_cast<size_t>(y * 4 + x);out.weights[index] += w;
            for (int d = 0; d < grid.channels; ++d) out.cells[index][d] += w * token[d];
        }
    }
    if (!retrieval::normalize(out.global)) throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tight ROI has no evidence");
    for (auto& cell : out.cells) retrieval::normalize(cell);
    return out;
}
DinoRoi targetRoi(const DinoRoi& query, const DinoRect& target) {
    DinoRoi result;result.bbox = target;result.is_polygon = query.is_polygon;
    const auto source = query.boundingBox();
    for (auto p : query.polygon) result.polygon.push_back({
        target.x0 + (p.x - source.x0) / source.width() * target.width(),
        target.y0 + (p.y - source.y0) / source.height() * target.height()});
    return result;
}
float tightScore(const TightDescription& query, const TightDescription& target) {
    const auto cosine01 = [](float x) {return std::clamp((x + 1.f) * .5f, 0.f, 1.f);};
    float global = cosine01(retrieval::dot(query.global.data(), target.global.data(), static_cast<int>(query.global.size())));
    double total = 0, score = 0;
    for (size_t i = 0; i < query.cells.size(); ++i) {
        total += query.weights[i];
        if (target.weights[i] > 0)
            score += query.weights[i] * cosine01(retrieval::dot(query.cells[i].data(), target.cells[i].data(), static_cast<int>(query.global.size())));
    }
    return .6f * global + .4f * (total > 0 ? float(score / total) : 0.f);
}
} // namespace

DinoFineMatchOutcome dinoFineMatch(const DinoIndexReader& reader, const DinoCanonicalImage& query_image,
                                   const DinoQuery& query, const std::vector<DinoCandidate>& candidates,
                                   const DinoRegionSearchConfig& config, DinoBackbone& backbone,
                                   const DinoViewPlanner& planner, DinoImageCache& image_cache,
                                   DinoFeatureGridCache& feature_cache, const std::string& extractor_signature,
                                   const DinoDeadline& deadline,
                                   const std::function<std::filesystem::path(int64_t)>& image_resolver,
                                   const DinoSearchProgressCallback& progress_callback) {
    DinoFineMatchOutcome outcome;outcome.total_candidates = candidates.size();
    const auto spec = dinoViewPreprocessSpec(planner.encoderEdge(), planner.patchSize());
    const size_t batch = std::max<size_t>(1, backbone.maxBatchSize());
    std::vector<FineQuery> fine_queries;
    for (const auto& q : query.views) {
        FineQuery item;item.roi = q.roi_bbox;
        for (const auto& t : q.tokens) {
            item.evidence.push_back({t.source, t.cell, t.weight});
            item.tokens.insert(item.tokens.end(), t.vector.begin(), t.vector.end());
        }
        if (!item.evidence.empty()) fine_queries.push_back(std::move(item));
    }
    retrieval::FineOptions options;
    options.max_sizes = config.fine_template_max_sizes;
    options.scale_step = config.fine_template_scale_step;
    options.peaks = config.fine_peaks_per_candidate;
    options.refinement_rounds = config.fine_refinement_rounds;
    options.nms = config.fine_nms_iou;
    options.match_cosine = static_cast<float>(config.fine_match_cosine_threshold);
    options.appearance_weight = static_cast<float>(config.score_weight_template);
    options.coverage_weight = static_cast<float>(config.score_weight_coverage);
    options.consistency_weight = static_cast<float>(config.score_weight_consistency);
    // Continuous sub-patch support is deliberate. The old integer support-size gate is obsolete.
    options.min_short = .75;
    std::vector<std::vector<const DinoCandidate*>> groups(reader.images().size());
    for (const auto& c : candidates) {
        if (c.view_id < 0 || static_cast<size_t>(c.view_id) >= reader.views().size()) {outcome.incomplete = true;continue;}
        int image = reader.views()[static_cast<size_t>(c.view_id)].image_index;
        if (image < 0 || static_cast<size_t>(image) >= groups.size()) {outcome.incomplete = true;continue;}
        groups[static_cast<size_t>(image)].push_back(&c);
    }
    size_t candidate_batch_index = 0;
    size_t processed_candidates = 0;
    for (size_t image_id = 0; image_id < groups.size(); ++image_id) {
        const auto& group = groups[image_id];if (group.empty()) continue;
        if (deadline.expired()) {outcome.incomplete = true;break;}
        const auto& record = reader.images()[image_id];
        std::shared_ptr<const DinoCanonicalImage> image;
        auto started = dinoNowMs();
        try {
            if (!image_resolver) {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "image_resolver is required for fine matching");
            }
            const auto image_path = image_resolver(record.image_id);
            if (image_path.empty() || !std::filesystem::exists(image_path)) {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Resolved image path does not exist");
            }
            image = DinoImageLoader::loadCached(image_path, image_cache, record.image_id);
        }
        catch (const std::exception&) {outcome.incomplete = true;continue;}
        outcome.extract_ms += dinoNowMs() - started;
        for (size_t begin = 0; begin < group.size(); begin += batch) {
            if (deadline.expired()) {outcome.incomplete = true;break;}
            started = dinoNowMs();
            const size_t count = std::min(batch, group.size() - begin);
            std::vector<DinoViewRaster> rasters;
            std::vector<std::shared_ptr<const DinoFeatureGrid>> grids(count);
            std::vector<size_t> misses;
            std::vector<std::string> keys(count);
            for (size_t i = 0; i < count; ++i) {
                auto crop = dinoExpandRect(group[begin + i]->source_bbox, config.fine_candidate_expand, image->record.width, image->record.height);
                if (crop.empty()) {outcome.incomplete = true;continue;}
                keys[i] = dinoFeatureCacheKey(std::to_string(record.image_id) + "|" + std::to_string(record.file_size) + "|" + std::to_string(record.mtime_ns), extractor_signature, crop);
                grids[i] = feature_cache.find(keys[i]);
                if (!grids[i]) {
                    misses.push_back(i);
                    rasters.push_back(dinoRenderView(image->image, planner.makePlan(crop, image->record.width, image->record.height, false, -3), spec));
                }
            }
            if (!rasters.empty()) {
                size_t before = backbone.forwardCount();
                auto extracted = backbone.extract(rasters);
                outcome.model_forwards += backbone.forwardCount() - before;
                if (extracted.size() != misses.size()) throw irt::Exception(irt::Status::ERROR_INTERNAL, "Fine output batch mismatch");
                for (size_t j = 0; j < extracted.size(); ++j) {
                    auto grid = std::make_shared<DinoFeatureGrid>(std::move(extracted[j]));
                    grids[misses[j]] = grid;feature_cache.insert(keys[misses[j]], grid);
                }
            }
            outcome.extract_ms += dinoNowMs() - started;
            for (size_t i = 0; i < count; ++i) {
                if (deadline.expired()) {outcome.incomplete = true;break;}
                if (!grids[i]) continue;
                const auto& grid = *grids[i];const auto& candidate = *group[begin + i];
                started = dinoNowMs();
                bool completed = true;
                // Both query contexts compete; candidate merging cannot lock fine matching to a wrong context.
                for (const auto& q : fine_queries) {
                    if (deadline.expired()) {completed = false;break;}
                    const size_t patches = static_cast<size_t>(grid.plan.patchCount());
                    std::vector<float> similarity(q.evidence.size() * patches);
                    cv::parallel_for_(cv::Range(0, static_cast<int>(q.evidence.size())), [&](const cv::Range& range) {
                        for (int t = range.start; t < range.end; ++t) for (size_t p = 0; p < patches; ++p)
                            similarity[static_cast<size_t>(t) * patches + p] = grid.valid_area[p] > 0 ? retrieval::dot(
                                q.tokens.data() + static_cast<size_t>(t) * grid.channels,
                                grid.tokens.data() + p * grid.channels, grid.channels) : -1.f;
                    });
                    retrieval::SparseMatcher matcher(grid.plan.grid_height, grid.plan.grid_width, q.evidence, q.roi, std::move(similarity), grid.valid_area);
                    auto input = grid.plan.canonical_to_input.apply(candidate.source_bbox);
                    double patch = static_cast<double>(grid.plan.patch_size);
                    DinoRect hint{input.x0 / patch, input.y0 / patch, input.x1 / patch, input.y1 / patch};
                    auto located = matcher.locate(options, hint, [&] {return deadline.expired();});
                    if (deadline.expired()) completed = false;
                    for (const auto& p : located) {
                        DinoRect box{p.box.x0 * patch, p.box.y0 * patch, p.box.x1 * patch, p.box.y1 * patch};
                        DinoMatchResult result;result.image_id = record.image_id;
                        result.bbox = dinoClampRect(grid.plan.input_to_canonical.apply(box), image->record.width, image->record.height);
                        result.score = p.score;result.template_similarity = p.appearance;
                        result.query_coverage = p.coverage;result.spatial_consistency = p.consistency;
                        result.from_region_channel = candidate.from_region;result.from_local_channel = candidate.from_local;result.coarse_view_id = candidate.view_id;
                        if (!result.bbox.empty()) outcome.results.push_back(std::move(result));
                    }
                }
                outcome.match_ms += dinoNowMs() - started;
                if (completed) ++outcome.completed_candidates;else outcome.incomplete = true;
            }
            if (progress_callback) {
                progress_callback(DinoSearchProgress{
                    DinoSearchStage::FineMatch,
                    candidate_batch_index++,
                    processed_candidates,
                    count,
                    processed_candidates + count,
                    outcome.total_candidates,
                    "matching candidate crops"
                });
            }
            processed_candidates += count;
        }
    }
    if (outcome.completed_candidates < outcome.total_candidates) outcome.incomplete = true;
    std::stable_sort(outcome.results.begin(), outcome.results.end(), [](const auto& a, const auto& b) {return a.score > b.score;});
    outcome.localized_results = outcome.results;
    if (config.fine_verify_k == 0) return outcome;
    // Beam limits are visible acceptance metrics: localization and verification have different scores.
    dinoNmsWithinImages(outcome.results, .85);
    if (outcome.results.size() > config.fine_verify_k) outcome.results.resize(config.fine_verify_k);
    outcome.verification_input = outcome.results;
    auto pending = std::move(outcome.results);outcome.results.clear();
    if (pending.empty()) return outcome;
    if (deadline.expired()) {outcome.incomplete = true;return outcome;}
    auto started = dinoNowMs();size_t before = backbone.forwardCount();
    auto query_crop = dinoExpandRect(query.roi.boundingBox(), 1.04, query_image.record.width, query_image.record.height);
    auto query_plan = planner.makePlan(query_crop, query_image.record.width, query_image.record.height, false, -4);
    auto query_grids = backbone.extract({dinoRenderView(query_image.image, query_plan, spec)});
    auto tight_query = describeTight(query_grids.at(0), query.roi);
    outcome.model_forwards += backbone.forwardCount() - before;outcome.extract_ms += dinoNowMs() - started;
    size_t verify_batch_index = 0;
    for (size_t begin = 0; begin < pending.size(); begin += batch) {
        if (deadline.expired()) {outcome.incomplete = true;break;}
        started = dinoNowMs();
        const size_t count = std::min(pending.size(), begin + batch) - begin;
        std::vector<DinoViewRaster> rasters;std::vector<size_t> ids;std::vector<DinoRoi> rois;
        for (size_t i = begin; i < begin + count; ++i) {
            try {
                if (!image_resolver) {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "image_resolver is required for verification");
                }
                const auto image_path = image_resolver(pending[i].image_id);
                if (image_path.empty() || !std::filesystem::exists(image_path)) {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Resolved image path does not exist");
                }
                auto image = DinoImageLoader::loadCached(image_path, image_cache, pending[i].image_id);
                auto roi = targetRoi(query.roi, pending[i].bbox);
                auto crop = dinoExpandRect(pending[i].bbox, 1.04, image->record.width, image->record.height);
                rasters.push_back(dinoRenderView(image->image, planner.makePlan(crop, image->record.width, image->record.height, false, -4), spec));
                ids.push_back(i);rois.push_back(std::move(roi));
            } catch (const std::exception&) {outcome.incomplete = true;}
        }
        if (rasters.empty()) continue;
        before = backbone.forwardCount();auto grids = backbone.extract(rasters);
        outcome.model_forwards += backbone.forwardCount() - before;outcome.extract_ms += dinoNowMs() - started;
        if (grids.size() != ids.size()) throw irt::Exception(irt::Status::ERROR_INTERNAL, "Verification batch mismatch");
        started = dinoNowMs();
        for (size_t j = 0; j < grids.size(); ++j) {
            auto result = pending[ids[j]];
            try {result.score = tightScore(tight_query, describeTight(grids[j], rois[j]));outcome.results.push_back(std::move(result));}
            catch (const std::exception&) {outcome.incomplete = true;}
        }
        outcome.match_ms += dinoNowMs() - started;
        if (progress_callback) {
            progress_callback(DinoSearchProgress{
                DinoSearchStage::FineExtract,
                verify_batch_index++,
                begin,
                count,
                begin + count,
                pending.size(),
                "verifying tight candidate crops"
            });
        }
    }
    std::stable_sort(outcome.results.begin(), outcome.results.end(), [](const auto& a, const auto& b) {return a.score > b.score;});
    return outcome;
}
} // namespace irt::features::priv
