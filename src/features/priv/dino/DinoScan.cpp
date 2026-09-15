/**
 * @file DinoScan.cpp
 * @brief 分块精确扫描与双路粗选实现。
 */

#include "DinoScan.hpp"
#include "DinoDescriptors.hpp"
#include "DinoSimilarity.hpp"
#include "DinoTime.hpp"
#include "DinoRetrievalCore.hpp"
#include <opencv2/core.hpp>
#include <queue>
#include <unordered_set>

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

namespace irt::features::priv {

namespace {

inline float dotProduct(const float *a, const float *b, const size_t dimension)
{
    float sum = 0.0F;
    for (size_t index = 0; index < dimension; ++index)
    {
        sum += a[index] * b[index];
    }
    return sum;
}
struct ViewKey
{
    int view_id{0};
    int x0{0};
    int y0{0};
    int x1{0};
    int y1{0};

    bool operator<(const ViewKey &other) const noexcept
    {
        return std::tie(view_id, x0, y0, x1, y1) < std::tie(other.view_id, other.x0, other.y0, other.x1, other.y1);
    }
};

int quantize(const double value) noexcept
{
    return static_cast<int>(std::lround(value));
}

} // namespace

DinoScanOutcome dinoScan(const DinoIndexReader &reader, const DinoQuery &query,
                         const DinoRegionSearchConfig &config, const DinoDeadline &deadline,
                         const int64_t excluded_image_id,
                         const std::optional<std::vector<int64_t>> &allowed_image_ids,
                         const DinoOperationControl &control)
{
    DinoScanOutcome outcome;
    const auto        scan_started = dinoNowMs();
    const auto        dimension    = reader.descriptorDim();
    const auto &views           = reader.views();
    const bool  quantized_index = reader.quantized();
    const auto        similarity_backend
        = quantized_index ? dinoSelectSimilarityBackend() : DinoSimilarityBackend::Cpu;
    outcome.view_survival_truncated = false;
    outcome.similarity_backend      = dinoSimilarityBackendName(similarity_backend);
    outcome.compact_scan            = quantized_index;
    std::vector<uint8_t> excluded_views(views.size(), 0U);
    if (allowed_image_ids.has_value())
    {
        const auto &allowed_list = allowed_image_ids.value();
        std::unordered_set<int64_t> allowed_set(allowed_list.begin(), allowed_list.end());
        for (size_t view_id = 0; view_id < views.size(); ++view_id)
        {
            const auto image_index = views[view_id].image_index;
            if (image_index >= 0 && static_cast<size_t>(image_index) < reader.images().size())
            {
                if (!allowed_set.contains(reader.images()[static_cast<size_t>(image_index)].image_id))
                {
                    excluded_views[view_id] = 1U;
                }
            }
            else
            {
                excluded_views[view_id] = 1U;
            }
        }
    }
    if (excluded_image_id >= 0)
    {
        for (size_t view_id = 0; view_id < views.size(); ++view_id)
        {
            const auto image_index = views[view_id].image_index;
            if (image_index >= 0 && static_cast<size_t>(image_index) < reader.images().size()
                && reader.images()[static_cast<size_t>(image_index)].image_id == excluded_image_id)
            {
                excluded_views[view_id] = 1U;
            }
        }
    }

    const bool all_excluded = !views.empty() && std::all_of(excluded_views.begin(), excluded_views.end(), [](uint8_t v) { return v != 0U; });
    if (all_excluded)
    {
        outcome.region_scan_ms = dinoNowMs() - scan_started;
        return outcome;
    }

    // ---------------- 区域通道 ----------------
    std::map<ViewKey, DinoCandidate> region_best;
    const int query_view_count = static_cast<int>(query.views.size());
    std::vector<DinoTopK> region_tops;
    region_tops.reserve(query.views.size());
    for (size_t query_view = 0; query_view < query.views.size(); ++query_view)
    {
        region_tops.emplace_back(config.region_topk);
    }
    std::vector<float> buffer;
    std::vector<float> compact_scores;
    std::vector<float> roi_vectors;
    roi_vectors.reserve(query.views.size() * dimension);
    for (const auto &query_view : query.views)
    {
        if (query_view.coarse_roi_vector.size() != dimension)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                 "Query ROI descriptor dimension does not match the index descriptor dimension");
        }
        roi_vectors.insert(roi_vectors.end(), query_view.coarse_roi_vector.begin(), query_view.coarse_roi_vector.end());
    }
    DinoCompactScratch compact_scratch;
    std::unique_ptr<DinoSimilarityReducer> region_reducer;
    if (quantized_index)
    {
        region_reducer = std::make_unique<DinoSimilarityReducer>(similarity_backend, static_cast<int>(dimension));
        // 显式请求不可用 CUDA 时门面会回退 CPU；诊断必须记录实际执行后端。
        outcome.similarity_backend = dinoSimilarityBackendName(region_reducer->backend());
    }

    for (size_t begin = 0; begin < reader.regionCount(); begin += config.region_scan_block)
    {
        if (control.cancelled && control.cancelled())
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
        }
        if (deadline.expired())
        {
            outcome.incomplete = true;
            break;
        }
        const size_t count = std::min(config.region_scan_block, reader.regionCount() - begin);
        if (quantized_index)
        {
            const auto block = reader.readRegionCompact(begin, count, compact_scratch);
            compact_scores.resize(count * static_cast<size_t>(query_view_count));
            region_reducer->reduceRegionScores(block, roi_vectors.data(), query_view_count, compact_scores.data());
            for (size_t index = 0; index < count; ++index)
            {
                const auto view_id = reader.viewOfRegionDescriptor(begin + index);
                if (view_id >= 0 && static_cast<size_t>(view_id) < excluded_views.size()
                    && excluded_views[static_cast<size_t>(view_id)] != 0U)
                {
                    continue;
                }
                for (int query_view = 0; query_view < query_view_count; ++query_view)
                {
                    region_tops[static_cast<size_t>(query_view)].push(
                        compact_scores[index * static_cast<size_t>(query_view_count) + query_view], begin + index);
                }
            }
        }
        else
        {
            reader.readRegionVectors(begin, count, buffer);
            for (size_t query_view = 0; query_view < query.views.size(); ++query_view)
            {
                const auto &roi_vector = query.views[query_view].coarse_roi_vector;
                if (roi_vector.empty())
                {
                    continue;
                }
                for (size_t index = 0; index < count; ++index)
                {
                    const auto view_id = reader.viewOfRegionDescriptor(begin + index);
                    if (view_id >= 0 && static_cast<size_t>(view_id) < excluded_views.size()
                        && excluded_views[static_cast<size_t>(view_id)] != 0U)
                    {
                        continue;
                    }
                    const float score = dotProduct(roi_vector.data(), buffer.data() + index * dimension, dimension);
                    region_tops[query_view].push(score, begin + index);
                }
            }
        }
        outcome.scanned_region_descriptors += count;
    }

    if (region_reducer)
    {
        outcome.device_allocated_bytes = std::max(outcome.device_allocated_bytes,
                                                  region_reducer->deviceAllocatedBytes());
        outcome.device_reserved_delta_bytes
            = std::max(outcome.device_reserved_delta_bytes, region_reducer->deviceReservedDeltaBytes());
    }

    outcome.region_scan_ms = dinoNowMs() - scan_started;

    for (size_t query_view = 0; query_view < region_tops.size(); ++query_view)
    {
        for (const auto &[score, index] : region_tops[query_view].sorted())
        {
            const auto view_id = reader.viewOfRegionDescriptor(index);
            const auto meta    = reader.regionMetaAt(index);
            const auto plan    = dinoPlanFromIndexView(views[static_cast<size_t>(view_id)]);
            const auto rect    = DinoRect::unite(plan.patchRect(meta.grid_row, meta.grid_col),
                                                 plan.patchRect(meta.grid_row + meta.grid_height - 1,
                                                                meta.grid_col + meta.grid_width - 1));
            ViewKey    key{view_id, quantize(rect.x0), quantize(rect.y0), quantize(rect.x1), quantize(rect.y1)};
            auto       it = region_best.find(key);
            if (it == region_best.end())
            {
                DinoCandidate candidate;
                candidate.view_id       = view_id;
                candidate.query_view_id = static_cast<int>(query_view);
                candidate.from_region   = true;
                candidate.region_score  = score;
                candidate.score         = score;
                candidate.source_bbox   = rect;
                region_best.emplace(key, candidate);
            }
            else if (score > it->second.region_score)
            {
                it->second.region_score  = score;
                it->second.score         = score;
                it->second.query_view_id = static_cast<int>(query_view);
            }
        }
    }

    outcome.region_scan_ms = dinoNowMs() - scan_started;

    const auto image_id_of = [&](const int view_id) -> int64_t
    {
        const auto image_index = views[static_cast<size_t>(view_id)].image_index;
        if (image_index < 0 || static_cast<size_t>(image_index) >= reader.images().size())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Index view references an unknown image");
        }
        return reader.images()[static_cast<size_t>(image_index)].image_id;
    };

    for (const auto &[key, candidate] : region_best)
    {
        (void)key;
        DinoCandidate enriched = candidate;
        enriched.image_id      = image_id_of(candidate.view_id);
        outcome.region_candidates.push_back(std::move(enriched));
    }
    std::sort(outcome.region_candidates.begin(), outcome.region_candidates.end(),
              [](const DinoCandidate &a, const DinoCandidate &b) { return a.score > b.score; });
    if (outcome.region_candidates.size() > config.channel_candidate_limit)
    {
        outcome.region_candidates.resize(config.channel_candidate_limit);
    }

    // Spatial evidence is evaluated for EVERY view before the global local Top-K.
    // Keep only a block of descriptors and two poses per view in memory.
    const auto local_started = dinoNowMs();
    std::vector<float> tokens;
    std::vector<size_t> query_offsets{0};
    std::vector<std::vector<retrieval::Evidence>> evidence(query.views.size());
    for (size_t q = 0; q < query.views.size(); ++q) {
        for (const auto &token : query.views[q].tokens) {
            if (token.coarse_vector.size() != dimension)
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query coarse dimension mismatch");
            tokens.insert(tokens.end(), token.coarse_vector.begin(), token.coarse_vector.end());
            evidence[q].push_back({token.source, token.cell, token.weight});
        }
        query_offsets.push_back(tokens.size() / dimension);
    }
    const int token_count = static_cast<int>(tokens.size() / dimension);
    if (token_count == 0) return outcome;
    const auto worse = [](const DinoCandidate &a, const DinoCandidate &b) {return a.score > b.score;};
    std::priority_queue<DinoCandidate, std::vector<DinoCandidate>, decltype(worse)> top(worse);
    DinoSimilarityReducer reducer(similarity_backend, static_cast<int>(dimension));
    DinoCompactScratch scratch;
    std::vector<float> decoded;
    std::vector<retrieval::Pair> pairs;
    size_t next_view = 0;
    while (next_view < views.size()) {
        if (control.cancelled && control.cancelled()) {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
        }
        if (deadline.expired()) {outcome.incomplete = true;break;}
        while (next_view < views.size() && (excluded_views[next_view] || reader.localRange(next_view).count == 0)) ++next_view;
        if (next_view == views.size()) break;
        size_t block_begin = reader.localRange(next_view).begin, count = 0;
        std::vector<size_t> ids, offsets, counts;
        while (next_view < views.size()) {
            auto range = reader.localRange(next_view);
            if (excluded_views[next_view] || range.count == 0) break;
            if (count && (count + range.count > config.region_scan_block || range.begin != block_begin + count)) break;
            ids.push_back(next_view);offsets.push_back(count);counts.push_back(range.count);
            count += range.count;++next_view;
            if (count >= config.region_scan_block) break;
        }
        // Whole views are never split: top-2 indices must remain view-relative.
        pairs.resize(ids.size() * static_cast<size_t>(token_count));
        if (quantized_index) {
            auto block = reader.readLocalCompact(block_begin, count, scratch);
            reducer.matchViewGroup(block, offsets.data(), counts.data(), ids.size(), tokens.data(), token_count, pairs.data());
        } else {
            reader.readLocalVectors(block_begin, count, decoded);
            cv::parallel_for_(cv::Range(0, static_cast<int>(ids.size())), [&](const cv::Range &range) {
                for (int g = range.start; g < range.end; ++g) {
                    auto found = retrieval::nearestTwo(decoded.data() + offsets[g] * dimension, counts[g],
                                                       tokens.data(), static_cast<size_t>(token_count), static_cast<int>(dimension));
                    std::copy(found.begin(), found.end(), pairs.begin() + static_cast<size_t>(g) * token_count);
                }
            });
        }
        const auto meta = reader.readLocalMeta(block_begin, count);
        std::vector<std::vector<DinoCandidate>> block_candidates(ids.size());
        const auto vote_started = dinoNowMs();
        cv::parallel_for_(cv::Range(0, static_cast<int>(ids.size())), [&](const cv::Range &range) {
            for (int g = range.start; g < range.end; ++g) {
                const auto view_id = ids[g];
                const auto plan = dinoPlanFromIndexView(views[view_id]);
                std::vector<DinoPoint> points;points.reserve(counts[g]);
                for (size_t i = 0; i < counts[g]; ++i) {
                    const auto &m = meta[offsets[g] + i];
                    auto b = plan.patchRect(m.rep_row, m.rep_col);
                    points.push_back({(b.x0 + b.x1) * .5, (b.y0 + b.y1) * .5});
                }
                auto &kept = block_candidates[g];
                for (size_t q = 0; q < query.views.size(); ++q) {
                    auto first = pairs.begin() + static_cast<size_t>(g) * token_count + query_offsets[q];
                    std::vector<retrieval::Pair> local_pairs(first, first + evidence[q].size());
                    const auto &qv = query.views[q];
                    const double gp = std::sqrt(plan.sourcePxPerPatchX() * plan.sourcePxPerPatchY());
                    const double qp = std::sqrt(qv.plan.sourcePxPerPatchX() * qv.plan.sourcePxPerPatchY());
                    const auto poses = retrieval::vote(evidence[q], local_pairs, points, qv.roi_bbox, gp / qp, gp, 2);
                    for (const auto &pose : poses) {
                        DinoCandidate candidate;
                        candidate.image_id = image_id_of(static_cast<int>(view_id));
                        candidate.view_id = static_cast<int>(view_id);candidate.query_view_id = static_cast<int>(q);
                        candidate.from_local = true;candidate.local_score = candidate.score = pose.score;
                        candidate.source_bbox = pose.box;
                        kept.push_back(std::move(candidate));
                    }
                }
                std::stable_sort(kept.begin(), kept.end(), [](const DinoCandidate &a,const DinoCandidate &b){return a.score>b.score;});
                std::vector<DinoCandidate> distinct;
                for (const auto &c : kept) {
                    bool duplicate = false;
                    for (const auto &d : distinct) if (retrieval::iou(c.source_bbox, d.source_bbox) > .5) duplicate = true;
                    if (!duplicate) distinct.push_back(c);
                    if (distinct.size() == 2) break;
                }
                kept = std::move(distinct);
            }
        });
        outcome.window_rescore_ms += dinoNowMs() - vote_started;
        for (const auto &group : block_candidates) for (const auto &candidate : group) {
            if (top.size() < config.channel_candidate_limit) top.push(candidate);
            else if (candidate.score > top.top().score) {top.pop();top.push(candidate);}
        }
        outcome.scanned_local_descriptors += count;
        outcome.retained_local_views += ids.size();
    }
    while (!top.empty()) {outcome.local_candidates.push_back(top.top());top.pop();}
    std::reverse(outcome.local_candidates.begin(), outcome.local_candidates.end());
    outcome.local_scan_ms = dinoNowMs() - local_started - outcome.window_rescore_ms;
    outcome.view_survival_truncated = false;
    outcome.device_allocated_bytes = std::max(outcome.device_allocated_bytes, reducer.deviceAllocatedBytes());
    outcome.device_reserved_delta_bytes = std::max(outcome.device_reserved_delta_bytes, reducer.deviceReservedDeltaBytes());
    return outcome;
}

} // namespace irt::features::priv
