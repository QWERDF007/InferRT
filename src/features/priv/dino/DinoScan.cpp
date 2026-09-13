/**
 * @file DinoScan.cpp
 * @brief 分块精确扫描与双路粗选实现。
 */

#include "DinoScan.hpp"
#include "DinoDescriptors.hpp"
#include "DinoSimilarity.hpp"
#include "DinoTime.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

namespace irt::features::priv {

namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

inline float dotProduct(const float *a, const float *b, const size_t dimension)
{
    float sum = 0.0F;
    for (size_t index = 0; index < dimension; ++index)
    {
        sum += a[index] * b[index];
    }
    return sum;
}
/** @brief 多个查询视图拼接后的行优先 token 矩阵。 */
struct TokenMatrix
{
    std::vector<float> tokens{};
    std::vector<int>   slot_of_token{};
    std::vector<int>   query_view_of_token{};

    int tokenCount() const noexcept { return static_cast<int>(slot_of_token.size()); }
};

TokenMatrix buildTokenMatrix(const DinoQuery &query, const size_t dimension)
{
    TokenMatrix matrix;
    if (dimension == 0U)
    {
        return matrix;
    }
    size_t token_count = 0U;
    for (const auto &query_view : query.views)
    {
        token_count += query_view.tokens.size();
    }
    matrix.tokens.reserve(token_count * dimension);
    matrix.slot_of_token.reserve(token_count);
    matrix.query_view_of_token.reserve(token_count);
    for (size_t query_view_index = 0; query_view_index < query.views.size(); ++query_view_index)
    {
        for (const auto &token : query.views[query_view_index].tokens)
        {
            if (token.vector.size() != dimension)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "Query token dimension does not match the index descriptor dimension");
            }
            matrix.tokens.insert(matrix.tokens.end(), token.vector.begin(), token.vector.end());
            matrix.slot_of_token.push_back(static_cast<int>(token.cell) * 2 + static_cast<int>(token.slot));
            matrix.query_view_of_token.push_back(static_cast<int>(query_view_index));
        }
    }
    return matrix;
}

/** @brief 有效格子等权平均；每格里的最多两个查询 token 先各自取最大，再取均值。 */
bool reduceCells(const std::vector<float> &maxima, const int cells, float &score)
{
    int    valid_cells = 0;
    double cell_sum    = 0.0;
    for (int cell = 0; cell < cells * cells; ++cell)
    {
        const float first  = maxima[static_cast<size_t>(cell) * 2U];
        const float second = maxima[static_cast<size_t>(cell) * 2U + 1U];
        if (first == kNegativeInfinity && second == kNegativeInfinity)
        {
            continue;
        }
        double value = 0.0;
        int    parts = 0;
        if (first != kNegativeInfinity)
        {
            value += first;
            ++parts;
        }
        if (second != kNegativeInfinity)
        {
            value += second;
            ++parts;
        }
        cell_sum += value / static_cast<double>(parts);
        ++valid_cells;
    }
    if (valid_cells == 0)
    {
        return false;
    }
    score = static_cast<float>(cell_sum / static_cast<double>(valid_cells));
    return true;
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
                         const DinoRegionSearchConfig &config, const DinoDeadline &deadline)
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
        if (query_view.roi_vector.size() != dimension)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                 "Query ROI descriptor dimension does not match the index descriptor dimension");
        }
        roi_vectors.insert(roi_vectors.end(), query_view.roi_vector.begin(), query_view.roi_vector.end());
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
                const auto &roi_vector = query.views[query_view].roi_vector;
                if (roi_vector.empty())
                {
                    continue;
                }
                for (size_t index = 0; index < count; ++index)
                {
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

    const auto image_id_of = [&](const int view_id) -> const std::string &
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

    // ---------------- 局部通道：视图级查询格子聚合 ----------------
    const auto   local_started = dinoNowMs();
    const size_t view_count = views.size();
    std::vector<float> view_scores(view_count, kNegativeInfinity);
    std::vector<int>   view_best_query(view_count, -1);
    const auto token_matrix = buildTokenMatrix(query, dimension);
    if (!token_matrix.tokens.empty())
    {
        const auto token_count = static_cast<size_t>(token_matrix.tokenCount());
        std::vector<int> identity_slots(token_count);
        for (size_t token = 0; token < token_count; ++token)
        {
            identity_slots[token] = static_cast<int>(token);
        }
        std::vector<size_t> group_offsets;
        std::vector<size_t> group_counts;
        std::vector<size_t> group_ids;
        std::vector<float>  group_maxima;
        std::vector<float>  active_maxima(token_count, kNegativeInfinity);
        std::vector<float>  cell_maxima(static_cast<size_t>(query.cells) * static_cast<size_t>(query.cells) * 2U,
                                        kNegativeInfinity);
        DinoCompactScratch  scratch;
        std::vector<float>  descriptor_buffer;
        std::unique_ptr<DinoSimilarityReducer> local_reducer;
        if (quantized_index)
        {
            // 归约器跨所有查询视图和索引视图复用设备缓冲；输出仅保留当前描述块的 view x token 最大值。
            local_reducer = std::make_unique<DinoSimilarityReducer>(similarity_backend, static_cast<int>(dimension));
        }

        int active_view = -1;
        const auto finish_view = [&](const int view_id)
        {
            if (view_id < 0)
            {
                return;
            }
            for (size_t query_view = 0; query_view < query.views.size(); ++query_view)
            {
                std::fill(cell_maxima.begin(), cell_maxima.end(), kNegativeInfinity);
                for (size_t token = 0; token < token_count; ++token)
                {
                    if (token_matrix.query_view_of_token[token] != static_cast<int>(query_view))
                    {
                        continue;
                    }
                    const auto slot = static_cast<size_t>(token_matrix.slot_of_token[token]);
                    cell_maxima[slot] = std::max(cell_maxima[slot], active_maxima[token]);
                }
                float score = 0.0F;
                if (reduceCells(cell_maxima, query.cells, score) && score > view_scores[static_cast<size_t>(view_id)])
                {
                    view_scores[static_cast<size_t>(view_id)] = score;
                    view_best_query[static_cast<size_t>(view_id)] = static_cast<int>(query_view);
                }
            }
        };

        size_t next_view   = 0U;
        size_t next_offset = 0U;
        while (next_view < view_count)
        {
            if (deadline.expired())
            {
                outcome.incomplete = true;
                break;
            }
            const auto first_range = reader.localRange(next_view);
            if (next_offset >= first_range.count)
            {
                finish_view(active_view);
                active_view = -1;
                ++next_view;
                next_offset = 0U;
                continue;
            }

            const size_t block_begin = first_range.begin + next_offset;
            size_t       block_count = 0U;
            size_t       cursor_view = next_view;
            size_t       cursor_offset = next_offset;
            group_offsets.clear();
            group_counts.clear();
            group_ids.clear();
            while (cursor_view < view_count && block_count < config.region_scan_block)
            {
                const auto range = reader.localRange(cursor_view);
                if (cursor_offset >= range.count)
                {
                    ++cursor_view;
                    cursor_offset = 0U;
                    continue;
                }
                const size_t descriptor_begin = range.begin + cursor_offset;
                if (block_count > 0U && descriptor_begin != block_begin + block_count)
                {
                    break;
                }
                const size_t available = range.count - cursor_offset;
                const size_t take       = std::min(available, config.region_scan_block - block_count);
                group_offsets.push_back(block_count);
                group_counts.push_back(take);
                group_ids.push_back(cursor_view);
                block_count += take;
                cursor_offset += take;
                if (cursor_offset < range.count)
                {
                    break;
                }
                ++cursor_view;
                cursor_offset = 0U;
            }
            if (block_count == 0U)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "Unable to form a contiguous local descriptor block");
            }

            group_maxima.assign(group_ids.size() * token_count, kNegativeInfinity);
            if (quantized_index)
            {
                const auto block = reader.readLocalCompact(block_begin, block_count, scratch);
                local_reducer->reduceViewGroup(block, group_offsets.data(), group_counts.data(), group_ids.size(),
                                               token_matrix.tokens.data(), token_matrix.tokenCount(),
                                               group_maxima.data());
            }
            else
            {
                // FP32 参考索引（压缩消融基线）：一次读取连续块，再按 view 边界归约。
                reader.readLocalVectors(block_begin, block_count, descriptor_buffer);
                for (size_t group = 0; group < group_ids.size(); ++group)
                {
                    dinoAccumulateSimilarityMaxima(
                        descriptor_buffer.data() + group_offsets[group] * dimension, group_counts[group],
                        token_matrix.tokens.data(), token_matrix.tokenCount(), static_cast<int>(dimension),
                        identity_slots.data(), group_maxima.data() + group * token_count);
                }
            }

            for (size_t group = 0; group < group_ids.size(); ++group)
            {
                const int group_view = static_cast<int>(group_ids[group]);
                if (active_view != group_view)
                {
                    finish_view(active_view);
                    active_view = group_view;
                    std::fill(active_maxima.begin(), active_maxima.end(), kNegativeInfinity);
                }
                const auto *source = group_maxima.data() + group * token_count;
                for (size_t token = 0; token < token_count; ++token)
                {
                    active_maxima[token] = std::max(active_maxima[token], source[token]);
                }
            }
            outcome.scanned_local_descriptors += block_count;
            next_view   = cursor_view;
            next_offset = cursor_offset;
        }
        finish_view(active_view);

        if (local_reducer)
        {
            outcome.device_allocated_bytes = std::max(outcome.device_allocated_bytes,
                                                      local_reducer->deviceAllocatedBytes());
            outcome.device_reserved_delta_bytes
                = std::max(outcome.device_reserved_delta_bytes, local_reducer->deviceReservedDeltaBytes());
        }
    }

    outcome.local_scan_ms = dinoNowMs() - local_started;

    // ---------------- 局部通道：窗口内重打分 ----------------
    const auto window_started = dinoNowMs();
    std::vector<std::pair<float, size_t>> ranked_views;
    ranked_views.reserve(view_count);
    for (size_t view_id = 0; view_id < view_count; ++view_id)
    {
        if (view_scores[view_id] > kNegativeInfinity)
        {
            ranked_views.emplace_back(view_scores[view_id], view_id);
        }
    }
    std::sort(ranked_views.begin(), ranked_views.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b)
              {
                  if (a.first != b.first)
                  {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });
    outcome.view_survival_truncated = ranked_views.size() > config.local_view_topk;
    if (ranked_views.size() > config.local_view_topk)
    {
        ranked_views.resize(config.local_view_topk);
    }
    outcome.retained_local_views = ranked_views.size();

    std::map<ViewKey, DinoCandidate> local_best;
    for (const auto &[view_score, view_id] : ranked_views)
    {
        if (deadline.expired())
        {
            outcome.incomplete = true;
            break;
        }
        const auto region_range = reader.regionRange(view_id);
        const auto local_range  = reader.localRange(view_id);
        if (region_range.count == 0U || local_range.count == 0U)
        {
            continue;
        }

        const auto plan     = dinoPlanFromIndexView(views[view_id]);
        const auto windows  = reader.readRegionMeta(region_range.begin, region_range.count);
        const auto leaves   = reader.readLocalMeta(local_range.begin, local_range.count);
        std::vector<float> leaf_vectors;
        reader.readLocalVectors(local_range.begin, local_range.count, leaf_vectors);

        const auto query_view_index = static_cast<size_t>(
            view_best_query[view_id] >= 0 ? view_best_query[view_id] : 0);
        const auto &query_view      = query.views[std::min(query_view_index, query.views.size() - 1U)];
        const auto &tokens          = query_view.tokens;
        const auto  roi_bbox        = query_view.roi_bbox;

        for (const auto &window : windows)
        {
            const auto window_rect = DinoRect::unite(plan.patchRect(window.grid_row, window.grid_col),
                                                     plan.patchRect(window.grid_row + window.grid_height - 1,
                                                                    window.grid_col + window.grid_width - 1));
            if (window_rect.empty() || roi_bbox.empty())
            {
                continue;
            }

            DinoWindowHypothesis hypothesis;
            hypothesis.scale = std::min(window_rect.width() / roi_bbox.width(),
                                        window_rect.height() / roi_bbox.height());
            if (!(hypothesis.scale > 0.0))
            {
                continue;
            }
            hypothesis.window_center = DinoPoint{window_rect.x0 + window_rect.width() * 0.5,
                                                window_rect.y0 + window_rect.height() * 0.5};
            hypothesis.roi_center    = DinoPoint{roi_bbox.x0 + roi_bbox.width() * 0.5,
                                              roi_bbox.y0 + roi_bbox.height() * 0.5};

            std::vector<float> maxima(static_cast<size_t>(query.cells) * static_cast<size_t>(query.cells) * 2U,
                                      kNegativeInfinity);
            const double cell_width  = roi_bbox.width() / static_cast<double>(query.cells);
            const double cell_height = roi_bbox.height() / static_cast<double>(query.cells);

            for (size_t leaf_index = 0; leaf_index < leaves.size(); ++leaf_index)
            {
                const auto &leaf      = leaves[leaf_index];
                const auto  leaf_rect = plan.patchRect(leaf.rep_row, leaf.rep_col);
                if (DinoRect::intersect(leaf_rect, window_rect).empty())
                {
                    continue;
                }
                const auto center = DinoPoint{leaf_rect.x0 + leaf_rect.width() * 0.5,
                                              leaf_rect.y0 + leaf_rect.height() * 0.5};
                const auto roi_point = hypothesis.toRoiSpace(center.x, center.y);
                if (!roi_bbox.contains(roi_point.x, roi_point.y))
                {
                    continue;
                }
                const int cell_col = std::min(query.cells - 1,
                                              std::max(0, static_cast<int>((roi_point.x - roi_bbox.x0) / cell_width)));
                const int cell_row = std::min(query.cells - 1,
                                              std::max(0, static_cast<int>((roi_point.y - roi_bbox.y0) / cell_height)));
                const int cell     = cell_row * query.cells + cell_col;

                const float *descriptor = leaf_vectors.data() + leaf_index * dimension;
                for (size_t token_index = 0; token_index < tokens.size(); ++token_index)
                {
                    if (tokens[token_index].cell != cell)
                    {
                        continue;
                    }
                    const auto  slot  = static_cast<size_t>(cell) * 2U + static_cast<size_t>(tokens[token_index].slot);
                    const float score = dotProduct(descriptor, tokens[token_index].vector.data(), dimension);
                    if (score > maxima[slot])
                    {
                        maxima[slot] = score;
                    }
                }
            }

            float score = 0.0F;
            if (!reduceCells(maxima, query.cells, score))
            {
                continue;
            }
            ++outcome.window_rescore_windows;

            const ViewKey key{static_cast<int>(view_id), quantize(window_rect.x0), quantize(window_rect.y0),
                              quantize(window_rect.x1), quantize(window_rect.y1)};
            auto          it = local_best.find(key);
            if (it == local_best.end())
            {
                DinoCandidate candidate;
                candidate.view_id       = static_cast<int>(view_id);
                candidate.query_view_id = static_cast<int>(query_view_index);
                candidate.from_local    = true;
                candidate.local_score   = score;
                candidate.score         = score;
                candidate.source_bbox   = window_rect;
                local_best.emplace(key, candidate);
            }
            else if (score > it->second.local_score)
            {
                it->second.local_score = score;
                it->second.score       = score;
            }
        }
    }

    outcome.window_rescore_ms = dinoNowMs() - window_started;

    for (const auto &[key, candidate] : local_best)
    {
        (void)key;
        DinoCandidate enriched = candidate;
        enriched.image_id      = image_id_of(candidate.view_id);
        outcome.local_candidates.push_back(std::move(enriched));
    }
    std::sort(outcome.local_candidates.begin(), outcome.local_candidates.end(),
              [](const DinoCandidate &a, const DinoCandidate &b) { return a.score > b.score; });
    if (outcome.local_candidates.size() > config.channel_candidate_limit)
    {
        outcome.local_candidates.resize(config.channel_candidate_limit);
    }
    return outcome;
}

} // namespace irt::features::priv
