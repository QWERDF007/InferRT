/**
 * @file DinoFineMatch.cpp
 * @brief 候选局部精算实现。
 */

#include "DinoFineMatch.hpp"
#include "DinoDescriptors.hpp"
#include "DinoGeometry.hpp"
#include "DinoPaths.hpp"
#include "DinoProfile.hpp"
#include "DinoTime.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <memory>

namespace irt::features::priv {

namespace {

/** @brief 4 路累加点积；同一形状的点积在一次查询里会被调用数十万次。 */
float dotProduct(const float *lhs, const float *rhs, const int count)
{
    float acc0 = 0.0F;
    float acc1 = 0.0F;
    float acc2 = 0.0F;
    float acc3 = 0.0F;
    int   index = 0;
    for (; index + 4 <= count; index += 4)
    {
        acc0 += lhs[index] * rhs[index];
        acc1 += lhs[index + 1] * rhs[index + 1];
        acc2 += lhs[index + 2] * rhs[index + 2];
        acc3 += lhs[index + 3] * rhs[index + 3];
    }
    for (; index < count; ++index)
    {
        acc0 += lhs[index] * rhs[index];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

/** @brief 查询模板：ROI 覆盖的 patch 块、权重与归一化位置。 */
struct QueryTemplate
{
    std::vector<float> tokens{};   ///< ``height * width * channels``
    std::vector<float> mask{};     ///< ``height * width``，ROI 覆盖权重
    int                height{0};
    int                width{0};
    int                channels{0};
    DinoRect           roi_bbox{};
};

bool buildQueryTemplate(const DinoQueryView &query_view, const int channels, QueryTemplate &out)
{
    const auto &grid = query_view.grid;
    int min_row = grid.plan.grid_height;
    int max_row = -1;
    int min_col = grid.plan.grid_width;
    int max_col = -1;
    for (int row = 0; row < grid.plan.grid_height; ++row)
    {
        for (int col = 0; col < grid.plan.grid_width; ++col)
        {
            const auto index = static_cast<size_t>(row) * static_cast<size_t>(grid.plan.grid_width)
                             + static_cast<size_t>(col);
            if (!(query_view.roi_weights[index] > 0.0F) || !grid.patchValid(row, col))
            {
                continue;
            }
            min_row = std::min(min_row, row);
            max_row = std::max(max_row, row);
            min_col = std::min(min_col, col);
            max_col = std::max(max_col, col);
        }
    }
    if (max_row < 0 || max_col < 0)
    {
        return false;
    }

    out.height   = max_row - min_row + 1;
    out.width    = max_col - min_col + 1;
    out.channels = channels;
    out.roi_bbox = query_view.roi_bbox;
    out.tokens.assign(static_cast<size_t>(out.height) * static_cast<size_t>(out.width) * static_cast<size_t>(channels),
                      0.0F);
    out.mask.assign(static_cast<size_t>(out.height) * static_cast<size_t>(out.width), 0.0F);

    for (int row = 0; row < out.height; ++row)
    {
        for (int col = 0; col < out.width; ++col)
        {
            const int   grid_row = min_row + row;
            const int   grid_col = min_col + col;
            const auto  index    = static_cast<size_t>(grid_row) * static_cast<size_t>(grid.plan.grid_width)
                               + static_cast<size_t>(grid_col);
            const float weight   = query_view.roi_weights[index];
            const auto  target   = static_cast<size_t>(row) * static_cast<size_t>(out.width)
                               + static_cast<size_t>(col);
            out.mask[target] = weight;
            const auto *token = grid.token(grid_row, grid_col);
            std::copy(token, token + channels, out.tokens.begin() + static_cast<std::ptrdiff_t>(target * channels));
        }
    }
    return true;
}

/**
 * @brief 已按 ROI 覆盖权重预加权的模板。
 *
 * 滑动相似度的定义是 ``sum(mask*u·v)/sum(mask)``。把 mask 预先乘进模板后，热循环里只剩下一次
 * 点积累加，既省掉每个滑窗位置的权重分支，也避免重复读取 mask；权重为 0 的 patch 在定义上恒为
 * 0，直接标记为失效并在滑动时跳过。
 */
struct PreparedTemplate
{
    int                  height{0};
    int                  width{0};
    int                  channels{0};
    float                mask_sum{0.0F};
    std::vector<float>   weighted_tokens{};
    std::vector<uint8_t> active{};
};

/** @brief 模板缩放：token 双线性重采样后重新 L2 归一化，mask 按面积重采样并并入 token。 */
void prepareTemplate(const QueryTemplate &source, const int target_height, const int target_width,
                     PreparedTemplate &out)
{
    out.height   = target_height;
    out.width    = target_width;
    out.channels = source.channels;

    const auto channels = static_cast<size_t>(source.channels);
    const auto patches  = static_cast<size_t>(target_height) * static_cast<size_t>(target_width);
    out.weighted_tokens.assign(patches * channels, 0.0F);
    out.active.assign(patches, 0);
    out.mask_sum = 0.0F;

    const double scale_y = static_cast<double>(source.height) / static_cast<double>(target_height);
    const double scale_x = static_cast<double>(source.width) / static_cast<double>(target_width);

    for (int row = 0; row < target_height; ++row)
    {
        const double source_row  = (static_cast<double>(row) + 0.5) * scale_y - 0.5;
        const double clamped_row = std::min(std::max(source_row, 0.0), static_cast<double>(source.height - 1));
        for (int col = 0; col < target_width; ++col)
        {
            const double source_col  = (static_cast<double>(col) + 0.5) * scale_x - 0.5;
            const double clamped_col = std::min(std::max(source_col, 0.0), static_cast<double>(source.width - 1));
            const int    row0        = static_cast<int>(clamped_row);
            const int    col0        = static_cast<int>(clamped_col);
            const int    row1        = std::min(row0 + 1, source.height - 1);
            const int    col1        = std::min(col0 + 1, source.width - 1);
            const double dy          = clamped_row - static_cast<double>(row0);
            const double dx          = clamped_col - static_cast<double>(col0);

            const auto  target = static_cast<size_t>(row) * static_cast<size_t>(target_width)
                              + static_cast<size_t>(col);
            auto *target_token = out.weighted_tokens.data() + target * channels;

            double       mask_value = 0.0;
            const int    rows[2]{row0, row1};
            const int    cols[2]{col0, col1};
            const double weight_y[2]{1.0 - dy, dy};
            const double weight_x[2]{1.0 - dx, dx};
            for (int r = 0; r < 2; ++r)
            {
                for (int c = 0; c < 2; ++c)
                {
                    const double factor = weight_y[r] * weight_x[c];
                    if (factor <= 0.0)
                    {
                        continue;
                    }
                    const auto source_index = static_cast<size_t>(rows[r]) * static_cast<size_t>(source.width)
                                            + static_cast<size_t>(cols[c]);
                    mask_value += factor * static_cast<double>(source.mask[source_index]);
                    const auto *source_token = source.tokens.data() + source_index * channels;
                    for (size_t channel = 0; channel < channels; ++channel)
                    {
                        target_token[channel]
                            += static_cast<float>(factor * static_cast<double>(source_token[channel]));
                    }
                }
            }
            if (!dinoNormalizeVector(target_token, channels))
            {
                continue;
            }
            const auto weight = static_cast<float>(std::min(1.0, std::max(0.0, mask_value)));
            if (!(weight > 0.0F))
            {
                continue;
            }
            out.active[target] = 1U;
            out.mask_sum += weight;
            for (size_t channel = 0; channel < channels; ++channel)
            {
                target_token[channel] *= weight;
            }
        }
    }
}

/** @brief 一个空间峰值；尺寸以 patch 为单位。 */
struct Peak
{
    double row{0.0};
    double col{0.0};
    float  score{0.0F};
    int    height{0};
    int    width{0};
};

/** @brief 两个峰值框在网格空间的 IoU。 */
double peakIoU(const Peak &a, const Peak &b)
{
    const double x0        = std::max(a.col, b.col);
    const double y0        = std::max(a.row, b.row);
    const double x1        = std::min(a.col + a.width, b.col + b.width);
    const double y1        = std::min(a.row + a.height, b.row + b.height);
    const double inter     = std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
    const double area_a    = static_cast<double>(a.width) * static_cast<double>(a.height);
    const double area_b    = static_cast<double>(b.width) * static_cast<double>(b.height);
    const double union_area = area_a + area_b - inter;
    return union_area > 0.0 ? inter / union_area : 0.0;
}

/** @brief 把各尺寸收集到的峰值合并为不超过 ``max_peaks`` 个空间上互不重叠的峰值。 */
void mergePeaks(std::vector<Peak> &candidates, const int max_peaks, const double nms_iou)
{
    // 分数降序、同分按网格位置升序：与顺序扫描的择优结果一致，且不受并行调度影响。
    std::sort(candidates.begin(), candidates.end(),
              [](const Peak &a, const Peak &b)
              {
                  if (a.score != b.score)
                  {
                      return a.score > b.score;
                  }
                  if (a.row != b.row)
                  {
                      return a.row < b.row;
                  }
                  return a.col < b.col;
              });

    std::vector<Peak> kept;
    kept.reserve(static_cast<size_t>(std::max(1, max_peaks)));
    for (const auto &candidate : candidates)
    {
        bool overlaps = false;
        for (const auto &existing : kept)
        {
            if (peakIoU(candidate, existing) >= nms_iou)
            {
                overlaps = true;
                break;
            }
        }
        if (overlaps)
        {
            continue;
        }
        kept.push_back(candidate);
        if (kept.size() >= static_cast<size_t>(max_peaks))
        {
            break;
        }
    }
    candidates = std::move(kept);
}

/** @brief 一行的滑动结果；行之间没有共享状态，行循环因此可以安全并行。 */
struct RowResult
{
    float best{-std::numeric_limits<float>::infinity()};
    int   count{0};
    Peak  peaks[kDinoMaxPeaksPerCandidate]{};
};

/** @brief 把一行的滑窗结果并入该行的峰值列表；规则与全局合并一致（高分挤掉重叠低分）。 */
void insertRowPeak(RowResult &row, const Peak &candidate, const int capacity, const double nms_iou)
{
    int count = 0;
    for (int index = 0; index < row.count; ++index)
    {
        const Peak &existing = row.peaks[index];
        if (peakIoU(candidate, existing) >= nms_iou)
        {
            if (existing.score >= candidate.score)
            {
                return;
            }
            continue;
        }
        row.peaks[count++] = existing;
    }
    row.count = count;

    int position = std::min(row.count, capacity - 1);
    while (position > 0 && row.peaks[position - 1].score < candidate.score)
    {
        row.peaks[position] = row.peaks[position - 1];
        --position;
    }
    row.peaks[position] = candidate;
    if (row.count < capacity)
    {
        ++row.count;
    }
}

/** @brief 对一行网格执行滑窗：累加加权点积并记录该行的最优分数与空间峰值。 */
void slideRow(const DinoFeatureGrid &grid, const PreparedTemplate &tpl, const int row, const int max_col,
              const int capacity, const double nms_iou, RowResult &out)
{
    const int    channels   = grid.channels;
    const auto   row_stride = static_cast<size_t>(grid.plan.grid_width) * static_cast<size_t>(channels);
    const auto   patch_stride = static_cast<size_t>(channels);

    for (int col = 0; col <= max_col; ++col)
    {
        const float *base = grid.tokens.data() + static_cast<size_t>(row) * row_stride
                          + static_cast<size_t>(col) * patch_stride;
        float accumulator = 0.0F;
        for (int t_row = 0; t_row < tpl.height; ++t_row)
        {
            const auto *template_row = tpl.weighted_tokens.data()
                                     + static_cast<size_t>(t_row) * static_cast<size_t>(tpl.width) * patch_stride;
            const auto *grid_row = base + static_cast<size_t>(t_row) * row_stride;
            for (int t_col = 0; t_col < tpl.width; ++t_col)
            {
                const auto patch = static_cast<size_t>(t_row) * static_cast<size_t>(tpl.width)
                                 + static_cast<size_t>(t_col);
                if (tpl.active[patch] == 0U)
                {
                    continue;
                }
                const auto offset = static_cast<size_t>(t_col) * patch_stride;
                accumulator += dotProduct(template_row + offset, grid_row + offset, channels);
            }
        }
        const auto score = accumulator / tpl.mask_sum;
        if (score > out.best)
        {
            out.best = score;
        }
        if (capacity == 0 || !(score > 0.0F))
        {
            continue;
        }
        insertRowPeak(out, Peak{static_cast<double>(row), static_cast<double>(col), score, tpl.height, tpl.width},
                      capacity, nms_iou);
    }
}

/**
 * @brief 在候选网格上滑动模板，收集至多 ``max_peaks`` 个空间上互不重叠的峰值。
 *
 * 滑动本身是逐行独立的点积累加，行之间没有任何共享状态，因此按行切分并行执行；峰值先按行收集，
 * 再以与顺序扫描一致的规则全局合并，结果与单线程版本等价。
 */
float slidePrepared(const DinoFeatureGrid &grid, const PreparedTemplate &tpl, const int max_peaks,
                    const double nms_iou, std::vector<Peak> &peaks, std::vector<RowResult> &rows)
{
    if (!(tpl.mask_sum > 0.0F))
    {
        return -std::numeric_limits<float>::infinity();
    }
    const int max_row = grid.plan.grid_height - tpl.height;
    const int max_col = grid.plan.grid_width - tpl.width;
    if (max_row < 0 || max_col < 0)
    {
        return -std::numeric_limits<float>::infinity();
    }

    const int capacity = std::min(max_peaks, kDinoMaxPeaksPerCandidate);
    rows.assign(static_cast<size_t>(max_row) + 1U, RowResult{});
    cv::parallel_for_(cv::Range(0, max_row + 1),
                      [&](const cv::Range &range)
                      {
                          for (int row = range.start; row < range.end; ++row)
                          {
                              slideRow(grid, tpl, row, max_col, capacity, nms_iou,
                                       rows[static_cast<size_t>(row)]);
                          }
                      });

    float best = -std::numeric_limits<float>::infinity();
    for (const auto &row : rows)
    {
        best = std::max(best, row.best);
        for (int index = 0; index < row.count; ++index)
        {
            peaks.push_back(row.peaks[index]);
        }
    }
    return best;
}

/**
 * @brief 互为近邻的查询覆盖率与归一化位置一致性。
 *
 * 模板 patch 与匹配框 patch 的最近邻都在**几何对应位置的 ±1 patch** 邻域内搜索：这个半径容得下取整
 * 误差与半 patch 偏移，又不允许跨多个 patch 去“凑”近邻。双向最近邻要求一致，因此当匹配框只覆盖
 * ROI 的一部分时，ROI 上未被覆盖的 patch 会因回代不一致而失配，覆盖率随之下降，尺度选择也不会退化
 * 为最小模板。覆盖率按 ROI 覆盖权重加权，一致性取互为近邻点在各自归一化 ROI 内的位置距离。
 */
void measureCoverage(const DinoFeatureGrid &grid, const QueryTemplate &query_template, const Peak &peak,
                     const double tolerance, float &coverage, float &consistency)
{
    coverage    = 0.0F;
    consistency = 0.0F;

    const int q_height = query_template.height;
    const int q_width  = query_template.width;
    if (q_height <= 0 || q_width <= 0 || peak.height <= 0 || peak.width <= 0)
    {
        return;
    }

    const int channels = query_template.channels;
    const int grid_w   = grid.plan.grid_width;
    const int origin_row = std::max(0, std::min(static_cast<int>(std::floor(peak.row)),
                                                grid.plan.grid_height - peak.height));
    const int origin_col = std::max(0, std::min(static_cast<int>(std::floor(peak.col)), grid_w - peak.width));

    const double scale_y = static_cast<double>(peak.height) / static_cast<double>(q_height);
    const double scale_x = static_cast<double>(peak.width) / static_cast<double>(q_width);
    if (!(scale_y > 0.0) || !(scale_x > 0.0))
    {
        return;
    }

    constexpr int kNeighborRadius = 1;

    const auto patchIndex = [](const int row, const int col, const int width)
    { return static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col); };

    const auto queryToken = [&](const int row, const int col)
    {
        return query_template.tokens.data() + patchIndex(row, col, q_width)
             * static_cast<size_t>(channels);
    };
    const auto boxToken = [&](const int row, const int col)
    { return grid.token(origin_row + row, origin_col + col); };

    struct Neighbor
    {
        int   row{-1};
        int   col{-1};
        float similarity{-std::numeric_limits<float>::infinity()};
    };

    // 查询模板 patch -> 匹配框 patch。
    const auto forwardNeighbor = [&](const int q_row, const int q_col)
    {
        Neighbor best;
        const int center_row = static_cast<int>(
            std::lround((static_cast<double>(q_row) + 0.5) * scale_y - 0.5));
        const int center_col = static_cast<int>(
            std::lround((static_cast<double>(q_col) + 0.5) * scale_x - 0.5));
        for (int delta_row = -kNeighborRadius; delta_row <= kNeighborRadius; ++delta_row)
        {
            const int box_row = center_row + delta_row;
            if (box_row < 0 || box_row >= peak.height)
            {
                continue;
            }
            for (int delta_col = -kNeighborRadius; delta_col <= kNeighborRadius; ++delta_col)
            {
                const int box_col = center_col + delta_col;
                if (box_col < 0 || box_col >= peak.width)
                {
                    continue;
                }
                const float similarity = dotProduct(queryToken(q_row, q_col), boxToken(box_row, box_col), channels);
                if (similarity > best.similarity)
                {
                    best = Neighbor{box_row, box_col, similarity};
                }
            }
        }
        return best;
    };

    std::vector<Neighbor> forward(patchIndex(q_height - 1, q_width - 1, q_width) + 1U);
    double                total_weight = 0.0;
    for (int row = 0; row < q_height; ++row)
    {
        for (int col = 0; col < q_width; ++col)
        {
            const float weight = query_template.mask[patchIndex(row, col, q_width)];
            if (!(weight > 0.0F))
            {
                continue;
            }
            total_weight += static_cast<double>(weight);
            forward[patchIndex(row, col, q_width)] = forwardNeighbor(row, col);
        }
    }
    if (!(total_weight > 0.0))
    {
        return;
    }

    double mutual_weight  = 0.0;
    double distance_sum   = 0.0;
    int    mutual_count   = 0;
    for (int row = 0; row < q_height; ++row)
    {
        for (int col = 0; col < q_width; ++col)
        {
            const auto  patch  = patchIndex(row, col, q_width);
            const float weight = query_template.mask[patch];
            if (!(weight > 0.0F))
            {
                continue;
            }
            const auto &neighbor = forward[patch];
            if (neighbor.row < 0)
            {
                continue;
            }

            // 反向最近邻必须回到同一个模板 patch，且只在 ROI 覆盖的 patch 之间比较。
            Neighbor back;
            const int center_row = static_cast<int>(
                std::lround((static_cast<double>(neighbor.row) + 0.5) / scale_y - 0.5));
            const int center_col = static_cast<int>(
                std::lround((static_cast<double>(neighbor.col) + 0.5) / scale_x - 0.5));
            for (int delta_row = -kNeighborRadius; delta_row <= kNeighborRadius; ++delta_row)
            {
                const int back_row = center_row + delta_row;
                if (back_row < 0 || back_row >= q_height)
                {
                    continue;
                }
                for (int delta_col = -kNeighborRadius; delta_col <= kNeighborRadius; ++delta_col)
                {
                    const int back_col = center_col + delta_col;
                    if (back_col < 0 || back_col >= q_width
                        || !(query_template.mask[patchIndex(back_row, back_col, q_width)] > 0.0F))
                    {
                        continue;
                    }
                    const float similarity
                        = dotProduct(boxToken(neighbor.row, neighbor.col), queryToken(back_row, back_col), channels);
                    if (similarity > back.similarity)
                    {
                        back = Neighbor{back_row, back_col, similarity};
                    }
                }
            }
            if (back.row != row || back.col != col)
            {
                continue;
            }

            mutual_weight += static_cast<double>(weight);
            ++mutual_count;

            const double query_u = (static_cast<double>(col) + 0.5) / static_cast<double>(q_width);
            const double query_v = (static_cast<double>(row) + 0.5) / static_cast<double>(q_height);
            const double box_u   = (static_cast<double>(neighbor.col) + 0.5) / static_cast<double>(peak.width);
            const double box_v   = (static_cast<double>(neighbor.row) + 0.5) / static_cast<double>(peak.height);
            distance_sum += std::hypot(query_u - box_u, query_v - box_v);
        }
    }

    coverage = static_cast<float>(mutual_weight / total_weight);
    if (mutual_count > 0)
    {
        const double mean_distance = distance_sum / static_cast<double>(mutual_count);
        consistency = static_cast<float>(std::min(1.0, std::max(0.0, 1.0 - mean_distance / tolerance)));
    }
}

} // namespace

DinoFineMatchOutcome dinoFineMatch(const DinoIndexReader &reader, const DinoCanonicalImage &query_image,
                                   const DinoQuery &query, const std::vector<DinoCandidate> &candidates,
                                   const DinoRegionSearchConfig &config, DinoBackbone &backbone,
                                   const DinoViewPlanner &planner, DinoImageCache &image_cache,
                                   DinoFeatureGridCache &feature_cache, const std::string &extractor_signature,
                                   const DinoDeadline &deadline)
{
    (void)query_image;

    DinoFineMatchOutcome outcome;
    outcome.total_candidates = candidates.size();

    const auto &views  = reader.views();
    const auto &images = reader.images();
    const auto  spec   = dinoViewPreprocessSpec(planner.encoderEdge(), planner.patchSize());

    const double tolerance = std::max(1e-6, config.fine_position_tolerance);

    // 候选按来源图分组的理由：同一张图库图片常常被多个候选命中，按图解码一次、把裁剪视图凑成
    // 一个批量再前向一次，可以同时省掉重复解码与逐候选单图前向的固定开销。
    struct FineTask
    {
        const DinoCandidate *candidate{nullptr};
        int                  image_index{-1};
    };

    std::vector<std::vector<FineTask>> groups;
    std::vector<int>                   group_of_image(images.size(), -1);
    for (const auto &candidate : candidates)
    {
        if (candidate.view_id < 0 || static_cast<size_t>(candidate.view_id) >= views.size())
        {
            outcome.incomplete = true;
            continue;
        }
        const auto image_index = views[static_cast<size_t>(candidate.view_id)].image_index;
        if (image_index < 0 || static_cast<size_t>(image_index) >= images.size())
        {
            outcome.incomplete = true;
            continue;
        }
        if (group_of_image[static_cast<size_t>(image_index)] < 0)
        {
            group_of_image[static_cast<size_t>(image_index)] = static_cast<int>(groups.size());
            groups.emplace_back();
        }
        groups[static_cast<size_t>(group_of_image[static_cast<size_t>(image_index)])].push_back(
            FineTask{&candidate, image_index});
    }

    const size_t batch_limit = std::max<size_t>(1U, backbone.maxBatchSize());

    for (const auto &group : groups)
    {
        if (deadline.expired())
        {
            outcome.incomplete = true;
            break;
        }
        const auto &record = images[static_cast<size_t>(group.front().image_index)];

        const auto                                decode_started = dinoNowMs();
        std::shared_ptr<const DinoCanonicalImage> gallery_image;
        try
        {
            gallery_image = DinoImageLoader::loadCached(dinoPathFromUtf8(record.source_path), image_cache);
        }
        catch (const std::exception &)
        {
            // 无法解码的图库图片不参与精算，但必须让响应保持 incomplete。
            outcome.incomplete = true;
            continue;
        }
        outcome.extract_ms += dinoNowMs() - decode_started;

        const int image_width  = gallery_image->record.width;
        const int image_height = gallery_image->record.height;

        for (size_t begin = 0; begin < group.size(); begin += batch_limit)
        {
            if (deadline.expired())
            {
                outcome.incomplete = true;
                break;
            }
            const size_t count = std::min(batch_limit, group.size() - begin);

            struct PreparedCandidate
            {
                const DinoCandidate *candidate{nullptr};
                DinoViewPlan          plan{};
                std::shared_ptr<const DinoFeatureGrid> grid{};
                std::string          cache_key{};
            };

            std::vector<PreparedCandidate> batch_items;
            std::vector<DinoViewRaster>    rasters;
            std::vector<size_t>             uncached_indices;
            batch_items.reserve(count);
            rasters.reserve(count);
            uncached_indices.reserve(count);

            const auto extract_started = dinoNowMs();
            for (size_t index = 0; index < count; ++index)
            {
                const auto &task = group[begin + index];
                const DinoRect crop = dinoExpandRect(task.candidate->source_bbox, config.fine_candidate_expand,
                                                     image_width, image_height);
                if (crop.empty())
                {
                    outcome.incomplete = true;
                    continue;
                }
                PreparedCandidate item;
                item.candidate = task.candidate;
                item.plan      = planner.makePlan(crop, image_width, image_height, false, -3);
                item.cache_key = dinoFeatureCacheKey(
                    record.source_path + "|" + std::to_string(record.file_size) + "|" + std::to_string(record.mtime_ns),
                    extractor_signature, crop);
                item.grid      = feature_cache.find(item.cache_key);
                const auto item_index = batch_items.size();
                batch_items.push_back(std::move(item));
                if (batch_items.back().grid == nullptr)
                {
                    uncached_indices.push_back(item_index);
                    rasters.push_back(dinoRenderView(gallery_image->image, batch_items.back().plan, spec));
                }
            }
            if (batch_items.empty())
            {
                outcome.incomplete = true;
                continue;
            }
            if (!rasters.empty())
            {
                const auto forward_start = backbone.forwardCount();
                std::vector<DinoFeatureGrid> grids;
                try
                {
                    grids = backbone.extract(rasters);
                }
                catch (const irt::Exception &error)
                {
                    if (error.code() != irt::Status::ERROR_OUT_OF_MEMORY)
                    {
                        throw;
                    }
                    outcome.model_forwards += backbone.forwardCount() - forward_start;
                    outcome.incomplete = true;
                    break;
                }
                catch (const std::bad_alloc &)
                {
                    outcome.model_forwards += backbone.forwardCount() - forward_start;
                    outcome.incomplete = true;
                    break;
                }
                outcome.model_forwards += backbone.forwardCount() - forward_start;
                outcome.extract_ms += dinoNowMs() - extract_started;
                if (grids.size() != uncached_indices.size())
                {
                    outcome.incomplete = true;
                    break;
                }
                for (size_t index = 0; index < grids.size(); ++index)
                {
                    const auto prepared_index = uncached_indices[index];
                    auto       grid = std::make_shared<DinoFeatureGrid>(std::move(grids[index]));
                    feature_cache.insert(batch_items[prepared_index].cache_key, grid);
                    batch_items[prepared_index].grid = std::move(grid);
                }
            }

            for (const auto &item : batch_items)
            {
                if (deadline.expired())
                {
                    outcome.incomplete = true;
                    break;
                }
                if (item.grid == nullptr)
                {
                    outcome.incomplete = true;
                    continue;
                }
                const auto &candidate = *item.candidate;
                const auto &plan      = item.plan;
                const auto &grid      = *item.grid;
                const auto  match_started = dinoNowMs();

                const auto query_view_index = candidate.query_view_id >= 0
                                                && static_cast<size_t>(candidate.query_view_id) < query.views.size()
                                                ? static_cast<size_t>(candidate.query_view_id)
                                                : 0U;
                QueryTemplate template_source;
                if (!buildQueryTemplate(query.views[query_view_index], grid.channels, template_source))
                {
                    outcome.incomplete = true;
                    continue;
                }

                const int    short_side  = std::min(template_source.height, template_source.width);
                const double short_ratio = static_cast<double>(short_side);
                std::vector<std::pair<int, int>> sizes;
                for (int level = 0; level < config.fine_template_max_sizes; ++level)
                {
                    const double factor = std::pow(config.fine_template_scale_step, static_cast<double>(level));
                    const int    target_short = static_cast<int>(std::lround(
                        static_cast<double>(config.fine_template_min_short_patches) * factor));
                    if (target_short < 1)
                    {
                        continue;
                    }
                    const double scale  = static_cast<double>(target_short) / short_ratio;
                    const int    height = std::max(1, static_cast<int>(std::lround(
                        static_cast<double>(template_source.height) * scale)));
                    const int    width  = std::max(1, static_cast<int>(std::lround(
                        static_cast<double>(template_source.width) * scale)));
                    if (height > grid.plan.grid_height || width > grid.plan.grid_width)
                    {
                        break;
                    }
                    const auto key = std::make_pair(height, width);
                    if (std::find(sizes.begin(), sizes.end(), key) == sizes.end())
                    {
                        sizes.emplace_back(key);
                    }
                }

                const int         peak_budget = std::max(1, config.fine_peaks_per_candidate);
                std::vector<Peak> candidates_peaks;
                candidates_peaks.reserve(static_cast<size_t>(peak_budget) * sizes.size());

                PreparedTemplate       prepared;
                std::vector<RowResult> slide_rows;

                for (const auto &[height, width] : sizes)
                {
                    prepareTemplate(template_source, height, width, prepared);
                    slidePrepared(grid, prepared, peak_budget, config.fine_nms_iou, candidates_peaks, slide_rows);
                }

                mergePeaks(candidates_peaks, peak_budget, config.fine_nms_iou);
                if (candidates_peaks.empty())
                {
                    outcome.match_ms += dinoNowMs() - match_started;
                    ++outcome.completed_candidates;
                    continue;
                }

                const auto scoreOf = [&](const float cosine, const float cov, const float cons)
                {
                    const auto similarity = std::min(1.0F, std::max(0.0F, (cosine + 1.0F) * 0.5F));
                    return static_cast<float>(config.score_weight_template * similarity
                                              + config.score_weight_coverage * cov
                                              + config.score_weight_consistency * cons);
                };

                // 峰值先按原始余弦相似度排序，再用互为近邻覆盖率决定最终尺度。
                Peak  chosen           = candidates_peaks.front();
                float chosen_cosine    = chosen.score;
                float chosen_coverage  = 0.0F;
                float chosen_consistency = 0.0F;
                float chosen_final     = -std::numeric_limits<float>::infinity();
                for (const auto &peak : candidates_peaks)
                {
                    float coverage    = 0.0F;
                    float consistency = 0.0F;
                    measureCoverage(grid, template_source, peak, tolerance, coverage, consistency);
                    const auto final_score = scoreOf(peak.score, coverage, consistency);
                    if (final_score > chosen_final)
                    {
                        chosen             = peak;
                        chosen_cosine      = peak.score;
                        chosen_coverage    = coverage;
                        chosen_consistency = consistency;
                        chosen_final       = final_score;
                    }
                }

                // 有界细化：在相邻两档之间二分尺度，并对峰值位置做半 patch 步长搜索，仍以最终分数择优。
                const int  refinement_rounds = config.fine_refinement_rounds;
                const auto patch_size        = static_cast<double>(grid.plan.patch_size);
                for (int round = 0; round < refinement_rounds; ++round)
                {
                    const double refine_ratio  = std::sqrt(config.fine_template_scale_step);
                    const int    refine_height = std::max(1, static_cast<int>(std::lround(
                        static_cast<double>(chosen.height) * refine_ratio)));
                    const int    refine_width  = std::max(1, static_cast<int>(std::lround(
                        static_cast<double>(chosen.width) * refine_ratio)));
                    if (refine_height > grid.plan.grid_height || refine_width > grid.plan.grid_width
                        || (refine_height == chosen.height && refine_width == chosen.width))
                    {
                        continue;
                    }
                    prepareTemplate(template_source, refine_height, refine_width, prepared);
                    if (!(prepared.mask_sum > 0.0F))
                    {
                        continue;
                    }

                    Peak                   refined;
                    float                  refined_best = -std::numeric_limits<float>::infinity();
                    std::vector<Peak>      refined_peaks;
                    std::vector<RowResult> refine_rows;
                    for (double delta_row = -0.5; delta_row <= 0.5; delta_row += 0.5)
                    {
                        for (double delta_col = -0.5; delta_col <= 0.5; delta_col += 0.5)
                        {
                            const double origin_row = std::max(0.0, chosen.row + delta_row);
                            const double origin_col = std::max(0.0, chosen.col + delta_col);
                            if (origin_row < 0.0 || origin_col < 0.0
                                || origin_row + refine_height > grid.plan.grid_height
                                || origin_col + refine_width > grid.plan.grid_width)
                            {
                                continue;
                            }
                            refined_peaks.clear();
                            const auto score
                                = slidePrepared(grid, prepared, 1, config.fine_nms_iou, refined_peaks, refine_rows);
                            if (score > refined_best && !refined_peaks.empty())
                            {
                                refined_best = score;
                                refined      = refined_peaks.front();
                            }
                        }
                    }
                    if (refined_best <= chosen_cosine)
                    {
                        continue;
                    }
                    float coverage    = 0.0F;
                    float consistency = 0.0F;
                    measureCoverage(grid, template_source, refined, tolerance, coverage, consistency);
                    const auto final_score = scoreOf(refined.score, coverage, consistency);
                    if (final_score > chosen_final)
                    {
                        chosen             = refined;
                        chosen_cosine      = refined.score;
                        chosen_coverage    = coverage;
                        chosen_consistency = consistency;
                        chosen_final       = final_score;
                    }
                }

                if (!(chosen_cosine >= static_cast<float>(config.fine_match_cosine_threshold)))
                {
                    outcome.match_ms += dinoNowMs() - match_started;
                    ++outcome.completed_candidates;
                    continue;
                }

                // 峰值与模板尺寸以 patch 为单位；映射回原图前必须换算到模型输入像素。
                const DinoRect matched_input_box{chosen.col * patch_size, chosen.row * patch_size,
                                                 (chosen.col + chosen.width) * patch_size,
                                                 (chosen.row + chosen.height) * patch_size};
                const DinoRect matched_source = plan.input_to_canonical.apply(matched_input_box);

                DinoMatchResult result;
                result.image_id            = record.image_id;
                result.source_path         = record.source_path;
                result.bbox                = dinoClampRect(matched_source, image_width, image_height);
                result.template_similarity = static_cast<float>(std::min(1.0, std::max(0.0,
                    (static_cast<double>(chosen_cosine) + 1.0) * 0.5)));
                result.query_coverage      = chosen_coverage;
                result.spatial_consistency = chosen_consistency;
                result.score               = chosen_final;
                result.from_region_channel = candidate.from_region;
                result.from_local_channel  = candidate.from_local;
                result.coarse_view_id      = candidate.view_id;

                if (!result.bbox.empty())
                {
                    outcome.results.push_back(std::move(result));
                }
                outcome.match_ms += dinoNowMs() - match_started;
                ++outcome.completed_candidates;
            }
        }
    }

    std::sort(outcome.results.begin(), outcome.results.end(),
              [](const DinoMatchResult &a, const DinoMatchResult &b) { return a.score > b.score; });
    return outcome;
}

} // namespace irt::features::priv
