/**
 * @file DinoDescriptors.cpp
 * @brief 区域池化、局部相邻合并与 INT8 编码实现。
 */

#include "DinoDescriptors.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace irt::features::priv {

namespace {

constexpr float kMinimumNorm = 1e-8F;

/** @brief 收集矩形内的有效 patch 下标（行优先）。 */
std::vector<int> collectValidPatches(const DinoFeatureGrid &grid, const int row0, const int col0, const int height,
                                     const int width)
{
    std::vector<int> ids;
    for (int row = row0; row < row0 + height; ++row)
    {
        for (int col = col0; col < col0 + width; ++col)
        {
            if (grid.patchValid(row, col))
            {
                ids.push_back(row * grid.plan.grid_width + col);
            }
        }
    }
    return ids;
}

int closestValidToCenter(const DinoFeatureGrid &grid, const std::vector<int> &ids, const int row0, const int col0,
                         const int height, const int width)
{
    const double center_row = static_cast<double>(row0) + static_cast<double>(height) * 0.5;
    const double center_col = static_cast<double>(col0) + static_cast<double>(width) * 0.5;
    int          best       = ids.front();
    double       best_distance = std::numeric_limits<double>::infinity();
    for (const auto id : ids)
    {
        const int row = id / grid.plan.grid_width;
        const int col = id % grid.plan.grid_width;
        const double dr = static_cast<double>(row) + 0.5 - center_row;
        const double dc = static_cast<double>(col) + 0.5 - center_col;
        const double distance = dr * dr + dc * dc;
        if (distance < best_distance)
        {
            best_distance = distance;
            best          = id;
        }
    }
    return best;
}

double maxDistanceToRepresentative(const DinoFeatureGrid &grid, const std::vector<int> &ids, const int representative)
{
    const float *reference = grid.tokens.data() + static_cast<size_t>(representative) * static_cast<size_t>(grid.channels);
    double       maximum   = 0.0;
    for (const auto id : ids)
    {
        if (id == representative)
        {
            continue;
        }
        const float *candidate = grid.tokens.data() + static_cast<size_t>(id) * static_cast<size_t>(grid.channels);
        double       sum       = 0.0;
        for (int channel = 0; channel < grid.channels; ++channel)
        {
            const double difference = static_cast<double>(candidate[channel]) - static_cast<double>(reference[channel]);
            sum += difference * difference;
        }
        maximum = std::max(maximum, std::sqrt(sum));
    }
    return maximum;
}

/**
 * @brief 矩形四叉划分的误差受控合并。
 *
 * 距离在原始 FP32 归一化特征上计算，不使用子节点均值；只有组内全部原始 patch 与
 * 代表的距离都不超过阈值时才允许该矩形成为叶节点。
 */
void compressRect(const DinoFeatureGrid &grid, const int row0, const int col0, const int height, const int width,
                  const DinoDescriptorBuildConfig &config, int view_id, DinoViewDescriptors &out,
                  DinoViewDescriptorStats &stats)
{
    const auto ids = collectValidPatches(grid, row0, col0, height, width);
    if (ids.empty())
    {
        return;
    }

    const int    representative = closestValidToCenter(grid, ids, row0, col0, height, width);
    const double radius         = maxDistanceToRepresentative(grid, ids, representative);
    const bool   fits_leaf_limit
        = height <= config.max_leaf_side_patches && width <= config.max_leaf_side_patches;

    if (fits_leaf_limit && radius <= config.merge_epsilon)
    {
        DinoLocalLeaf leaf;
        leaf.view_id      = view_id;
        leaf.grid_row     = row0;
        leaf.grid_col     = col0;
        leaf.grid_height  = height;
        leaf.grid_width   = width;
        leaf.rep_row      = representative / grid.plan.grid_width;
        leaf.rep_col      = representative % grid.plan.grid_width;
        leaf.member_count = static_cast<int>(ids.size());
        leaf.radius       = static_cast<float>(radius);
        out.local_meta.push_back(leaf);

        const float *source = grid.tokens.data()
                            + static_cast<size_t>(representative) * static_cast<size_t>(grid.channels);
        out.local_vectors.emplace_back(source, source + grid.channels);

        stats.max_radius = std::max(stats.max_radius, static_cast<float>(radius));
        stats.mean_radius += radius;
        return;
    }

    if (height == 1 && width == 1)
    {
        // 单 patch 无法再细分；以自身为代表保留，不丢弃任何有效证据。
        DinoLocalLeaf leaf;
        leaf.view_id      = view_id;
        leaf.grid_row     = row0;
        leaf.grid_col     = col0;
        leaf.grid_height  = 1;
        leaf.grid_width   = 1;
        leaf.rep_row      = row0;
        leaf.rep_col      = col0;
        leaf.member_count = 1;
        leaf.radius       = 0.0F;
        out.local_meta.push_back(leaf);
        const float *source
            = grid.tokens.data() + static_cast<size_t>(representative) * static_cast<size_t>(grid.channels);
        out.local_vectors.emplace_back(source, source + grid.channels);
        return;
    }

    if (height > 1 && width > 1)
    {
        const int top_height    = height / 2;
        const int top_width     = width / 2;
        const int bottom_height = height - top_height;
        const int right_width   = width - top_width;
        compressRect(grid, row0, col0, top_height, top_width, config, view_id, out, stats);
        compressRect(grid, row0, col0 + top_width, top_height, right_width, config, view_id, out, stats);
        compressRect(grid, row0 + top_height, col0, bottom_height, top_width, config, view_id, out, stats);
        compressRect(grid, row0 + top_height, col0 + top_width, bottom_height, right_width, config, view_id, out,
                     stats);
        return;
    }

    if (height > 1)
    {
        const int top_height    = height / 2;
        const int bottom_height = height - top_height;
        compressRect(grid, row0, col0, top_height, width, config, view_id, out, stats);
        compressRect(grid, row0 + top_height, col0, bottom_height, width, config, view_id, out, stats);
        return;
    }

    const int left_width  = width / 2;
    const int right_width = width - left_width;
    compressRect(grid, row0, col0, height, left_width, config, view_id, out, stats);
    compressRect(grid, row0, col0 + left_width, height, right_width, config, view_id, out, stats);
}

std::vector<int> axisPositions(const int extent, const int window, const int stride)
{
    std::vector<int> positions;
    if (window >= extent)
    {
        positions.push_back(0);
        return positions;
    }
    for (int start = 0; start + window <= extent; start += stride)
    {
        positions.push_back(start);
    }
    const int last = extent - window;
    if (positions.empty() || positions.back() != last)
    {
        positions.push_back(last);
    }
    return positions;
}

} // namespace

bool dinoNormalizeVector(float *values, const size_t dimension) noexcept
{
    if (values == nullptr || dimension == 0U)
    {
        return false;
    }
    double norm = 0.0;
    for (size_t index = 0; index < dimension; ++index)
    {
        norm += static_cast<double>(values[index]) * static_cast<double>(values[index]);
    }
    norm = std::sqrt(norm);
    if (!(norm > kMinimumNorm))
    {
        return false;
    }
    const float inverse = static_cast<float>(1.0 / norm);
    for (size_t index = 0; index < dimension; ++index)
    {
        values[index] *= inverse;
    }
    return true;
}

std::vector<DinoRegionDescriptor> dinoEnumerateRegionWindows(const DinoFeatureGrid &grid,
                                                             const DinoRegionWindowConfig &config,
                                                             const int view_id)
{
    std::vector<DinoRegionDescriptor> windows;
    const auto &plan = grid.plan;
    if (plan.grid_height <= 0 || plan.grid_width <= 0)
    {
        return windows;
    }

    const double valid_width_source  = plan.valid_input_rect.width() * plan.canonicalPerInputPxX();
    const double valid_height_source = plan.valid_input_rect.height() * plan.canonicalPerInputPxY();
    const double base_edge_source    = std::min(valid_width_source, valid_height_source);
    if (!(base_edge_source > 0.0))
    {
        return windows;
    }

    const double interval_x = plan.sourcePxPerPatchX();
    const double interval_y = plan.sourcePxPerPatchY();
    if (!(interval_x > 0.0) || !(interval_y > 0.0))
    {
        return windows;
    }

    struct WindowShape
    {
        int height{0};
        int width{0};
        int stride_rows{0};
        int stride_cols{0};
    };

    std::vector<WindowShape> shapes;
    shapes.reserve(config.ratios.size());
    for (const auto ratio : config.ratios)
    {
        const double edge_source = ratio * base_edge_source;
        const int    rows = std::max(1, std::min(plan.grid_height,
                                                 static_cast<int>(std::lround(edge_source / interval_y))));
        const int    cols = std::max(1, std::min(plan.grid_width,
                                                 static_cast<int>(std::lround(edge_source / interval_x))));
        const int    stride_rows = std::max(1, static_cast<int>(std::lround(static_cast<double>(rows)
                                                                           * config.stride_ratio)));
        const int    stride_cols = std::max(1, static_cast<int>(std::lround(static_cast<double>(cols)
                                                                           * config.stride_ratio)));
        shapes.push_back(WindowShape{rows, cols, stride_rows, stride_cols});
    }

    for (const auto &shape : shapes)
    {
        const auto row_positions = axisPositions(plan.grid_height, shape.height, shape.stride_rows);
        const auto col_positions = axisPositions(plan.grid_width, shape.width, shape.stride_cols);
        for (const int row0 : row_positions)
        {
            for (const int col0 : col_positions)
            {
                double valid_sum = 0.0;
                for (int row = row0; row < row0 + shape.height; ++row)
                {
                    for (int col = col0; col < col0 + shape.width; ++col)
                    {
                        valid_sum += static_cast<double>(grid.valid_area[static_cast<size_t>(row)
                                                                        * static_cast<size_t>(plan.grid_width)
                                                                        + static_cast<size_t>(col)]);
                    }
                }
                const double window_patches = static_cast<double>(shape.height) * static_cast<double>(shape.width);
                const double fraction       = valid_sum / window_patches;
                if (fraction < config.min_valid_fraction)
                {
                    continue;
                }

                DinoRegionDescriptor descriptor;
                descriptor.view_id        = view_id;
                descriptor.grid_row       = row0;
                descriptor.grid_col       = col0;
                descriptor.grid_height    = shape.height;
                descriptor.grid_width     = shape.width;
                descriptor.valid_fraction = static_cast<float>(fraction);
                descriptor.source_rect    = DinoRect::unite(plan.patchRect(row0, col0),
                                                            plan.patchRect(row0 + shape.height - 1,
                                                                           col0 + shape.width - 1));
                windows.push_back(descriptor);
            }
        }
    }
    return windows;
}

std::vector<float> dinoPoolRegionWindow(const DinoFeatureGrid &grid, const DinoRegionDescriptor &window)
{
    const auto dimension = static_cast<size_t>(grid.channels);
    std::vector<float> pooled(dimension, 0.0F);
    double             weight_sum = 0.0;

    for (int row = window.grid_row; row < window.grid_row + window.grid_height; ++row)
    {
        for (int col = window.grid_col; col < window.grid_col + window.grid_width; ++col)
        {
            const auto  index  = static_cast<size_t>(row) * static_cast<size_t>(grid.plan.grid_width)
                              + static_cast<size_t>(col);
            const float weight = grid.valid_area[index];
            if (!(weight > 0.0F))
            {
                continue;
            }
            const float *token = grid.tokens.data() + index * dimension;
            for (size_t channel = 0; channel < dimension; ++channel)
            {
                pooled[channel] += token[channel] * weight;
            }
            weight_sum += weight;
        }
    }

    if (!(weight_sum > 0.0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region window has no valid area to pool");
    }
    for (auto &value : pooled)
    {
        value = static_cast<float>(value / weight_sum);
    }
    if (!dinoNormalizeVector(pooled.data(), pooled.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region window pooled vector is degenerate");
    }
    return pooled;
}

DinoViewDescriptors dinoBuildViewDescriptors(const int view_id, const DinoFeatureGrid &grid,
                                             const DinoDescriptorBuildConfig &config)
{
    DinoViewDescriptors descriptors;
    descriptors.view_id = view_id;

    const auto patch_count = static_cast<uint64_t>(grid.plan.patchCount());
    descriptors.stats.original_patch_count = patch_count;
    for (const auto valid : grid.valid_area)
    {
        if (valid > 0.0F)
        {
            ++descriptors.stats.valid_patch_count;
        }
    }

    const auto windows = dinoEnumerateRegionWindows(grid, config.window, view_id);
    descriptors.region_meta = windows;
    descriptors.region_vectors.reserve(windows.size());
    for (const auto &window : windows)
    {
        descriptors.region_vectors.push_back(dinoPoolRegionWindow(grid, window));
    }
    descriptors.stats.region_count = descriptors.region_vectors.size();

    if (config.merge_enabled)
    {
        compressRect(grid, 0, 0, grid.plan.grid_height, grid.plan.grid_width, config, view_id, descriptors,
                     descriptors.stats);
    }
    else
    {
        // 不合并基线：逐有效 patch 保留独立描述与真实空间坐标。
        for (int row = 0; row < grid.plan.grid_height; ++row)
        {
            for (int col = 0; col < grid.plan.grid_width; ++col)
            {
                if (!grid.patchValid(row, col))
                {
                    continue;
                }
                DinoLocalLeaf leaf;
                leaf.view_id      = view_id;
                leaf.grid_row     = row;
                leaf.grid_col     = col;
                leaf.grid_height  = 1;
                leaf.grid_width   = 1;
                leaf.rep_row      = row;
                leaf.rep_col      = col;
                leaf.member_count = 1;
                leaf.radius       = 0.0F;
                descriptors.local_meta.push_back(leaf);

                const auto   index  = static_cast<size_t>(row) * static_cast<size_t>(grid.plan.grid_width)
                                  + static_cast<size_t>(col);
                const float *source = grid.tokens.data() + index * static_cast<size_t>(grid.channels);
                descriptors.local_vectors.emplace_back(source, source + grid.channels);
            }
        }
    }

    descriptors.stats.local_count = descriptors.local_meta.size();
    if (!descriptors.local_meta.empty())
    {
        descriptors.stats.mean_radius /= static_cast<double>(descriptors.local_meta.size());
    }
    return descriptors;
}

DinoQuantizedVector dinoQuantizeInt8(const float *values, const size_t dimension)
{
    if (values == nullptr || dimension == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "INT8 quantization requires a non-empty vector");
    }

    float maximum = 0.0F;
    for (size_t index = 0; index < dimension; ++index)
    {
        maximum = std::max(maximum, std::abs(values[index]));
    }
    if (!(maximum > kMinimumNorm))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "INT8 quantization input is degenerate");
    }

    DinoQuantizedVector quantized;
    quantized.codes.resize(dimension);
    quantized.scale = maximum / 127.0F;

    double squared_norm = 0.0;
    for (size_t index = 0; index < dimension; ++index)
    {
        // 默认舍入模式即 ties-to-even，与原值保持确定的一致行为。
        const double scaled = static_cast<double>(values[index]) / static_cast<double>(quantized.scale);
        double       rounded = std::nearbyint(scaled);
        rounded = std::min(127.0, std::max(-127.0, rounded));
        quantized.codes[index] = static_cast<int8_t>(rounded);
        const double restored  = static_cast<double>(quantized.codes[index]) * static_cast<double>(quantized.scale);
        squared_norm += restored * restored;
    }
    if (!(squared_norm > 0.0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "INT8 quantization produced a zero vector");
    }
    quantized.inv_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    return quantized;
}

void dinoDequantizeInt8(const DinoQuantizedVector &quantized, float *output)
{
    if (output == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "INT8 dequantization output is null");
    }
    const float factor = quantized.scale * quantized.inv_norm;
    for (size_t index = 0; index < quantized.codes.size(); ++index)
    {
        output[index] = static_cast<float>(quantized.codes[index]) * factor;
    }
}

float dinoQuantizationDelta(const float *values, const DinoQuantizedVector &quantized)
{
    if (values == nullptr || quantized.codes.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "INT8 delta requires a non-empty vector");
    }
    std::vector<float> restored(quantized.codes.size());
    dinoDequantizeInt8(quantized, restored.data());

    double sum = 0.0;
    for (size_t index = 0; index < restored.size(); ++index)
    {
        const double difference = static_cast<double>(values[index]) - static_cast<double>(restored[index]);
        sum += difference * difference;
    }
    return static_cast<float>(std::sqrt(sum));
}

} // namespace irt::features::priv
