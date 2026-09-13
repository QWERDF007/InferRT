/**
 * @file DinoViews.cpp
 * @brief 视图几何规划与输入光栅渲染实现。
 */

#include "DinoViews.hpp"
#include "DinoGeometry.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

namespace irt::features::priv {

namespace {

constexpr double kMinimumScale = 1e-9;

/**
 * @brief 把源矩形向外吸附到整数像素。
 *
 * 吸附在渲染前完成，之后的所有尺寸与偏移都由该整数矩形派生，避免“渲染时再取整”
 * 造成的 mask 与光栅错位。
 */
DinoRect snapSourceRect(const DinoRect &source_rect)
{
    return DinoRect{std::floor(source_rect.x0), std::floor(source_rect.y0), std::ceil(source_rect.x1),
                    std::ceil(source_rect.y1)};
}

} // namespace

DinoViewPlanner::DinoViewPlanner(const int patch_size, const int encoder_edge, const double view_overlap,
                                 std::vector<int> tile_edges, std::vector<double> roi_target_lengths)
    : patch_size_(patch_size)
    , encoder_edge_(encoder_edge)
    , view_overlap_(view_overlap)
    , tile_edges_(std::move(tile_edges))
    , roi_target_lengths_(std::move(roi_target_lengths))
{
    if (patch_size_ <= 0 || encoder_edge_ <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View planner requires positive patch and edge sizes");
    }
    if (encoder_edge_ % patch_size_ != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Encoder edge %d must be divisible by patch size %d", encoder_edge_, patch_size_);
    }
    if (tile_edges_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View planner requires at least one tile edge");
    }
    if (roi_target_lengths_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View planner requires at least one ROI target");
    }
}

DinoViewPlan DinoViewPlanner::makePlan(DinoRect source_rect, const int image_width, const int image_height,
                                       const bool is_full_view, const int scale_index) const
{
    if (image_width <= 0 || image_height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View planning requires positive image dimensions");
    }
    if (source_rect.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View source rect must have positive extent");
    }

    const DinoRect snapped        = snapSourceRect(source_rect);
    const DinoRect image_rect{0.0, 0.0, static_cast<double>(image_width), static_cast<double>(image_height)};
    const DinoRect valid_source   = DinoRect::intersect(snapped, image_rect);

    const double   source_width  = snapped.width();
    const double   source_height = snapped.height();
    const double   scale         = std::min(static_cast<double>(encoder_edge_) / source_width,
                                            static_cast<double>(encoder_edge_) / source_height);
    if (!(scale > kMinimumScale))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View scale resolved to zero");
    }

    const int resized_width = std::min(
        encoder_edge_,
        std::max(1, static_cast<int>(std::lround(source_width * scale))));
    const int resized_height = std::min(
        encoder_edge_,
        std::max(1, static_cast<int>(std::lround(source_height * scale))));

    DinoViewPlan plan;
    plan.input_width  = encoder_edge_;
    plan.input_height = encoder_edge_;
    plan.patch_size   = patch_size_;
    plan.grid_width   = encoder_edge_ / patch_size_;
    plan.grid_height  = encoder_edge_ / patch_size_;
    plan.source_rect  = snapped;
    plan.is_full_view = is_full_view;
    plan.scale_index  = scale_index;

    // 光栅尺度按取整后的输出尺寸定义，保证 mask 与光栅使用同一组换算关系。
    const double scale_x = static_cast<double>(resized_width) / source_width;
    const double scale_y = static_cast<double>(resized_height) / source_height;
    plan.canonical_to_input = DinoAffine::scaleTranslate(scale_x, scale_y, -snapped.x0 * scale_x, -snapped.y0 * scale_y);
    plan.input_to_canonical = plan.canonical_to_input.inverse();

    if (valid_source.empty())
    {
        plan.valid_input_rect = DinoRect{};
    }
    else
    {
        plan.valid_input_rect = plan.canonical_to_input.apply(valid_source);
        plan.valid_input_rect = DinoRect::intersect(
            plan.valid_input_rect,
            DinoRect{0.0, 0.0, static_cast<double>(encoder_edge_), static_cast<double>(encoder_edge_)});
    }
    return plan;
}

std::vector<DinoViewPlan> DinoViewPlanner::planGalleryViews(const int image_width, const int image_height) const
{
    std::vector<DinoViewPlan> plans;

    // 整图视图：保持纵横比编码，作为全局上下文。
    plans.push_back(makePlan(DinoRect{0.0, 0.0, static_cast<double>(image_width), static_cast<double>(image_height)},
                             image_width, image_height, true, -1));

    std::set<std::tuple<int, int, int, int>> seen;
    seen.emplace(0, 0, image_width, image_height);

    for (size_t scale_index = 0; scale_index < tile_edges_.size(); ++scale_index)
    {
        const int tile_edge = tile_edges_[scale_index];
        const int tile_w    = std::min(tile_edge, image_width);
        const int tile_h    = std::min(tile_edge, image_height);
        const int stride    = std::max(1, static_cast<int>(std::lround(static_cast<double>(tile_edge)
                                                                     * (1.0 - view_overlap_))));

        const auto axis_starts = [&](const int full_length, const int tile_length)
        {
            std::vector<int> starts;
            if (full_length <= tile_length)
            {
                starts.push_back(0);
                return starts;
            }
            for (int start = 0; start + tile_length <= full_length; start += stride)
            {
                starts.push_back(start);
            }
            const int last = std::max(0, full_length - tile_length);
            if (starts.empty() || starts.back() != last)
            {
                starts.push_back(last);
            }
            return starts;
        };

        const auto x_starts = axis_starts(image_width, tile_w);
        const auto y_starts = axis_starts(image_height, tile_h);
        for (const int y : y_starts)
        {
            for (const int x : x_starts)
            {
                const auto key = std::make_tuple(x, y, x + tile_w, y + tile_h);
                if (!seen.insert(key).second)
                {
                    continue;
                }
                plans.push_back(makePlan(DinoRect{static_cast<double>(x), static_cast<double>(y),
                                                  static_cast<double>(x + tile_w), static_cast<double>(y + tile_h)},
                                         image_width, image_height, false, static_cast<int>(scale_index)));
            }
        }
    }
    return plans;
}

std::vector<DinoViewPlan> DinoViewPlanner::planQueryViews(const DinoRoi &roi, const int image_width,
                                                          const int image_height) const
{
    const auto   bounds    = roi.boundingBox();
    const double roi_long  = dinoRoiLongSide(roi);
    if (!(roi_long > 0.0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query ROI has zero extent");
    }

    const double center_x = bounds.x0 + bounds.width() * 0.5;
    const double center_y = bounds.y0 + bounds.height() * 0.5;

    std::vector<DinoViewPlan> plans;
    std::set<std::tuple<int, int, int, int>> seen;
    for (size_t index = 0; index < roi_target_lengths_.size(); ++index)
    {
        // 目标长度不能超过输入光栅本身；超过时按输入长边封顶并记录。
        const double target = std::min(roi_target_lengths_[index], static_cast<double>(encoder_edge_));
        const double context_edge = roi_long * static_cast<double>(encoder_edge_) / target;
        if (!(context_edge > 0.0))
        {
            continue;
        }
        DinoRect source_rect{center_x - context_edge * 0.5, center_y - context_edge * 0.5,
                             center_x + context_edge * 0.5, center_y + context_edge * 0.5};
        const auto plan = makePlan(source_rect, image_width, image_height, false, -2 - static_cast<int>(index));
        const auto key  = std::make_tuple(static_cast<int>(plan.source_rect.x0), static_cast<int>(plan.source_rect.y0),
                                          static_cast<int>(plan.source_rect.x1), static_cast<int>(plan.source_rect.y1));
        if (!seen.insert(key).second)
        {
            continue;
        }
        plans.push_back(plan);
    }
    if (plans.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query ROI produced no usable context view");
    }
    return plans;
}

irt::PreprocessSpec dinoViewPreprocessSpec(const int encoder_edge, const int patch_size)
{
    if (encoder_edge <= 0 || patch_size <= 0 || encoder_edge % patch_size != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid encoder edge/patch size for view preprocess");
    }
    irt::PreprocessSpec spec;
    spec.input_width       = encoder_edge;
    spec.input_height      = encoder_edge;
    spec.input_channels    = 3;
    spec.source_channels   = 3;
    spec.src_color         = irt::ColorFormat::BGR;
    spec.dst_color         = irt::ColorFormat::RGB;
    spec.interpolation     = irt::Interpolation::Linear;
    spec.padding_mode      = irt::PaddingMode::Letterbox;
    spec.padding_alignment = irt::PaddingAlignment::TopLeft;
    spec.pad_after_normalize = true;
    spec.mean              = {0.485F, 0.456F, 0.406F};
    spec.stddev            = {0.229F, 0.224F, 0.225F};
    spec.scale             = 1.0F / 255.0F;
    spec.pad_value         = 0.0F;
    spec.validate();
    return spec;
}

std::vector<float> dinoPatchValidArea(const DinoViewPlan &plan)
{
    const auto count = static_cast<size_t>(plan.patchCount());
    std::vector<float> area(count, 0.0F);
    if (count == 0U || plan.valid_input_rect.empty())
    {
        return area;
    }

    const DinoRect input_rect{0.0, 0.0, static_cast<double>(plan.input_width),
                              static_cast<double>(plan.input_height)};
    const DinoRect valid = DinoRect::intersect(plan.valid_input_rect, input_rect);
    if (valid.empty())
    {
        return area;
    }

    const double patch_area = static_cast<double>(plan.patch_size) * static_cast<double>(plan.patch_size);
    for (int row = 0; row < plan.grid_height; ++row)
    {
        for (int col = 0; col < plan.grid_width; ++col)
        {
            const double x0 = static_cast<double>(col) * plan.patch_size;
            const double y0 = static_cast<double>(row) * plan.patch_size;
            const DinoRect patch{x0, y0, x0 + plan.patch_size, y0 + plan.patch_size};
            const double covered = dinoRectIntersectionArea(patch, valid);
            const double fraction = std::min(1.0, std::max(0.0, covered / patch_area));
            area[static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width) + static_cast<size_t>(col)]
                = static_cast<float>(fraction);
        }
    }
    return area;
}

DinoViewRaster dinoRenderView(const cv::Mat &canonical_bgr, const DinoViewPlan &plan,
                             const irt::PreprocessSpec &spec)
{
    if (canonical_bgr.type() != CV_8UC3)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Canonical image must be 8-bit three channel BGR");
    }

    DinoViewRaster raster;
    raster.plan = plan;

    const auto element_count = static_cast<size_t>(plan.input_width) * static_cast<size_t>(plan.input_height) * 3U;
    raster.chw.assign(element_count, 0.0F);

    const DinoRect valid_source{plan.valid_input_rect.empty()
                                    ? DinoRect{}
                                    : plan.input_to_canonical.apply(plan.valid_input_rect)};
    const DinoRect image_rect{0.0, 0.0, static_cast<double>(canonical_bgr.cols),
                              static_cast<double>(canonical_bgr.rows)};
    const DinoRect integer_source = DinoRect::intersect(
        DinoRect{std::floor(valid_source.x0), std::floor(valid_source.y0), std::ceil(valid_source.x1),
                 std::ceil(valid_source.y1)},
        image_rect);
    if (integer_source.empty())
    {
        return raster;
    }

    const int source_x0 = static_cast<int>(integer_source.x0);
    const int source_y0 = static_cast<int>(integer_source.y0);
    const int source_w  = static_cast<int>(integer_source.width());
    const int source_h  = static_cast<int>(integer_source.height());
    const int target_w  = std::max(1, static_cast<int>(std::lround(plan.valid_input_rect.width())));
    const int target_h  = std::max(1, static_cast<int>(std::lround(plan.valid_input_rect.height())));
    const int offset_x  = std::max(0, static_cast<int>(std::lround(plan.valid_input_rect.x0)));
    const int offset_y  = std::max(0, static_cast<int>(std::lround(plan.valid_input_rect.y0)));

    const cv::Mat crop = canonical_bgr(cv::Rect(source_x0, source_y0, source_w, source_h));
    cv::Mat       resized;
    cv::resize(crop, resized, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    const float scale = spec.scale;
    const int   width   = plan.input_width;
    const int   height  = plan.input_height;
    const int   plane   = width * height;

    for (int row = 0; row < target_h; ++row)
    {
        const int canvas_row = offset_y + row;
        if (canvas_row >= height)
        {
            break;
        }
        const auto *source_row = rgb.ptr<cv::Vec3b>(row);
        for (int col = 0; col < target_w; ++col)
        {
            const int canvas_col = offset_x + col;
            if (canvas_col >= width)
            {
                break;
            }
            const auto       index  = static_cast<size_t>(canvas_row) * static_cast<size_t>(width)
                                    + static_cast<size_t>(canvas_col);
            const cv::Vec3b &pixel = source_row[col];
            for (int channel = 0; channel < 3; ++channel)
            {
                const float normalized = (static_cast<float>(pixel[channel]) * scale - spec.mean[static_cast<size_t>(channel)])
                                       / spec.stddev[static_cast<size_t>(channel)];
                raster.chw[static_cast<size_t>(channel) * static_cast<size_t>(plane) + index] = normalized;
            }
        }
    }
    return raster;
}

} // namespace irt::features::priv
