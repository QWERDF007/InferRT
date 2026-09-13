/**
 * @file DinoGeometry.cpp
 * @brief 区域检索几何契约实现。
 */

#include "DinoGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace irt::features::priv {

namespace {

constexpr double kCoordinateEpsilon = 1e-4;

inline bool isFinite(const double value) noexcept
{
    return std::isfinite(value);
}

/** @brief 用 Sutherland–Hodgman 算法把多边形裁剪到轴对齐矩形。 */
std::vector<DinoPoint> clipPolygonToRect(const std::vector<DinoPoint> &polygon, const DinoRect &rect)
{
    std::vector<DinoPoint> output = polygon;

    struct Edge
    {
        int    axis;     ///< 0 = x, 1 = y
        double value;    ///< 裁剪边界值
        bool   keep_greater; ///< true 表示保留 >= value 的一侧
    };

    const Edge edges[4]{
        {0, rect.x0, true},
        {0, rect.x1, false},
        {1, rect.y0, true},
        {1, rect.y1, false},
    };

    for (const auto &edge : edges)
    {
        if (output.empty())
        {
            return {};
        }
        std::vector<DinoPoint> input = output;
        output.clear();
        output.reserve(input.size() + 1U);

        const auto coordinate = [&](const DinoPoint &point)
        {
            return edge.axis == 0 ? point.x : point.y;
        };
        const auto inside = [&](const DinoPoint &point)
        {
            const double value = coordinate(point);
            return edge.keep_greater ? value >= edge.value : value <= edge.value;
        };
        const auto interpolate = [&](const DinoPoint &from, const DinoPoint &to)
        {
            const double from_value = coordinate(from);
            const double to_value   = coordinate(to);
            const double denominator = to_value - from_value;
            const double t           = std::abs(denominator) < 1e-12 ? 0.0 : (edge.value - from_value) / denominator;
            DinoPoint    point;
            point.x = from.x + (to.x - from.x) * t;
            point.y = from.y + (to.y - from.y) * t;
            // 裁剪产生的顶点必须精确落在边界上，避免后续面积出现负值或 NaN。
            if (edge.axis == 0)
            {
                point.x = edge.value;
            }
            else
            {
                point.y = edge.value;
            }
            return point;
        };

        for (size_t index = 0; index < input.size(); ++index)
        {
            const DinoPoint &current = input[index];
            const DinoPoint &previous = input[(index + input.size() - 1U) % input.size()];
            const bool current_inside  = inside(current);
            const bool previous_inside = inside(previous);

            if (current_inside)
            {
                if (!previous_inside)
                {
                    output.push_back(interpolate(previous, current));
                }
                output.push_back(current);
            }
            else if (previous_inside)
            {
                output.push_back(interpolate(previous, current));
            }
        }
    }

    return output;
}

inline double cross(const DinoPoint &origin, const DinoPoint &a, const DinoPoint &b) noexcept
{
    return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
}

inline bool onSegment(const DinoPoint &a, const DinoPoint &b, const DinoPoint &point) noexcept
{
    const double tolerance = 1e-9;
    if (std::abs(cross(a, b, point)) > tolerance)
    {
        return false;
    }
    return point.x <= std::max(a.x, b.x) + tolerance && point.x >= std::min(a.x, b.x) - tolerance
        && point.y <= std::max(a.y, b.y) + tolerance && point.y >= std::min(a.y, b.y) - tolerance;
}

} // namespace

void dinoValidateRect(const DinoRect &rect, const char *owner)
{
    if (!isFinite(rect.x0) || !isFinite(rect.y0) || !isFinite(rect.x1) || !isFinite(rect.y1))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s rect contains non-finite coordinate", owner);
    }
    if (!(rect.x1 > rect.x0) || !(rect.y1 > rect.y0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s rect must have positive area", owner);
    }
}

bool dinoValidatePolygon(const std::vector<DinoPoint> &polygon, std::string &message)
{
    if (polygon.size() < 3U)
    {
        message = "polygon requires at least 3 vertices";
        return false;
    }
    for (const auto &point : polygon)
    {
        if (!isFinite(point.x) || !isFinite(point.y))
        {
            message = "polygon contains non-finite vertex";
            return false;
        }
    }
    if (std::abs(dinoPolygonArea(polygon)) < 1e-6)
    {
        message = "polygon has zero area";
        return false;
    }

    const size_t count = polygon.size();
    for (size_t i = 0; i < count; ++i)
    {
        const auto &a1 = polygon[i];
        const auto &a2 = polygon[(i + 1U) % count];
        for (size_t j = i + 1U; j < count; ++j)
        {
            if (j == i || (j + 1U) % count == i || j == (i + 1U) % count)
            {
                continue;
            }
            const auto &b1 = polygon[j];
            const auto &b2 = polygon[(j + 1U) % count];
            const double d1 = cross(a1, a2, b1);
            const double d2 = cross(a1, a2, b2);
            const double d3 = cross(b1, b2, a1);
            const double d4 = cross(b1, b2, a2);
            const bool   intersects = ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0));
            if (intersects || onSegment(a1, a2, b1) || onSegment(a1, a2, b2) || onSegment(b1, b2, a1)
                || onSegment(b1, b2, a2))
            {
                message = "polygon must not self-intersect";
                return false;
            }
        }
    }
    return true;
}

double dinoPolygonArea(const std::vector<DinoPoint> &polygon)
{
    if (polygon.size() < 3U)
    {
        return 0.0;
    }
    double twice_area = 0.0;
    for (size_t index = 0; index < polygon.size(); ++index)
    {
        const auto &current = polygon[index];
        const auto &next    = polygon[(index + 1U) % polygon.size()];
        twice_area += current.x * next.y - next.x * current.y;
    }
    return std::abs(twice_area) * 0.5;
}

double dinoPolygonRectArea(const std::vector<DinoPoint> &polygon, const DinoRect &rect)
{
    if (polygon.size() < 3U || rect.empty())
    {
        return 0.0;
    }
    const auto clipped = clipPolygonToRect(polygon, rect);
    return dinoPolygonArea(clipped);
}

double dinoRectIntersectionArea(const DinoRect &a, const DinoRect &b) noexcept
{
    const auto intersection = DinoRect::intersect(a, b);
    return intersection.area();
}

double dinoIoU(const DinoRect &a, const DinoRect &b) noexcept
{
    const double intersection = dinoRectIntersectionArea(a, b);
    if (intersection <= 0.0)
    {
        return 0.0;
    }
    const double uni = a.area() + b.area() - intersection;
    return uni > 0.0 ? intersection / uni : 0.0;
}

DinoRoi dinoToInternalRoi(const DinoSearchRoi &roi, const int image_width, const int image_height)
{
    if (roi.has_bbox == roi.has_polygon)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query request must contain exactly one of bbox or polygon");
    }
    if (image_width <= 0 || image_height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Canonical image dimensions must be positive");
    }

    const DinoRect image_rect{0.0, 0.0, static_cast<double>(image_width), static_cast<double>(image_height)};
    DinoRoi        result;

    if (roi.has_bbox)
    {
        DinoRect rect{roi.bbox.x0, roi.bbox.y0, roi.bbox.x1, roi.bbox.y1};
        if (!isFinite(rect.x0) || !isFinite(rect.y0) || !isFinite(rect.x1) || !isFinite(rect.y1))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI bbox contains non-finite coordinate");
        }
        if (!(rect.x1 > rect.x0) || !(rect.y1 > rect.y0))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI bbox must have positive area");
        }

        // 只容忍浮点舍入级别的越界；明显超出图像范围视为非法请求。
        const double tolerance = kCoordinateEpsilon;
        if (rect.x0 < -tolerance || rect.y0 < -tolerance || rect.x1 > image_rect.x1 + tolerance
            || rect.y1 > image_rect.y1 + tolerance)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI bbox is outside the canonical image: [%g, %g, %g, %g] vs %dx%d", rect.x0,
                                 rect.y0, rect.x1, rect.y1, image_width, image_height);
        }
        rect.x0 = std::max(rect.x0, 0.0);
        rect.y0 = std::max(rect.y0, 0.0);
        rect.x1 = std::min(rect.x1, image_rect.x1);
        rect.y1 = std::min(rect.y1, image_rect.y1);
        if (rect.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI bbox collapses to zero area after clamping");
        }
        result.is_polygon = false;
        result.bbox       = rect;
        return result;
    }

    std::string message;
    std::vector<DinoPoint> points;
    points.reserve(roi.polygon.size());
    for (const auto &point : roi.polygon)
    {
        points.push_back(DinoPoint{point.x, point.y});
    }
    if (!dinoValidatePolygon(points, message))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI polygon: %s", message.c_str());
    }

    const auto bounds = DinoRoi{true, DinoRect{}, points}.boundingBox();
    const double tolerance = kCoordinateEpsilon;
    if (bounds.x0 < -tolerance || bounds.y0 < -tolerance || bounds.x1 > image_rect.x1 + tolerance
        || bounds.y1 > image_rect.y1 + tolerance)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI polygon is outside the canonical image: [%g, %g, %g, %g] vs %dx%d", bounds.x0,
                             bounds.y0, bounds.x1, bounds.y1, image_width, image_height);
    }

    result.is_polygon = true;
    result.polygon    = std::move(points);
    return result;
}

double dinoRoiLongSide(const DinoRoi &roi) noexcept
{
    const auto bounds = roi.boundingBox();
    const auto width  = bounds.width();
    const auto height = bounds.height();
    return width > height ? width : height;
}

DinoRect dinoClampRect(const DinoRect &rect, const int image_width, const int image_height) noexcept
{
    DinoRect clamped = rect;
    clamped.x0       = std::max(clamped.x0, 0.0);
    clamped.y0       = std::max(clamped.y0, 0.0);
    clamped.x1       = std::min(clamped.x1, static_cast<double>(image_width));
    clamped.y1       = std::min(clamped.y1, static_cast<double>(image_height));
    if (clamped.x1 < clamped.x0)
    {
        clamped.x1 = clamped.x0;
    }
    if (clamped.y1 < clamped.y0)
    {
        clamped.y1 = clamped.y0;
    }
    return clamped;
}

DinoRect dinoExpandRect(const DinoRect &rect, const double expand_ratio, const int image_width,
                        const int image_height) noexcept
{
    const double width    = rect.width();
    const double height   = rect.height();
    const double center_x = rect.x0 + width * 0.5;
    const double center_y = rect.y0 + height * 0.5;
    const double half_w   = width * 0.5 * expand_ratio;
    const double half_h   = height * 0.5 * expand_ratio;
    return dinoClampRect(DinoRect{center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h},
                         image_width, image_height);
}

std::vector<float> dinoPatchRoiWeights(const DinoViewPlan &plan, const DinoRoi &roi)
{
    const auto count = static_cast<size_t>(plan.patchCount());
    std::vector<float> weights(count, 0.0F);
    if (count == 0U)
    {
        return weights;
    }

    for (int row = 0; row < plan.grid_height; ++row)
    {
        for (int col = 0; col < plan.grid_width; ++col)
        {
            const auto patch_rect = plan.patchRect(row, col);
            const double patch_source_area = patch_rect.area();
            if (patch_source_area <= 0.0)
            {
                continue;
            }
            double covered = 0.0;
            if (roi.is_polygon)
            {
                covered = dinoPolygonRectArea(roi.polygon, patch_rect);
            }
            else
            {
                covered = dinoRectIntersectionArea(roi.bbox, patch_rect);
            }
            const double fraction = std::min(1.0, std::max(0.0, covered / patch_source_area));
            weights[static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width) + static_cast<size_t>(col)]
                = static_cast<float>(fraction);
        }
    }
    return weights;
}

bool dinoWithinValidatedProfile(const DinoRoi &roi, const int image_width, const int image_height,
                                const DinoValidatedRange &range, std::string &message)
{
    const int long_edge  = std::max(image_width, image_height);
    const int short_edge = std::min(image_width, image_height);
    if (long_edge > range.max_image_edge || short_edge < range.min_image_edge)
    {
        message = "image edge outside the validated " + std::to_string(range.min_image_edge) + ".."
                + std::to_string(range.max_image_edge) + " px range";
        return false;
    }

    const auto bounds      = roi.boundingBox();
    const double width     = bounds.width();
    const double height    = bounds.height();
    const double short_side = std::min(width, height);
    const double long_side  = std::max(width, height);
    if (short_side < range.min_target_short_px)
    {
        message = "ROI short side below the validated minimum of " + std::to_string(range.min_target_short_px) + " px";
        return false;
    }
    if (long_side / short_side > range.max_target_aspect)
    {
        message = "ROI aspect ratio outside the validated 1:A..A:1 range with A="
                + std::to_string(range.max_target_aspect);
        return false;
    }
    return true;
}

void dinoNmsWithinImages(std::vector<DinoMatchResult> &results, const double iou_threshold)
{
    if (results.size() < 2U)
    {
        return;
    }

    std::vector<DinoMatchResult> ordered = results;
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const DinoMatchResult &a, const DinoMatchResult &b) { return a.score > b.score; });

    // 只在同一图片内抑制；不同图片之间的框不互相影响。
    std::vector<DinoMatchResult> kept;
    kept.reserve(ordered.size());
    for (const auto &candidate : ordered)
    {
        bool suppressed = false;
        for (const auto &existing : kept)
        {
            if (existing.image_id != candidate.image_id)
            {
                continue;
            }
            if (dinoIoU(existing.bbox, candidate.bbox) >= iou_threshold)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
        {
            kept.push_back(candidate);
        }
    }
    results = std::move(kept);
}

} // namespace irt::features::priv
