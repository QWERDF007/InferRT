#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/NMS.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace irt::ops {
namespace {

void validateInputs(const float *boxes, const float *scores, int64_t num_boxes, float iou_threshold)
{
    if (num_boxes < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_boxes must be non-negative, got %lld",
                        static_cast<long long>(num_boxes));
    }
    if (!std::isfinite(iou_threshold))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "iou_threshold must be finite");
    }
    if (num_boxes > 0 && boxes == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "boxes must not be null");
    }
    if (num_boxes > 0 && scores == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "scores must not be null");
    }
}

float boxArea(const float *box)
{
    const float width  = std::max(box[2] - box[0], 0.0f);
    const float height = std::max(box[3] - box[1], 0.0f);
    return width * height;
}

float intersectionOverUnion(const float *lhs, float lhs_area, const float *rhs, float rhs_area)
{
    const float xx1 = std::max(lhs[0], rhs[0]);
    const float yy1 = std::max(lhs[1], rhs[1]);
    const float xx2 = std::min(lhs[2], rhs[2]);
    const float yy2 = std::min(lhs[3], rhs[3]);

    const float width      = std::max(xx2 - xx1, 0.0f);
    const float height     = std::max(yy2 - yy1, 0.0f);
    const float inter      = width * height;
    const float union_area = lhs_area + rhs_area - inter;
    if (union_area <= 0.0f)
    {
        return 0.0f;
    }
    return inter / union_area;
}

} // namespace

std::vector<int64_t> nms(const float *boxes, const float *scores, int64_t num_boxes, float iou_threshold)
{
    validateInputs(boxes, scores, num_boxes, iou_threshold);
    if (num_boxes == 0)
    {
        return {};
    }

    std::vector<float> areas(static_cast<size_t>(num_boxes));
    for (int64_t i = 0; i < num_boxes; ++i)
    {
        const float *box = boxes + i * 4;
        for (int coord = 0; coord < 4; ++coord)
        {
            if (!std::isfinite(box[coord]))
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "boxes[%lld, %d] must be finite",
                                static_cast<long long>(i), coord);
            }
        }
        if (!std::isfinite(scores[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "scores[%lld] must be finite", static_cast<long long>(i));
        }
        areas[static_cast<size_t>(i)] = boxArea(box);
    }

    std::vector<int64_t> order(static_cast<size_t>(num_boxes));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) { return scores[lhs] > scores[rhs]; });

    std::vector<int64_t> keep;
    keep.reserve(order.size());
    for (const int64_t candidate : order)
    {
        bool         suppressed     = false;
        const float *candidate_box  = boxes + candidate * 4;
        const float  candidate_area = areas[static_cast<size_t>(candidate)];

        for (const int64_t kept : keep)
        {
            const float *kept_box  = boxes + kept * 4;
            const float  kept_area = areas[static_cast<size_t>(kept)];
            if (intersectionOverUnion(candidate_box, candidate_area, kept_box, kept_area) > iou_threshold)
            {
                suppressed = true;
                break;
            }
        }

        if (!suppressed)
        {
            keep.push_back(candidate);
        }
    }

    return keep;
}

} // namespace irt::ops
