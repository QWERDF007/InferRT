/**
 * @file ShapeTemplateMatcherImpl.cpp
 * @brief 形状模板匹配 PIMPL 实现。
 */

#include "ShapeTemplateMatcherImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

#if defined(_M_X64) || defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define IRT_SHAPE_TEMPLATE_HAS_SSE2 1
#include <emmintrin.h>
#else
#define IRT_SHAPE_TEMPLATE_HAS_SSE2 0
#endif

namespace fs = std::filesystem;

namespace irt::features {
namespace {

constexpr unsigned char kInvalidLabel = 255;     ///< 无效方向标签，用于跳过弱梯度点。
constexpr float         kEps          = 1.0e-6f; ///< 浮点区间比较容差。
constexpr int           kOrientationBins = 8;    ///< 梯度方向量化 bin 数量。

/**
 * @brief 源图梯度量化结果。
 */
struct QuantizedGradient
{
    cv::Mat labels;    ///< ``CV_8U`` 方向标签，255 表示无效点。
    cv::Mat magnitude; ///< ``CV_32F`` 梯度幅值。
    cv::Mat angle;     ///< ``CV_32F`` 梯度方向角，单位为度。
};

/**
 * @brief 每个模板方向对应的源图响应图集合。
 *
 * ``maps[label]`` 中的像素值是该源图像素与模板方向 ``label`` 的匹配分子。分母由
 * ``denominator_per_feature`` 提供，因此匹配分数可保持与浮点公式完全一致。
 */
struct ResponseMaps
{
    std::array<cv::Mat, kOrientationBins> maps; ///< 8 个方向的 ``CV_8U`` 响应图。
    int denominator_per_feature{1};             ///< 单个特征点满分对应的分母。
};

using ResponseTable = std::array<std::array<unsigned char, kOrientationBins>, kOrientationBins>;

/**
 * @brief 模板训练阶段的候选特征点。
 */
struct Candidate
{
    int   x{0};                 ///< 候选点 x 坐标。
    int   y{0};                 ///< 候选点 y 坐标。
    int   label{0};             ///< 量化方向标签。
    float angle_degrees{0.0f};  ///< 原始梯度方向角，单位为度。
    float score{0.0f};          ///< 候选点评分，当前使用梯度幅值。
};

/**
 * @brief 校验分数阈值范围。
 * @param value 待校验值。
 * @param name 错误消息中的字段名。
 */
void validateScoreRange(float value, const char *name)
{
    if (!std::isfinite(value) || value < 0.0f || value > 100.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be finite and in [0, 100]", name);
    }
}

/**
 * @brief 校验模板匹配器配置。
 * @param config 待校验配置。
 */
void validateConfig(const ShapeTemplateMatcherConfig &config)
{
    if (config.num_features <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ShapeTemplateMatcher num_features must be positive");
    }
    if (config.min_features <= 0 || config.min_features > config.num_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher min_features must be positive and <= num_features");
    }
    if (!std::isfinite(config.weak_threshold) || config.weak_threshold < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher weak_threshold must be finite and non-negative");
    }
    if (!std::isfinite(config.strong_threshold) || config.strong_threshold < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher strong_threshold must be finite and non-negative");
    }
    if (config.max_label_difference < 0 || config.max_label_difference > 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_label_difference must be in [0, 4]");
    }
    validateScoreRange(config.match_threshold, "ShapeTemplateMatcher match_threshold");
    if (!std::isfinite(config.nms_threshold) || config.nms_threshold > 1.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher nms_threshold must be finite and <= 1");
    }
    if (config.max_results < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_results must be non-negative");
    }
    if (config.scan_step <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ShapeTemplateMatcher scan_step must be positive");
    }
    if (!std::isfinite(config.min_feature_distance) || config.min_feature_distance < 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher min_feature_distance must be finite and non-negative");
    }
}

/**
 * @brief 校验 OpenCV 图像基本形状。
 * @param image 待校验图像。
 * @param name 错误消息中的图像名称。
 */
void validateImage(const cv::Mat &image, const char *name)
{
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must not be empty", name);
    }
    if (image.dims != 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be a 2D image", name);
    }
    if (image.cols <= 0 || image.rows <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s has invalid size", name);
    }
}

/**
 * @brief 从磁盘读取图像并转换读取失败为 InferRT 异常。
 * @param image_file 图像路径。
 * @param flags OpenCV ``imread`` 标志。
 * @param name 错误消息中的图像名称。
 * @return 读取到的图像。
 */
cv::Mat loadImage(const fs::path &image_file, int flags, const char *name)
{
    if (image_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s path must not be empty", name);
    }
    std::error_code ec;
    if (!fs::is_regular_file(image_file, ec))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s path does not exist: %s", name,
                             image_file.string().c_str());
    }

    cv::Mat image = cv::imread(image_file.string(), flags);
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load %s: %s", name,
                             image_file.string().c_str());
    }
    return image;
}

/**
 * @brief 将输入图像规范化为 8 位灰度图。
 * @param image 输入图像，支持灰度、BGR 或 BGRA。
 * @return ``CV_8UC1`` 灰度图。
 */
cv::Mat toGray8(const cv::Mat &image)
{
    validateImage(image, "image");

    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported image channel count: %d",
                             image.channels());
    }

    cv::Mat gray8;
    if (gray.depth() == CV_8U)
    {
        gray8 = gray;
    }
    else
    {
        gray.convertTo(gray8, CV_8U);
    }
    return gray8;
}

/**
 * @brief 将可选掩膜规范化为二值 8 位单通道图。
 * @param mask 输入掩膜；为空时返回全 255 掩膜。
 * @param size 期望掩膜尺寸。
 * @param name 错误消息中的掩膜名称。
 * @return ``CV_8UC1`` 二值掩膜。
 */
cv::Mat normalizeMask(const cv::Mat &mask, cv::Size size, const char *name)
{
    if (mask.empty())
    {
        return cv::Mat(size, CV_8UC1, cv::Scalar(255));
    }
    if (mask.dims != 2 || mask.size() != size)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s size must match image size", name);
    }

    cv::Mat gray;
    if (mask.channels() == 1)
    {
        gray = mask;
    }
    else if (mask.channels() == 3)
    {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    }
    else if (mask.channels() == 4)
    {
        cv::cvtColor(mask, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s channel count: %d", name,
                             mask.channels());
    }

    cv::Mat mask8;
    if (gray.depth() == CV_8U)
    {
        mask8 = gray;
    }
    else
    {
        gray.convertTo(mask8, CV_8U);
    }

    cv::Mat binary;
    cv::threshold(mask8, binary, 0.0, 255.0, cv::THRESH_BINARY);
    return binary;
}

/**
 * @brief 将梯度方向角量化为 8 个方向 bin。
 * @param angle_degrees 梯度方向角，单位为度。
 * @return 方向标签，范围为 ``[0, 7]``。
 */
int quantizeAngle(float angle_degrees) noexcept
{
    int label = static_cast<int>(std::floor((angle_degrees + 22.5f) / 45.0f));
    label %= kOrientationBins;
    if (label < 0)
    {
        label += kOrientationBins;
    }
    return label;
}

/**
 * @brief 使用标量路径填充方向标签图。
 */
void fillQuantizedLabelsScalar(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask, float threshold,
                               cv::Mat &labels)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *mag_row   = magnitude.ptr<float>(y);
        const auto *angle_row = angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        auto       *label_row = labels.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            if (mask_row[x] != 0 && mag_row[x] >= threshold)
            {
                label_row[x] = static_cast<unsigned char>(quantizeAngle(angle_row[x]));
            }
        }
    }
}

#if IRT_SHAPE_TEMPLATE_HAS_SSE2
/**
 * @brief 使用 SSE2 批量量化梯度方向标签。
 *
 * 每次读取 4 个 ``float`` 方向角，通过 8 个阈值比较得到方向 bin，避免训练阶段逐点调用
 * ``floor``。掩膜和幅值阈值仍以位掩码方式合并，保证无效点写为 ``kInvalidLabel``。
 */
void fillQuantizedLabelsSse2(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask, float threshold,
                             cv::Mat &labels)
{
    const __m128 threshold_vec = _mm_set1_ps(threshold);
    const std::array<__m128, kOrientationBins> angle_thresholds{
        _mm_set1_ps(22.5f),  _mm_set1_ps(67.5f),  _mm_set1_ps(112.5f), _mm_set1_ps(157.5f),
        _mm_set1_ps(202.5f), _mm_set1_ps(247.5f), _mm_set1_ps(292.5f), _mm_set1_ps(337.5f),
    };

    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *mag_row   = magnitude.ptr<float>(y);
        const auto *angle_row = angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        auto       *label_row = labels.ptr<unsigned char>(y);

        int x = 0;
        for (; x <= labels.cols - 4; x += 4)
        {
            const __m128 mag_values   = _mm_loadu_ps(mag_row + x);
            const int    valid_mag    = _mm_movemask_ps(_mm_cmpge_ps(mag_values, threshold_vec));
            const __m128 angle_values = _mm_loadu_ps(angle_row + x);

            int labels4[4]{0, 0, 0, 0};
            for (int threshold_index = 0; threshold_index < kOrientationBins; ++threshold_index)
            {
                const int ge_mask = _mm_movemask_ps(_mm_cmpge_ps(angle_values, angle_thresholds[threshold_index]));
                for (int lane = 0; lane < 4; ++lane)
                {
                    labels4[lane] += (ge_mask >> lane) & 1;
                }
            }

            for (int lane = 0; lane < 4; ++lane)
            {
                if (((valid_mag >> lane) & 1) != 0 && mask_row[x + lane] != 0)
                {
                    label_row[x + lane]
                        = static_cast<unsigned char>(labels4[lane] == kOrientationBins ? 0 : labels4[lane]);
                }
            }
        }

        for (; x < labels.cols; ++x)
        {
            if (mask_row[x] != 0 && mag_row[x] >= threshold)
            {
                label_row[x] = static_cast<unsigned char>(quantizeAngle(angle_row[x]));
            }
        }
    }
}
#endif

/**
 * @brief 计算图像梯度幅值、方向角和量化方向标签。
 * @param image 输入图像。
 * @param mask 有效区域掩膜。
 * @param weak_threshold 弱梯度阈值。
 * @return 梯度量化结果。
 */
QuantizedGradient computeQuantizedGradient(const cv::Mat &image, const cv::Mat &mask, float weak_threshold)
{
    const cv::Mat gray = toGray8(image);
    const cv::Mat mask8 = normalizeMask(mask, gray.size(), "mask");

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    QuantizedGradient gradient;
    cv::cartToPolar(grad_x, grad_y, gradient.magnitude, gradient.angle, true);
    gradient.labels = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(kInvalidLabel));

#if IRT_SHAPE_TEMPLATE_HAS_SSE2
    fillQuantizedLabelsSse2(gradient.magnitude, gradient.angle, mask8, weak_threshold, gradient.labels);
#else
    fillQuantizedLabelsScalar(gradient.magnitude, gradient.angle, mask8, weak_threshold, gradient.labels);
#endif
    return gradient;
}

/**
 * @brief 将候选点按梯度幅值和坐标稳定排序。
 */
void sortCandidates(std::vector<Candidate> &candidates)
{
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b)
              {
                  if (a.score != b.score)
                  {
                      return a.score > b.score;
                  }
                  if (a.y != b.y)
                  {
                      return a.y < b.y;
                  }
                  return a.x < b.x;
              });
}

/**
 * @brief 使用标量路径收集模板训练候选点。
 */
std::vector<Candidate> collectCandidatesScalar(const QuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
    const cv::Mat mask8 = normalizeMask(mask, gradient.labels.size(), "mask");

    std::vector<Candidate> candidates;
    for (int y = 1; y < gradient.labels.rows - 1; ++y)
    {
        const auto *mag_row   = gradient.magnitude.ptr<float>(y);
        const auto *angle_row = gradient.angle.ptr<float>(y);
        const auto *mask_row  = mask8.ptr<unsigned char>(y);
        const auto *label_row = gradient.labels.ptr<unsigned char>(y);
        for (int x = 1; x < gradient.labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || label_row[x] == kInvalidLabel || mag_row[x] < threshold)
            {
                continue;
            }

            candidates.push_back(
                Candidate{x, y, static_cast<int>(label_row[x]), angle_row[x], std::max(mag_row[x], 0.0f)});
        }
    }

    sortCandidates(candidates);
    return candidates;
}

#if IRT_SHAPE_TEMPLATE_HAS_SSE2
/**
 * @brief 使用 SSE2 块级过滤收集模板训练候选点。
 *
 * 每 16 个像素先用 SIMD 合并 ``label != invalid``、``mask != 0`` 和 ``magnitude >= threshold`` 三个条件。
 * 对完全无候选的块直接跳过，只对有效 lane 读取角度和幅值并创建 ``Candidate``。
 */
std::vector<Candidate> collectCandidatesSse2(const QuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
    const cv::Mat mask8 = normalizeMask(mask, gradient.labels.size(), "mask");

    std::vector<Candidate> candidates;
    const __m128  threshold_vec = _mm_set1_ps(threshold);
    const __m128i invalid_vec   = _mm_set1_epi8(static_cast<char>(kInvalidLabel));
    const __m128i zero_vec      = _mm_setzero_si128();

    for (int y = 1; y < gradient.labels.rows - 1; ++y)
    {
        const auto *mag_row   = gradient.magnitude.ptr<float>(y);
        const auto *angle_row = gradient.angle.ptr<float>(y);
        const auto *mask_row  = mask8.ptr<unsigned char>(y);
        const auto *label_row = gradient.labels.ptr<unsigned char>(y);

        int x = 1;
        for (; x <= gradient.labels.cols - 17; x += 16)
        {
            const __m128i label_values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(label_row + x));
            const __m128i mask_values  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(mask_row + x));
            const __m128i label_valid  = _mm_andnot_si128(_mm_cmpeq_epi8(label_values, invalid_vec),
                                                         _mm_set1_epi8(static_cast<char>(0xFF)));
            const __m128i mask_valid
                = _mm_andnot_si128(_mm_cmpeq_epi8(mask_values, zero_vec), _mm_set1_epi8(static_cast<char>(0xFF)));
            const int byte_valid = _mm_movemask_epi8(_mm_and_si128(label_valid, mask_valid));
            if (byte_valid == 0)
            {
                continue;
            }

            int mag_valid = 0;
            for (int group = 0; group < 4; ++group)
            {
                const __m128 mag_values = _mm_loadu_ps(mag_row + x + group * 4);
                mag_valid |= _mm_movemask_ps(_mm_cmpge_ps(mag_values, threshold_vec)) << (group * 4);
            }

            const int valid = byte_valid & mag_valid;
            if (valid == 0)
            {
                continue;
            }

            for (int lane = 0; lane < 16; ++lane)
            {
                if (((valid >> lane) & 1) == 0)
                {
                    continue;
                }
                const int col = x + lane;
                candidates.push_back(
                    Candidate{col, y, static_cast<int>(label_row[col]), angle_row[col], std::max(mag_row[col], 0.0f)});
            }
        }

        for (; x < gradient.labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || label_row[x] == kInvalidLabel || mag_row[x] < threshold)
            {
                continue;
            }
            candidates.push_back(
                Candidate{x, y, static_cast<int>(label_row[x]), angle_row[x], std::max(mag_row[x], 0.0f)});
        }
    }

    sortCandidates(candidates);
    return candidates;
}
#endif

/**
 * @brief 从梯度量化图中收集模板训练候选点。
 * @param gradient 梯度量化结果。
 * @param mask 有效区域掩膜。
 * @param threshold 候选点梯度幅值阈值。
 * @return 按评分从高到低排序的候选点列表。
 */
std::vector<Candidate> collectCandidates(const QuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
#if IRT_SHAPE_TEMPLATE_HAS_SSE2
    return collectCandidatesSse2(gradient, mask, threshold);
#else
    return collectCandidatesScalar(gradient, mask, threshold);
#endif
}

/**
 * @brief 根据模板面积和期望特征数估算默认特征点间距。
 * @param size 模板图像尺寸。
 * @param num_features 期望特征点数量。
 * @return 贪心选点使用的最小间距。
 */
float autoFeatureDistance(cv::Size size, int num_features)
{
    const auto area = static_cast<float>(std::max(1, size.width) * std::max(1, size.height));
    return std::max(1.0f, std::sqrt(area / static_cast<float>(std::max(1, num_features))) * 0.5f);
}

/**
 * @brief 在候选点中贪心选择分散的强梯度特征。
 *
 * 若初始间距导致特征点不足，会逐步降低间距，直到满足最少特征点数量或退化到 1 像素间距。
 *
 * @param candidates 已按评分排序的候选点。
 * @param num_features 最多选择的特征点数。
 * @param min_features 最少需要的特征点数。
 * @param min_distance 初始特征点间距。
 * @return 选中的模板特征候选点。
 */
std::vector<Candidate> greedySelectCandidates(const std::vector<Candidate> &candidates, int num_features,
                                              int min_features, float min_distance)
{
    std::vector<Candidate> best;
    if (candidates.empty())
    {
        return best;
    }

    float distance = std::max(1.0f, min_distance);
    for (;;)
    {
        std::vector<Candidate> selected;
        selected.reserve(static_cast<size_t>(num_features));
        const float min_distance_sq = distance * distance;

        for (const auto &candidate : candidates)
        {
            bool keep = true;
            for (const auto &existing : selected)
            {
                const float dx = static_cast<float>(candidate.x - existing.x);
                const float dy = static_cast<float>(candidate.y - existing.y);
                if (dx * dx + dy * dy < min_distance_sq)
                {
                    keep = false;
                    break;
                }
            }
            if (keep)
            {
                selected.push_back(candidate);
                if (static_cast<int>(selected.size()) >= num_features)
                {
                    break;
                }
            }
        }

        if (selected.size() > best.size())
        {
            best = std::move(selected);
        }
        if (static_cast<int>(best.size()) >= min_features
            || distance <= 1.0f + std::numeric_limits<float>::epsilon())
        {
            break;
        }
        distance *= 0.75f;
        if (distance < 1.0f)
        {
            distance = 1.0f;
        }
    }

    if (static_cast<int>(best.size()) > num_features)
    {
        best.resize(static_cast<size_t>(num_features));
    }
    return best;
}

/**
 * @brief 根据选中特征点构造可持久化的模板信息。
 * @param class_id 模板类别 ID。
 * @param template_id 类别内模板 ID。
 * @param features 已选中的训练特征点。
 * @param variant 旋转/缩放变体元数据。
 * @return 裁剪到特征包围盒后的模板信息。
 */
ShapeTemplateInfo makeTemplateInfo(const std::string &class_id, int template_id,
                                   const std::vector<Candidate> &features, ShapeTemplateVariant variant)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    for (const auto &feature : features)
    {
        min_x = std::min(min_x, feature.x);
        min_y = std::min(min_y, feature.y);
        max_x = std::max(max_x, feature.x);
        max_y = std::max(max_y, feature.y);
    }

    ShapeTemplateInfo info;
    info.class_id      = class_id;
    info.template_id   = template_id;
    info.width         = max_x - min_x + 1;
    info.height        = max_y - min_y + 1;
    info.tl_x          = min_x;
    info.tl_y          = min_y;
    info.angle_degrees = variant.angle_degrees;
    info.scale         = variant.scale;
    info.features.reserve(features.size());

    for (const auto &feature : features)
    {
        info.features.push_back(ShapeTemplateFeature{feature.x - min_x, feature.y - min_y, feature.label,
                                                     feature.angle_degrees});
    }
    return info;
}

/**
 * @brief 计算两个 8-bin 方向标签的环形距离。
 * @param a 方向标签 A。
 * @param b 方向标签 B。
 * @return 最短环形 bin 距离。
 */
int labelDistance(int a, int b) noexcept
{
    const int diff = std::abs(a - b);
    return std::min(diff, kOrientationBins - diff);
}

/**
 * @brief 构造方向响应查找表。
 *
 * 表中值是某个源图方向标签对某个模板方向标签的整数贡献分子。这样可以用字节图保存响应，
 * 最终分数再除以 ``denominator_per_feature``，保持与原始方向距离公式一致。
 *
 * @param max_label_difference 最大允许方向标签差。
 * @return ``table[template_label][source_label]`` 响应表。
 */
ResponseTable makeResponseTable(int max_label_difference)
{
    ResponseTable table{};
    for (int templ_label = 0; templ_label < kOrientationBins; ++templ_label)
    {
        for (int source_label = 0; source_label < kOrientationBins; ++source_label)
        {
            const int diff = labelDistance(templ_label, source_label);
            if (diff > max_label_difference)
            {
                table[templ_label][source_label] = 0;
            }
            else if (max_label_difference == 0)
            {
                table[templ_label][source_label] = 1;
            }
            else
            {
                table[templ_label][source_label]
                    = static_cast<unsigned char>(2 * max_label_difference - diff);
            }
        }
    }
    return table;
}

/**
 * @brief 返回单个特征点满分对应的整数分母。
 * @param max_label_difference 最大允许方向标签差。
 * @return 单点分母。
 */
int responseDenominatorPerFeature(int max_label_difference) noexcept
{
    return max_label_difference == 0 ? 1 : 2 * max_label_difference;
}

/**
 * @brief 使用标量路径生成单个模板方向的响应图。
 */
void fillResponseMapScalar(const cv::Mat &labels, cv::Mat &response, int template_label,
                           const ResponseTable &table)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *src = labels.ptr<unsigned char>(y);
        auto       *dst = response.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            const auto label = src[x];
            dst[x] = label < kOrientationBins ? table[template_label][label] : 0;
        }
    }
}

#if IRT_SHAPE_TEMPLATE_HAS_SSE2
/**
 * @brief 使用 SSE2 指令生成单个模板方向的响应图。
 *
 * SSE2 没有字节查表指令，这里按 8 个方向逐一比较并 OR 合并结果；每次处理 16 个像素。
 */
void fillResponseMapSse2(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ResponseTable &table)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *src = labels.ptr<unsigned char>(y);
        auto       *dst = response.ptr<unsigned char>(y);

        int x = 0;
        for (; x <= labels.cols - 16; x += 16)
        {
            const __m128i source = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + x));
            __m128i       out    = _mm_setzero_si128();
            for (int label = 0; label < kOrientationBins; ++label)
            {
                const auto value = table[template_label][label];
                if (value == 0)
                {
                    continue;
                }

                const __m128i eq = _mm_cmpeq_epi8(source, _mm_set1_epi8(static_cast<char>(label)));
                out              = _mm_or_si128(out, _mm_and_si128(eq, _mm_set1_epi8(static_cast<char>(value))));
            }
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + x), out);
        }

        for (; x < labels.cols; ++x)
        {
            const auto label = src[x];
            dst[x] = label < kOrientationBins ? table[template_label][label] : 0;
        }
    }
}
#endif

/**
 * @brief 为源图方向标签预计算 8 张响应图。
 * @param labels 源图方向标签图。
 * @param max_label_difference 最大允许方向标签差。
 * @return 方向响应图集合。
 */
ResponseMaps buildResponseMaps(const cv::Mat &labels, int max_label_difference)
{
    ResponseMaps response_maps;
    response_maps.denominator_per_feature = responseDenominatorPerFeature(max_label_difference);
    const auto table = makeResponseTable(max_label_difference);

    for (int label = 0; label < kOrientationBins; ++label)
    {
        response_maps.maps[label].create(labels.size(), CV_8UC1);
#if IRT_SHAPE_TEMPLATE_HAS_SSE2
        fillResponseMapSse2(labels, response_maps.maps[label], label, table);
#else
        fillResponseMapScalar(labels, response_maps.maps[label], label, table);
#endif
    }
    return response_maps;
}

/**
 * @brief 基于预计算响应图计算某个滑窗位置的匹配分数。
 * @param response_maps 源图方向响应图。
 * @param templ 待匹配模板。
 * @param x 滑窗左上角 x 坐标。
 * @param y 滑窗左上角 y 坐标。
 * @return 匹配分数，范围为 ``[0, 100]``。
 */
float similarityAt(const ResponseMaps &response_maps, const ShapeTemplateInfo &templ, int x, int y)
{
    uint64_t sum = 0;
    for (const auto &feature : templ.features)
    {
        const auto &map = response_maps.maps[static_cast<size_t>(feature.label)];
        sum += map.ptr<unsigned char>(y + feature.y)[x + feature.x];
    }

    const auto denominator = static_cast<float>(response_maps.denominator_per_feature)
                           * static_cast<float>(templ.features.size());
    return 100.0f * static_cast<float>(sum) / denominator;
}

/**
 * @brief 计算两个匹配框的 IoU。
 */
float intersectionOverUnion(const ShapeTemplateMatch &a, const ShapeTemplateMatch &b) noexcept
{
    const int ax2 = a.x + a.width;
    const int ay2 = a.y + a.height;
    const int bx2 = b.x + b.width;
    const int by2 = b.y + b.height;

    const int ix1 = std::max(a.x, b.x);
    const int iy1 = std::max(a.y, b.y);
    const int ix2 = std::min(ax2, bx2);
    const int iy2 = std::min(ay2, by2);
    const int iw  = std::max(0, ix2 - ix1);
    const int ih  = std::max(0, iy2 - iy1);

    const float intersection = static_cast<float>(iw * ih);
    const float union_area = static_cast<float>(a.width * a.height + b.width * b.height) - intersection;
    return union_area <= 0.0f ? 0.0f : intersection / union_area;
}

/**
 * @brief 按相似度、类别、模板 ID 和坐标稳定排序匹配结果。
 */
void sortMatches(std::vector<ShapeTemplateMatch> &matches)
{
    std::sort(matches.begin(), matches.end(),
              [](const ShapeTemplateMatch &a, const ShapeTemplateMatch &b)
              {
                  if (a.similarity != b.similarity)
                  {
                      return a.similarity > b.similarity;
                  }
                  if (a.class_id != b.class_id)
                  {
                      return a.class_id < b.class_id;
                  }
                  if (a.template_id != b.template_id)
                  {
                      return a.template_id < b.template_id;
                  }
                  if (a.y != b.y)
                  {
                      return a.y < b.y;
                  }
                  return a.x < b.x;
              });
}

/**
 * @brief 对匹配结果执行同类别 NMS 和最大数量限制。
 * @param matches 待处理匹配结果。
 * @param nms_threshold NMS IoU 阈值；小于 0 表示关闭 NMS。
 * @param max_results 最大返回数量；0 表示不限制。
 * @return 过滤后的匹配结果。
 */
std::vector<ShapeTemplateMatch> applyNms(std::vector<ShapeTemplateMatch> matches, float nms_threshold,
                                         int max_results)
{
    sortMatches(matches);
    if (nms_threshold < 0.0f)
    {
        if (max_results > 0 && static_cast<int>(matches.size()) > max_results)
        {
            matches.resize(static_cast<size_t>(max_results));
        }
        return matches;
    }

    std::vector<ShapeTemplateMatch> kept;
    kept.reserve(matches.size());
    for (const auto &match : matches)
    {
        bool keep = true;
        for (const auto &existing : kept)
        {
            if (match.class_id == existing.class_id && intersectionOverUnion(match, existing) > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            kept.push_back(match);
            if (max_results > 0 && static_cast<int>(kept.size()) >= max_results)
            {
                break;
            }
        }
    }
    return kept;
}

/**
 * @brief 校验从文件加载出的单个模板是否完整。
 * @param info 待校验模板。
 * @param min_features 最少特征点数量。
 */
void validateTemplateInfo(const ShapeTemplateInfo &info, int min_features)
{
    if (info.class_id.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template class_id must not be empty");
    }
    if (info.width <= 0 || info.height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid size");
    }
    if (static_cast<int>(info.features.size()) < min_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has too few features");
    }
    if (!std::isfinite(info.angle_degrees) || !std::isfinite(info.scale) || info.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid variant metadata");
    }
    for (const auto &feature : info.features)
    {
        if (feature.x < 0 || feature.x >= info.width || feature.y < 0 || feature.y >= info.height)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template feature is out of bounds");
        }
        if (feature.label < 0 || feature.label > 7 || !std::isfinite(feature.angle_degrees))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template feature label is invalid");
        }
    }
}

/**
 * @brief 从 OpenCV 文件节点读取可选字段。
 * @tparam T 字段类型。
 * @param node 父节点。
 * @param key 字段名称。
 * @param value 字段存在时写入的目标变量。
 */
template<typename T>
void readIfPresent(const cv::FileNode &node, const char *key, T &value)
{
    const cv::FileNode child = node[key];
    if (!child.empty())
    {
        child >> value;
    }
}

} // namespace

/**
 * @brief 构造匹配器实现并完成配置校验。
 */
ShapeTemplateMatcher::Impl::Impl(ShapeTemplateMatcherConfig config)
    : config_(std::move(config))
{
    validateConfig(config_);
}

/**
 * @brief 提取模板图中的强梯度方向特征并写入模板库。
 */
int ShapeTemplateMatcher::Impl::addTemplate(const cv::Mat &image, const std::string &class_id,
                                            const cv::Mat &object_mask, ShapeTemplateVariant variant)
{
    validateImage(image, "template image");
    if (class_id.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template class_id must not be empty");
    }
    if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
    }

    const cv::Mat mask = normalizeMask(object_mask, image.size(), "object_mask");
    const auto    gradient = computeQuantizedGradient(image, mask, config_.weak_threshold);
    auto          candidates = collectCandidates(gradient, mask, config_.strong_threshold);
    if (static_cast<int>(candidates.size()) < config_.min_features && config_.weak_threshold < config_.strong_threshold)
    {
        candidates = collectCandidates(gradient, mask, config_.weak_threshold);
    }

    const float distance = config_.min_feature_distance > 0.0f
                             ? config_.min_feature_distance
                             : autoFeatureDistance(image.size(), config_.num_features);
    auto selected = greedySelectCandidates(candidates, config_.num_features, config_.min_features, distance);
    if (static_cast<int>(selected.size()) < config_.min_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Not enough gradient features to build a shape template");
    }

    auto &class_templates = templates_[class_id];
    const int template_id = static_cast<int>(class_templates.size());
    class_templates.push_back(makeTemplateInfo(class_id, template_id, selected, variant));
    return template_id;
}

/**
 * @brief 从文件读取训练图和可选掩膜后训练模板。
 */
int ShapeTemplateMatcher::Impl::addTemplateFile(const fs::path &image_file, const std::string &class_id,
                                                const fs::path &mask_file, ShapeTemplateVariant variant)
{
    const cv::Mat image = loadImage(image_file, cv::IMREAD_UNCHANGED, "template image");
    cv::Mat       mask;
    if (!mask_file.empty())
    {
        mask = loadImage(mask_file, cv::IMREAD_GRAYSCALE, "template mask");
    }
    return addTemplate(image, class_id, mask, variant);
}

/**
 * @brief 对训练图和掩膜逐个执行中心旋转/缩放并训练模板。
 */
std::vector<int> ShapeTemplateMatcher::Impl::addTemplateVariants(
    const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
    const std::vector<ShapeTemplateVariant> &variants)
{
    validateImage(image, "template image");
    if (variants.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variants must not be empty");
    }

    const cv::Mat mask = normalizeMask(object_mask, image.size(), "object_mask");
    std::vector<int> ids;
    ids.reserve(variants.size());

    const cv::Point2f center(static_cast<float>(image.cols) * 0.5f, static_cast<float>(image.rows) * 0.5f);
    for (const auto &variant : variants)
    {
        if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
        }

        const cv::Mat transform = cv::getRotationMatrix2D(center, variant.angle_degrees, variant.scale);
        cv::Mat       transformed_image;
        cv::Mat       transformed_mask;
        cv::warpAffine(image, transformed_image, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       cv::Scalar());
        cv::warpAffine(mask, transformed_mask, transform, mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
        ids.push_back(addTemplate(transformed_image, class_id, transformed_mask, variant));
    }
    return ids;
}

/**
 * @brief 在源图中滑窗计算模板方向一致性并应用同类别 NMS。
 */
std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::Impl::match(const cv::Mat &image, float threshold,
                                                                  const std::vector<std::string> &class_ids,
                                                                  const cv::Mat &search_mask) const
{
    validateImage(image, "source image");
    if (empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ShapeTemplateMatcher has no templates");
    }

    const float effective_threshold = threshold < 0.0f ? config_.match_threshold : threshold;
    validateScoreRange(effective_threshold, "ShapeTemplateMatcher match threshold");

    const cv::Mat mask = normalizeMask(search_mask, image.size(), "search_mask");
    const auto    gradient = computeQuantizedGradient(image, mask, config_.weak_threshold);
    const auto    response_maps = buildResponseMaps(gradient.labels, config_.max_label_difference);

    std::vector<ShapeTemplateMatch> matches;
    auto match_class = [&](const std::string &class_id, const std::vector<ShapeTemplateInfo> &templates)
    {
        for (const auto &templ : templates)
        {
            if (templ.width > image.cols || templ.height > image.rows || templ.features.empty())
            {
                continue;
            }

            const int max_y = image.rows - templ.height;
            const int max_x = image.cols - templ.width;
            for (int y = 0; y <= max_y; y += config_.scan_step)
            {
                for (int x = 0; x <= max_x; x += config_.scan_step)
                {
                    const int center_x = std::min(image.cols - 1, x + templ.width / 2);
                    const int center_y = std::min(image.rows - 1, y + templ.height / 2);
                    if (mask.ptr<unsigned char>(center_y)[center_x] == 0)
                    {
                        continue;
                    }

                    const float score = similarityAt(response_maps, templ, x, y);
                    if (score >= effective_threshold)
                    {
                        matches.push_back(ShapeTemplateMatch{x, y, templ.width, templ.height, score, class_id,
                                                             templ.template_id, templ.angle_degrees, templ.scale});
                    }
                }
            }
        }
    };

    if (class_ids.empty())
    {
        for (const auto &item : templates_)
        {
            match_class(item.first, item.second);
        }
    }
    else
    {
        for (const auto &class_id : class_ids)
        {
            const auto it = templates_.find(class_id);
            if (it != templates_.end())
            {
                match_class(it->first, it->second);
            }
        }
    }

    return applyNms(std::move(matches), config_.nms_threshold, config_.max_results);
}

/**
 * @brief 从文件读取源图和可选搜索掩膜后执行匹配。
 */
std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::Impl::matchFile(const fs::path &image_file, float threshold,
                                                                      const std::vector<std::string> &class_ids,
                                                                      const fs::path &mask_file) const
{
    const cv::Mat image = loadImage(image_file, cv::IMREAD_UNCHANGED, "source image");
    cv::Mat       mask;
    if (!mask_file.empty())
    {
        mask = loadImage(mask_file, cv::IMREAD_GRAYSCALE, "search mask");
    }
    return match(image, threshold, class_ids, mask);
}

void ShapeTemplateMatcher::Impl::clear()
{
    templates_.clear();
}

bool ShapeTemplateMatcher::Impl::empty() const noexcept
{
    return templates_.empty();
}

int ShapeTemplateMatcher::Impl::numClasses() const noexcept
{
    return static_cast<int>(templates_.size());
}

int ShapeTemplateMatcher::Impl::numTemplates() const noexcept
{
    int count = 0;
    for (const auto &item : templates_)
    {
        count += static_cast<int>(item.second.size());
    }
    return count;
}

int ShapeTemplateMatcher::Impl::numTemplates(const std::string &class_id) const noexcept
{
    const auto it = templates_.find(class_id);
    return it == templates_.end() ? 0 : static_cast<int>(it->second.size());
}

std::vector<std::string> ShapeTemplateMatcher::Impl::classIds() const
{
    std::vector<std::string> ids;
    ids.reserve(templates_.size());
    for (const auto &item : templates_)
    {
        ids.push_back(item.first);
    }
    return ids;
}

const ShapeTemplateInfo &ShapeTemplateMatcher::Impl::getTemplate(const std::string &class_id, int template_id) const
{
    const auto it = templates_.find(class_id);
    if (it == templates_.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown shape template class_id: %s",
                             class_id.c_str());
    }
    if (template_id < 0 || template_id >= static_cast<int>(it->second.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid shape template id: %d", template_id);
    }
    return it->second[static_cast<size_t>(template_id)];
}

const ShapeTemplateMatcherConfig &ShapeTemplateMatcher::Impl::config() const noexcept
{
    return config_;
}

/**
 * @brief 将配置、模板元数据和特征点写入 OpenCV FileStorage。
 *
 * 文件格式使用版本号保护，模板 ID 在加载时会按类别内顺序重新归一化，避免外部文件中的
 * 非连续 ID 破坏后续索引访问。
 */
void ShapeTemplateMatcher::Impl::save(const fs::path &template_file) const
{
    if (template_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file path must not be empty");
    }
    if (!template_file.parent_path().empty())
    {
        fs::create_directories(template_file.parent_path());
    }

    cv::FileStorage storage(template_file.string(), cv::FileStorage::WRITE);
    if (!storage.isOpened())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open shape template file for write: %s",
                             template_file.string().c_str());
    }

    storage << "version" << 1;
    storage << "config" << "{";
    storage << "num_features" << config_.num_features;
    storage << "min_features" << config_.min_features;
    storage << "weak_threshold" << config_.weak_threshold;
    storage << "strong_threshold" << config_.strong_threshold;
    storage << "max_label_difference" << config_.max_label_difference;
    storage << "match_threshold" << config_.match_threshold;
    storage << "nms_threshold" << config_.nms_threshold;
    storage << "max_results" << config_.max_results;
    storage << "scan_step" << config_.scan_step;
    storage << "min_feature_distance" << config_.min_feature_distance;
    storage << "}";

    storage << "templates" << "[";
    for (const auto &item : templates_)
    {
        for (const auto &templ : item.second)
        {
            storage << "{";
            storage << "class_id" << templ.class_id;
            storage << "template_id" << templ.template_id;
            storage << "width" << templ.width;
            storage << "height" << templ.height;
            storage << "tl_x" << templ.tl_x;
            storage << "tl_y" << templ.tl_y;
            storage << "angle_degrees" << templ.angle_degrees;
            storage << "scale" << templ.scale;
            storage << "features" << "[";
            for (const auto &feature : templ.features)
            {
                storage << "{";
                storage << "x" << feature.x;
                storage << "y" << feature.y;
                storage << "label" << feature.label;
                storage << "angle_degrees" << feature.angle_degrees;
                storage << "}";
            }
            storage << "]";
            storage << "}";
        }
    }
    storage << "]";
}

/**
 * @brief 从 OpenCV FileStorage 读取模板库并替换当前状态。
 *
 * 加载流程先在临时容器中校验完整性，全部成功后再替换 ``config_`` 和 ``templates_``，
 * 避免部分加载失败时留下半初始化状态。
 */
void ShapeTemplateMatcher::Impl::load(const fs::path &template_file)
{
    if (template_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file path must not be empty");
    }
    std::error_code ec;
    if (!fs::is_regular_file(template_file, ec))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file does not exist: %s",
                             template_file.string().c_str());
    }

    cv::FileStorage storage(template_file.string(), cv::FileStorage::READ);
    if (!storage.isOpened())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open shape template file for read: %s",
                             template_file.string().c_str());
    }

    int version = 0;
    storage["version"] >> version;
    if (version != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported shape template file version: %d",
                             version);
    }

    ShapeTemplateMatcherConfig loaded_config = config_;
    const cv::FileNode         config_node   = storage["config"];
    if (!config_node.empty())
    {
        readIfPresent(config_node, "num_features", loaded_config.num_features);
        readIfPresent(config_node, "min_features", loaded_config.min_features);
        readIfPresent(config_node, "weak_threshold", loaded_config.weak_threshold);
        readIfPresent(config_node, "strong_threshold", loaded_config.strong_threshold);
        readIfPresent(config_node, "max_label_difference", loaded_config.max_label_difference);
        readIfPresent(config_node, "match_threshold", loaded_config.match_threshold);
        readIfPresent(config_node, "nms_threshold", loaded_config.nms_threshold);
        readIfPresent(config_node, "max_results", loaded_config.max_results);
        readIfPresent(config_node, "scan_step", loaded_config.scan_step);
        readIfPresent(config_node, "min_feature_distance", loaded_config.min_feature_distance);
    }
    validateConfig(loaded_config);

    TemplateMap loaded_templates;
    const auto  templates_node = storage["templates"];
    if (!templates_node.empty() && !templates_node.isSeq())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file templates node must be a list");
    }

    for (const auto &node : templates_node)
    {
        ShapeTemplateInfo info;
        node["class_id"] >> info.class_id;
        readIfPresent(node, "template_id", info.template_id);
        node["width"] >> info.width;
        node["height"] >> info.height;
        node["tl_x"] >> info.tl_x;
        node["tl_y"] >> info.tl_y;
        node["angle_degrees"] >> info.angle_degrees;
        node["scale"] >> info.scale;

        const cv::FileNode features_node = node["features"];
        if (features_node.empty() || !features_node.isSeq())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template features must be a list");
        }

        for (const auto &feature_node : features_node)
        {
            ShapeTemplateFeature feature;
            feature_node["x"] >> feature.x;
            feature_node["y"] >> feature.y;
            feature_node["label"] >> feature.label;
            feature_node["angle_degrees"] >> feature.angle_degrees;
            info.features.push_back(feature);
        }

        auto &class_templates = loaded_templates[info.class_id];
        info.template_id      = static_cast<int>(class_templates.size());
        validateTemplateInfo(info, loaded_config.min_features);
        class_templates.push_back(std::move(info));
    }

    config_    = loaded_config;
    templates_ = std::move(loaded_templates);
}

} // namespace irt::features
