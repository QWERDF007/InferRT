/**
 * @file ShapeTemplateMatcherEngine.cpp
 * @brief 形状模板匹配公共流程核心的实现。
 */

#include "ShapeTemplateMatcherEngine.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

constexpr unsigned char kInvalidLabel = detail::kShapeTemplateInvalidLabel;     ///< 无效方向标签，用于跳过弱梯度点。
constexpr float         kEps          = 1.0e-6f;                                ///< 浮点区间比较容差。
constexpr int           kOrientationBins = detail::kShapeTemplateOrientationBins; ///< 梯度方向量化 bin 数量。
constexpr int           kShapeTemplateFileFormatVersion = 4; ///< 强制保存训练画布尺寸的紧凑二维特征数组模板文件格式版本。

using QuantizedGradient = detail::ShapeTemplateQuantizedGradient;
using Candidate          = detail::ShapeTemplateCandidate;

using ResponseMaps = detail::ShapeTemplateResponseMaps;

using ResponseTable = detail::ShapeTemplateResponseTable;

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
    if (config.max_parallelism < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_parallelism must be non-negative");
    }
    if (config.max_training_parallelism < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatcher max_training_parallelism must be non-negative");
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
 * @param gray8 可复用的 ``CV_8UC1`` 输出缓冲区。
 */
void toGray8(const cv::Mat &image, cv::Mat &gray8)
{
    validateImage(image, "image");

    if (image.channels() == 1)
    {
        if (image.depth() == CV_8U)
        {
            gray8 = image;
        }
        else
        {
            image.convertTo(gray8, CV_8U);
        }
        return;
    }

    int conversion_code = 0;
    if (image.channels() == 3)
    {
        conversion_code = cv::COLOR_BGR2GRAY;
    }
    else if (image.channels() == 4)
    {
        conversion_code = cv::COLOR_BGRA2GRAY;
    }
    else
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported image channel count: %d",
                             image.channels());
    }

    if (image.depth() == CV_8U)
    {
        cv::cvtColor(image, gray8, conversion_code);
    }
    else
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, conversion_code);
        gray.convertTo(gray8, CV_8U);
    }
}

/**
 * @brief 返回 8 位灰度图的便捷包装。
 * @param image 输入图像。
 * @return ``CV_8UC1`` 灰度图。
 */
cv::Mat toGray8(const cv::Mat &image)
{
    cv::Mat gray8;
    toGray8(image, gray8);
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
 * @brief 对量化方向执行可选的 3x3 多数滤波。
 *
 * 只有在显式启用近似预处理时调用；默认路径完全不进入该分支。滤波以原始标签副本为
 * 输入，避免扫描顺序影响邻域统计，并保留掩膜外及不足多数的像素。
 */
void applyOrientationHistogram(cv::Mat &labels, const cv::Mat &mask)
{
    cv::Mat filtered = labels.clone();
    for (int y = 1; y < labels.rows - 1; ++y)
    {
        const auto *mask_row = mask.ptr<unsigned char>(y);
        auto       *dst_row  = filtered.ptr<unsigned char>(y);
        for (int x = 1; x < labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || labels.ptr<unsigned char>(y)[x] >= kOrientationBins)
                continue;

            std::array<int, kOrientationBins> counts{};
            for (int dy = -1; dy <= 1; ++dy)
            {
                const auto *neighbor = labels.ptr<unsigned char>(y + dy);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int label = neighbor[x + dx];
                    if (label < kOrientationBins)
                        ++counts[static_cast<size_t>(label)];
                }
            }
            int best_label = labels.ptr<unsigned char>(y)[x];
            int best_count = counts[static_cast<size_t>(best_label)];
            for (int label = 0; label < kOrientationBins; ++label)
            {
                if (counts[static_cast<size_t>(label)] > best_count)
                {
                    best_label = label;
                    best_count = counts[static_cast<size_t>(label)];
                }
            }
            if (best_count >= 5)
                dst_row[x] = static_cast<unsigned char>(best_label);
        }
    }
    labels = std::move(filtered);
}

/** @brief 沿量化梯度方向执行 2 邻域非极大值抑制；仅用于显式近似配置。 */
void applyEdgeNonMaximumSuppression(cv::Mat &labels, const cv::Mat &magnitude, const cv::Mat &angle,
                                    const cv::Mat &mask)
{
    cv::Mat filtered = labels.clone();
    for (int y = 1; y < labels.rows - 1; ++y)
    {
        const auto *mask_row = mask.ptr<unsigned char>(y);
        const auto *mag_row  = magnitude.ptr<float>(y);
        const auto *ang_row  = angle.ptr<float>(y);
        auto       *dst_row  = filtered.ptr<unsigned char>(y);
        for (int x = 1; x < labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || labels.ptr<unsigned char>(y)[x] >= kOrientationBins)
                continue;

            const int bin = quantizeAngle(ang_row[x]) & 3;
            int       dx  = 0;
            int       dy  = 0;
            switch (bin)
            {
            case 0:
                dx = 1;
                break;
            case 1:
                dx = 1;
                dy = 1;
                break;
            case 2:
                dy = 1;
                break;
            default:
                dx = 1;
                dy = -1;
                break;
            }
            const float forward  = magnitude.at<float>(y + dy, x + dx);
            const float backward = magnitude.at<float>(y - dy, x - dx);
            if (mag_row[x] < forward || mag_row[x] < backward)
                dst_row[x] = kInvalidLabel;
        }
    }
    labels = std::move(filtered);
}

/** @brief 从强梯度种子出发保留连通的弱梯度边缘；仅用于显式近似配置。 */
void applyEdgeConnectivity(cv::Mat &labels, const cv::Mat &magnitude, const cv::Mat &mask, float strong_threshold)
{
    cv::Mat reachable(labels.size(), CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> frontier;
    frontier.reserve(static_cast<size_t>(labels.total() / 16U + 1U));
    for (int y = 1; y < labels.rows - 1; ++y)
    {
        const auto *mask_row = mask.ptr<unsigned char>(y);
        const auto *mag_row  = magnitude.ptr<float>(y);
        const auto *label_row = labels.ptr<unsigned char>(y);
        auto       *seen_row = reachable.ptr<unsigned char>(y);
        for (int x = 1; x < labels.cols - 1; ++x)
        {
            if (mask_row[x] != 0 && label_row[x] < kOrientationBins && mag_row[x] >= strong_threshold)
            {
                seen_row[x] = 1;
                frontier.emplace_back(x, y);
            }
        }
    }

    for (size_t index = 0; index < frontier.size(); ++index)
    {
        const cv::Point point = frontier[index];
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0)
                    continue;
                const int nx = point.x + dx;
                const int ny = point.y + dy;
                if (nx <= 0 || nx >= labels.cols - 1 || ny <= 0 || ny >= labels.rows - 1)
                    continue;
                if (reachable.at<unsigned char>(ny, nx) != 0 || mask.at<unsigned char>(ny, nx) == 0
                    || labels.at<unsigned char>(ny, nx) >= kOrientationBins)
                    continue;
                reachable.at<unsigned char>(ny, nx) = 1;
                frontier.emplace_back(nx, ny);
            }
        }
    }

    for (int y = 1; y < labels.rows - 1; ++y)
    {
        auto *label_row = labels.ptr<unsigned char>(y);
        const auto *seen_row = reachable.ptr<unsigned char>(y);
        for (int x = 1; x < labels.cols - 1; ++x)
        {
            if (label_row[x] < kOrientationBins && seen_row[x] == 0)
                label_row[x] = kInvalidLabel;
        }
    }
}

/** @brief 将邻域中已有方向标签扩散到孤立位置；仅用于显式近似配置。 */
void applySpatialSpread(cv::Mat &labels, const cv::Mat &mask)
{
    cv::Mat spread = labels.clone();
    for (int y = 1; y < labels.rows - 1; ++y)
    {
        const auto *mask_row = mask.ptr<unsigned char>(y);
        auto       *dst_row  = spread.ptr<unsigned char>(y);
        for (int x = 1; x < labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || labels.ptr<unsigned char>(y)[x] < kOrientationBins)
                continue;
            std::array<int, kOrientationBins> counts{};
            for (int dy = -1; dy <= 1; ++dy)
            {
                const auto *neighbor = labels.ptr<unsigned char>(y + dy);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int label = neighbor[x + dx];
                    if (label < kOrientationBins)
                        ++counts[static_cast<size_t>(label)];
                }
            }
            int best_label = 0;
            int best_count = 0;
            for (int label = 0; label < kOrientationBins; ++label)
            {
                if (counts[static_cast<size_t>(label)] > best_count)
                {
                    best_label = label;
                    best_count = counts[static_cast<size_t>(label)];
                }
            }
            if (best_count >= 2)
                dst_row[x] = static_cast<unsigned char>(best_label);
        }
    }
    labels = std::move(spread);
}

/** @brief 折叠相反方向极性；仅用于显式近似配置。 */
void applyPolarityInvariant(cv::Mat &labels)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        auto *row = labels.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            if (row[x] < kOrientationBins)
                row[x] = static_cast<unsigned char>(row[x] & 3);
        }
    }
}

void applyOptionalLabelPreprocessing(QuantizedGradient &gradient, const cv::Mat &mask,
                                     const ShapeTemplateMatcherConfig &config)
{
    if (config.use_edge_nms)
        applyEdgeNonMaximumSuppression(gradient.labels, gradient.magnitude, gradient.angle, mask);
    if (config.use_edge_connectivity)
        applyEdgeConnectivity(gradient.labels, gradient.magnitude, mask, config.strong_threshold);
    if (config.use_polarity_invariant)
        applyPolarityInvariant(gradient.labels);
    if (config.use_orientation_histogram)
        applyOrientationHistogram(gradient.labels, mask);
    if (config.use_spatial_spread)
        applySpatialSpread(gradient.labels, mask);
}

/**
 * @brief 计算图像梯度幅值、方向角和量化方向标签。
 * @param image 输入图像。
 * @param mask 有效区域掩膜。
 * @param config 梯度阈值和可选 v1 预处理配置。
 * @return 梯度量化结果。
 */
QuantizedGradient computeQuantizedGradient(const cv::Mat &image, const cv::Mat &mask,
                                           const ShapeTemplateMatcherConfig &config,
                                           const detail::ShapeTemplateMatcherKernel &kernel)
{
    const cv::Mat gray = toGray8(image);
    const cv::Mat mask8 = normalizeMask(mask, gray.size(), "mask");

    cv::Mat filtered_gray;
    const cv::Mat *gradient_input = &gray;
    if (config.use_gaussian_gradient)
    {
        cv::GaussianBlur(gray, filtered_gray, cv::Size(5, 5), 0.0, 0.0, cv::BORDER_DEFAULT);
        gradient_input = &filtered_gray;
    }

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(*gradient_input, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(*gradient_input, grad_y, CV_32F, 0, 1, 3);

    QuantizedGradient gradient;
    cv::cartToPolar(grad_x, grad_y, gradient.magnitude, gradient.angle, true);
    gradient.labels = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(kInvalidLabel));

    kernel.fillQuantizedLabels(gradient.magnitude, gradient.angle, mask8, config.weak_threshold, gradient.labels);
    applyOptionalLabelPreprocessing(gradient, mask8, config);
    return gradient;
}

/**
 * @brief 将候选点按梯度幅值和坐标稳定排序。
 */
bool candidateComesBefore(const Candidate &a, const Candidate &b) noexcept
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
}

void sortCandidates(std::vector<Candidate> &candidates)
{
    std::sort(candidates.begin(), candidates.end(), candidateComesBefore);
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
 * @brief 为 v1/v2 训练路径执行复用候选缓冲和严格等价的排序。
 *
 * 特征选择继续调用 v0 原有的贪心实现。每个工作线程跨旋转/缩放变体复用候选数组，避免
 * ``std::vector`` 在候选收集阶段反复分配、扩容和释放；候选比较器及模板特征顺序不变。
 */
std::vector<Candidate> greedySelectCandidatesOptimized(std::vector<Candidate> &candidates, int num_features,
                                                        int min_features, float min_distance)
{
    sortCandidates(candidates);

    if (candidates.empty())
        return {};

    // 将候选点放入边长等于最小距离的网格。只检查 3x3 邻域即可覆盖所有可能
    // 小于距离的点；候选遍历顺序、距离比较和逐步减小距离规则均与 v0 相同，
    // 因而这是严格等价的空间容差优化，而不是改变选点策略。
    const auto make_cell_key = [](int cell_x, int cell_y) noexcept
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell_x)) << 32U)
             | static_cast<std::uint32_t>(cell_y);
    };
    std::vector<Candidate> best;
    float distance = std::max(1.0f, min_distance);
    for (;;)
    {
        std::vector<Candidate> selected;
        selected.reserve(static_cast<size_t>(num_features));
        std::unordered_map<std::uint64_t, std::vector<size_t>> buckets;
        buckets.reserve(static_cast<size_t>(num_features) * 2U + 1U);
        const float min_distance_sq = distance * distance;

        for (const auto &candidate : candidates)
        {
            const int cell_x = static_cast<int>(std::floor(static_cast<float>(candidate.x) / distance));
            const int cell_y = static_cast<int>(std::floor(static_cast<float>(candidate.y) / distance));
            bool keep = true;
            for (int dy = -1; dy <= 1 && keep; ++dy)
            {
                for (int dx = -1; dx <= 1 && keep; ++dx)
                {
                    const auto it = buckets.find(make_cell_key(cell_x + dx, cell_y + dy));
                    if (it == buckets.end())
                        continue;
                    for (const size_t selected_index : it->second)
                    {
                        const auto &existing = selected[selected_index];
                        const float delta_x = static_cast<float>(candidate.x - existing.x);
                        const float delta_y = static_cast<float>(candidate.y - existing.y);
                        if (delta_x * delta_x + delta_y * delta_y < min_distance_sq)
                        {
                            keep = false;
                            break;
                        }
                    }
                }
            }
            if (!keep)
                continue;

            const size_t selected_index = selected.size();
            selected.push_back(candidate);
            buckets[make_cell_key(cell_x, cell_y)].push_back(selected_index);
            if (static_cast<int>(selected.size()) >= num_features)
                break;
        }

        if (selected.size() > best.size())
            best = std::move(selected);
        if (static_cast<int>(best.size()) >= min_features
            || distance <= 1.0f + std::numeric_limits<float>::epsilon())
            break;
        distance *= 0.75f;
        if (distance < 1.0f)
            distance = 1.0f;
    }
    if (static_cast<int>(best.size()) > num_features)
        best.resize(static_cast<size_t>(num_features));
    return best;
}

/** @brief 单个训练工作线程复用的图像与梯度缓冲区。 */
struct ShapeTemplateTrainingWorkspace
{
    cv::Mat transformed_image;
    cv::Mat transformed_mask;
    cv::Mat grad_x;
    cv::Mat grad_y;
    QuantizedGradient gradient;
    std::vector<Candidate> candidates; ///< 跨变体复用的候选缓冲，避免反复扩容和释放。
};

/**
 * @brief 基于已规范化掩膜计算模板训练所需的量化梯度，并复用工作缓冲区。
 *
 * 调用方保证 ``normalized_mask`` 为同尺寸的二值 ``CV_8UC1`` 图。训练变体的掩膜
 * 由最近邻仿射变换生成，跳过重复的 ``normalizeMask()`` 不改变任一像素值。
 * @param config 梯度阈值和可选 v1 预处理配置。
 */
void computeTemplateQuantizedGradient(const cv::Mat &image, const cv::Mat &normalized_mask,
                                      const ShapeTemplateMatcherConfig &config,
                                      const detail::ShapeTemplateMatcherKernel &kernel,
                                      ShapeTemplateTrainingWorkspace &workspace)
{
    const cv::Mat gray = toGray8(image);
    cv::Mat filtered_gray;
    const cv::Mat *gradient_input = &gray;
    if (config.use_gaussian_gradient)
    {
        cv::GaussianBlur(gray, filtered_gray, cv::Size(5, 5), 0.0, 0.0, cv::BORDER_DEFAULT);
        gradient_input = &filtered_gray;
    }
    cv::Sobel(*gradient_input, workspace.grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(*gradient_input, workspace.grad_y, CV_32F, 0, 1, 3);

    cv::cartToPolar(workspace.grad_x, workspace.grad_y, workspace.gradient.magnitude, workspace.gradient.angle,
                    true);
    workspace.gradient.labels.create(gray.size(), CV_8UC1);
    workspace.gradient.labels.setTo(cv::Scalar(kInvalidLabel));
    kernel.fillQuantizedLabels(workspace.gradient.magnitude, workspace.gradient.angle, normalized_mask,
                               config.weak_threshold, workspace.gradient.labels);
    applyOptionalLabelPreprocessing(workspace.gradient, normalized_mask, config);
}

/**
 * @brief 提取一个已完成仿射变换的模板变体的特征点。
 *
 * 此函数无引擎状态写入，因而多个变体可并行调用；候选排序和特征选择规则与
 * ``addTemplate()`` 的原实现完全相同。
 */
std::vector<Candidate> extractTemplateCandidates(const cv::Mat &image, const cv::Mat &normalized_mask,
                                                 const ShapeTemplateMatcherConfig &config,
                                                 const detail::ShapeTemplateMatcherKernel &kernel,
                                                 ShapeTemplateTrainingWorkspace &workspace)
{
    computeTemplateQuantizedGradient(image, normalized_mask, config, kernel, workspace);
    kernel.collectCandidatesUnsorted(workspace.gradient, normalized_mask, config.strong_threshold,
                                     workspace.candidates);
    if (static_cast<int>(workspace.candidates.size()) < config.min_features
        && config.weak_threshold < config.strong_threshold)
    {
        kernel.collectCandidatesUnsorted(workspace.gradient, normalized_mask, config.weak_threshold,
                                         workspace.candidates);
    }

    const float distance = config.min_feature_distance > 0.0f
                             ? config.min_feature_distance
                             : autoFeatureDistance(image.size(), config.num_features);
    auto selected = kernel.usesIndexedFeatureSelection()
                      ? greedySelectCandidatesOptimized(workspace.candidates, config.num_features,
                                                        config.min_features, distance)
                      : greedySelectCandidates(workspace.candidates, config.num_features, config.min_features,
                                                distance);
    if (static_cast<int>(selected.size()) < config.min_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Not enough gradient features to build a shape template");
    }
    return selected;
}

/**
 * @brief 根据选中特征点构造可持久化的模板信息。
 * @param template_id 模板文件内的全局模板 ID。
 * @param features 已选中的训练特征点。
 * @param variant 旋转/缩放变体元数据。
 * @return 裁剪到特征包围盒后的模板信息。
 */
ShapeTemplateInfo makeTemplateInfo(int template_id, const std::vector<Candidate> &features,
                                   ShapeTemplateVariant variant, cv::Size template_size)
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
    info.template_id   = template_id;
    info.width         = max_x - min_x + 1;
    info.height        = max_y - min_y + 1;
    info.template_width  = template_size.width;
    info.template_height = template_size.height;
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
int labelDistance(int a, int b, bool polarity_invariant) noexcept
{
    if (polarity_invariant)
    {
        constexpr int kPolarityBins = kOrientationBins / 2;
        a &= (kPolarityBins - 1);
        b &= (kPolarityBins - 1);
        const int diff = std::abs(a - b);
        return std::min(diff, kPolarityBins - diff);
    }
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
ResponseTable makeResponseTable(int max_label_difference, bool polarity_invariant)
{
    ResponseTable table{};
    for (int templ_label = 0; templ_label < kOrientationBins; ++templ_label)
    {
        for (int source_label = 0; source_label < kOrientationBins; ++source_label)
        {
            const int diff = labelDistance(templ_label, source_label, polarity_invariant);
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
 * @brief 为源图方向标签预计算 8 张响应图。
 * @param labels 源图方向标签图。
 * @param max_label_difference 最大允许方向标签差。
 * @return 方向响应图集合。
 */
ResponseMaps buildResponseMaps(const cv::Mat &labels, int max_label_difference, bool polarity_invariant,
                               const detail::ShapeTemplateMatcherKernel &kernel,
                               const std::array<bool, kOrientationBins> &required_labels)
{
    ResponseMaps response_maps;
    response_maps.denominator_per_feature = responseDenominatorPerFeature(max_label_difference);
    const auto table = makeResponseTable(max_label_difference, polarity_invariant);
    response_maps.quantized_labels = labels;
    response_maps.response_table = table;
    for (int template_label = 0; template_label < kOrientationBins; ++template_label)
    {
        for (int source_label = 0; source_label < kOrientationBins; ++source_label)
        {
            response_maps.response_lookup128[static_cast<size_t>(template_label)][static_cast<size_t>(source_label)]
                = table[static_cast<size_t>(template_label)][static_cast<size_t>(source_label)];
        }
    }
    if (!kernel.needsMaterializedResponseMaps())
    {
        std::array<std::uint64_t, kOrientationBins> label_counts{};
        for (int y = 0; y < labels.rows; ++y)
        {
            const auto *row = labels.ptr<unsigned char>(y);
            for (int x = 0; x < labels.cols; ++x)
            {
                if (row[x] < kOrientationBins)
                    ++label_counts[static_cast<size_t>(row[x])];
            }
        }
        const auto pixel_count = static_cast<float>(labels.total());
        for (int template_label = 0; template_label < kOrientationBins; ++template_label)
        {
            std::uint64_t response_sum = 0;
            for (int source_label = 0; source_label < kOrientationBins; ++source_label)
            {
                response_sum += label_counts[static_cast<size_t>(source_label)]
                              * table[static_cast<size_t>(template_label)][static_cast<size_t>(source_label)];
            }
            response_maps.average_responses[static_cast<size_t>(template_label)]
                = pixel_count > 0.0f ? static_cast<float>(response_sum) / pixel_count : 0.0f;
        }
        return response_maps;
    }
    for (int label = 0; label < kOrientationBins; ++label)
    {
        if (!required_labels[static_cast<size_t>(label)])
            continue;
        response_maps.maps[label].create(labels.size(), CV_8UC1);
        kernel.fillResponseMap(labels, response_maps.maps[label], label, table);
        response_maps.average_responses[static_cast<size_t>(label)]
            = static_cast<float>(cv::mean(response_maps.maps[label])[0]);
    }
    return response_maps;
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
 * @brief 按相似度、模板 ID 和坐标稳定排序匹配结果。
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
 * @brief 对匹配结果执行 NMS 和最大数量限制。
 * @param matches 待处理匹配结果。
 * @param nms_threshold NMS IoU 阈值；小于 0 表示关闭 NMS。
 * @param max_results 最大返回数量；0 表示不限制。
 * @return 过滤后的匹配结果。
 */
int floorCellCoordinate(int value, int cell_size) noexcept
{
    if (value >= 0)
        return value / cell_size;
    return -static_cast<int>((static_cast<std::int64_t>(-value) + cell_size - 1) / cell_size);
}

std::uint64_t makeNmsCellKey(int cell_x, int cell_y) noexcept
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell_x)) << 32U)
         | static_cast<std::uint32_t>(cell_y);
}

std::vector<ShapeTemplateMatch> applyNms(std::vector<ShapeTemplateMatch> matches, float nms_threshold,
                                         int max_results, bool use_spatial_index)
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
    if (!use_spatial_index)
    {
        for (const auto &match : matches)
        {
            bool keep = true;
            for (const auto &existing : kept)
            {
                if (intersectionOverUnion(match, existing) > nms_threshold)
                {
                    keep = false;
                    break;
                }
            }
            if (keep)
            {
                kept.push_back(match);
                if (max_results > 0 && static_cast<int>(kept.size()) >= max_results)
                    break;
            }
        }
        return kept;
    }

    int cell_size = 1;
    for (const auto &match : matches)
        cell_size = std::max(cell_size, std::max(match.width, match.height));

    std::unordered_map<std::uint64_t, std::vector<size_t>> cells;
    cells.reserve(matches.size() * 2U + 1U);
    for (const auto &match : matches)
    {
        bool keep = true;
        const int min_cell_x = floorCellCoordinate(match.x, cell_size);
        const int max_cell_x = floorCellCoordinate(match.x + std::max(0, match.width - 1), cell_size);
        const int min_cell_y = floorCellCoordinate(match.y, cell_size);
        const int max_cell_y = floorCellCoordinate(match.y + std::max(0, match.height - 1), cell_size);
        for (int cell_y = min_cell_y; cell_y <= max_cell_y && keep; ++cell_y)
        {
            for (int cell_x = min_cell_x; cell_x <= max_cell_x && keep; ++cell_x)
            {
                const auto it = cells.find(makeNmsCellKey(cell_x, cell_y));
                if (it == cells.end())
                    continue;
                for (const size_t kept_index : it->second)
                {
                    if (intersectionOverUnion(match, kept[kept_index]) > nms_threshold)
                    {
                        keep = false;
                        break;
                    }
                }
            }
        }
        if (!keep)
            continue;

        const size_t kept_index = kept.size();
        kept.push_back(match);
        for (int cell_y = min_cell_y; cell_y <= max_cell_y; ++cell_y)
        {
            for (int cell_x = min_cell_x; cell_x <= max_cell_x; ++cell_x)
                cells[makeNmsCellKey(cell_x, cell_y)].push_back(kept_index);
        }
        if (max_results > 0 && static_cast<int>(kept.size()) >= max_results)
            break;
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
    if (info.width <= 0 || info.height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid size");
    }
    if (static_cast<int>(info.features.size()) < min_features)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has too few features");
    }
    if (info.template_width <= 0 || info.template_height <= 0 || info.template_width < info.width
        || info.template_height < info.height)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Loaded template has invalid canvas size");
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
 * @brief 从 YAML 映射读取可选字段。
 * @tparam T 字段类型。
 * @param node 父节点。
 * @param key 字段名称。
 * @param value 字段存在时写入的目标变量。
 */
template<typename T>
void readYamlIfPresent(const YAML::Node &node, const char *key, T &value)
{
    const YAML::Node child = node[key];
    if (child && !child.IsNull())
    {
        value = child.as<T>();
    }
}

/**
 * @brief 从 YAML 映射读取必填字段。
 * @tparam T 字段类型。
 * @param node 父节点。
 * @param key 字段名称。
 * @return 解析后的字段值。
 */
template<typename T>
T readRequiredYamlValue(const YAML::Node &node, const char *key)
{
    const YAML::Node child = node[key];
    if (!child || child.IsNull())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template YAML is missing required field: %s",
                             key);
    }
    return child.as<T>();
}

} // namespace

int detail::quantizeShapeTemplateAngle(float angle_degrees) noexcept
{
    return quantizeAngle(angle_degrees);
}

void detail::sortShapeTemplateCandidates(std::vector<ShapeTemplateCandidate> &candidates)
{
    sortCandidates(candidates);
}

/**
 * @brief 构造公共流程核心并注入版本私有热点内核。
 */
detail::ShapeTemplateMatcherEngine::ShapeTemplateMatcherEngine(
    ShapeTemplateMatcherConfig config, std::unique_ptr<ShapeTemplateMatcherKernel> kernel)
    : config_(std::move(config)), kernel_(std::move(kernel))
{
    validateConfig(config_);
    if (!kernel_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ShapeTemplateMatcher kernel must not be null");
    }
    if ((config_.use_gaussian_gradient || config_.use_orientation_histogram || config_.use_edge_nms
         || config_.use_edge_connectivity || config_.use_polarity_invariant || config_.use_spatial_spread
         || config_.reuse_base_features_for_variants)
        && !kernel_->supportsApproximatePreprocessing())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Approximate v1 preprocessing/variant-reuse options are supported by v1 only");
    }
}

/** @brief 校验仅影响本次调用的近似匹配策略。 */
void validateMatchOptions(const ShapeTemplateMatchOptions &options)
{
    if (options.template_stride <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatchOptions template_stride must be positive");
    }
    if (options.scan_step < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatchOptions scan_step must be non-negative");
    }
    if (options.max_parallelism < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ShapeTemplateMatchOptions max_parallelism must be non-negative");
    }
}

/**
 * @brief 将内部的特征包围盒命中转换为训练画布上的输出框。
 *
 * 模板保存了原始训练画布尺寸，因此输出框与 shapeMatchV2 使用的 ROI 语义一致。
 */
ShapeTemplateMatch makeShapeTemplateMatch(const ShapeTemplateInfo &templ,
                                          const detail::ShapeTemplateScoredPosition &position)
{
    const int output_width  = std::max(1, static_cast<int>(std::lround(templ.template_width * templ.scale)));
    const int output_height = std::max(1, static_cast<int>(std::lround(templ.template_height * templ.scale)));
    const int output_x = position.x - templ.tl_x
                       + static_cast<int>(std::lround((templ.template_width - output_width) * 0.5));
    const int output_y = position.y - templ.tl_y
                       + static_cast<int>(std::lround((templ.template_height - output_height) * 0.5));
    return ShapeTemplateMatch{output_x, output_y, output_width, output_height, position.score, templ.template_id,
                              templ.angle_degrees, templ.scale};
}

detail::ShapeTemplateMatcherEngine::~ShapeTemplateMatcherEngine() = default;
detail::ShapeTemplateMatcherEngine::ShapeTemplateMatcherEngine(ShapeTemplateMatcherEngine &&) noexcept = default;
detail::ShapeTemplateMatcherEngine &detail::ShapeTemplateMatcherEngine::operator=(ShapeTemplateMatcherEngine &&) noexcept = default;

/**
 * @brief 提取模板图中的强梯度方向特征并写入模板库。
 */
int detail::ShapeTemplateMatcherEngine::addTemplate(const cv::Mat &image, const cv::Mat &object_mask,
                                                     ShapeTemplateVariant variant)
{
    validateImage(image, "template image");
    if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
    }

    std::vector<Candidate> selected;
    if (kernel_->usesOptimizedTemplateTraining())
    {
        const cv::Mat mask = normalizeMask(object_mask, image.size(), "object_mask");
        ShapeTemplateTrainingWorkspace workspace;
        selected = extractTemplateCandidates(image, mask, config_, *kernel_, workspace);
    }
    else
    {
        // v0 保持原始标量训练流程：保留掩膜的完整规范化、梯度提取与候选选择顺序。
        const cv::Mat mask = normalizeMask(object_mask, image.size(), "object_mask");
        const auto gradient = computeQuantizedGradient(image, mask, config_, *kernel_);
        auto candidates = kernel_->collectCandidates(gradient, mask, config_.strong_threshold);
        if (static_cast<int>(candidates.size()) < config_.min_features
            && config_.weak_threshold < config_.strong_threshold)
        {
            candidates = kernel_->collectCandidates(gradient, mask, config_.weak_threshold);
        }

        const float distance = config_.min_feature_distance > 0.0f
                                 ? config_.min_feature_distance
                                 : autoFeatureDistance(image.size(), config_.num_features);
        selected = greedySelectCandidates(candidates, config_.num_features, config_.min_features, distance);
        if (static_cast<int>(selected.size()) < config_.min_features)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Not enough gradient features to build a shape template");
        }
    }

    const int template_id = static_cast<int>(templates_.size());
    templates_.push_back(makeTemplateInfo(template_id, selected, variant, image.size()));
    return template_id;
}

/**
 * @brief 从文件读取训练图和可选掩膜后训练模板。
 */
int detail::ShapeTemplateMatcherEngine::addTemplateFile(const fs::path &image_file, const fs::path &mask_file,
                                                         ShapeTemplateVariant variant)
{
    const cv::Mat image = loadImage(image_file, cv::IMREAD_UNCHANGED, "template image");
    cv::Mat       mask;
    if (!mask_file.empty())
    {
        mask = loadImage(mask_file, cv::IMREAD_GRAYSCALE, "template mask");
    }
    return addTemplate(image, mask, variant);
}

/**
 * @brief 对一个训练输入执行角度/尺度变体训练。
 *
 * @details v0 保持原始的逐变体串行路径；v1/v2 则转发到全局批量调度器，以便单输入和多输入
 * 使用完全相同的优化训练语义。
 */
std::vector<int> detail::ShapeTemplateMatcherEngine::addTemplateVariants(
    const cv::Mat &image, const cv::Mat &object_mask, const std::vector<ShapeTemplateVariant> &variants)
{
    validateImage(image, "template image");
    if (variants.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variants must not be empty");
    }

    if (!kernel_->usesOptimizedTemplateTraining())
    {
        // v0 保持与原始实现一致的串行训练过程，不复用缓冲区且逐变体立即提交模板。
        const cv::Mat mask = normalizeMask(object_mask, image.size(), "object_mask");
        std::vector<int> ids;
        ids.reserve(variants.size());

        const cv::Point2f center(static_cast<float>(image.cols) * 0.5f,
                                 static_cast<float>(image.rows) * 0.5f);
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
            ids.push_back(addTemplate(transformed_image, transformed_mask, variant));
        }
        return ids;
    }

    ShapeTemplateTrainingInput input{image, object_mask};
    auto ids = addTemplateVariantsBatch({input}, variants);
    return std::move(ids.front());
}

/**
 * @brief 统一调度多个训练输入及其共同的角度/尺度变体。
 *
 * @details v1/v2 先构建 ``输入 × 变体`` 的全局任务队列。每个工作线程跨 ROI 复用仿射、梯度和
 * 候选缓冲区；并行阶段不写 ``templates_``，所有结果会在结束后按输入顺序、再按变体顺序提交。
 * 因而与逐输入调用 ``addTemplateVariants()`` 相比，模板 ID、YAML 顺序和匹配结果严格一致。
 */
std::vector<std::vector<int>> detail::ShapeTemplateMatcherEngine::addTemplateVariantsBatch(
    const std::vector<ShapeTemplateTrainingInput> &inputs,
    const std::vector<ShapeTemplateVariant> &variants)
{
    if (inputs.empty())
    {
        return {};
    }
    if (variants.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variants must not be empty");
    }

    if (!kernel_->usesOptimizedTemplateTraining())
    {
        // v0 不改变参考实现：按输入顺序分别执行原有的串行变体训练。
        std::vector<std::vector<int>> ids;
        ids.reserve(inputs.size());
        for (const auto &input : inputs)
        {
            ids.push_back(addTemplateVariants(input.image, input.object_mask, variants));
        }
        return ids;
    }

    for (const auto &variant : variants)
    {
        if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
        }
    }

    struct PreparedTemplate
    {
        std::vector<Candidate> selected_features;
    };
    struct PreparedInput
    {
        const ShapeTemplateTrainingInput *input{nullptr};
        cv::Mat                            mask;
        cv::Point2f                        center;
        std::vector<Candidate>             base_features;
        std::vector<PreparedTemplate>      templates;
    };
    struct TrainingTask
    {
        size_t input_index{0};
        size_t variant_index{0};
    };

    std::vector<PreparedInput> prepared_inputs;
    prepared_inputs.reserve(inputs.size());
    std::vector<TrainingTask> tasks;
    tasks.reserve(inputs.size() * variants.size());
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index)
    {
        const auto &input = inputs[input_index];
        validateImage(input.image, "template image");
        PreparedInput prepared;
        prepared.input = &input;
        prepared.mask = normalizeMask(input.object_mask, input.image.size(), "object_mask");
        prepared.center = cv::Point2f(static_cast<float>(input.image.cols) * 0.5f,
                                      static_cast<float>(input.image.rows) * 0.5f);
        if (config_.reuse_base_features_for_variants)
        {
            // 近似模式只对每个 ROI 提取一次梯度特征；后续变体仅变换坐标和方向标签。
            // 这会忽略缩放/旋转后的插值梯度变化，因此显式配置且默认关闭。
            ShapeTemplateTrainingWorkspace base_workspace;
            prepared.base_features = extractTemplateCandidates(input.image, prepared.mask, config_, *kernel_,
                                                               base_workspace);
        }
        prepared.templates.resize(variants.size());
        prepared_inputs.push_back(std::move(prepared));

        for (size_t variant_index = 0; variant_index < variants.size(); ++variant_index)
        {
            tasks.push_back(TrainingTask{input_index, variant_index});
        }
    }

    std::vector<std::exception_ptr> errors(tasks.size());
    // 每项任务只写入自身的 prepared_templates 和 errors 元素，因此 worker 之间无共享写入。
    auto prepare_task = [&](size_t task_index, ShapeTemplateTrainingWorkspace &workspace)
    {
        const auto task = tasks[task_index];
        auto &prepared_input = prepared_inputs[task.input_index];
        const auto &input = *prepared_input.input;
        const auto &variant = variants[task.variant_index];
        const cv::Mat transform = cv::getRotationMatrix2D(prepared_input.center, variant.angle_degrees, variant.scale);
        if (config_.reuse_base_features_for_variants)
        {
            cv::warpAffine(prepared_input.mask, workspace.transformed_mask, transform,
                           prepared_input.mask.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            auto &selected = prepared_input.templates[task.variant_index].selected_features;
            selected.clear();
            selected.reserve(prepared_input.base_features.size());
            for (const auto &base : prepared_input.base_features)
            {
                const float transformed_x = static_cast<float>(transform.at<double>(0, 0)) * base.x
                                          + static_cast<float>(transform.at<double>(0, 1)) * base.y
                                          + static_cast<float>(transform.at<double>(0, 2));
                const float transformed_y = static_cast<float>(transform.at<double>(1, 0)) * base.x
                                          + static_cast<float>(transform.at<double>(1, 1)) * base.y
                                          + static_cast<float>(transform.at<double>(1, 2));
                const int x = cvRound(transformed_x);
                const int y = cvRound(transformed_y);
                if (x <= 0 || x >= input.image.cols - 1 || y <= 0 || y >= input.image.rows - 1
                    || workspace.transformed_mask.at<unsigned char>(y, x) == 0)
                    continue;

                bool duplicate = false;
                for (const auto &existing : selected)
                {
                    if (existing.x == x && existing.y == y)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate)
                    continue;

                Candidate transformed = base;
                transformed.x = x;
                transformed.y = y;
                transformed.angle_degrees = base.angle_degrees + variant.angle_degrees;
                transformed.label = quantizeAngle(transformed.angle_degrees);
                if (config_.use_polarity_invariant)
                    transformed.label &= 3;
                selected.push_back(transformed);
            }
            if (static_cast<int>(selected.size()) < config_.min_features)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Not enough transformed features to build a reused shape template");
            }
            return;
        }
        cv::warpAffine(input.image, workspace.transformed_image, transform, input.image.size(), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar());
        cv::warpAffine(prepared_input.mask, workspace.transformed_mask, transform, prepared_input.mask.size(),
                       cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        prepared_input.templates[task.variant_index].selected_features = extractTemplateCandidates(
            workspace.transformed_image, workspace.transformed_mask, config_, *kernel_, workspace);
    };

    unsigned int worker_count = config_.max_training_parallelism > 0
                                  ? static_cast<unsigned int>(config_.max_training_parallelism)
                                  : std::thread::hardware_concurrency();
    worker_count = std::max(1U, worker_count);
    worker_count = std::min(worker_count, static_cast<unsigned int>(tasks.size()));

    if (worker_count == 1)
    {
        ShapeTemplateTrainingWorkspace workspace;
        for (size_t task_index = 0; task_index < tasks.size(); ++task_index)
        {
            try
            {
                prepare_task(task_index, workspace);
            }
            catch (...)
            {
                errors[task_index] = std::current_exception();
                break;
            }
        }
    }
    else
    {
        std::atomic<size_t> next_task{0};
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (unsigned int worker = 0; worker < worker_count; ++worker)
        {
            workers.emplace_back([&]
            {
                ShapeTemplateTrainingWorkspace workspace;
                for (;;)
                {
                    const size_t task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size())
                        break;
                    try
                    {
                        prepare_task(task_index, workspace);
                    }
                    catch (...)
                    {
                        errors[task_index] = std::current_exception();
                    }
                }
            });
        }
        for (auto &worker : workers)
            worker.join();
    }

    for (const auto &error : errors)
    {
        if (error)
            std::rethrow_exception(error);
    }

    std::vector<std::vector<int>> ids(inputs.size());
    for (size_t input_index = 0; input_index < prepared_inputs.size(); ++input_index)
    {
        const auto &prepared_input = prepared_inputs[input_index];
        const int first_template_id = static_cast<int>(templates_.size());
        templates_.reserve(templates_.size() + prepared_input.templates.size());
        auto &input_ids = ids[input_index];
        input_ids.reserve(variants.size());
        for (size_t variant_index = 0; variant_index < variants.size(); ++variant_index)
        {
            const int template_id = first_template_id + static_cast<int>(variant_index);
            templates_.push_back(makeTemplateInfo(template_id,
                                                  prepared_input.templates[variant_index].selected_features,
                                                  variants[variant_index], prepared_input.input->image.size()));
            input_ids.push_back(template_id);
        }
    }
    return ids;
}

/**
 * @brief 在源图中滑窗计算模板方向一致性并应用 NMS。
 */
std::vector<ShapeTemplateMatch> detail::ShapeTemplateMatcherEngine::match(
    const cv::Mat &image, float threshold, const cv::Mat &search_mask, ShapeTemplateMatchOptions options) const
{
    validateImage(image, "source image");
    if (empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ShapeTemplateMatcher has no templates");
    }

    const float effective_threshold = threshold < 0.0f ? config_.match_threshold : threshold;
    validateScoreRange(effective_threshold, "ShapeTemplateMatcher match threshold");
    validateMatchOptions(options);
    const int effective_scan_step = options.scan_step > 0 ? options.scan_step : config_.scan_step;

    const cv::Mat mask = normalizeMask(search_mask, image.size(), "search_mask");
    // 空掩膜是常见的全图匹配入口。显式传入的全非零掩膜也等价于全图匹配，后端可跳过
    // 每个候选位置的掩膜读取和判断；这不会改变任一候选的评分或扫描范围。
    const bool full_search_mask = search_mask.empty()
                               || static_cast<size_t>(cv::countNonZero(mask)) == mask.total();

    struct MatchWork
    {
        const ShapeTemplateInfo *templ;
    };
    std::vector<MatchWork> work_items;
    work_items.reserve(templates_.size());
    for (const auto &templ : templates_)
    {
        if (templ.template_id % options.template_stride == 0)
            work_items.push_back(MatchWork{&templ});
    }
    if (work_items.empty())
    {
        return {};
    }

    // 仅物化实际会扫描的方向响应图。stride=1 时集合与原实现完全相同；近似模式还能避免
    // 为已跳过的模板变体做无用工作。
    std::array<bool, kOrientationBins> required_labels{};
    for (const auto &work : work_items)
    {
        for (const auto &feature : work.templ->features)
            required_labels[static_cast<size_t>(feature.label)] = true;
    }
    const auto gradient = computeQuantizedGradient(image, mask, config_, *kernel_);
    const auto response_maps = buildResponseMaps(gradient.labels, config_.max_label_difference,
                                                 config_.use_polarity_invariant, *kernel_, required_labels);

    auto scan_template = [&](const MatchWork &work, detail::ShapeTemplateScanWorkspace &workspace,
        std::vector<ShapeTemplateMatch> &local_matches)
    {
        const auto &templ = *work.templ;
        std::vector<detail::ShapeTemplateScoredPosition> owned_positions;
        const std::vector<detail::ShapeTemplateScoredPosition> *scored_positions = nullptr;
        if (kernel_->usesReusableScanWorkspace())
        {
            kernel_->scanTemplate(response_maps, templ, mask, full_search_mask, image.size(), effective_scan_step,
                                  effective_threshold, workspace);
            scored_positions = &workspace.scored_positions;
        }
        else
        {
            owned_positions = kernel_->scanTemplate(response_maps, templ, mask, full_search_mask, image.size(),
                                                    effective_scan_step, effective_threshold);
            scored_positions = &owned_positions;
        }
        for (const auto &position : *scored_positions)
        {
            local_matches.push_back(makeShapeTemplateMatch(templ, position));
        }
    };

    const int requested_parallelism = options.max_parallelism > 0 ? options.max_parallelism : config_.max_parallelism;
    unsigned int worker_count = requested_parallelism > 0
                                  ? static_cast<unsigned int>(requested_parallelism)
                                  : std::thread::hardware_concurrency();
    worker_count = std::max(1U, worker_count);
    worker_count = std::min(worker_count, static_cast<unsigned int>(work_items.size()));

    std::vector<ShapeTemplateMatch> matches;
    if (worker_count <= 1)
    {
        if (kernel_->usesReusableScanWorkspace())
        {
            detail::ShapeTemplateScanWorkspace workspace;
            std::vector<ShapeTemplateMatch> local_matches;
            for (const auto &work : work_items)
                scan_template(work, workspace, local_matches);
            matches = std::move(local_matches);
        }
        else
        {
            // v0/v2 保持原始的“每个模板返回一个 vector、随后移动合并”调度方式。
            for (const auto &work : work_items)
            {
                std::vector<ShapeTemplateMatch> local_matches;
                const auto scored_positions = kernel_->scanTemplate(
                    response_maps, *work.templ, mask, full_search_mask, image.size(), effective_scan_step,
                    effective_threshold);
                local_matches.reserve(scored_positions.size());
                for (const auto &position : scored_positions)
                    local_matches.push_back(makeShapeTemplateMatch(*work.templ, position));
                matches.insert(matches.end(), std::make_move_iterator(local_matches.begin()),
                               std::make_move_iterator(local_matches.end()));
            }
        }
    }
    else
    {
        std::atomic<size_t> next_work{0};
        std::vector<std::vector<ShapeTemplateMatch>> partial_matches(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (unsigned int worker = 0; worker < worker_count; ++worker)
        {
            workers.emplace_back([&, worker]
            {
                auto &local_matches = partial_matches[worker];
                detail::ShapeTemplateScanWorkspace workspace;
                for (;;)
                {
                    const size_t index = next_work.fetch_add(1, std::memory_order_relaxed);
                    if (index >= work_items.size())
                        break;
                    scan_template(work_items[index], workspace, local_matches);
                }
            });
        }
        for (auto &worker : workers)
            worker.join();
        for (auto &local_matches : partial_matches)
            matches.insert(matches.end(), std::make_move_iterator(local_matches.begin()),
                           std::make_move_iterator(local_matches.end()));
    }
    return applyNms(std::move(matches), config_.nms_threshold, config_.max_results, kernel_->usesIndexedNms());
}

/**
 * @brief 从文件读取源图和可选搜索掩膜后执行匹配。
 */
std::vector<ShapeTemplateMatch> detail::ShapeTemplateMatcherEngine::matchFile(
    const fs::path &image_file, float threshold, const fs::path &mask_file, ShapeTemplateMatchOptions options) const
{
    const cv::Mat image = loadImage(image_file, cv::IMREAD_UNCHANGED, "source image");
    cv::Mat       mask;
    if (!mask_file.empty())
    {
        mask = loadImage(mask_file, cv::IMREAD_GRAYSCALE, "search mask");
    }
    return match(image, threshold, mask, options);
}

void detail::ShapeTemplateMatcherEngine::clear()
{
    templates_.clear();
}

bool detail::ShapeTemplateMatcherEngine::empty() const noexcept
{
    return templates_.empty();
}

int detail::ShapeTemplateMatcherEngine::numTemplates() const noexcept
{
    return static_cast<int>(templates_.size());
}

const ShapeTemplateInfo &detail::ShapeTemplateMatcherEngine::getTemplate(int template_id) const
{
    if (template_id < 0 || template_id >= static_cast<int>(templates_.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid shape template id: %d", template_id);
    }
    return templates_[static_cast<size_t>(template_id)];
}

const ShapeTemplateMatcherConfig &detail::ShapeTemplateMatcherEngine::config() const noexcept
{
    return config_;
}

/**
 * @brief 将配置、模板元数据和特征点写入标准 YAML。
 *
 * 使用 yaml-cpp 发射器输出。v4 将每个特征存为 flow 风格的
 * ``[x, y, label, angle_degrees]``，避免对每个特征重复写入字段名；模板 ID 在加载时会按
 * 文件内顺序重新归一化，避免外部文件中的非连续 ID 破坏后续索引访问。
 */
void detail::ShapeTemplateMatcherEngine::save(const fs::path &template_file) const
{
    if (template_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template file path must not be empty");
    }
    if (!template_file.parent_path().empty())
    {
        fs::create_directories(template_file.parent_path());
    }

    std::ofstream output(template_file);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open shape template file for write: %s",
                             template_file.string().c_str());
    }

    YAML::Emitter emitter;
    emitter.SetFloatPrecision(std::numeric_limits<float>::max_digits10);
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "version" << YAML::Value << kShapeTemplateFileFormatVersion;
    emitter << YAML::Key << "config" << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "num_features" << YAML::Value << config_.num_features;
    emitter << YAML::Key << "min_features" << YAML::Value << config_.min_features;
    emitter << YAML::Key << "weak_threshold" << YAML::Value << config_.weak_threshold;
    emitter << YAML::Key << "strong_threshold" << YAML::Value << config_.strong_threshold;
    emitter << YAML::Key << "max_label_difference" << YAML::Value << config_.max_label_difference;
    emitter << YAML::Key << "match_threshold" << YAML::Value << config_.match_threshold;
    emitter << YAML::Key << "nms_threshold" << YAML::Value << config_.nms_threshold;
    emitter << YAML::Key << "max_results" << YAML::Value << config_.max_results;
    emitter << YAML::Key << "scan_step" << YAML::Value << config_.scan_step;
    emitter << YAML::Key << "max_parallelism" << YAML::Value << config_.max_parallelism;
    emitter << YAML::Key << "min_feature_distance" << YAML::Value << config_.min_feature_distance;
    emitter << YAML::Key << "use_gaussian_gradient" << YAML::Value << config_.use_gaussian_gradient;
    emitter << YAML::Key << "use_orientation_histogram" << YAML::Value << config_.use_orientation_histogram;
    emitter << YAML::Key << "use_edge_nms" << YAML::Value << config_.use_edge_nms;
    emitter << YAML::Key << "use_edge_connectivity" << YAML::Value << config_.use_edge_connectivity;
    emitter << YAML::Key << "use_polarity_invariant" << YAML::Value << config_.use_polarity_invariant;
    emitter << YAML::Key << "use_spatial_spread" << YAML::Value << config_.use_spatial_spread;
    emitter << YAML::Key << "reuse_base_features_for_variants" << YAML::Value
            << config_.reuse_base_features_for_variants;
    emitter << YAML::EndMap;

    emitter << YAML::Key << "templates" << YAML::Value << YAML::BeginSeq;
    for (const auto &templ : templates_)
    {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "template_id" << YAML::Value << templ.template_id;
        emitter << YAML::Key << "width" << YAML::Value << templ.width;
        emitter << YAML::Key << "height" << YAML::Value << templ.height;
        emitter << YAML::Key << "template_width" << YAML::Value << templ.template_width;
        emitter << YAML::Key << "template_height" << YAML::Value << templ.template_height;
        emitter << YAML::Key << "tl_x" << YAML::Value << templ.tl_x;
        emitter << YAML::Key << "tl_y" << YAML::Value << templ.tl_y;
        emitter << YAML::Key << "angle_degrees" << YAML::Value << templ.angle_degrees;
        emitter << YAML::Key << "scale" << YAML::Value << templ.scale;
        emitter << YAML::Key << "features" << YAML::Value << YAML::BeginSeq;
        for (const auto &feature : templ.features)
        {
            emitter << YAML::Flow << YAML::BeginSeq << feature.x << feature.y << feature.label
                    << feature.angle_degrees << YAML::EndSeq;
        }
        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;

    if (!emitter.good())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to serialize shape template YAML: %s",
                             emitter.GetLastError().c_str());
    }
    output << emitter.c_str() << '\n';
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write shape template file: %s",
                             template_file.string().c_str());
    }
}

/**
 * @brief 从标准 YAML 读取模板库并替换当前状态。
 *
 * 加载流程先在临时容器中校验完整性，全部成功后再替换 ``config_`` 和 ``templates_``，
 * 避免部分加载失败时留下半初始化状态。
 */
void detail::ShapeTemplateMatcherEngine::load(const fs::path &template_file)
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

    try
    {
        const YAML::Node root = YAML::LoadFile(template_file.string());
        if (!root || root.IsNull() || !root.IsMap())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Shape template YAML root must be a mapping");
        }

        const int version = readRequiredYamlValue<int>(root, "version");
        if (version != kShapeTemplateFileFormatVersion)
        {
            throw irt::Exception(
                irt::Status::ERROR_INVALID_ARGUMENT,
                "Unsupported shape template file version: %d; only v%d compact feature arrays are supported", version,
                kShapeTemplateFileFormatVersion);
        }

        ShapeTemplateMatcherConfig loaded_config = config_;
        const YAML::Node           config_node   = root["config"];
        if (config_node && !config_node.IsNull())
        {
            if (!config_node.IsMap())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Shape template YAML config node must be a mapping");
            }
            readYamlIfPresent(config_node, "num_features", loaded_config.num_features);
            readYamlIfPresent(config_node, "min_features", loaded_config.min_features);
            readYamlIfPresent(config_node, "weak_threshold", loaded_config.weak_threshold);
            readYamlIfPresent(config_node, "strong_threshold", loaded_config.strong_threshold);
            readYamlIfPresent(config_node, "max_label_difference", loaded_config.max_label_difference);
            readYamlIfPresent(config_node, "match_threshold", loaded_config.match_threshold);
            readYamlIfPresent(config_node, "nms_threshold", loaded_config.nms_threshold);
            readYamlIfPresent(config_node, "max_results", loaded_config.max_results);
            readYamlIfPresent(config_node, "scan_step", loaded_config.scan_step);
            readYamlIfPresent(config_node, "max_parallelism", loaded_config.max_parallelism);
            readYamlIfPresent(config_node, "min_feature_distance", loaded_config.min_feature_distance);
            readYamlIfPresent(config_node, "use_gaussian_gradient", loaded_config.use_gaussian_gradient);
            readYamlIfPresent(config_node, "use_orientation_histogram", loaded_config.use_orientation_histogram);
            readYamlIfPresent(config_node, "use_edge_nms", loaded_config.use_edge_nms);
            readYamlIfPresent(config_node, "use_edge_connectivity", loaded_config.use_edge_connectivity);
            readYamlIfPresent(config_node, "use_polarity_invariant", loaded_config.use_polarity_invariant);
            readYamlIfPresent(config_node, "use_spatial_spread", loaded_config.use_spatial_spread);
            readYamlIfPresent(config_node, "reuse_base_features_for_variants",
                              loaded_config.reuse_base_features_for_variants);
        }
        validateConfig(loaded_config);
        if ((loaded_config.use_gaussian_gradient || loaded_config.use_orientation_histogram
             || loaded_config.use_edge_nms || loaded_config.use_edge_connectivity
             || loaded_config.use_polarity_invariant || loaded_config.use_spatial_spread
             || loaded_config.reuse_base_features_for_variants)
            && !kernel_->supportsApproximatePreprocessing())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Approximate v1 preprocessing/variant-reuse options are supported by v1 only");
        }

        std::vector<ShapeTemplateInfo> loaded_templates;
        const YAML::Node templates_node = root["templates"];
        if (templates_node && !templates_node.IsNull() && !templates_node.IsSequence())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Shape template YAML templates node must be a list");
        }

        if (templates_node && !templates_node.IsNull())
        {
            for (const auto &node : templates_node)
            {
                if (!node.IsMap())
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Each shape template YAML entry must be a mapping");
                }

                ShapeTemplateInfo info;
                readYamlIfPresent(node, "template_id", info.template_id);
                info.width         = readRequiredYamlValue<int>(node, "width");
                info.height        = readRequiredYamlValue<int>(node, "height");
                info.template_width  = readRequiredYamlValue<int>(node, "template_width");
                info.template_height = readRequiredYamlValue<int>(node, "template_height");
                info.tl_x          = readRequiredYamlValue<int>(node, "tl_x");
                info.tl_y          = readRequiredYamlValue<int>(node, "tl_y");
                info.angle_degrees = readRequiredYamlValue<float>(node, "angle_degrees");
                info.scale         = readRequiredYamlValue<float>(node, "scale");

                const YAML::Node features_node = node["features"];
                if (!features_node || features_node.IsNull() || !features_node.IsSequence())
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Shape template features must be a list of [x, y, label, angle_degrees] arrays");
                }

                for (const auto &feature_node : features_node)
                {
                    if (!feature_node.IsSequence() || feature_node.size() != 4U)
                    {
                        throw irt::Exception(
                            irt::Status::ERROR_INVALID_ARGUMENT,
                            "Each shape template feature must contain exactly [x, y, label, angle_degrees]");
                    }
                    ShapeTemplateFeature feature;
                    feature.x             = feature_node[0].as<int>();
                    feature.y             = feature_node[1].as<int>();
                    feature.label         = feature_node[2].as<int>();
                    feature.angle_degrees = feature_node[3].as<float>();
                    info.features.push_back(feature);
                }

                info.template_id = static_cast<int>(loaded_templates.size());
                validateTemplateInfo(info, loaded_config.min_features);
                loaded_templates.push_back(std::move(info));
            }
        }

        config_    = loaded_config;
        templates_ = std::move(loaded_templates);
    }
    catch (const YAML::Exception &exception)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to parse shape template YAML '%s': %s",
                             template_file.string().c_str(), exception.what());
    }
}

std::vector<ShapeTemplateVariant>
detail::makeShapeTemplateAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees,
                                            float angle_step_degrees, float scale_begin, float scale_end,
                                            float scale_step)
{
    if (!std::isfinite(angle_begin_degrees) || !std::isfinite(angle_end_degrees)
        || !std::isfinite(angle_step_degrees) || angle_step_degrees <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Angle range must be finite and angle_step_degrees must be positive");
    }
    if (!std::isfinite(scale_begin) || !std::isfinite(scale_end) || !std::isfinite(scale_step)
        || scale_begin <= 0.0f || scale_end <= 0.0f || scale_step <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Scale range must be finite, positive, and scale_step must be positive");
    }
    if (angle_end_degrees + kEps < angle_begin_degrees)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Angle range end must be >= begin");
    }
    if (scale_end + kEps < scale_begin)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Scale range end must be >= begin");
    }

    std::vector<ShapeTemplateVariant> variants;
    for (float scale = scale_begin; scale <= scale_end + kEps; scale += scale_step)
    {
        for (float angle = angle_begin_degrees; angle <= angle_end_degrees + kEps; angle += angle_step_degrees)
        {
            variants.push_back(ShapeTemplateVariant{angle, scale});
        }
    }
    return variants;
}

cv::Mat detail::transformShapeTemplateImage(const cv::Mat &image, ShapeTemplateVariant variant,
                                            const cv::Scalar &border_value)
{
    validateImage(image, "image");
    if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
    }

    const cv::Point2f center(static_cast<float>(image.cols) * 0.5f, static_cast<float>(image.rows) * 0.5f);
    const cv::Mat transform = cv::getRotationMatrix2D(center, variant.angle_degrees, variant.scale);
    cv::Mat output;
    cv::warpAffine(image, output, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
    return output;
}

} // namespace irt::features
