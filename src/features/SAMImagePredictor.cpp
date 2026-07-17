/**
 * @file SAMImagePredictor.cpp
 * @brief SAMImagePredictor implementation.
 */

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/SAMImagePredictor.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/ModelFactory.h>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

using irt::model::checkCuda;
using irt::model::DeviceBuffer;
using irt::model::dimsToCsv;
using irt::model::elementCount;

constexpr std::array<const char *, 5> kSamInputNames{
    "image", "point_coords", "point_labels", "mask_input", "has_mask_input",
};

constexpr std::array<const char *, 3> kSamOutputNames{
    "masks",
    "iou_predictions",
    "low_res_masks",
};

struct PreprocessedSAMImage
{
    std::vector<float> tensor;             ///< 预处理后的 1x3xHxW 输入张量。
    int                original_width{0};  ///< 原图宽度。
    int                original_height{0}; ///< 原图高度。
    int                resized_width{0};   ///< padding 前或拉伸后的图像宽度。
    int                resized_height{0};  ///< padding 前或拉伸后的图像高度。
};

/**
 * @brief 判断模型后端是否为 TensorRT。
 * @param backend 模型后端。
 * @return 使用 TensorRT 时返回 true。
 */
bool usesTensorRt(const irt::model::ModelRuntime &runtime) noexcept
{
    return runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
}

/**
 * @brief 校验 SAMImagePredictor 配置。
 * @param config 待校验配置。
 */
void validateConfig(const SAMImagePredictorConfig &config)
{
    if (config.model_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAMImagePredictor model_name must not be empty");
    }

    config.model_runtime.validate();

    switch (config.model_precision)
    {
    case irt::model::ModelPrecision::FP32:
    case irt::model::ModelPrecision::FP16:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Unsupported SAMImagePredictor model precision");
    }

    switch (config.resize_mode)
    {
    case SAMImageResizeMode::ResizeLongestSide:
    case SAMImageResizeMode::StretchSquare:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported SAMImagePredictor resize mode");
    }
}

/**
 * @brief 校验 SAM mask 后处理选项。
 * @param options 待校验后处理选项。
 */
void validateOptions(const SAMImagePredictOptions &options)
{
    if (options.max_hole_area < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM max_hole_area must be >= 0");
    }
    if (options.max_sprinkle_area < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM max_sprinkle_area must be >= 0");
    }
    switch (options.mask_output_mode)
    {
    case SAMMaskOutputMode::Single:
    case SAMMaskOutputMode::Multimask:
    case SAMMaskOutputMode::All:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported SAM mask output mode");
    }
}

/**
 * @brief 按最长边缩放规则计算 padding 前尺寸。
 * @param original_height 原图高度。
 * @param original_width 原图宽度。
 * @param target_size 目标最长边大小。
 * @return padding 前的高度和宽度。
 */
std::pair<int, int> computeResizeShape(int original_height, int original_width, int target_size)
{
    const float scale = static_cast<float>(target_size) / static_cast<float>(std::max(original_height, original_width));
    return {static_cast<int>(std::floor(static_cast<float>(original_height) * scale + 0.5F)),
            static_cast<int>(std::floor(static_cast<float>(original_width) * scale + 0.5F))};
}

/**
 * @brief 校验 SAM 图像输入张量形状。
 * @param dims 图像输入张量形状。
 */
void validateImageInputShape(const nvinfer1::Dims &dims)
{
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3 || dims.d[2] <= 0 || dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "SAMImagePredictor expects image input shape 1x3xHxW, got %s", dimsToCsv(dims).c_str());
    }
    if (dims.d[2] != dims.d[3])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "SAMImagePredictor expects square image input, got %s", dimsToCsv(dims).c_str());
    }
}

/**
 * @brief 校验 SAM prompt 输入张量形状。
 * @param dims prompt 输入张量形状。
 * @param name 输入张量名称。
 */
void validatePromptInputShape(const nvinfer1::Dims &dims, const char *name)
{
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM input %s has unsupported shape %s", name,
                             dimsToCsv(dims).c_str());
    }
}

/**
 * @brief 使用 SAM/SAM2 对应规则预处理图像。
 * @param image OpenCV BGR 输入图像。
 * @param input_height 模型输入高度。
 * @param input_width 模型输入宽度。
 * @param mode 图像缩放模式。
 * @return 预处理后的输入张量和几何信息。
 */
PreprocessedSAMImage preprocessImage(const cv::Mat &image, int input_height, int input_width, SAMImageResizeMode mode)
{
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot preprocess empty SAM image");
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    PreprocessedSAMImage output;
    output.original_width  = image.cols;
    output.original_height = image.rows;

    if (mode == SAMImageResizeMode::StretchSquare)
    {
        static constexpr std::array<float, 3> kMean{0.485F, 0.456F, 0.406F};
        static constexpr std::array<float, 3> kStd{0.229F, 0.224F, 0.225F};

        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
        output.resized_width  = input_width;
        output.resized_height = input_height;
        output.tensor.resize(static_cast<size_t>(3) * input_height * input_width);

        for (int y = 0; y < input_height; ++y)
        {
            const auto *row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < input_width; ++x)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const size_t offset = static_cast<size_t>(c) * input_height * input_width
                                        + static_cast<size_t>(y) * input_width + static_cast<size_t>(x);
                    output.tensor[offset] = (static_cast<float>(row[x][c]) / 255.0F - kMean[c]) / kStd[c];
                }
            }
        }
        return output;
    }

    if (mode != SAMImageResizeMode::ResizeLongestSide)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported SAMImagePredictor resize mode");
    }

    static constexpr std::array<float, 3> kMean{123.675F, 116.28F, 103.53F};
    static constexpr std::array<float, 3> kStd{58.395F, 57.12F, 57.375F};

    const auto [resized_height, resized_width] = computeResizeShape(image.rows, image.cols, input_height);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    output.resized_width  = resized_width;
    output.resized_height = resized_height;
    output.tensor.assign(static_cast<size_t>(3) * input_height * input_width, 0.0F);
    for (int y = 0; y < resized_height; ++y)
    {
        const auto *row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < resized_width; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                const size_t offset = static_cast<size_t>(c) * input_height * input_width
                                    + static_cast<size_t>(y) * input_width + static_cast<size_t>(x);
                output.tensor[offset] = (static_cast<float>(row[x][c]) - kMean[c]) / kStd[c];
            }
        }
    }
    return output;
}

/**
 * @brief 将 prompt 坐标映射到预处理后图像坐标系。
 * @param value 原始 prompt 坐标值。
 * @param original_size 原图对应方向尺寸。
 * @param resized_size 预处理后对应方向尺寸。
 * @param mode prompt 坐标解释方式。
 * @return 预处理后图像坐标。
 */
float scaleCoordinate(float value, int original_size, int resized_size, SAMPromptCoordinateMode mode)
{
    if (mode == SAMPromptCoordinateMode::NormalizedImage)
    {
        return value * static_cast<float>(resized_size);
    }
    return value * static_cast<float>(resized_size) / static_cast<float>(std::max(1, original_size));
}

/**
 * @brief 构造 SAM point_coords 输入。
 * @param prompt 用户 prompt。
 * @param image 预处理图像几何信息。
 * @param max_points 模型支持的最大点数量。
 * @return 扁平化 point_coords 数组，逻辑形状为 1xmax_pointsx2x1。
 */
std::vector<float> makePointCoords(const SAMImagePrompt &prompt, const PreprocessedSAMImage &image, int max_points)
{
    std::vector<float> coords(static_cast<size_t>(max_points) * 2, 0.0F);
    int                index = 0;

    auto add_point = [&](float x, float y)
    {
        if (index >= max_points)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM prompt has more than %d points", max_points);
        }
        coords[static_cast<size_t>(index) * 2]
            = scaleCoordinate(x, image.original_width, image.resized_width, prompt.coordinate_mode);
        coords[static_cast<size_t>(index) * 2 + 1]
            = scaleCoordinate(y, image.original_height, image.resized_height, prompt.coordinate_mode);
        ++index;
    };

    if (prompt.box)
    {
        add_point(prompt.box->x1, prompt.box->y1);
        add_point(prompt.box->x2, prompt.box->y2);
    }

    for (const auto &point : prompt.points)
    {
        add_point(point.x, point.y);
    }

    return coords;
}

/**
 * @brief 构造 SAM point_labels 输入。
 * @param prompt 用户 prompt。
 * @param max_points 模型支持的最大点数量。
 * @return 扁平化 point_labels 数组，逻辑形状为 1xmax_pointsx1x1。
 */
std::vector<float> makePointLabels(const SAMImagePrompt &prompt, int max_points)
{
    std::vector<float> labels(static_cast<size_t>(max_points), -1.0F);
    int                index = 0;

    auto add_label = [&](int label)
    {
        if (index >= max_points)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM prompt has more than %d points", max_points);
        }
        labels[static_cast<size_t>(index)] = static_cast<float>(label);
        ++index;
    };

    if (prompt.box)
    {
        add_label(2);
        add_label(3);
    }

    for (const auto &point : prompt.points)
    {
        add_label(point.label);
    }

    return labels;
}

/**
 * @brief mask 输出模式对应的通道切片。
 */
struct MaskChannelSlice
{
    int offset{0}; ///< 起始 mask 通道。
    int count{0};  ///< 输出 mask 通道数。
};

/**
 * @brief 根据输出模式解析 mask 通道切片。
 *        Single: 取第 0 个通道; Multimask: 取第 1-3 个通道; All: 取所有通道
 *
 * @param mask_count 模型实际输出的 mask 通道数。
 * @param mode 已解析的输出模式。
 * @return 需要保留的通道切片。
 */
MaskChannelSlice resolveMaskChannelSlice(int mask_count, SAMMaskOutputMode mode)
{
    if (mask_count <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM mask_count must be positive");
    }

    switch (mode)
    {
    case SAMMaskOutputMode::Single:
        if (mask_count == 1 || mask_count >= 4)
        {
            return {0, 1};
        }
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "SAM single-mask output requires a 4-mask raw model output; rebuild the SAM engine");
    case SAMMaskOutputMode::Multimask:
        if (mask_count >= 4)
        {
            return {1, 3};
        }
        if (mask_count == 3)
        {
            return {0, 3};
        }
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "SAM multimask output requires at least 3 masks, got %d", mask_count);
    case SAMMaskOutputMode::All:
        return {0, mask_count};
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported SAM mask output mode");
    }
}

/**
 * @brief 复制指定 mask 通道范围。
 *        Single: 取第 0 个通道; Multimask: 取第 1-3 个通道; All: 取所有通道
 */
std::vector<float> selectMaskChannels(const std::vector<float> &values, int source_count, int height, int width,
                                      MaskChannelSlice slice)
{
    if (slice.offset < 0 || slice.count <= 0 || slice.offset + slice.count > source_count)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid SAM mask channel slice offset=%d count=%d",
                             slice.offset, slice.count);
    }
    const size_t plane_size = static_cast<size_t>(height) * width;
    const size_t expected   = static_cast<size_t>(source_count) * plane_size;
    if (values.size() != expected)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM low_res_masks has %zu values, expected %zu",
                             values.size(), expected);
    }

    std::vector<float> selected(static_cast<size_t>(slice.count) * plane_size);
    const auto         first = values.begin() + static_cast<std::ptrdiff_t>(slice.offset) * plane_size;
    const auto         last  = first + static_cast<std::ptrdiff_t>(slice.count) * plane_size;
    std::copy(first, last, selected.begin());
    return selected;
}

/**
 * @brief 复制指定 mask 通道对应的 IoU 预测。
 */
std::vector<float> selectIouChannels(const std::vector<float> &values, int source_count, MaskChannelSlice slice)
{
    if (values.empty())
    {
        return {};
    }
    if (slice.offset < 0 || slice.count <= 0 || slice.offset + slice.count > source_count)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid SAM IoU channel slice offset=%d count=%d",
                             slice.offset, slice.count);
    }
    if (static_cast<int>(values.size()) < source_count)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM iou_predictions has %zu values, expected >= %d",
                             values.size(), source_count);
    }

    std::vector<float> selected(static_cast<size_t>(slice.count));
    const auto         first = values.begin() + slice.offset;
    const auto         last  = first + slice.count;
    std::copy(first, last, selected.begin());
    return selected;
}

/**
 * @brief 校验 SAM mask 输出张量形状。
 * @param dims 输出张量形状。
 * @param name 输出张量名称。
 */
void validateOutputMaskShape(const nvinfer1::Dims &dims, const char *name)
{
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM output %s must be 1xCxHxW, got %s", name,
                             dimsToCsv(dims).c_str());
    }
}

/**
 * @brief 查找张量名下标，未找到时使用默认下标。
 * @param names 张量名称列表。
 * @param name 目标张量名称。
 * @param fallback 未找到时使用的默认下标。
 * @return 实际张量下标。
 */
size_t tensorIndexOrDefault(const std::vector<std::string> &names, const char *name, size_t fallback)
{
    const auto it = std::find(names.begin(), names.end(), name);
    if (it != names.end())
    {
        return static_cast<size_t>(std::distance(names.begin(), it));
    }
    if (fallback >= names.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM output tensor %s is missing", name);
    }
    return fallback;
}

/**
 * @brief 将指定连通域对应的像素写成替换值。
 * @param values 待修改的 logits 图。
 * @param labels 连通域标签图。
 * @param component 连通域编号。
 * @param replacement 替换后的 logits 值。
 */
void setComponentPixels(cv::Mat &values, const cv::Mat &labels, int component, float replacement)
{
    for (int y = 0; y < values.rows; ++y)
    {
        auto       *value_row = values.ptr<float>(y);
        const auto *label_row = labels.ptr<int>(y);
        for (int x = 0; x < values.cols; ++x)
        {
            if (label_row[x] == component)
            {
                value_row[x] = replacement;
            }
        }
    }
}

/**
 * @brief 按面积阈值修改低分辨率 mask 中的小连通域。
 * @param values 待修改的 logits 图。
 * @param source 用于计算连通域的原始 logits 图。
 * @param foreground 为 true 时处理前景噪点，为 false 时处理背景洞。
 * @param max_area 最大连通域面积，0 表示不处理。
 * @param threshold 前景/背景划分阈值。
 * @param replacement 替换后的 logits 值。
 */
void applyComponentEdit(cv::Mat &values, const cv::Mat &source, bool foreground, int max_area, float threshold,
                        float replacement)
{
    if (max_area <= 0)
    {
        return;
    }

    cv::Mat mask(source.rows, source.cols, CV_8UC1);
    for (int y = 0; y < source.rows; ++y)
    {
        const auto *src_row  = source.ptr<float>(y);
        auto       *mask_row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < source.cols; ++x)
        {
            const bool selected = foreground ? (src_row[x] > threshold) : (src_row[x] <= threshold);
            mask_row[x]         = selected ? 255U : 0U;
        }
    }

    cv::Mat   labels;
    cv::Mat   stats;
    cv::Mat   centroids;
    const int components = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    for (int component = 1; component < components; ++component)
    {
        const int area = stats.at<int>(component, cv::CC_STAT_AREA);
        if (area <= max_area)
        {
            setComponentPixels(values, labels, component, replacement);
        }
    }
}

/**
 * @brief 对低分辨率 mask logits 执行小洞填充和小噪点移除。
 * @param masks 待原地修改的低分辨率 logits，按 CxHxW 排列。
 * @param mask_count mask 数量。
 * @param height 低分辨率 logits 高度。
 * @param width 低分辨率 logits 宽度。
 * @param options mask 后处理选项。
 */
void applyLowResMaskCleanup(std::vector<float> &masks, int mask_count, int height, int width,
                            const SAMImagePredictOptions &options)
{
    if (options.max_hole_area <= 0 && options.max_sprinkle_area <= 0)
    {
        return;
    }

    const size_t plane_size = static_cast<size_t>(height) * width;
    for (int mask_index = 0; mask_index < mask_count; ++mask_index)
    {
        auto   *data = masks.data() + static_cast<size_t>(mask_index) * plane_size;
        cv::Mat values(height, width, CV_32FC1, data);
        cv::Mat original = values.clone();

        applyComponentEdit(values, original, false, options.max_hole_area, options.mask_threshold,
                           options.mask_threshold + 10.0F);
        applyComponentEdit(values, original, true, options.max_sprinkle_area, options.mask_threshold,
                           options.mask_threshold - 10.0F);
    }
}

/**
 * @brief 将低分辨率 logits 双线性 resize 到原图尺寸。
 * @param low_res_masks 低分辨率 logits，按 CxHxW 排列。
 * @param mask_count mask 数量。
 * @param low_res_height 低分辨率 logits 高度。
 * @param low_res_width 低分辨率 logits 宽度。
 * @param geometry 后处理几何信息。
 * @return 原图尺寸 logits，按 CxHxW 排列。
 */
std::vector<float> resizeMasks(const std::vector<float> &low_res_masks, int mask_count, int low_res_height,
                               int low_res_width, const SAMMaskPostprocessGeometry &geometry)
{
    const size_t       output_plane = static_cast<size_t>(geometry.original_height) * geometry.original_width;
    std::vector<float> output(static_cast<size_t>(mask_count) * output_plane);
    const size_t       low_res_plane = static_cast<size_t>(low_res_height) * low_res_width;

    for (int mask_index = 0; mask_index < mask_count; ++mask_index)
    {
        const auto *mask_data = low_res_masks.data() + static_cast<size_t>(mask_index) * low_res_plane;
        cv::Mat     low_res(low_res_height, low_res_width, CV_32FC1, const_cast<float *>(mask_data));
        cv::Mat     resized;

        if (geometry.resize_mode == SAMImageResizeMode::StretchSquare)
        {
            cv::resize(low_res, resized, cv::Size(geometry.original_width, geometry.original_height), 0, 0,
                       cv::INTER_LINEAR);
        }
        else if (geometry.resize_mode == SAMImageResizeMode::ResizeLongestSide)
        {
            cv::Mat padded;
            cv::resize(low_res, padded, cv::Size(geometry.model_width, geometry.model_height), 0, 0, cv::INTER_LINEAR);
            const int crop_width  = std::min(geometry.resized_width, geometry.model_width);
            const int crop_height = std::min(geometry.resized_height, geometry.model_height);
            cv::Mat   cropped     = padded(cv::Rect(0, 0, crop_width, crop_height));
            cv::resize(cropped, resized, cv::Size(geometry.original_width, geometry.original_height), 0, 0,
                       cv::INTER_LINEAR);
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "SAM postprocess geometry must resolve resize mode");
        }

        const auto *resized_data = resized.ptr<float>(0);
        std::copy(resized_data, resized_data + output_plane,
                  output.begin() + static_cast<std::ptrdiff_t>(mask_index) * static_cast<std::ptrdiff_t>(output_plane));
    }
    return output;
}

} // namespace

/**
 * @brief SAMImagePredictor 的私有实现，隐藏模型运行时和缓冲区组织细节。
 */
class SAMImagePredictor::Impl
{
public:
    /**
     * @brief 构造私有实现并校验配置。
     * @param config 预测器配置。
     */
    explicit Impl(SAMImagePredictorConfig config)
        : config_(std::move(config))
    {
        validateConfig(config_);
        resize_mode_ = config_.resize_mode;
    }

    /**
     * @brief 析构私有实现并先释放模型上下文。
     */
    ~Impl()
    {
        model_.reset();
    }

    /**
     * @brief 构建或加载 SAM 模型并校验 I/O 契约。
     * @param weights_file 权重、engine 或图模型文件路径。
     */
    void load(const fs::path &weights_file)
    {
        auto model_config = std::make_unique<irt::model::IModelConfig>();
        model_config->setRuntime(config_.model_runtime);
        model_config->setPrecision(config_.model_precision);

        const std::string runtime_model_name = usesTensorRt(config_.model_runtime) ? config_.model_name : "onnx";
        model_                               = irt::model::CreateModel(runtime_model_name, std::move(model_config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create SAM model: %s",
                                 runtime_model_name.c_str());
        }

        model_->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        model_->buildOrLoad(weights_file.string());

        input_names_ = model_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        if (input_names_.size() != kSamInputNames.size())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "SAMImagePredictor expects the 5-input SAM contract");
        }
        for (size_t i = 0; i < kSamInputNames.size(); ++i)
        {
            if (input_names_[i] != kSamInputNames[i])
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "SAMImagePredictor input %zu must be %s, got %s", i, kSamInputNames[i],
                                     input_names_[i].c_str());
            }
        }

        output_names_ = model_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (output_names_.size() < 2)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "SAMImagePredictor expects SAM mask and IoU outputs");
        }

        image_dims_ = model_->tensorShape(input_names_[0]);
        validateImageInputShape(image_dims_);
        prompt_coords_dims_ = model_->tensorShape(input_names_[1]);
        prompt_labels_dims_ = model_->tensorShape(input_names_[2]);
        mask_input_dims_    = model_->tensorShape(input_names_[3]);
        has_mask_dims_      = model_->tensorShape(input_names_[4]);
        validatePromptInputShape(prompt_coords_dims_, input_names_[1].c_str());
        validatePromptInputShape(prompt_labels_dims_, input_names_[2].c_str());
        validatePromptInputShape(mask_input_dims_, input_names_[3].c_str());
        validatePromptInputShape(has_mask_dims_, input_names_[4].c_str());
        if (prompt_coords_dims_.d[2] != 2)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM point_coords must have XY dimension 2");
        }
        if (prompt_labels_dims_.d[1] != prompt_coords_dims_.d[1])
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "SAM point_labels must match point_coords max point count");
        }
        if (mask_input_dims_.d[1] != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM mask_input must have one channel");
        }

        ready_ = true;
    }

    /**
     * @brief 对单张图像执行模型推理和 mask 后处理。
     * @param image_path 输入图像路径。
     * @param prompt 用户 prompt。
     * @param options mask 后处理选项。
     * @return SAM 单图预测结果。
     */
    SAMImagePrediction predict(const fs::path &image_path, const SAMImagePrompt &prompt,
                               const SAMImagePredictOptions &options)
    {
        validateOptions(options);
        if (!ready_ || !model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                 "SAMImagePredictor must be loaded before predict");
        }

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const int input_height = static_cast<int>(image_dims_.d[2]);
        const int input_width  = static_cast<int>(image_dims_.d[3]);
        auto      preprocessed = preprocessImage(image, input_height, input_width, resize_mode_);

        const int max_points   = static_cast<int>(prompt_coords_dims_.d[1]);
        auto      point_coords = makePointCoords(prompt, preprocessed, max_points);
        auto      point_labels = makePointLabels(prompt, max_points);

        const size_t       mask_input_count = elementCount(mask_input_dims_);
        std::vector<float> mask_input(mask_input_count, 0.0F);
        std::vector<float> has_mask_input(elementCount(has_mask_dims_), prompt.mask_input.empty() ? 0.0F : 1.0F);
        if (!prompt.mask_input.empty())
        {
            if (prompt.mask_input.size() != mask_input_count)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM mask_input has %zu values, expected %zu",
                                     prompt.mask_input.size(), mask_input_count);
            }
            mask_input = prompt.mask_input;
        }

        const std::vector<std::vector<float> *> input_vectors{
            &preprocessed.tensor, &point_coords, &point_labels, &mask_input, &has_mask_input,
        };

        std::vector<std::vector<float>> output_vectors(output_names_.size());
        for (size_t i = 0; i < output_names_.size(); ++i)
        {
            output_vectors[i].resize(elementCount(model_->tensorShape(output_names_[i])));
        }

        runModel(input_vectors, output_vectors);

        const size_t mask_output_index = tensorIndexOrDefault(output_names_, kSamOutputNames[0], 0);
        const size_t iou_output_index
            = tensorIndexOrDefault(output_names_, kSamOutputNames[1], std::min<size_t>(1, output_names_.size() - 1));
        const size_t low_res_output_index
            = output_names_.size() > 2 ? tensorIndexOrDefault(output_names_, kSamOutputNames[2], 2) : mask_output_index;

        const auto low_res_dims = model_->tensorShape(output_names_[low_res_output_index]);
        validateOutputMaskShape(low_res_dims, output_names_[low_res_output_index].c_str());
        const int mask_count     = static_cast<int>(low_res_dims.d[1]);
        const int low_res_height = static_cast<int>(low_res_dims.d[2]);
        const int low_res_width  = static_cast<int>(low_res_dims.d[3]);

        SAMMaskPostprocessGeometry geometry;
        geometry.original_width  = preprocessed.original_width;
        geometry.original_height = preprocessed.original_height;
        geometry.model_width     = input_width;
        geometry.model_height    = input_height;
        geometry.resized_width   = preprocessed.resized_width;
        geometry.resized_height  = preprocessed.resized_height;
        geometry.resize_mode     = resize_mode_;

        (void)mask_output_index;
        return SAMImagePredictor::postprocessMasks(output_vectors[low_res_output_index], mask_count, low_res_height,
                                                   low_res_width, output_vectors[iou_output_index], geometry, options);
    }

    /**
     * @brief 查询模型是否已经加载完成。
     * @return 已加载时返回 true。
     */
    bool isReady() const noexcept
    {
        return ready_;
    }

    /**
     * @brief 获取预测器配置。
     * @return 配置常量引用。
     */
    const SAMImagePredictorConfig &config() const noexcept
    {
        return config_;
    }

private:
    /**
     * @brief 执行一次 SAM 模型推理。
     * @param input_vectors 主机侧输入数组列表。
     * @param output_vectors 主机侧输出数组列表，函数会写入推理结果。
     */
    void runModel(const std::vector<std::vector<float> *> &input_vectors,
                  std::vector<std::vector<float>>         &output_vectors)
    {
        const bool use_trt = usesTensorRt(config_.model_runtime);
        if (!use_trt)
        {
            std::vector<void *> buffers;
            buffers.reserve(input_vectors.size() + output_vectors.size());
            for (const auto *input : input_vectors)
            {
                buffers.push_back(const_cast<float *>(input->data()));
            }
            for (auto &output : output_vectors)
            {
                buffers.push_back(output.data());
            }
            model_->infer(buffers, nullptr, false);
            return;
        }

        irt::model::setCudaDevice(config_.model_runtime.deviceId());
        const auto                stream = model_->resolveExecutionStream();
        std::vector<DeviceBuffer> device_inputs;
        std::vector<DeviceBuffer> device_outputs;
        std::vector<void *>       buffers;
        device_inputs.reserve(input_vectors.size());
        device_outputs.reserve(output_vectors.size());
        buffers.reserve(input_vectors.size() + output_vectors.size());

        for (const auto *input : input_vectors)
        {
            device_inputs.emplace_back(input->size(), nvinfer1::DataType::kFLOAT);
            buffers.push_back(device_inputs.back().data());
        }
        for (const auto &output : output_vectors)
        {
            device_outputs.emplace_back(output.size(), nvinfer1::DataType::kFLOAT);
            buffers.push_back(device_outputs.back().data());
        }

        for (size_t i = 0; i < input_vectors.size(); ++i)
        {
            const auto *input = input_vectors[i];
            checkCuda(cudaMemcpyAsync(device_inputs[i].data(), input->data(), input->size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(SAM predictor input)");
        }

        model_->infer(buffers, stream, true);

        for (size_t i = 0; i < output_vectors.size(); ++i)
        {
            auto &output = output_vectors[i];
            checkCuda(cudaMemcpyAsync(output.data(), device_outputs[i].data(), output.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync(SAM predictor output)");
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SAM predictor)");
    }

    SAMImagePredictorConfig config_{};                                       ///< 预测器配置。
    SAMImageResizeMode      resize_mode_{SAMImageResizeMode::StretchSquare}; ///< 预处理和后处理几何模式。
    bool                    ready_{false};                                   ///< 模型是否已经加载完成。

    std::unique_ptr<irt::model::IModel> model_;        ///< SAM 推理模型。
    std::vector<std::string>            input_names_;  ///< 模型输入张量名称列表。
    std::vector<std::string>            output_names_; ///< 模型输出张量名称列表。

    nvinfer1::Dims image_dims_{};         ///< 图像输入张量形状。
    nvinfer1::Dims prompt_coords_dims_{}; ///< point_coords 输入张量形状。
    nvinfer1::Dims prompt_labels_dims_{}; ///< point_labels 输入张量形状。
    nvinfer1::Dims mask_input_dims_{};    ///< mask_input 输入张量形状。
    nvinfer1::Dims has_mask_dims_{};      ///< has_mask_input 输入张量形状。
};

/**
 * @brief 构造 SAM 单图预测器。
 * @param config 预测器配置。
 */
SAMImagePredictor::SAMImagePredictor(SAMImagePredictorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

/**
 * @brief 析构预测器。
 */
SAMImagePredictor::~SAMImagePredictor() = default;

/**
 * @brief 移动构造预测器。
 * @param other 被移动的预测器。
 */
SAMImagePredictor::SAMImagePredictor(SAMImagePredictor &&other) noexcept = default;

/**
 * @brief 移动赋值预测器。
 * @param other 被移动的预测器。
 * @return 当前预测器引用。
 */
SAMImagePredictor &SAMImagePredictor::operator=(SAMImagePredictor &&other) noexcept = default;

/**
 * @brief 构建或加载 SAM 模型。
 * @param weights_file 权重、engine 或图模型文件路径。
 */
void SAMImagePredictor::load(const fs::path &weights_file)
{
    impl_->load(weights_file);
}

/**
 * @brief 对单张图像执行 prompt 分割预测。
 * @param image_path 输入图像路径。
 * @param prompt 用户 prompt。
 * @param options mask 后处理选项。
 * @return SAM 单图预测结果。
 */
SAMImagePrediction SAMImagePredictor::predict(const fs::path &image_path, const SAMImagePrompt &prompt,
                                              const SAMImagePredictOptions &options)
{
    return impl_->predict(image_path, prompt, options);
}

/**
 * @brief 查询预测器是否已经加载模型。
 * @return 已加载可预测时返回 true。
 */
bool SAMImagePredictor::isReady() const noexcept
{
    return impl_ && impl_->isReady();
}

/**
 * @brief 获取当前预测器配置。
 * @return 配置常量引用。
 */
const SAMImagePredictorConfig &SAMImagePredictor::config() const noexcept
{
    return impl_->config();
}

/**
 * @brief 对低分辨率 mask logits 执行几何一致的 mask 后处理。
 * @param low_res_masks 低分辨率 mask logits，按 CxHxW 排列。
 * @param mask_count mask 数量。
 * @param low_res_height 低分辨率 logits 高度。
 * @param low_res_width 低分辨率 logits 宽度。
 * @param iou_predictions 模型输出的 IoU/质量预测。
 * @param geometry 还原到原图尺寸所需的几何信息。
 * @param options mask 后处理选项。
 * @return 原图尺寸预测结果和 clamp 后的低分辨率 logits。
 */
SAMImagePrediction SAMImagePredictor::postprocessMasks(const std::vector<float> &low_res_masks, int mask_count,
                                                       int low_res_height, int low_res_width,
                                                       const std::vector<float>         &iou_predictions,
                                                       const SAMMaskPostprocessGeometry &geometry,
                                                       const SAMImagePredictOptions     &options)
{
    validateOptions(options);
    if (mask_count <= 0 || low_res_height <= 0 || low_res_width <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM low_res_masks shape must be positive");
    }
    if (geometry.original_width <= 0 || geometry.original_height <= 0 || geometry.model_width <= 0
        || geometry.model_height <= 0 || geometry.resized_width <= 0 || geometry.resized_height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM postprocess geometry must be positive");
    }
    const auto         slice = resolveMaskChannelSlice(mask_count, options.mask_output_mode);
    std::vector<float> selected_masks
        = selectMaskChannels(low_res_masks, mask_count, low_res_height, low_res_width, slice);
    std::vector<float> selected_ious = selectIouChannels(iou_predictions, mask_count, slice);

    std::vector<float> cleaned_masks = selected_masks;
    applyLowResMaskCleanup(cleaned_masks, slice.count, low_res_height, low_res_width, options);
    auto high_res_logits = resizeMasks(cleaned_masks, slice.count, low_res_height, low_res_width, geometry);

    SAMImagePrediction prediction;
    prediction.mask_count       = slice.count;
    prediction.width            = geometry.original_width;
    prediction.height           = geometry.original_height;
    prediction.low_res_width    = low_res_width;
    prediction.low_res_height   = low_res_height;
    prediction.masks_are_logits = options.return_logits;
    prediction.low_res_masks    = std::move(selected_masks);
    prediction.iou_predictions  = std::move(selected_ious);

    for (auto &value : prediction.low_res_masks)
    {
        value = std::clamp(value, -32.0F, 32.0F);
    }

    if (options.return_logits)
    {
        prediction.masks = std::move(high_res_logits);
        return prediction;
    }

    prediction.masks.resize(high_res_logits.size());
    prediction.binary_masks.resize(high_res_logits.size());
    for (size_t i = 0; i < high_res_logits.size(); ++i)
    {
        const bool foreground      = high_res_logits[i] > options.mask_threshold;
        prediction.masks[i]        = foreground ? 1.0F : 0.0F;
        prediction.binary_masks[i] = foreground ? std::uint8_t{1} : std::uint8_t{0};
    }
    return prediction;
}

} // namespace irt::features
