#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using irt::model::checkCuda;
using irt::model::DeviceBuffer;
using irt::model::dimsToCsv;
using irt::model::elementCount;
using irt::model::HostBuffer;

constexpr std::array<int, 3> kYoloStrides{8, 16, 32};
constexpr int                kYoloMaskChannels = 32;
const fs::path               kDefaultImagePath{"assets/pics/dog.jpg"};
const fs::path               kDefaultLabelPath{"assets/coco80.names"};

/**
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

enum class SegmentationFamily
{
    RFDETR,
    YOLOV8,
};

/**
 * @brief RF-DETR-Seg 单图实例分割示例的命令行参数。
 */
struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    fs::path                 image_path;
    fs::path                 label_file;
    fs::path                 output_image;
    int                      input_size{0};
    bool                     input_size_explicit{false};
    int                      num_classes{80};
    float                    conf_threshold{0.35F};
    float                    mask_threshold{0.0F};
    float                    nms_threshold{0.50F};
    int                      max_instances{20};
    SegmentationFamily       family{SegmentationFamily::RFDETR};
    irt::model::ModelRuntime runtime{};
    int                      warmup{0};
    int                      repeat{1};
};

/**
 * @brief YOLO letterbox 预处理的缩放与填充信息。
 */
struct LetterboxInfo
{
    float scale{1.0F};
    int   pad_x{0};
    int   pad_y{0};
    int   original_w{0};
    int   original_h{0};
    int   input_w{0};
    int   input_h{0};
};

/**
 * @brief 单个实例分割结果，坐标使用原图像素坐标。
 */
struct Instance
{
    float              x1{0.0F};
    float              y1{0.0F};
    float              x2{0.0F};
    float              y2{0.0F};
    float              score{0.0F};
    int                class_id{0};
    int                query_index{0};
    std::vector<float> mask_coefficients;
};

struct IterationTiming
{
    double h2d_ms{0.0};
    double inference_ms{0.0};
    double d2h_ms{0.0};

    double totalMs() const noexcept
    {
        return h2d_ms + inference_ms + d2h_ms;
    }
};

struct TimingStats
{
    double total_ms{0.0};
    double avg_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string trim(std::string value)
{
    auto not_space = [](unsigned char ch)
    {
        return !std::isspace(ch);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool isRFDETRSegModelName(const std::string &model_name)
{
    const std::string normalized = toLower(model_name);
    return normalized.rfind("rfdetr", 0) == 0
        && (normalized.find("_seg") != std::string::npos || normalized.find("-seg") != std::string::npos);
}

bool isYOLOv8SegModelName(const std::string &model_name)
{
    const std::string normalized = toLower(model_name);
    return normalized.rfind("yolov8", 0) == 0
        && (normalized.find("_seg") != std::string::npos || normalized.find("-seg") != std::string::npos);
}

SegmentationFamily modelFamily(const std::string &model_name)
{
    if (isRFDETRSegModelName(model_name))
    {
        return SegmentationFamily::RFDETR;
    }
    if (isYOLOv8SegModelName(model_name))
    {
        return SegmentationFamily::YOLOV8;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "Segmentation sample expects RF-DETR-Seg or YOLOv8-Seg model");
}

const char *modelFamilyName(SegmentationFamily family)
{
    return family == SegmentationFamily::RFDETR ? "RF-DETR-Seg" : "YOLOv8-Seg";
}

/**
 * @brief 按名称定位 RF-DETR-Seg 输出，避免后处理依赖后端返回顺序。
 */
size_t requireOutputIndex(const std::vector<std::string> &output_names, const char *required_name)
{
    const auto iter = std::find(output_names.begin(), output_names.end(), required_name);
    if (iter == output_names.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Segmentation output is missing: %s", required_name);
    }
    return static_cast<size_t>(std::distance(output_names.begin(), iter));
}

float sigmoid(float value)
{
    if (value >= 0.0F)
    {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

float clampFloat(float value, float lower, float upper)
{
    return std::max(lower, std::min(value, upper));
}

TimingStats summarizeTimings(const std::vector<double> &values)
{
    if (values.empty())
    {
        return {};
    }
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const double total          = std::accumulate(values.begin(), values.end(), 0.0);
    return TimingStats{total, total / static_cast<double>(values.size()), *min_it, *max_it};
}

void printTimingStats(const char *name, const TimingStats &stats)
{
    std::cout << ", " << name << "_total=" << stats.total_ms << " ms, " << name << "_avg=" << stats.avg_ms << " ms, "
              << name << "_min=" << stats.min_ms << " ms, " << name << "_max=" << stats.max_ms << " ms";
}

std::vector<std::string> readLabels(const fs::path &label_file)
{
    std::vector<std::string> labels;
    if (label_file.empty() || !fs::exists(label_file))
    {
        std::cout << "Label file not found, class ids will be printed: " << label_file.generic_string() << std::endl;
        return labels;
    }

    std::ifstream input(label_file);
    std::string   line;
    while (std::getline(input, line))
    {
        line = trim(line);
        if (!line.empty())
        {
            labels.push_back(line);
        }
    }
    return labels;
}

std::string className(int class_id, const std::vector<std::string> &labels)
{
    if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size())
    {
        return labels[static_cast<size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}

/**
 * @brief 使用 YOLO letterbox 规则预处理输入图像。
 */
std::vector<float> preprocessLetterbox(const cv::Mat &bgr_image, int input_w, int input_h, LetterboxInfo &info)
{
    info.original_w = bgr_image.cols;
    info.original_h = bgr_image.rows;
    info.input_w    = input_w;
    info.input_h    = input_h;
    info.scale      = std::min(static_cast<float>(input_w) / static_cast<float>(bgr_image.cols),
                               static_cast<float>(input_h) / static_cast<float>(bgr_image.rows));

    const int resized_w = static_cast<int>(std::round(static_cast<float>(bgr_image.cols) * info.scale));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(bgr_image.rows) * info.scale));
    info.pad_x          = (input_w - resized_w) / 2;
    info.pad_y          = (input_h - resized_h) / 2;

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, resized_w, resized_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    std::vector<float> chw(static_cast<size_t>(3 * input_h * input_w));
    const size_t       plane_size = static_cast<size_t>(input_h * input_w);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(chw.data() + static_cast<size_t>(c) * plane_size, channels[c].ptr<float>(),
                    plane_size * sizeof(float));
    }
    return chw;
}

/**
 * @brief 使用 RF-DETR 官方 ImageNet 归一化预处理，并按 NCHW 展平。
 */
std::vector<float> preprocessRFDETR(const cv::Mat &bgr_image, const nvinfer1::Dims &input_dims)
{
    if (input_dims.nbDims != 4 || input_dims.d[1] != 3 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RF-DETR-Seg expects Nx3xHxW input, got %s",
                             dimsToCsv(input_dims).c_str());
    }

    const cv::Mat preprocessed = irt::model::ImageNetUtil::preprocess(
        bgr_image, cv::Size(static_cast<int>(input_dims.d[3]), static_cast<int>(input_dims.d[2])));
    return irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
}

float intersectionOverUnion(const Instance &a, const Instance &b)
{
    const float x1         = std::max(a.x1, b.x1);
    const float y1         = std::max(a.y1, b.y1);
    const float x2         = std::min(a.x2, b.x2);
    const float y2         = std::min(a.y2, b.y2);
    const float inter_w    = std::max(0.0F, x2 - x1);
    const float inter_h    = std::max(0.0F, y2 - y1);
    const float inter_area = inter_w * inter_h;
    const float area_a     = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b     = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float denom      = area_a + area_b - inter_area;
    return denom <= 0.0F ? 0.0F : inter_area / denom;
}

std::vector<Instance> nonMaximumSuppression(std::vector<Instance> instances, float threshold, int max_instances)
{
    std::sort(instances.begin(), instances.end(),
              [](const Instance &a, const Instance &b) { return a.score > b.score; });

    std::vector<Instance> kept;
    std::vector<bool>     removed(instances.size(), false);
    kept.reserve(std::min<int>(max_instances, static_cast<int>(instances.size())));
    for (size_t i = 0; i < instances.size() && static_cast<int>(kept.size()) < max_instances; ++i)
    {
        if (removed[i])
        {
            continue;
        }
        kept.push_back(instances[i]);
        for (size_t j = i + 1; j < instances.size(); ++j)
        {
            if (!removed[j] && instances[i].class_id == instances[j].class_id
                && intersectionOverUnion(instances[i], instances[j]) > threshold)
            {
                removed[j] = true;
            }
        }
    }
    return kept;
}

/**
 * @brief 解码 RF-DETR-Seg 的 boxes/logits/masks 输出。
 */
std::vector<Instance> decodeRFDETRSegOutputs(const HostBuffer &boxes_output, const nvinfer1::Dims &boxes_dims,
                                             const HostBuffer &logits_output, const nvinfer1::Dims &logits_dims,
                                             const nvinfer1::Dims &masks_dims, const Arguments &args,
                                             const cv::Size &image_size)
{
    if (boxes_dims.nbDims != 3 || logits_dims.nbDims != 3 || masks_dims.nbDims != 4 || boxes_dims.d[0] != 1
        || logits_dims.d[0] != 1 || masks_dims.d[0] != 1 || boxes_dims.d[1] != logits_dims.d[1]
        || boxes_dims.d[1] != masks_dims.d[1] || boxes_dims.d[2] != 4)
    {
        throw irt::Exception(
            irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected RF-DETR-Seg output shapes: dets=%s labels=%s masks=%s",
            dimsToCsv(boxes_dims).c_str(), dimsToCsv(logits_dims).c_str(), dimsToCsv(masks_dims).c_str());
    }

    const auto *boxes   = static_cast<const float *>(boxes_output.data());
    const auto *logits  = static_cast<const float *>(logits_output.data());
    const int   queries = static_cast<int>(boxes_dims.d[1]);
    const int   classes = static_cast<int>(logits_dims.d[2]);

    std::vector<Instance> instances;
    instances.reserve(static_cast<size_t>(queries));
    for (int q = 0; q < queries; ++q)
    {
        int   best_class = 0;
        float best_logit = logits[q * classes];
        for (int cls = 1; cls < classes; ++cls)
        {
            const float value = logits[q * classes + cls];
            if (value > best_logit)
            {
                best_logit = value;
                best_class = cls;
            }
        }

        const float score = sigmoid(best_logit);
        if (score < args.conf_threshold)
        {
            continue;
        }

        const float cx = boxes[q * 4 + 0] * static_cast<float>(image_size.width);
        const float cy = boxes[q * 4 + 1] * static_cast<float>(image_size.height);
        const float w  = boxes[q * 4 + 2] * static_cast<float>(image_size.width);
        const float h  = boxes[q * 4 + 3] * static_cast<float>(image_size.height);

        Instance instance;
        instance.x1          = clampFloat(cx - 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
        instance.y1          = clampFloat(cy - 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));
        instance.x2          = clampFloat(cx + 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
        instance.y2          = clampFloat(cy + 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));
        instance.score       = score;
        instance.class_id    = best_class;
        instance.query_index = q;
        instances.push_back(instance);
    }

    return nonMaximumSuppression(std::move(instances), args.nms_threshold, args.max_instances);
}

void remapToOriginalImage(Instance &instance, const LetterboxInfo &letterbox)
{
    instance.x1 = clampFloat((instance.x1 - static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0F,
                             static_cast<float>(letterbox.original_w - 1));
    instance.y1 = clampFloat((instance.y1 - static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0F,
                             static_cast<float>(letterbox.original_h - 1));
    instance.x2 = clampFloat((instance.x2 - static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0F,
                             static_cast<float>(letterbox.original_w - 1));
    instance.y2 = clampFloat((instance.y2 - static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0F,
                             static_cast<float>(letterbox.original_h - 1));
}

/**
 * @brief 解码 YOLOv8-Seg 的 DFL distance、class logits 和 mask coefficients。
 */
std::vector<Instance> decodeYoloV8SegOutputs(const std::vector<HostBuffer>     &outputs,
                                             const std::vector<nvinfer1::Dims> &output_dims,
                                             const std::array<size_t, 3> &branch_indices, const Arguments &args,
                                             const LetterboxInfo &letterbox)
{
    std::vector<Instance> instances;
    int                   query_index = 0;
    for (size_t branch = 0; branch < branch_indices.size(); ++branch)
    {
        const auto &dims              = output_dims[branch_indices[branch]];
        const int   stride            = kYoloStrides[branch];
        const int   grid_h            = letterbox.input_h / stride;
        const int   grid_w            = letterbox.input_w / stride;
        const int   grid              = grid_h * grid_w;
        const int   expected_channels = 4 + args.num_classes + kYoloMaskChannels;
        if (dims.nbDims != 3 || dims.d[0] != 1 || dims.d[1] != expected_channels || dims.d[2] != grid)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Unexpected YOLOv8-Seg output shape for branch %zu: %s", branch,
                                 dimsToCsv(dims).c_str());
        }

        const auto *data = static_cast<const float *>(outputs[branch_indices[branch]].data());
        for (int idx = 0; idx < grid; ++idx)
        {
            int   best_class = 0;
            float best_score = 0.0F;
            for (int cls = 0; cls < args.num_classes; ++cls)
            {
                const float score = sigmoid(data[(4 + cls) * grid + idx]);
                if (score > best_score)
                {
                    best_score = score;
                    best_class = cls;
                }
            }
            if (best_score < args.conf_threshold)
            {
                continue;
            }

            const int   x        = idx % grid_w;
            const int   y        = idx / grid_w;
            const float anchor_x = static_cast<float>(x) + 0.5F;
            const float anchor_y = static_cast<float>(y) + 0.5F;
            const float left     = data[0 * grid + idx];
            const float top      = data[1 * grid + idx];
            const float right    = data[2 * grid + idx];
            const float bottom   = data[3 * grid + idx];

            Instance instance;
            instance.x1          = (anchor_x - left) * static_cast<float>(stride);
            instance.y1          = (anchor_y - top) * static_cast<float>(stride);
            instance.x2          = (anchor_x + right) * static_cast<float>(stride);
            instance.y2          = (anchor_y + bottom) * static_cast<float>(stride);
            instance.score       = best_score;
            instance.class_id    = best_class;
            instance.query_index = query_index++;
            instance.mask_coefficients.resize(kYoloMaskChannels);
            for (int c = 0; c < kYoloMaskChannels; ++c)
            {
                instance.mask_coefficients[static_cast<size_t>(c)] = data[(4 + args.num_classes + c) * grid + idx];
            }
            remapToOriginalImage(instance, letterbox);
            instances.push_back(std::move(instance));
        }
    }

    return nonMaximumSuppression(std::move(instances), args.nms_threshold, args.max_instances);
}

/**
 * @brief 根据 YOLOv8-Seg proto 和实例 mask coefficients 生成原图尺寸 mask logits。
 */
std::vector<cv::Mat> buildYoloV8SegMasks(const std::vector<Instance> &instances, const HostBuffer &proto_output,
                                         const nvinfer1::Dims &proto_dims, const LetterboxInfo &letterbox)
{
    if (proto_dims.nbDims != 4 || proto_dims.d[0] != 1 || proto_dims.d[1] != kYoloMaskChannels || proto_dims.d[2] <= 0
        || proto_dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected YOLOv8-Seg proto shape: %s",
                             dimsToCsv(proto_dims).c_str());
    }

    const auto    *proto     = static_cast<const float *>(proto_output.data());
    const int      proto_h   = static_cast<int>(proto_dims.d[2]);
    const int      proto_w   = static_cast<int>(proto_dims.d[3]);
    const int      area      = proto_h * proto_w;
    const int      resized_w = static_cast<int>(std::round(static_cast<float>(letterbox.original_w) * letterbox.scale));
    const int      resized_h = static_cast<int>(std::round(static_cast<float>(letterbox.original_h) * letterbox.scale));
    const cv::Rect content_rect(letterbox.pad_x, letterbox.pad_y, resized_w, resized_h);

    std::vector<cv::Mat> masks;
    masks.reserve(instances.size());
    for (const auto &instance : instances)
    {
        cv::Mat low_res = cv::Mat::zeros(proto_h, proto_w, CV_32FC1);
        for (int c = 0; c < kYoloMaskChannels; ++c)
        {
            const cv::Mat channel(proto_h, proto_w, CV_32FC1,
                                  const_cast<float *>(proto + static_cast<size_t>(c) * area));
            low_res += instance.mask_coefficients[static_cast<size_t>(c)] * channel;
        }

        const float net_x1 = instance.x1 * letterbox.scale + static_cast<float>(letterbox.pad_x);
        const float net_y1 = instance.y1 * letterbox.scale + static_cast<float>(letterbox.pad_y);
        const float net_x2 = instance.x2 * letterbox.scale + static_cast<float>(letterbox.pad_x);
        const float net_y2 = instance.y2 * letterbox.scale + static_cast<float>(letterbox.pad_y);
        const int   crop_x1
            = std::max(0, std::min(proto_w - 1, static_cast<int>(std::floor(net_x1 / letterbox.input_w * proto_w))));
        const int crop_y1
            = std::max(0, std::min(proto_h - 1, static_cast<int>(std::floor(net_y1 / letterbox.input_h * proto_h))));
        const int crop_x2
            = std::max(0, std::min(proto_w, static_cast<int>(std::ceil(net_x2 / letterbox.input_w * proto_w))));
        const int crop_y2
            = std::max(0, std::min(proto_h, static_cast<int>(std::ceil(net_y2 / letterbox.input_h * proto_h))));

        cv::Mat cropped = cv::Mat::zeros(proto_h, proto_w, CV_32FC1);
        if (crop_x2 > crop_x1 && crop_y2 > crop_y1)
        {
            const cv::Rect crop_rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
            low_res(crop_rect).copyTo(cropped(crop_rect));
        }

        cv::Mat input_mask;
        cv::resize(cropped, input_mask, cv::Size(letterbox.input_w, letterbox.input_h), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat original_mask;
        cv::resize(input_mask(content_rect), original_mask, cv::Size(letterbox.original_w, letterbox.original_h), 0.0,
                   0.0, cv::INTER_LINEAR);
        masks.push_back(std::move(original_mask));
    }
    return masks;
}

void printInstances(const std::vector<Instance> &instances, const std::vector<std::string> &labels)
{
    std::cout << "\nInstances: " << instances.size() << std::endl;
    for (size_t i = 0; i < instances.size(); ++i)
    {
        const auto &instance = instances[i];
        std::cout << std::fixed << std::setprecision(4) << "#" << i << " score=" << instance.score
                  << " class=" << instance.class_id << " (" << className(instance.class_id, labels) << ") box=["
                  << instance.x1 << ", " << instance.y1 << ", " << instance.x2 << ", " << instance.y2
                  << "] query=" << instance.query_index << std::endl;
    }
}

/**
 * @brief 将实例 mask 叠加到原图，并绘制检测框和类别标签。
 */
void drawInstances(const cv::Mat &image, const std::vector<Instance> &instances, const HostBuffer &mask_buffer,
                   const nvinfer1::Dims &mask_dims, const std::vector<std::string> &labels, float mask_threshold,
                   const fs::path &output_path)
{
    if (output_path.empty())
    {
        return;
    }
    if (mask_dims.nbDims != 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RF-DETR-Seg masks output must be 4D");
    }

    const auto *masks  = static_cast<const float *>(mask_buffer.data());
    const int   mask_h = static_cast<int>(mask_dims.d[2]);
    const int   mask_w = static_cast<int>(mask_dims.d[3]);

    cv::Mat visual;
    image.convertTo(visual, CV_32FC3);

    for (const auto &instance : instances)
    {
        const int       color_seed = instance.class_id * 37 + instance.query_index * 13;
        const cv::Vec3f color(static_cast<float>((color_seed * 3 + 80) % 255),
                              static_cast<float>((color_seed * 7 + 120) % 255),
                              static_cast<float>((color_seed * 11 + 160) % 255));
        const float    *mask_ptr = masks + static_cast<size_t>(instance.query_index) * mask_h * mask_w;
        const cv::Mat   low_res(mask_h, mask_w, CV_32FC1, const_cast<float *>(mask_ptr));
        cv::Mat         resized;
        cv::resize(low_res, resized, image.size(), 0.0, 0.0, cv::INTER_LINEAR);

        for (int y = 0; y < visual.rows; ++y)
        {
            auto       *pixel_row = visual.ptr<cv::Vec3f>(y);
            const auto *mask_row  = resized.ptr<float>(y);
            for (int x = 0; x < visual.cols; ++x)
            {
                if (mask_row[x] > mask_threshold)
                {
                    pixel_row[x] = pixel_row[x] * 0.55F + color * 0.45F;
                }
            }
        }
    }

    visual.convertTo(visual, CV_8UC3);
    for (const auto &instance : instances)
    {
        const int        color_seed = instance.class_id * 37 + instance.query_index * 13;
        const cv::Scalar color((color_seed * 3 + 80) % 255, (color_seed * 7 + 120) % 255,
                               (color_seed * 11 + 160) % 255);
        cv::rectangle(
            visual, cv::Point(static_cast<int>(std::round(instance.x1)), static_cast<int>(std::round(instance.y1))),
            cv::Point(static_cast<int>(std::round(instance.x2)), static_cast<int>(std::round(instance.y2))), color, 2);

        std::ostringstream label;
        label << className(instance.class_id, labels) << ' ' << std::fixed << std::setprecision(2) << instance.score;
        int            baseline  = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int      text_x    = static_cast<int>(std::round(instance.x1));
        const int      text_y    = std::max(text_size.height + 4, static_cast<int>(std::round(instance.y1)));
        cv::rectangle(
            visual,
            cv::Rect(text_x, text_y - text_size.height - 4, text_size.width + 4, text_size.height + baseline + 4),
            color, cv::FILLED);
        cv::putText(visual, label.str(), cv::Point(text_x + 2, text_y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    fs::create_directories(output_path.parent_path().empty() ? fs::path{"."} : output_path.parent_path());
    if (!cv::imwrite(output_path.string(), visual))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                             output_path.string().c_str());
    }
    std::cout << "Saved segmentation image to: " << fs::absolute(output_path).string() << std::endl;
}

/**
 * @brief 将 YOLOv8-Seg 已还原到原图尺寸的 mask logits 叠加到图像。
 */
void drawYoloInstances(const cv::Mat &image, const std::vector<Instance> &instances, const std::vector<cv::Mat> &masks,
                       const std::vector<std::string> &labels, float mask_threshold, const fs::path &output_path)
{
    if (output_path.empty())
    {
        return;
    }
    if (masks.size() != instances.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "YOLOv8-Seg mask count mismatch: got %zu masks for %zu instances", masks.size(),
                             instances.size());
    }

    cv::Mat visual;
    image.convertTo(visual, CV_32FC3);

    for (size_t i = 0; i < instances.size(); ++i)
    {
        const auto     &instance   = instances[i];
        const int       color_seed = instance.class_id * 37 + instance.query_index * 13;
        const cv::Vec3f color(static_cast<float>((color_seed * 3 + 80) % 255),
                              static_cast<float>((color_seed * 7 + 120) % 255),
                              static_cast<float>((color_seed * 11 + 160) % 255));
        for (int y = 0; y < visual.rows; ++y)
        {
            auto       *pixel_row = visual.ptr<cv::Vec3f>(y);
            const auto *mask_row  = masks[i].ptr<float>(y);
            for (int x = 0; x < visual.cols; ++x)
            {
                if (mask_row[x] > mask_threshold)
                {
                    pixel_row[x] = pixel_row[x] * 0.55F + color * 0.45F;
                }
            }
        }
    }

    visual.convertTo(visual, CV_8UC3);
    for (const auto &instance : instances)
    {
        const int        color_seed = instance.class_id * 37 + instance.query_index * 13;
        const cv::Scalar color((color_seed * 3 + 80) % 255, (color_seed * 7 + 120) % 255,
                               (color_seed * 11 + 160) % 255);
        cv::rectangle(
            visual, cv::Point(static_cast<int>(std::round(instance.x1)), static_cast<int>(std::round(instance.y1))),
            cv::Point(static_cast<int>(std::round(instance.x2)), static_cast<int>(std::round(instance.y2))), color, 2);

        std::ostringstream label;
        label << className(instance.class_id, labels) << ' ' << std::fixed << std::setprecision(2) << instance.score;
        int            baseline  = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int      text_x    = static_cast<int>(std::round(instance.x1));
        const int      text_y    = std::max(text_size.height + 4, static_cast<int>(std::round(instance.y1)));
        cv::rectangle(
            visual,
            cv::Rect(text_x, text_y - text_size.height - 4, text_size.width + 4, text_size.height + baseline + 4),
            color, cv::FILLED);
        cv::putText(visual, label.str(), cv::Point(text_x + 2, text_y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    fs::create_directories(output_path.parent_path().empty() ? fs::path{"."} : output_path.parent_path());
    if (!cv::imwrite(output_path.string(), visual))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                             output_path.string().c_str());
    }
    std::cout << "Saved segmentation image to: " << fs::absolute(output_path).string() << std::endl;
}

cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run image-only instance segmentation models");
    options.add_options()("model,m", "Model key, e.g. rfdetr_seg_nano or yolov8n_seg (required)",
                          cxxopts::value<std::string>())("weights-file,w", "Native .wts file (required)",
                                                         cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "label-file,l", "Optional class label file", cxxopts::value<std::string>()->default_value(""))(
        "output-image,o", "Path to save mask overlay", cxxopts::value<std::string>()->default_value(""))(
        "input-size", "Optional square network input size; default uses the registered model size",
        cxxopts::value<int>())("conf-threshold", "Instance confidence threshold",
                               cxxopts::value<float>()->default_value("0.35"))(
        "num-classes", "Number of YOLO classes; ignored by RF-DETR-Seg", cxxopts::value<int>()->default_value("80"))(
        "mask-threshold", "Mask logit threshold", cxxopts::value<float>()->default_value("0.0"))(
        "nms-threshold", "Class-wise NMS IoU threshold", cxxopts::value<float>()->default_value("0.50"))(
        "max-instances", "Maximum instances to draw", cxxopts::value<int>()->default_value("20"))(
        "runtime", "Model runtime: gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
        cxxopts::value<std::string>()->default_value("tensorrt:0"))(
        "warmup", "Warmup iterations before timing", cxxopts::value<int>()->default_value("0"))(
        "repeat", "Timed inference iterations", cxxopts::value<int>()->default_value("1"))("h,help", "Show help");
    return options;
}

Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << kDefaultImagePath.generic_string() << std::endl;
        std::cout << "Default labels: " << kDefaultLabelPath.generic_string() << std::endl;
        throw HelpRequested{};
    }
    if (!result.count("model") || !result.count("weights-file"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--model and --weights-file are required");
    }

    Arguments args;
    args.model_name          = result["model"].as<std::string>();
    args.weights_file        = result["weights-file"].as<std::string>();
    args.image_path          = result["image-path"].as<std::string>();
    args.label_file          = result["label-file"].as<std::string>();
    args.output_image        = result["output-image"].as<std::string>();
    args.input_size_explicit = result.count("input-size") != 0U;
    args.input_size          = args.input_size_explicit ? result["input-size"].as<int>() : 0;
    args.num_classes         = result["num-classes"].as<int>();
    args.conf_threshold      = result["conf-threshold"].as<float>();
    args.mask_threshold      = result["mask-threshold"].as<float>();
    args.nms_threshold       = result["nms-threshold"].as<float>();
    args.max_instances       = result["max-instances"].as<int>();
    args.runtime             = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    args.warmup              = result["warmup"].as<int>();
    args.repeat              = result["repeat"].as<int>();
    args.family              = modelFamily(args.model_name);

    if (args.runtime.backend() != irt::model::ModelRuntime::Backend::TensorRT || !args.runtime.isGpu())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Segmentation sample currently requires TensorRT/GPU");
    }
    if (args.input_size_explicit && args.input_size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--input-size must be positive");
    }
    if (args.family == SegmentationFamily::YOLOV8 && args.input_size_explicit && args.input_size % 32 != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "YOLO input size must be divisible by 32");
    }
    if (args.num_classes <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--num-classes must be positive");
    }
    if (args.conf_threshold < 0.0F || args.conf_threshold > 1.0F || args.nms_threshold < 0.0F
        || args.nms_threshold > 1.0F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "thresholds must be in [0, 1]");
    }
    if (args.max_instances <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--max-instances must be positive");
    }
    if (args.warmup < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--warmup must be >= 0");
    }
    if (args.repeat <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--repeat must be > 0");
    }
    return args;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const Arguments args = parseArguments(argc, argv);
        if (!irt::model::isSupportedModel(args.model_name))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", args.model_name.c_str());
        }

        const fs::path project_root
            = irt::util::findProjectRoot(argv[0], {kDefaultImagePath, kDefaultLabelPath}, __FILE__);
        const fs::path image_path  = args.image_path.empty() ? (project_root / kDefaultImagePath) : args.image_path;
        const fs::path label_path  = args.label_file.empty() ? (project_root / kDefaultLabelPath) : args.label_file;
        const fs::path output_path = args.output_image.empty()
                                       ? (fs::current_path() / (args.model_name + "_segmentation.jpg"))
                                       : args.output_image;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setRuntime(args.runtime);
        if (args.input_size_explicit)
        {
            config->setInputShape(nvinfer1::Dims4{1, 3, args.input_size, args.input_size});
        }
        if (args.family == SegmentationFamily::YOLOV8)
        {
            config->setNumClasses(args.num_classes);
        }

        auto model = irt::model::CreateModel(args.model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 args.model_name.c_str());
        }
        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or loading " << modelFamilyName(args.family) << " model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(args.weights_file.string());
        const auto build_end = Clock::now();
        std::cout << "Model loaded successfully." << std::endl;

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto   input_names      = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        const auto   output_names     = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        const size_t expected_outputs = args.family == SegmentationFamily::RFDETR ? 3U : 4U;
        if (input_names.size() != 1 || output_names.size() != expected_outputs)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "%s sample expects one image input and %zu outputs", modelFamilyName(args.family),
                                 expected_outputs);
        }
        size_t                dets_index   = 0;
        size_t                labels_index = 0;
        size_t                masks_index  = 0;
        size_t                proto_index  = 0;
        std::array<size_t, 3> yolo_branch_indices{0, 1, 2};
        if (args.family == SegmentationFamily::RFDETR)
        {
            dets_index   = requireOutputIndex(output_names, "dets");
            labels_index = requireOutputIndex(output_names, "labels");
            masks_index  = requireOutputIndex(output_names, "masks");
        }
        else
        {
            yolo_branch_indices
                = {requireOutputIndex(output_names, "output0"), requireOutputIndex(output_names, "output1"),
                   requireOutputIndex(output_names, "output2")};
            proto_index = requireOutputIndex(output_names, "proto");
        }

        const auto input_dims       = model->tensorShape(input_names.front());
        const auto preprocess_start = Clock::now();
        if (input_dims.nbDims != 4 || input_dims.d[1] != 3 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s expects Nx3xHxW input, got %s",
                                 modelFamilyName(args.family), dimsToCsv(input_dims).c_str());
        }
        LetterboxInfo letterbox;
        auto          input_tensor   = args.family == SegmentationFamily::RFDETR
                                         ? preprocessRFDETR(image, input_dims)
                                         : preprocessLetterbox(image, static_cast<int>(input_dims.d[3]),
                                                               static_cast<int>(input_dims.d[2]), letterbox);
        const auto    preprocess_end = Clock::now();

        const auto   stream = model->resolveExecutionStream();
        DeviceBuffer device_input(elementCount(input_dims), nvinfer1::DataType::kFLOAT);

        std::vector<nvinfer1::Dims> output_dims;
        std::vector<DeviceBuffer>   device_outputs;
        std::vector<HostBuffer>     host_outputs;
        output_dims.reserve(output_names.size());
        device_outputs.reserve(output_names.size());
        host_outputs.reserve(output_names.size());
        for (const auto &output_name : output_names)
        {
            const auto dims = model->tensorShape(output_name);
            const auto type = model->tensorDataType(output_name);
            if (type != nvinfer1::DataType::kFLOAT)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Segmentation sample expects float32 outputs");
            }
            output_dims.push_back(dims);
            device_outputs.emplace_back(elementCount(dims), type);
            host_outputs.emplace_back(elementCount(dims), type);
            std::cout << "Output " << output_name << " dims=[" << dimsToCsv(dims) << "]" << std::endl;
        }

        std::vector<void *> buffers;
        buffers.reserve(1 + output_names.size());
        buffers.push_back(device_input.data());
        for (auto &buffer : device_outputs)
        {
            buffers.push_back(buffer.data());
        }

        auto run_inference_once = [&]() -> IterationTiming
        {
            IterationTiming timing;

            const auto h2d_start = Clock::now();
            checkCuda(cudaMemcpyAsync(device_input.data(), input_tensor.data(), device_input.sizeBytes(),
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(RF-DETR-Seg input)");
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(RF-DETR-Seg input)");
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->infer(buffers, stream, true);
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(RF-DETR-Seg inference)");
            const auto inference_end = Clock::now();
            timing.inference_ms      = elapsedMs(inference_start, inference_end);

            const auto d2h_start = Clock::now();
            for (size_t i = 0; i < host_outputs.size(); ++i)
            {
                checkCuda(cudaMemcpyAsync(host_outputs[i].data(), device_outputs[i].data(), host_outputs[i].sizeBytes(),
                                          cudaMemcpyDeviceToHost, stream),
                          "cudaMemcpyAsync(RF-DETR-Seg output)");
            }
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(RF-DETR-Seg output)");
            const auto d2h_end = Clock::now();
            timing.d2h_ms      = elapsedMs(d2h_start, d2h_end);
            return timing;
        };

        std::cout << "Running inference warmup=" << args.warmup << ", repeat=" << args.repeat << "..." << std::endl;
        for (int i = 0; i < args.warmup; ++i)
        {
            (void)run_inference_once();
        }

        std::vector<double> h2d_times_ms;
        std::vector<double> inference_times_ms;
        std::vector<double> d2h_times_ms;
        std::vector<double> end_to_end_times_ms;
        h2d_times_ms.reserve(static_cast<size_t>(args.repeat));
        inference_times_ms.reserve(static_cast<size_t>(args.repeat));
        d2h_times_ms.reserve(static_cast<size_t>(args.repeat));
        end_to_end_times_ms.reserve(static_cast<size_t>(args.repeat));
        const auto infer_start = Clock::now();
        for (int i = 0; i < args.repeat; ++i)
        {
            const auto timing = run_inference_once();
            h2d_times_ms.push_back(timing.h2d_ms);
            inference_times_ms.push_back(timing.inference_ms);
            d2h_times_ms.push_back(timing.d2h_ms);
            end_to_end_times_ms.push_back(timing.totalMs());
        }
        const auto infer_end = Clock::now();

        const auto            post_start = Clock::now();
        const auto            labels     = readLabels(label_path);
        std::vector<Instance> instances;
        if (args.family == SegmentationFamily::RFDETR)
        {
            instances
                = decodeRFDETRSegOutputs(host_outputs[dets_index], output_dims[dets_index], host_outputs[labels_index],
                                         output_dims[labels_index], output_dims[masks_index], args, image.size());
        }
        else
        {
            instances = decodeYoloV8SegOutputs(host_outputs, output_dims, yolo_branch_indices, args, letterbox);
        }
        printInstances(instances, labels);
        if (args.family == SegmentationFamily::RFDETR)
        {
            drawInstances(image, instances, host_outputs[masks_index], output_dims[masks_index], labels,
                          args.mask_threshold, output_path);
        }
        else
        {
            const auto masks
                = buildYoloV8SegMasks(instances, host_outputs[proto_index], output_dims[proto_index], letterbox);
            drawYoloInstances(image, instances, masks, labels, args.mask_threshold, output_path);
        }
        const auto post_end = Clock::now();

        std::cout << "Runtime: " << args.runtime.toString() << std::endl;
        const auto h2d_stats        = summarizeTimings(h2d_times_ms);
        const auto inference_stats  = summarizeTimings(inference_times_ms);
        const auto d2h_stats        = summarizeTimings(d2h_times_ms);
        const auto end_to_end_stats = summarizeTimings(end_to_end_times_ms);
        std::cout << "Timing: build_or_load=" << elapsedMs(build_start, build_end)
                  << " ms, preprocess=" << elapsedMs(preprocess_start, preprocess_end) << " ms";
        printTimingStats("h2d", h2d_stats);
        printTimingStats("inference", inference_stats);
        printTimingStats("d2h", d2h_stats);
        printTimingStats("end_to_end", end_to_end_stats);
        std::cout << ", timed_loop_wall=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(post_start, post_end) << " ms" << std::endl;
        return 0;
    }
    catch (const HelpRequested &)
    {
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
