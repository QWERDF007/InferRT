#include <SampleSupport.hpp>
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
#include <cstdint>
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

using Clock = irt::util::TimingClock;
using irt::model::checkCuda;
using irt::model::DeviceBuffer;
using irt::model::dimsToCsv;
using irt::model::elementCount;
using irt::model::HostBuffer;
using irt::samples::HelpRequested;
using irt::util::elapsedMs;
using irt::util::printTimingStats;
using irt::util::summarizeTimings;
using irt::util::TimingStats;
using irt::util::toLower;

constexpr std::array<int, 3>                  kYoloStrides{8, 16, 32};
constexpr std::array<std::array<float, 6>, 3> kDefaultYoloV5Anchors{
    {
     {10.0F, 13.0F, 16.0F, 30.0F, 33.0F, 23.0F},
     {30.0F, 61.0F, 62.0F, 45.0F, 59.0F, 119.0F},
     {116.0F, 90.0F, 156.0F, 198.0F, 373.0F, 326.0F},
     }
};
const fs::path kDefaultImagePath{"assets/pics/dog.jpg"};
const fs::path kDefaultLabelPath{"assets/coco80.names"};

enum class DetectionFamily
{
    YOLO,
    RFDETR,
};

/**
 * @brief detection sample 的命令行参数集合。
 */
struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    fs::path                 image_path;
    fs::path                 label_file;
    fs::path                 output_image;
    std::string              legacy_anchors;
    int                      input_size{640};
    bool                     input_size_explicit{false};
    int                      batch_min{1};
    int                      batch_opt{1};
    int                      batch_max{1};
    int                      batch_size{1};
    int                      num_classes{80};
    bool                     num_classes_explicit{false};
    int                      max_detections{100};
    float                    conf_threshold{0.25F};
    float                    nms_threshold{0.45F};
    DetectionFamily          family{DetectionFamily::YOLO};
    irt::model::ModelRuntime runtime{};
    irt::model::ModelPrecision precision{irt::model::ModelPrecision::FP32};
    int                      warmup{0};
    int                      repeat{1};
};

/**
 * @brief letterbox 预处理的缩放与填充信息。
 */
struct LetterboxInfo
{
    float scale{1.0F}; ///< 原图缩放到网络输入的比例。
    int   pad_x{0};    ///< 左侧填充像素。
    int   pad_y{0};    ///< 顶部填充像素。
    int   original_w{0};
    int   original_h{0};
    int   input_w{0};
    int   input_h{0};
};

/**
 * @brief 单个检测框，坐标均为原图像素坐标。
 */
struct Detection
{
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
    float confidence{0.0F};
    int   class_id{0};
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

DetectionFamily modelFamily(const std::string &model_name)
{
    const std::string normalized = toLower(model_name);
    if (normalized.rfind("rfdetr", 0) == 0 && normalized.find("_seg") == std::string::npos
        && normalized.find("-seg") == std::string::npos)
    {
        return DetectionFamily::RFDETR;
    }
    return DetectionFamily::YOLO;
}

bool isRFDETRSegModelName(const std::string &model_name)
{
    const std::string normalized = toLower(model_name);
    return normalized.rfind("rfdetr", 0) == 0
        && (normalized.find("_seg") != std::string::npos || normalized.find("-seg") != std::string::npos);
}

/**
 * @brief 判断权重文件是否为 ONNX 图。
 *
 * RF-DETR 家族使用 `.onnx` 权重时，sample 会路由到通用 ONNXModel
 * （onnx -> TensorRT parser），而不是原生 `.wts` 逐层构建器。
 */
bool isOnnxWeightsFile(const fs::path &weights_file)
{
    return toLower(weights_file.extension().string()) == ".onnx";
}

const char *modelFamilyName(DetectionFamily family)
{
    return family == DetectionFamily::RFDETR ? "RF-DETR" : "YOLO";
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

/**
 * @brief 将网络输入坐标映射回 letterbox 前的原图坐标。
 */
void remapToOriginalImage(Detection &det, const LetterboxInfo &info)
{
    det.x1 = clampFloat((det.x1 - static_cast<float>(info.pad_x)) / info.scale, 0.0F,
                        static_cast<float>(info.original_w - 1));
    det.y1 = clampFloat((det.y1 - static_cast<float>(info.pad_y)) / info.scale, 0.0F,
                        static_cast<float>(info.original_h - 1));
    det.x2 = clampFloat((det.x2 - static_cast<float>(info.pad_x)) / info.scale, 0.0F,
                        static_cast<float>(info.original_w - 1));
    det.y2 = clampFloat((det.y2 - static_cast<float>(info.pad_y)) / info.scale, 0.0F,
                        static_cast<float>(info.original_h - 1));
}

/**
 * @brief 解析旧版 YOLOv5 anchor；未提供时使用 COCO 默认 anchor。
 */
std::array<std::array<float, 6>, 3> parseLegacyAnchors(const std::string &text)
{
    if (text.empty())
    {
        return kDefaultYoloV5Anchors;
    }

    std::string normalized = text;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream iss(normalized);

    std::array<float, 18> flat{};
    for (float &value : flat)
    {
        if (!(iss >> value))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "--legacy-anchors expects 18 comma-separated numbers");
        }
    }

    std::array<std::array<float, 6>, 3> anchors{};
    for (size_t scale = 0; scale < anchors.size(); ++scale)
    {
        for (size_t i = 0; i < anchors[scale].size(); ++i)
        {
            anchors[scale][i] = flat[scale * anchors[scale].size() + i];
        }
    }
    return anchors;
}

/**
 * @brief 构造命令行选项。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run InferRT YOLO/RF-DETR detection models on a single image");
    options.add_options()("model,m", "Detection model name, e.g. yolov8n or rfdetr_nano (required)",
                          cxxopts::value<std::string>())(
        "weights-file,w", "Weights/model file (.wts, .onnx or OpenVINO IR) (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "label-file,l", "Class label file", cxxopts::value<std::string>()->default_value(""))(
        "output-image,o", "Optional path to save image with boxes", cxxopts::value<std::string>()->default_value(""))(
        "input-size", "Square network input size; YOLO defaults to 640, RF-DETR uses its registered default",
        cxxopts::value<int>())("batch-min", "Minimum native YOLO TensorRT batch profile size",
                               cxxopts::value<int>()->default_value("1"))(
        "batch-opt", "Optimal native YOLO TensorRT batch profile size", cxxopts::value<int>()->default_value("1"))(
        "batch-max", "Maximum native YOLO TensorRT batch profile size", cxxopts::value<int>()->default_value("1"))(
        "batch-size", "Inference batch size; RF-DETR native builds a dynamic 1..N profile, ONNX graphs are static",
        cxxopts::value<int>()->default_value("1"))(
        "classes", "Number of classes used by the YOLO head; for RF-DETR the foreground class count",
        cxxopts::value<int>()->default_value("80"))(
        "conf-threshold", "Confidence threshold", cxxopts::value<float>()->default_value("0.25"))(
        "nms-threshold", "Class-wise NMS IoU threshold", cxxopts::value<float>()->default_value("0.45"))(
        "max-detections", "Maximum detections printed after NMS", cxxopts::value<int>()->default_value("100"))(
        "legacy-anchors", "Legacy YOLOv5 anchors as 18 comma-separated numbers; empty uses COCO defaults",
        cxxopts::value<std::string>()->default_value(""))(
        "runtime", "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
        cxxopts::value<std::string>()->default_value("tensorrt:0"))(
        "precision", "TensorRT build precision: fp32 or fp16", cxxopts::value<std::string>()->default_value("fp32"));
    irt::samples::addTimingOptions(options, "Timed inference iterations");
    options.add_options()("h,help", "Show help");
    return options;
}

/**
 * @brief 解析并校验命令行参数。
 */
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
    args.model_name = result["model"].as<std::string>();
    if (isRFDETRSegModelName(args.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "RF-DETR segmentation models must be run with inferrt_sample_segmentation");
    }
    args.family              = modelFamily(args.model_name);
    args.weights_file        = result["weights-file"].as<std::string>();
    args.image_path          = result["image-path"].as<std::string>();
    args.label_file          = result["label-file"].as<std::string>();
    args.output_image        = result["output-image"].as<std::string>();
    args.input_size_explicit = result.count("input-size") != 0U;
    args.input_size          = args.input_size_explicit ? result["input-size"].as<int>() : 640;
    args.batch_min           = result["batch-min"].as<int>();
    args.batch_opt           = result["batch-opt"].as<int>();
    args.batch_max           = result["batch-max"].as<int>();
    args.batch_size          = result["batch-size"].as<int>();
    args.num_classes         = result["classes"].as<int>();
    args.num_classes_explicit = result.count("classes") != 0U;
    args.conf_threshold      = result["conf-threshold"].as<float>();
    args.nms_threshold       = result["nms-threshold"].as<float>();
    args.max_detections      = result["max-detections"].as<int>();
    args.legacy_anchors      = result["legacy-anchors"].as<std::string>();
    args.runtime             = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    const std::string precision_name = toLower(result["precision"].as<std::string>());
    if (precision_name == "fp16")
    {
        args.precision = irt::model::ModelPrecision::FP16;
    }
    else if (precision_name != "fp32")
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--precision must be fp32 or fp16");
    }
    const auto timing        = irt::samples::parseTimingOptions(result);
    args.warmup              = timing.warmup;
    args.repeat              = timing.repeat;

    if (args.family == DetectionFamily::RFDETR)
    {
        if (result.count("conf-threshold") == 0U)
        {
            args.conf_threshold = 0.35F;
        }
        if (result.count("nms-threshold") == 0U)
        {
            args.nms_threshold = 0.50F;
        }
    }

    if (args.family == DetectionFamily::YOLO && (args.input_size <= 0 || args.input_size % 32 != 0))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--input-size must be positive and divisible by 32");
    }
    if (args.family == DetectionFamily::RFDETR && args.input_size_explicit && args.input_size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--input-size must be positive");
    }
    if (args.batch_min <= 0 || args.batch_min > args.batch_opt || args.batch_opt > args.batch_max)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "batch profile must satisfy 1 <= --batch-min <= --batch-opt <= --batch-max");
    }
    if (args.batch_size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--batch-size must be positive");
    }
    if (!isOnnxWeightsFile(args.weights_file) && args.batch_size > args.batch_max)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--batch-size must not exceed --batch-max for the native profile");
    }
    if (args.family == DetectionFamily::RFDETR && args.batch_size > 1 && !isOnnxWeightsFile(args.weights_file)
        && !args.input_size_explicit)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "RF-DETR --batch-size > 1 requires an explicit --input-size");
    }
    if (args.batch_min != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "the single-image detection sample requires --batch-min=1");
    }
    if (args.num_classes <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--classes must be positive");
    }
    if (args.conf_threshold < 0.0F || args.conf_threshold > 1.0F || args.nms_threshold < 0.0F
        || args.nms_threshold > 1.0F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "thresholds must be in [0, 1]");
    }
    if (args.max_detections <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--max-detections must be positive");
    }
    return args;
}

/**
 * @brief 使用 YOLO letterbox 规则预处理输入图像。
 */
std::vector<float> preprocessLetterbox(const cv::Mat &bgr_image, int input_w, int input_h, LetterboxInfo &info)
{
    irt::PreprocessSpec spec;
    spec.input_width     = input_w;
    spec.input_height    = input_h;
    spec.input_channels  = 3;
    spec.source_channels = 3;
    spec.src_color       = irt::ColorFormat::BGR;
    spec.dst_color       = irt::ColorFormat::RGB;
    spec.padding_mode    = irt::PaddingMode::Letterbox;
    spec.pad_value       = 114.0F;
    spec.mean            = {0.0F, 0.0F, 0.0F};
    spec.stddev          = {1.0F, 1.0F, 1.0F};
    spec.scale            = 1.0F / 255.0F;

    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(bgr_image, spec);
    info.original_w   = result.geometry.original_width;
    info.original_h   = result.geometry.original_height;
    info.input_w      = input_w;
    info.input_h      = input_h;
    info.scale        = result.geometry.scale;
    info.pad_x        = result.geometry.pad_left;
    info.pad_y        = result.geometry.pad_top;
    return irt::model::ImageNetUtil::imageToTensorCHW(result.image);
}

/**
 * @brief 使用 RF-DETR 官方 ImageNet 归一化预处理，并按 NCHW 展平。
 */
std::vector<float> preprocessRFDETR(const cv::Mat &bgr_image, const irt::Shape &input_dims)
{
    if (input_dims.rank() != 4 || input_dims[1] != 3 || input_dims[2] <= 0 || input_dims[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RF-DETR expects Nx3xHxW input, got %s",
                             dimsToCsv(input_dims).c_str());
    }

    const cv::Mat preprocessed = irt::model::ImageNetUtil::preprocess(
        bgr_image, cv::Size(static_cast<int>(input_dims[3]), static_cast<int>(input_dims[2])));
    return irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
}

float intersectionOverUnion(const Detection &a, const Detection &b)
{
    const float x1         = std::max(a.x1, b.x1);
    const float y1         = std::max(a.y1, b.y1);
    const float x2         = std::min(a.x2, b.x2);
    const float y2         = std::min(a.y2, b.y2);
    const float inter_w    = std::max(0.0F, x2 - x1);
    const float inter_h    = std::max(0.0F, y2 - y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float denom  = area_a + area_b - inter_area;
    return denom <= 0.0F ? 0.0F : inter_area / denom;
}

/**
 * @brief 按类别执行 NMS，避免不同类别之间互相抑制。
 */
std::vector<Detection> nonMaximumSuppression(std::vector<Detection> detections, float threshold, int max_detections)
{
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) { return a.confidence > b.confidence; });

    std::vector<Detection> kept;
    std::vector<bool>      removed(detections.size(), false);
    kept.reserve(std::min<int>(max_detections, static_cast<int>(detections.size())));

    for (size_t i = 0; i < detections.size() && static_cast<int>(kept.size()) < max_detections; ++i)
    {
        if (removed[i])
        {
            continue;
        }
        kept.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (!removed[j] && detections[i].class_id == detections[j].class_id
                && intersectionOverUnion(detections[i], detections[j]) > threshold)
            {
                removed[j] = true;
            }
        }
    }
    return kept;
}

/**
 * @brief 判断输出是否为旧版 YOLOv5 raw anchor head。
 */
bool isLegacyYoloV5Output(const irt::Shape &dims, int num_classes)
{
    return dims.rank() == 4 && dims[0] > 0 && dims[1] == 3 * (num_classes + 5) && dims[2] > 0 && dims[3] > 0;
}

/**
 * @brief 判断输出是否为 YOLOv5u/YOLOv8 DFL head。
 */
bool isDflOutput(const irt::Shape &dims, int num_classes)
{
    return dims.rank() == 3 && dims[0] > 0 && dims[1] == num_classes + 4 && dims[2] > 0;
}

/**
 * @brief 解码旧版 YOLOv5 raw anchor head。
 */
void decodeLegacyYoloV5Output(const float *data, const irt::Shape &dims, int branch_index, int num_classes,
                              float conf_threshold, const LetterboxInfo &letterbox,
                              const std::array<std::array<float, 6>, 3> &anchors, std::vector<Detection> &detections)
{
    const int channels = static_cast<int>(dims[1]);
    const int grid_h   = static_cast<int>(dims[2]);
    const int grid_w   = static_cast<int>(dims[3]);
    const int info_len = num_classes + 5;
    if (channels != 3 * info_len)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected legacy YOLOv5 output shape: %s",
                             dimsToCsv(dims).c_str());
    }

    const float stride_x = static_cast<float>(letterbox.input_w) / static_cast<float>(grid_w);
    const float stride_y = static_cast<float>(letterbox.input_h) / static_cast<float>(grid_h);

    for (int anchor_idx = 0; anchor_idx < 3; ++anchor_idx)
    {
        const float anchor_w = anchors[branch_index][static_cast<size_t>(2 * anchor_idx)];
        const float anchor_h = anchors[branch_index][static_cast<size_t>(2 * anchor_idx + 1)];
        for (int b = 0; b < dims[0]; ++b)
        {
            const size_t batch_base = static_cast<size_t>(b) * static_cast<size_t>(channels) * grid_h * grid_w;
            for (int y = 0; y < grid_h; ++y)
            {
                for (int x = 0; x < grid_w; ++x)
                {
                    const int grid_index = y * grid_w + x;
                    auto      valueAt    = [&](int attr)
                    {
                        const int channel = anchor_idx * info_len + attr;
                        return data[batch_base + static_cast<size_t>(channel) * grid_h * grid_w + grid_index];
                    };

                    const float objectness = sigmoid(valueAt(4));
                    if (objectness < conf_threshold)
                    {
                        continue;
                    }

                    int   best_class = 0;
                    float best_score = 0.0F;
                    for (int cls = 0; cls < num_classes; ++cls)
                    {
                        const float score = sigmoid(valueAt(5 + cls));
                        if (score > best_score)
                        {
                            best_score = score;
                            best_class = cls;
                        }
                    }

                    const float confidence = objectness * best_score;
                    if (confidence < conf_threshold)
                    {
                        continue;
                    }

                    const float bx = (static_cast<float>(x) - 0.5F + 2.0F * sigmoid(valueAt(0))) * stride_x;
                    const float by = (static_cast<float>(y) - 0.5F + 2.0F * sigmoid(valueAt(1))) * stride_y;
                    const float bw = std::pow(2.0F * sigmoid(valueAt(2)), 2.0F) * anchor_w;
                    const float bh = std::pow(2.0F * sigmoid(valueAt(3)), 2.0F) * anchor_h;

                    Detection det;
                    det.x1         = bx - bw * 0.5F;
                    det.y1         = by - bh * 0.5F;
                    det.x2         = bx + bw * 0.5F;
                    det.y2         = by + bh * 0.5F;
                    det.confidence = confidence;
                    det.class_id   = best_class;
                    remapToOriginalImage(det, letterbox);
                    detections.push_back(det);
                }
            }
        }
    }
}

/**
 * @brief 解码 YOLOv5u/YOLOv8 的 DFL distance + class head。
 */
void decodeDflOutput(const float *data, const irt::Shape &dims, int branch_index, int num_classes,
                     float conf_threshold, const LetterboxInfo &letterbox, std::vector<Detection> &detections)
{
    const int stride = kYoloStrides[static_cast<size_t>(branch_index)];
    const int grid_h = letterbox.input_h / stride;
    const int grid_w = letterbox.input_w / stride;
    const int grid   = grid_h * grid_w;
    if (dims[2] != grid)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "DFL output grid mismatch for branch %d: expected %d, got %d", branch_index, grid,
                             dims[2]);
    }

    for (int b = 0; b < dims[0]; ++b)
    {
        const size_t batch_base = static_cast<size_t>(b) * static_cast<size_t>(num_classes + 4) * grid;
        for (int idx = 0; idx < grid; ++idx)
        {
            int   best_class = 0;
            float best_score = 0.0F;
            for (int cls = 0; cls < num_classes; ++cls)
            {
                const float score = sigmoid(data[batch_base + static_cast<size_t>(4 + cls) * grid + idx]);
                if (score > best_score)
                {
                    best_score = score;
                    best_class = cls;
                }
            }
            if (best_score < conf_threshold)
            {
                continue;
            }

            const int   x        = idx % grid_w;
            const int   y        = idx / grid_w;
            const float anchor_x = static_cast<float>(x) + 0.5F;
            const float anchor_y = static_cast<float>(y) + 0.5F;
            const float left     = data[batch_base + 0 * grid + idx];
            const float top      = data[batch_base + 1 * grid + idx];
            const float right    = data[batch_base + 2 * grid + idx];
            const float bottom   = data[batch_base + 3 * grid + idx];

            Detection det;
            det.x1         = (anchor_x - left) * static_cast<float>(stride);
            det.y1         = (anchor_y - top) * static_cast<float>(stride);
            det.x2         = (anchor_x + right) * static_cast<float>(stride);
            det.y2         = (anchor_y + bottom) * static_cast<float>(stride);
            det.confidence = best_score;
            det.class_id   = best_class;
            remapToOriginalImage(det, letterbox);
            detections.push_back(det);
        }
    }
}

/**
 * @brief 根据模型输出形态选择 YOLO 后处理路径。
 */
std::vector<Detection> postprocessYoloOutputs(const std::vector<HostBuffer>     &outputs,
                                              const std::vector<irt::Shape> &output_dims, const Arguments &args,
                                              const LetterboxInfo &letterbox)
{
    if (outputs.size() != 3 || output_dims.size() != 3)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "YOLO detection sample expects 3 outputs");
    }

    std::vector<Detection> detections;
    const bool             dfl_head    = isDflOutput(output_dims.front(), args.num_classes);
    const bool             legacy_head = isLegacyYoloV5Output(output_dims.front(), args.num_classes);
    const auto             anchors     = parseLegacyAnchors(args.legacy_anchors);

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        const auto *data = static_cast<const float *>(outputs[i].data());
        if (dfl_head)
        {
            if (!isDflOutput(output_dims[i], args.num_classes))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Mixed YOLO output head types");
            }
            decodeDflOutput(data, output_dims[i], static_cast<int>(i), args.num_classes, args.conf_threshold, letterbox,
                            detections);
        }
        else if (legacy_head)
        {
            if (!isLegacyYoloV5Output(output_dims[i], args.num_classes))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Mixed YOLO output head types");
            }
            decodeLegacyYoloV5Output(data, output_dims[i], static_cast<int>(i), args.num_classes, args.conf_threshold,
                                     letterbox, anchors, detections);
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Unsupported YOLO output shape: %s. Expected DFL [1,4+classes,grid] or legacy "
                                 "[1,3*(classes+5),H,W]",
                                 dimsToCsv(output_dims.front()).c_str());
        }
    }

    return nonMaximumSuppression(std::move(detections), args.nms_threshold, args.max_detections);
}

/**
 * @brief 解码 RF-DETR 原生 TensorRT 输出，boxes 为归一化 cxcywh，logits 为类别分数。
 */
std::vector<Detection> postprocessRFDETROutputs(const std::vector<HostBuffer>     &outputs,
                                                const std::vector<irt::Shape> &output_dims, const Arguments &args,
                                                const cv::Size &image_size)
{
    if (outputs.size() < 2 || output_dims.size() < 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "RF-DETR detection sample expects dets/labels outputs");
    }

    const auto &boxes_dims  = output_dims[0];
    const auto &logits_dims = output_dims[1];
    if (boxes_dims.rank() != 3 || logits_dims.rank() != 3 || boxes_dims[1] != logits_dims[1]
        || boxes_dims[2] != 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected RF-DETR output shapes: dets=%s labels=%s",
                             dimsToCsv(boxes_dims).c_str(), dimsToCsv(logits_dims).c_str());
    }

    const auto *boxes   = static_cast<const float *>(outputs[0].data());
    const auto *logits  = static_cast<const float *>(outputs[1].data());
    const int   batch   = static_cast<int>(boxes_dims[0]);
    const int   queries = static_cast<int>(boxes_dims[1]);
    const int   classes = static_cast<int>(logits_dims[2]);

    std::vector<Detection> detections;
    for (int b = 0; b < batch; ++b)
    {
        for (int q = 0; q < queries; ++q)
        {
            const float cx = boxes[(b * queries + q) * 4 + 0] * static_cast<float>(image_size.width);
            const float cy = boxes[(b * queries + q) * 4 + 1] * static_cast<float>(image_size.height);
            const float w  = boxes[(b * queries + q) * 4 + 2] * static_cast<float>(image_size.width);
            const float h  = boxes[(b * queries + q) * 4 + 3] * static_cast<float>(image_size.height);

            Detection base;
            base.x1 = clampFloat(cx - 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
            base.y1 = clampFloat(cy - 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));
            base.x2 = clampFloat(cx + 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
            base.y2 = clampFloat(cy + 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));

            for (int cls = 0; cls < classes; ++cls)
            {
                const float confidence = sigmoid(logits[(b * queries + q) * classes + cls]);
                if (confidence < args.conf_threshold)
                {
                    continue;
                }

                auto det       = base;
                det.confidence = confidence;
                det.class_id   = cls;
                detections.push_back(det);
            }
        }
    }

    return nonMaximumSuppression(std::move(detections), args.nms_threshold, args.max_detections);
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
 * @brief 打印检测结果。
 */
void printDetections(const std::vector<Detection> &detections, const std::vector<std::string> &labels)
{
    std::cout << "\nDetections: " << detections.size() << std::endl;
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto &det = detections[i];
        std::cout << std::fixed << std::setprecision(4) << "#" << i << " confidence=" << det.confidence
                  << " class=" << det.class_id << " (" << className(det.class_id, labels) << ") box=[" << det.x1 << ", "
                  << det.y1 << ", " << det.x2 << ", " << det.y2 << "]" << std::endl;
    }
}

/**
 * @brief 将检测框绘制到原图并保存。
 */
void drawDetections(const cv::Mat &image, const std::vector<Detection> &detections,
                    const std::vector<std::string> &labels, const fs::path &output_path)
{
    if (output_path.empty())
    {
        return;
    }

    cv::Mat visual = image.clone();
    for (const auto &det : detections)
    {
        const int  color_seed = det.class_id * 37;
        cv::Scalar color((color_seed * 3) % 255, (color_seed * 7 + 80) % 255, (color_seed * 11 + 160) % 255);
        cv::rectangle(visual, cv::Point(static_cast<int>(std::round(det.x1)), static_cast<int>(std::round(det.y1))),
                      cv::Point(static_cast<int>(std::round(det.x2)), static_cast<int>(std::round(det.y2))), color, 2);

        std::ostringstream label;
        label << className(det.class_id, labels) << ' ' << std::fixed << std::setprecision(2) << det.confidence;
        int            baseline  = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int      text_x    = static_cast<int>(std::round(det.x1));
        const int      text_y    = std::max(text_size.height + 4, static_cast<int>(std::round(det.y1)));
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
    std::cout << "Saved detection image to: " << fs::absolute(output_path).string() << std::endl;
}

} // namespace

/**
 * @brief 运行单张图片检测模型，并在 sample 层完成解码与 NMS。
 */
int main(int argc, char *argv[])
{
    try
    {
        const Arguments args = parseArguments(argc, argv);
        if (!irt::model::isSupportedModel(args.model_name))
        {
            std::cerr << "Unsupported model: " << args.model_name << std::endl;
            return -1;
        }
        const bool onnx_weights = isOnnxWeightsFile(args.weights_file);
        if (args.family == DetectionFamily::RFDETR && !onnx_weights
            && args.runtime.backend() != irt::model::ModelRuntime::Backend::TensorRT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "RF-DETR native .wts path in the detection sample requires TensorRT backend; .onnx "
                                 "weights can run on any backend");
        }

        fs::path project_root = irt::util::findProjectRoot(argv[0], {kDefaultImagePath, kDefaultLabelPath}, __FILE__);
        fs::path image_path   = args.image_path.empty() ? project_root / kDefaultImagePath : args.image_path;
        fs::path label_path   = args.label_file.empty() ? project_root / kDefaultLabelPath : args.label_file;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setRuntime(args.runtime);
        config->setPrecision(args.precision);
        if (args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT)
        {
            if (args.family == DetectionFamily::YOLO)
            {
                config->setInputShape(irt::Shape{args.batch_opt, 3, args.input_size, args.input_size});
                if (args.batch_min != args.batch_max)
                {
                    config->setDynamicBatchRange(args.batch_min, args.batch_opt, args.batch_max);
                }
                config->setNumClasses(args.num_classes);
                config->setOutputTensorNames({"output0", "output1", "output2"});
            }
            else if (args.num_classes_explicit)
            {
                config->setNumClasses(args.num_classes);
            }
            if (args.family == DetectionFamily::RFDETR && !onnx_weights && args.input_size_explicit)
            {
                config->setInputShape(irt::Shape{args.batch_max, 3, args.input_size, args.input_size});
                if (args.batch_max > 1)
                {
                    config->setDynamicBatchRange(1, args.batch_opt, args.batch_max);
                }
            }
        }

        const std::string runtime_model_name
            = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT
                  ? (onnx_weights ? "onnx" : args.model_name)
                  : "onnx";
        auto model = irt::model::CreateModel(runtime_model_name, std::move(config));
        if (!model)
        {
            std::cerr << "Failed to create model: " << runtime_model_name << std::endl;
            return -1;
        }
        model->setLogLevel(irt::model::LogLevel::Info);

        std::cout << "Building or loading model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(args.weights_file.string());
        const auto build_end = Clock::now();
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Loading image: " << image_path.generic_string() << std::endl;
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read image: %s",
                                 image_path.string().c_str());
        }

        const auto input_names  = model->ioTensorNames(irt::TensorIOMode::Input);
        const auto output_names = model->ioTensorNames(irt::TensorIOMode::Output);
        const bool output_contract_ok
            = args.family == DetectionFamily::RFDETR ? output_names.size() >= 2U : output_names.size() == 3U;
        if (input_names.size() != 1 || !output_contract_ok)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "%s detection sample expects one input and %s outputs", modelFamilyName(args.family),
                                 args.family == DetectionFamily::RFDETR ? "at least two" : "three");
        }

        if (args.family == DetectionFamily::YOLO)
        {
            model->setTensorShape(input_names.front(),
                                  irt::Shape{args.batch_size, 3, args.input_size, args.input_size});
        }
        else if (args.family == DetectionFamily::RFDETR && args.batch_max > 1)
        {
            const irt::Shape engine_dims = model->tensorShape(input_names.front());
            model->setTensorShape(input_names.front(),
                                  irt::Shape{args.batch_size, 3, engine_dims[2], engine_dims[3]});
        }
        const irt::Shape input_dims = model->tensorShape(input_names.front());
        if (input_dims.rank() != 4 || input_dims[0] != args.batch_size || input_dims[1] != 3)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Unexpected input shape: %s (expected batch=%d, 3 channels)", dimsToCsv(input_dims).c_str(),
                                 args.batch_size);
        }

        const auto         preprocess_start = Clock::now();
        LetterboxInfo      letterbox;
        std::vector<float> single_tensor = args.family == DetectionFamily::RFDETR
                                               ? preprocessRFDETR(image, input_dims)
                                               : preprocessLetterbox(image, static_cast<int>(input_dims[3]),
                                                                     static_cast<int>(input_dims[2]), letterbox);
        std::vector<float> input_tensor;
        if (args.batch_size > 1)
        {
            input_tensor.reserve(single_tensor.size() * static_cast<size_t>(args.batch_size));
            for (int b = 0; b < args.batch_size; ++b)
            {
                input_tensor.insert(input_tensor.end(), single_tensor.begin(), single_tensor.end());
            }
        }
        else
        {
            input_tensor = std::move(single_tensor);
        }
        const auto         preprocess_end = Clock::now();

        const bool uses_tensorrt = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
        const auto stream        = uses_tensorrt ? reinterpret_cast<cudaStream_t>(model->resolveExecutionStream()) : nullptr;

        DeviceBuffer device_input;
        if (uses_tensorrt)
        {
            device_input.resize(elementCount(input_dims), irt::TensorDataType::F32);
        }

        std::vector<irt::Shape> output_dims;
        std::vector<DeviceBuffer>   device_outputs;
        std::vector<HostBuffer>     host_outputs;
        output_dims.reserve(output_names.size());
        device_outputs.reserve(output_names.size());
        host_outputs.reserve(output_names.size());

        for (const auto &output_name : output_names)
        {
            const auto dims = model->tensorShape(output_name);
            const auto type = model->tensorDataType(output_name);
            if (type != irt::TensorDataType::F32)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Detection sample expects float32 outputs");
            }
            output_dims.push_back(dims);
            if (uses_tensorrt)
            {
                device_outputs.emplace_back(elementCount(dims), type);
            }
            host_outputs.emplace_back(elementCount(dims), type);
            std::cout << "Output " << output_name << " shape: " << dimsToCsv(dims) << std::endl;
        }

        std::vector<irt::BufferView> buffers;
        buffers.reserve(1 + output_names.size());
        buffers.push_back(uses_tensorrt
                              ? irt::BufferView::fromBytes(device_input.data(), device_input.sizeBytes(),
                                                           irt::MemoryKind::DEVICE, input_names.front())
                              : irt::BufferView::fromBytes(input_tensor.data(), input_tensor.size() * sizeof(float),
                                                           irt::MemoryKind::HOST, input_names.front()));
        for (size_t i = 0; i < output_names.size(); ++i)
        {
            buffers.push_back(uses_tensorrt
                              ? irt::BufferView::fromBytes(device_outputs[i].data(), device_outputs[i].sizeBytes(),
                                                               irt::MemoryKind::DEVICE, output_names[i])
                              : irt::BufferView::fromBytes(host_outputs[i].data(), host_outputs[i].sizeBytes(),
                                                               irt::MemoryKind::HOST, output_names[i]));
        }

        auto run_inference_once = [&]() -> IterationTiming
        {
            IterationTiming timing;

            const auto h2d_start = Clock::now();
            if (uses_tensorrt)
            {
                checkCuda(cudaMemcpyAsync(device_input.data(), input_tensor.data(), device_input.sizeBytes(),
                                          cudaMemcpyHostToDevice, stream),
                          "cudaMemcpyAsync(H2D input)");
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(H2D input)");
            }
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->infer(buffers, reinterpret_cast<std::uintptr_t>(stream), true);
            if (uses_tensorrt)
            {
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inference)");
            }
            const auto inference_end = Clock::now();
            timing.inference_ms      = elapsedMs(inference_start, inference_end);

            const auto d2h_start = Clock::now();
            if (uses_tensorrt)
            {
                for (size_t i = 0; i < host_outputs.size(); ++i)
                {
                    checkCuda(cudaMemcpyAsync(host_outputs[i].data(), device_outputs[i].data(),
                                              host_outputs[i].sizeBytes(), cudaMemcpyDeviceToHost, stream),
                              "cudaMemcpyAsync(D2H output)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H output)");
            }
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

        const auto postprocess_start = Clock::now();
        const auto labels            = irt::samples::readLabelNames(label_path);
        auto       detections        = args.family == DetectionFamily::RFDETR
                                         ? postprocessRFDETROutputs(host_outputs, output_dims, args, image.size())
                                         : postprocessYoloOutputs(host_outputs, output_dims, args, letterbox);
        const auto postprocess_end   = Clock::now();

        printDetections(detections, labels);
        drawDetections(image, detections, labels, args.output_image);

        std::cout << "Runtime: " << args.runtime.toString() << std::endl;
        const auto h2d_stats        = summarizeTimings(h2d_times_ms);
        const auto inference_stats  = summarizeTimings(inference_times_ms);
        const auto d2h_stats        = summarizeTimings(d2h_times_ms);
        const auto end_to_end_stats = summarizeTimings(end_to_end_times_ms);
        std::cout << "Timing: build_or_load=" << elapsedMs(build_start, build_end)
                  << " ms, preprocess=" << elapsedMs(preprocess_start, preprocess_end) << " ms";
        printTimingStats(std::cout, "h2d", h2d_stats);
        printTimingStats(std::cout, "inference", inference_stats);
        printTimingStats(std::cout, "d2h", d2h_stats);
        printTimingStats(std::cout, "end_to_end", end_to_end_stats);
        if (args.batch_size > 1)
        {
            std::cout << "Per-image (batch=" << args.batch_size << "): h2d="
                      << h2d_stats.avg_ms / static_cast<double>(args.batch_size)
                      << " ms, inference=" << inference_stats.avg_ms / static_cast<double>(args.batch_size)
                      << " ms, d2h=" << d2h_stats.avg_ms / static_cast<double>(args.batch_size)
                      << " ms, end_to_end=" << end_to_end_stats.avg_ms / static_cast<double>(args.batch_size) << " ms"
                      << std::endl;
        }
        std::cout << ", timed_loop_wall=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(postprocess_start, postprocess_end) << " ms" << std::endl;
        std::cout << "Done!" << std::endl;
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
