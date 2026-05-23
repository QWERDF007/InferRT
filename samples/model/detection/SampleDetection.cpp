#include <cxxopts.hpp>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using irt::model::DeviceBuffer;
using irt::model::HostBuffer;
using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

constexpr std::array<int, 3> kYoloStrides{8, 16, 32};
constexpr std::array<std::array<float, 6>, 3> kDefaultYoloV5Anchors{{
    {10.0F, 13.0F, 16.0F, 30.0F, 33.0F, 23.0F},
    {30.0F, 61.0F, 62.0F, 45.0F, 59.0F, 119.0F},
    {116.0F, 90.0F, 156.0F, 198.0F, 373.0F, 326.0F},
}};
const fs::path kDefaultImagePath{"assets/pics/dog.jpg"};
const fs::path kDefaultLabelPath{"assets/coco80.names"};

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

/**
 * @brief detection sample 的命令行参数集合。
 */
struct Arguments
{
    std::string model_name;
    fs::path    weights_file;
    fs::path    image_path;
    fs::path    label_file;
    fs::path    output_image;
    std::string legacy_anchors;
    int         input_size{640};
    int         num_classes{80};
    int         max_detections{100};
    float       conf_threshold{0.25F};
    float       nms_threshold{0.45F};
};

/**
 * @brief letterbox 预处理的缩放与填充信息。
 */
struct LetterboxInfo
{
    float scale{1.0F}; ///< 原图缩放到网络输入的比例。
    int   pad_x{0};   ///< 左侧填充像素。
    int   pad_y{0};   ///< 顶部填充像素。
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
 * @brief 从文本文件读取类别名称；文件缺失时返回空列表并打印提示。
 */
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
        if (!line.empty())
        {
            labels.push_back(line);
        }
    }
    return labels;
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
    cxxopts::Options options(program_name, "Run InferRT YOLO detection models on a single image");
    options.add_options()("model,m", "YOLO model name, e.g. yolov5n/yolov8n (required)",
                          cxxopts::value<std::string>())(
        "weights-file,w", "Weights file (.wts) (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "label-file,l", "Class label file", cxxopts::value<std::string>()->default_value(""))(
        "output-image,o", "Optional path to save image with boxes", cxxopts::value<std::string>()->default_value(""))(
        "input-size", "Square network input size, must be divisible by 32",
        cxxopts::value<int>()->default_value("640"))(
        "classes", "Number of classes used by the YOLO head", cxxopts::value<int>()->default_value("80"))(
        "conf-threshold", "Confidence threshold", cxxopts::value<float>()->default_value("0.25"))(
        "nms-threshold", "Class-wise NMS IoU threshold", cxxopts::value<float>()->default_value("0.45"))(
        "max-detections", "Maximum detections printed after NMS", cxxopts::value<int>()->default_value("100"))(
        "legacy-anchors",
        "Legacy YOLOv5 anchors as 18 comma-separated numbers; empty uses COCO defaults",
        cxxopts::value<std::string>()->default_value(""))("h,help", "Show help");
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
    args.model_name      = result["model"].as<std::string>();
    args.weights_file    = result["weights-file"].as<std::string>();
    args.image_path      = result["image-path"].as<std::string>();
    args.label_file      = result["label-file"].as<std::string>();
    args.output_image    = result["output-image"].as<std::string>();
    args.input_size      = result["input-size"].as<int>();
    args.num_classes     = result["classes"].as<int>();
    args.conf_threshold  = result["conf-threshold"].as<float>();
    args.nms_threshold   = result["nms-threshold"].as<float>();
    args.max_detections  = result["max-detections"].as<int>();
    args.legacy_anchors  = result["legacy-anchors"].as<std::string>();

    if (args.input_size <= 0 || args.input_size % 32 != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--input-size must be positive and divisible by 32");
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
    info.original_w = bgr_image.cols;
    info.original_h = bgr_image.rows;
    info.input_w    = input_w;
    info.input_h    = input_h;
    info.scale      = std::min(static_cast<float>(input_w) / static_cast<float>(bgr_image.cols),
                               static_cast<float>(input_h) / static_cast<float>(bgr_image.rows));

    const int resized_w = static_cast<int>(std::round(static_cast<float>(bgr_image.cols) * info.scale));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(bgr_image.rows) * info.scale));
    info.pad_x         = (input_w - resized_w) / 2;
    info.pad_y         = (input_h - resized_h) / 2;

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

float intersectionOverUnion(const Detection &a, const Detection &b)
{
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float inter_w = std::max(0.0F, x2 - x1);
    const float inter_h = std::max(0.0F, y2 - y1);
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
bool isLegacyYoloV5Output(const nvinfer1::Dims &dims, int num_classes)
{
    return dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3 * (num_classes + 5) && dims.d[2] > 0
        && dims.d[3] > 0;
}

/**
 * @brief 判断输出是否为 YOLOv5u/YOLOv8 DFL head。
 */
bool isDflOutput(const nvinfer1::Dims &dims, int num_classes)
{
    return dims.nbDims == 3 && dims.d[0] == 1 && dims.d[1] == num_classes + 4 && dims.d[2] > 0;
}

/**
 * @brief 解码旧版 YOLOv5 raw anchor head。
 */
void decodeLegacyYoloV5Output(const float *data, const nvinfer1::Dims &dims, int branch_index, int num_classes,
                              float conf_threshold, const LetterboxInfo &letterbox,
                              const std::array<std::array<float, 6>, 3> &anchors,
                              std::vector<Detection> &detections)
{
    const int channels = static_cast<int>(dims.d[1]);
    const int grid_h   = static_cast<int>(dims.d[2]);
    const int grid_w   = static_cast<int>(dims.d[3]);
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
        for (int y = 0; y < grid_h; ++y)
        {
            for (int x = 0; x < grid_w; ++x)
            {
                const int grid_index = y * grid_w + x;
                auto valueAt = [&](int attr)
                {
                    const int channel = anchor_idx * info_len + attr;
                    return data[channel * grid_h * grid_w + grid_index];
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

/**
 * @brief 解码 YOLOv5u/YOLOv8 的 DFL distance + class head。
 */
void decodeDflOutput(const float *data, const nvinfer1::Dims &dims, int branch_index, int num_classes,
                     float conf_threshold, const LetterboxInfo &letterbox, std::vector<Detection> &detections)
{
    const int stride = kYoloStrides[static_cast<size_t>(branch_index)];
    const int grid_h = letterbox.input_h / stride;
    const int grid_w = letterbox.input_w / stride;
    const int grid   = grid_h * grid_w;
    if (dims.d[2] != grid)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "DFL output grid mismatch for branch %d: expected %d, got %d", branch_index, grid,
                             dims.d[2]);
    }

    for (int idx = 0; idx < grid; ++idx)
    {
        int   best_class = 0;
        float best_score = 0.0F;
        for (int cls = 0; cls < num_classes; ++cls)
        {
            const float score = sigmoid(data[(4 + cls) * grid + idx]);
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

        const int   x = idx % grid_w;
        const int   y = idx / grid_w;
        const float anchor_x = static_cast<float>(x) + 0.5F;
        const float anchor_y = static_cast<float>(y) + 0.5F;
        const float left     = data[0 * grid + idx];
        const float top      = data[1 * grid + idx];
        const float right    = data[2 * grid + idx];
        const float bottom   = data[3 * grid + idx];

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

/**
 * @brief 根据模型输出形态选择 YOLO 后处理路径。
 */
std::vector<Detection> postprocessYoloOutputs(const std::vector<HostBuffer> &outputs,
                                              const std::vector<nvinfer1::Dims> &output_dims,
                                              const Arguments &args, const LetterboxInfo &letterbox)
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
            decodeDflOutput(data, output_dims[i], static_cast<int>(i), args.num_classes, args.conf_threshold,
                            letterbox, detections);
        }
        else if (legacy_head)
        {
            if (!isLegacyYoloV5Output(output_dims[i], args.num_classes))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Mixed YOLO output head types");
            }
            decodeLegacyYoloV5Output(data, output_dims[i], static_cast<int>(i), args.num_classes,
                                     args.conf_threshold, letterbox, anchors, detections);
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
                  << " class=" << det.class_id << " (" << className(det.class_id, labels) << ") box=["
                  << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]" << std::endl;
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
        const int color_seed = det.class_id * 37;
        cv::Scalar color((color_seed * 3) % 255, (color_seed * 7 + 80) % 255, (color_seed * 11 + 160) % 255);
        cv::rectangle(visual, cv::Point(static_cast<int>(std::round(det.x1)), static_cast<int>(std::round(det.y1))),
                      cv::Point(static_cast<int>(std::round(det.x2)), static_cast<int>(std::round(det.y2))), color, 2);

        std::ostringstream label;
        label << className(det.class_id, labels) << ' ' << std::fixed << std::setprecision(2) << det.confidence;
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int      text_x    = static_cast<int>(std::round(det.x1));
        const int      text_y    = std::max(text_size.height + 4, static_cast<int>(std::round(det.y1)));
        cv::rectangle(visual, cv::Rect(text_x, text_y - text_size.height - 4, text_size.width + 4,
                                       text_size.height + baseline + 4),
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
 * @brief 运行单张图片 YOLO 检测，并在 sample 层完成解码与 NMS。
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

        fs::path project_root = irt::util::findProjectRoot(argv[0], {kDefaultImagePath, kDefaultLabelPath}, __FILE__);
        fs::path image_path   = args.image_path.empty() ? project_root / kDefaultImagePath : args.image_path;
        fs::path label_path   = args.label_file.empty() ? project_root / kDefaultLabelPath : args.label_file;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setInputShape(nvinfer1::Dims4{1, 3, args.input_size, args.input_size});
        config->setNumClasses(args.num_classes);
        config->setOutputTensorNames({"output0", "output1", "output2"});

        auto model = irt::model::CreateModel(args.model_name, std::move(config));
        if (!model)
        {
            std::cerr << "Failed to create model: " << args.model_name << std::endl;
            return -1;
        }
        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or loading model..." << std::endl;
        model->buildOrLoad(args.weights_file.string());
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Loading image: " << image_path.generic_string() << std::endl;
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read image: %s",
                                 image_path.string().c_str());
        }

        const auto &input_names  = model->modelConfig().inputTensorNames();
        const auto &output_names = model->modelConfig().outputTensorNames();
        if (input_names.size() != 1 || output_names.size() != 3)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Detection sample expects one input and three outputs");
        }

        const nvinfer1::Dims input_dims = model->tensorShape(input_names.front());
        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected input shape: %s",
                                 dimsToCsv(input_dims).c_str());
        }

        const auto preprocess_start = Clock::now();
        LetterboxInfo letterbox;
        std::vector<float> input_tensor = preprocessLetterbox(image, static_cast<int>(input_dims.d[3]),
                                                              static_cast<int>(input_dims.d[2]), letterbox);
        const auto preprocess_end = Clock::now();

        DeviceBuffer device_input(elementCount(input_dims), nvinfer1::DataType::kFLOAT);
        checkCuda(cudaMemcpyAsync(device_input.data(), input_tensor.data(), device_input.sizeBytes(),
                                  cudaMemcpyHostToDevice, model->resolveExecutionStream()),
                  "cudaMemcpyAsync(H2D input)");

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
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "YOLO sample expects float32 outputs");
            }
            output_dims.push_back(dims);
            device_outputs.emplace_back(elementCount(dims), type);
            host_outputs.emplace_back(elementCount(dims), type);
            std::cout << "Output " << output_name << " shape: " << dimsToCsv(dims) << std::endl;
        }

        std::vector<void *> buffers;
        buffers.reserve(1 + device_outputs.size());
        buffers.push_back(device_input.data());
        for (auto &output : device_outputs)
        {
            buffers.push_back(output.data());
        }

        const auto stream = model->resolveExecutionStream();
        std::cout << "Running inference..." << std::endl;
        const auto infer_start = Clock::now();
        model->infer(buffers, stream, true);
        for (size_t i = 0; i < host_outputs.size(); ++i)
        {
            checkCuda(cudaMemcpyAsync(host_outputs[i].data(), device_outputs[i].data(), host_outputs[i].sizeBytes(),
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync(D2H output)");
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inference)");
        const auto infer_end = Clock::now();

        const auto postprocess_start = Clock::now();
        const auto labels = readLabels(label_path);
        auto detections = postprocessYoloOutputs(host_outputs, output_dims, args, letterbox);
        const auto postprocess_end = Clock::now();

        printDetections(detections, labels);
        drawDetections(image, detections, labels, args.output_image);

        std::cout << "Timing: preprocess=" << elapsedMs(preprocess_start, preprocess_end)
                  << " ms, inference=" << elapsedMs(infer_start, infer_end)
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
