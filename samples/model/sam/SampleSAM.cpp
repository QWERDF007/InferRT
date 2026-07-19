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
#include <cstddef>
#include <filesystem>
#include <iostream>
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

constexpr int  kSamMaxPoints = 16;
const fs::path kDefaultImagePath{"assets/pics/dog.jpg"};

/**
 * @brief SAM 分割 sample 的命令行参数。
 */
struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    fs::path                 image_path;
    fs::path                 output_image;
    float                    point_x{0.5F};
    float                    point_y{0.5F};
    float                    threshold{0.0F};
    bool                     has_box{false};
    std::array<float, 4>     box{0.0F, 0.0F, 1.0F, 1.0F};
    irt::model::ModelRuntime runtime{};
    int                      warmup{0};
    int                      repeat{1};
};

/**
 * @brief SAM 官方最长边缩放后的图像信息。
 */
struct PreprocessedSAMImage
{
    std::vector<float> tensor;    ///< `1x3x1024x1024` 输入张量。
    int                resized_h; ///< padding 前的缩放高度。
    int                resized_w; ///< padding 前的缩放宽度。
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

/**
 * @brief 判断当前模型是否属于 SAM2/SAM2.1。
 */
bool isSAM2Model(std::string model_name)
{
    model_name = toLower(model_name);
    return model_name.rfind("sam2", 0) == 0;
}

/**
 * @brief 判断模型 key 是否属于当前 sample 支持的 SAM 分割族。
 */
bool isSAMFamilyModel(std::string model_name)
{
    model_name = toLower(model_name);
    return model_name.rfind("sam", 0) == 0 || model_name == "edge_sam";
}

/**
 * @brief 解析 `x0,y0,x1,y1` 格式的 box prompt。
 */
std::array<float, 4> parseBoxPrompt(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream   input(value);
    std::array<float, 4> box{};
    if (!(input >> box[0] >> box[1] >> box[2] >> box[3]))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--box must be x0,y0,x1,y1");
    }
    return box;
}

/**
 * @brief 构造命令行选项。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run InferRT SAM/SAM2 prompt segmentation models");
    options.add_options()("model,m", "SAM/SAM2 model name, e.g. sam_vit_b/sam2_1_hiera_tiny (required)",
                          cxxopts::value<std::string>())(
        "weights-file,w", "Weights/model file (.wts, .onnx or OpenVINO IR) (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "output-image,o", "Path to save mask overlay", cxxopts::value<std::string>()->default_value(""))(
        "point-x,x", "Positive point prompt x; <=1 means normalized coordinate",
        cxxopts::value<float>()->default_value("0.5"))("point-y,y",
                                                       "Positive point prompt y; <=1 means normalized coordinate",
                                                       cxxopts::value<float>()->default_value("0.5"))(
        "box", "Optional box prompt x0,y0,x1,y1; <=1 means normalized coordinate",
        cxxopts::value<std::string>()->default_value(""))("threshold", "Mask logit threshold",
                                                          cxxopts::value<float>()->default_value("0.0"))(
        "runtime", "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
         cxxopts::value<std::string>()->default_value("tensorrt:0"));
    irt::samples::addTimingOptions(options, "Timed inference iterations");
    options.add_options()("h,help", "Show help");
    return options;
}

/**
 * @brief 解析命令行参数。
 */
Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << kDefaultImagePath.generic_string() << std::endl;
        std::cout << "Supported SAM models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            if (isSAMFamilyModel(model_name))
            {
                std::cout << ' ' << model_name;
            }
        }
        std::cout << std::endl;
        throw HelpRequested{};
    }

    if (!result.count("model") || !result.count("weights-file"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--model and --weights-file are required");
    }

    Arguments args;
    args.model_name   = result["model"].as<std::string>();
    args.weights_file = result["weights-file"].as<std::string>();
    args.image_path   = result["image-path"].as<std::string>();
    args.output_image = result["output-image"].as<std::string>();
    args.point_x      = result["point-x"].as<float>();
    args.point_y      = result["point-y"].as<float>();
    args.runtime      = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    const auto timing = irt::samples::parseTimingOptions(result);
    args.warmup       = timing.warmup;
    args.repeat       = timing.repeat;
    const auto box    = result["box"].as<std::string>();
    if (!box.empty())
    {
        args.has_box = true;
        args.box     = parseBoxPrompt(box);
    }
    args.threshold = result["threshold"].as<float>();
    return args;
}

/**
 * @brief 按官方最长边缩放规则计算 padding 前尺寸。
 */
std::pair<int, int> computeResizeShape(int original_h, int original_w, int target_size)
{
    const float scale = static_cast<float>(target_size) / static_cast<float>(std::max(original_h, original_w));
    return {static_cast<int>(std::floor(static_cast<float>(original_h) * scale + 0.5F)),
            static_cast<int>(std::floor(static_cast<float>(original_w) * scale + 0.5F))};
}

/**
 * @brief 将输入点转换到官方 SAM padding 前坐标系。
 */
float scalePointCoordinate(float value, int original_size, int resized_size)
{
    const float original_coord = value <= 1.0F ? value * static_cast<float>(original_size) : value;
    return original_coord * static_cast<float>(resized_size) / static_cast<float>(std::max(1, original_size));
}

/**
 * @brief 使用 SAM 官方均值方差与最长边缩放/padding 预处理图像。
 */
PreprocessedSAMImage preprocessSAMImage(const cv::Mat &image, int input_h, int input_w, bool use_sam2_preprocess)
{
    if (input_h != input_w)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM input must be square, got %dx%d", input_h,
                             input_w);
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    if (use_sam2_preprocess)
    {
        static constexpr float kMean[3]{0.485F, 0.456F, 0.406F};
        static constexpr float kStd[3]{0.229F, 0.224F, 0.225F};
        cv::Mat                resized;
        cv::resize(rgb, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

        std::vector<float> tensor(static_cast<size_t>(3) * input_h * input_w);
        for (int y = 0; y < input_h; ++y)
        {
            const auto *row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < input_w; ++x)
            {
                for (int c = 0; c < 3; ++c)
                {
                    const size_t offset
                        = static_cast<size_t>(c) * input_h * input_w + static_cast<size_t>(y) * input_w + x;
                    tensor[offset] = (static_cast<float>(row[x][c]) / 255.0F - kMean[c]) / kStd[c];
                }
            }
        }
        return {std::move(tensor), input_h, input_w};
    }

    static constexpr float kMean[3]{123.675F, 116.28F, 103.53F};
    static constexpr float kStd[3]{58.395F, 57.12F, 57.375F};
    const auto [resized_h, resized_w] = computeResizeShape(image.rows, image.cols, input_h);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    std::vector<float> tensor(static_cast<size_t>(3) * input_h * input_w, 0.0F);
    for (int y = 0; y < resized_h; ++y)
    {
        const auto *row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < resized_w; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                const size_t offset = static_cast<size_t>(c) * input_h * input_w + static_cast<size_t>(y) * input_w + x;
                tensor[offset]      = (static_cast<float>(row[x][c]) - kMean[c]) / kStd[c];
            }
        }
    }
    return {std::move(tensor), resized_h, resized_w};
}

/**
 * @brief 保存第一张 mask 的彩色叠加结果，并按官方后处理裁掉 padding。
 */
void saveMaskOverlay(const cv::Mat &image, const float *mask, int mask_h, int mask_w, int input_h, int input_w,
                     int resized_h, int resized_w, float threshold, const fs::path &output_image)
{
    cv::Mat low_res(mask_h, mask_w, CV_32FC1, const_cast<float *>(mask));
    cv::Mat padded;
    cv::resize(low_res, padded, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat cropped = padded(cv::Rect(0, 0, resized_w, resized_h));
    cv::Mat resized;
    cv::resize(cropped, resized, image.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat overlay = image.clone();
    for (int y = 0; y < overlay.rows; ++y)
    {
        auto       *pixel_row = overlay.ptr<cv::Vec3b>(y);
        const auto *mask_row  = resized.ptr<float>(y);
        for (int x = 0; x < overlay.cols; ++x)
        {
            if (mask_row[x] > threshold)
            {
                pixel_row[x][2] = static_cast<unsigned char>(0.65F * 255.0F + 0.35F * pixel_row[x][2]);
                pixel_row[x][1] = static_cast<unsigned char>(0.35F * pixel_row[x][1]);
                pixel_row[x][0] = static_cast<unsigned char>(0.35F * pixel_row[x][0]);
            }
        }
    }

    if (!cv::imwrite(output_image.string(), overlay))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                             output_image.string().c_str());
    }
}

} // namespace

/**
 * @brief 运行 SAM/SAM2/SAM3 单图分割示例。
 */
int main(int argc, char *argv[])
{
    try
    {
        const Arguments args = parseArguments(argc, argv);
        if (!irt::model::isSupportedModel(args.model_name))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", args.model_name.c_str());
        }

        const fs::path project_root = irt::util::findProjectRoot(argv[0], {kDefaultImagePath}, __FILE__);
        const fs::path image_path   = args.image_path.empty() ? (project_root / kDefaultImagePath) : args.image_path;
        const fs::path output_path  = args.output_image.empty()
                                        ? (fs::current_path() / (args.model_name + "_sam_mask.jpg"))
                                        : args.output_image;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setRuntime(args.runtime);
        const std::string runtime_model_name
            = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT ? args.model_name : "onnx";
        auto model = irt::model::CreateModel(runtime_model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 runtime_model_name.c_str());
        }
        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or loading SAM model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(args.weights_file.string());
        const auto build_end = Clock::now();

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto input_names  = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        const auto output_names = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (input_names.size() != 5 || output_names.size() != 3)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "SAM sample expects the default 5-input/3-output prompt contract");
        }

        const auto image_dims       = model->tensorShape(input_names[0]);
        const int  input_h          = static_cast<int>(image_dims.d[2]);
        const int  input_w          = static_cast<int>(image_dims.d[3]);
        const auto prompt_mask_dims = model->tensorShape(input_names[3]);
        const int  prompt_mask_h    = static_cast<int>(prompt_mask_dims.d[2]);
        const int  prompt_mask_w    = static_cast<int>(prompt_mask_dims.d[3]);
        const auto output_mask_dims = model->tensorShape(output_names[0]);
        if (output_mask_dims.nbDims != 4)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SAM masks output must be 4D, got %dD",
                                 output_mask_dims.nbDims);
        }
        const int output_mask_h = static_cast<int>(output_mask_dims.d[2]);
        const int output_mask_w = static_cast<int>(output_mask_dims.d[3]);

        const auto         preprocess_start = Clock::now();
        auto               preprocessed     = preprocessSAMImage(image, input_h, input_w, isSAM2Model(args.model_name));
        auto               image_tensor     = std::move(preprocessed.tensor);
        std::vector<float> point_coords(static_cast<size_t>(kSamMaxPoints) * 2, 0.0F);
        std::vector<float> point_labels(kSamMaxPoints, -1.0F);
        point_coords[0] = scalePointCoordinate(args.point_x, image.cols, preprocessed.resized_w);
        point_coords[1] = scalePointCoordinate(args.point_y, image.rows, preprocessed.resized_h);
        point_labels[0] = 1.0F;
        if (args.has_box)
        {
            point_coords[2] = scalePointCoordinate(args.box[0], image.cols, preprocessed.resized_w);
            point_coords[3] = scalePointCoordinate(args.box[1], image.rows, preprocessed.resized_h);
            point_labels[1] = 2.0F;
            point_coords[4] = scalePointCoordinate(args.box[2], image.cols, preprocessed.resized_w);
            point_coords[5] = scalePointCoordinate(args.box[3], image.rows, preprocessed.resized_h);
            point_labels[2] = 3.0F;
        }
        std::vector<float> mask_input(static_cast<size_t>(prompt_mask_h) * prompt_mask_w, 0.0F);
        std::vector<float> has_mask_input{0.0F};
        const auto         preprocess_end = Clock::now();

        const std::vector<std::vector<float> *> input_vectors{
            &image_tensor, &point_coords, &point_labels, &mask_input, &has_mask_input,
        };

        const auto                stream        = model->resolveExecutionStream();
        const bool                uses_tensorrt
            = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
        std::vector<DeviceBuffer> device_inputs;
        std::vector<DeviceBuffer> device_outputs;
        std::vector<HostBuffer>   host_outputs;
        std::vector<void *>       buffers;
        device_inputs.reserve(input_vectors.size());
        device_outputs.reserve(output_names.size());
        host_outputs.reserve(output_names.size());
        buffers.reserve(input_vectors.size() + output_names.size());

        for (const auto *values : input_vectors)
        {
            if (uses_tensorrt)
            {
                device_inputs.emplace_back(values->size(), nvinfer1::DataType::kFLOAT);
                buffers.push_back(device_inputs.back().data());
            }
            else
            {
                buffers.push_back(const_cast<float *>(values->data()));
            }
        }

        for (const auto &output_name : output_names)
        {
            const auto dims = model->tensorShape(output_name);
            std::cout << "Output " << output_name << " dims=[" << dimsToCsv(dims) << "]" << std::endl;
            const size_t count = elementCount(dims);
            if (uses_tensorrt)
            {
                device_outputs.emplace_back(count, nvinfer1::DataType::kFLOAT);
            }
            host_outputs.emplace_back(count, nvinfer1::DataType::kFLOAT);
            buffers.push_back(uses_tensorrt ? device_outputs.back().data() : host_outputs.back().data());
        }

        auto run_inference_once = [&]() -> IterationTiming
        {
            IterationTiming timing;

            const auto h2d_start = Clock::now();
            if (uses_tensorrt)
            {
                for (size_t i = 0; i < input_vectors.size(); ++i)
                {
                    const auto *values = input_vectors[i];
                    checkCuda(cudaMemcpyAsync(device_inputs[i].data(), values->data(), values->size() * sizeof(float),
                                              cudaMemcpyHostToDevice, stream),
                              "cudaMemcpyAsync(SAM input)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SAM input)");
            }
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->infer(buffers, stream, true);
            if (uses_tensorrt)
            {
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SAM inference)");
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
                              "cudaMemcpyAsync(SAM output)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SAM output)");
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

        const auto  post_start  = Clock::now();
        const auto *mask_values = static_cast<const float *>(host_outputs.front().data());
        saveMaskOverlay(image, mask_values, output_mask_h, output_mask_w, input_h, input_w, preprocessed.resized_h,
                        preprocessed.resized_w, args.threshold, output_path);
        const auto post_end = Clock::now();

        std::cout << "Saved mask overlay to: " << fs::absolute(output_path).string() << std::endl;
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
