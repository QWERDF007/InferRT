#include <cxxopts.hpp>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using irt::model::checkCuda;
using irt::model::dataTypeToString;
using irt::model::dimsToCsv;
using irt::model::elementCount;
using irt::model::elementSize;

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
 * @brief 分类 sample 的命令行参数集合。
 */
struct Arguments
{
    std::string model_name;
    fs::path    weights_file;
    fs::path    image_path;
    fs::path    label_file;
    fs::path    dump_dir;
};

std::string sanitizeFileStem(std::string_view value)
{
    std::string stem;
    stem.reserve(value.size());
    for (unsigned char ch : value)
    {
        if (std::isalnum(ch))
        {
            stem.push_back(static_cast<char>(ch));
        }
        else
        {
            stem.push_back('_');
        }
    }
    return stem.empty() ? "tensor" : stem;
}

void writeBinaryFile(const fs::path &file_path, const void *data, size_t num_bytes)
{
    std::ofstream output(file_path, std::ios::binary);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open output file: %s",
                             file_path.string().c_str());
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(num_bytes));
}

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run InferRT classification models on a single image");
    options.add_options()("model,m", "Built-in model name (required)", cxxopts::value<std::string>())(
        "weights-file,w", "Weights file (.wts) (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "label-file,l", "ImageNet label file", cxxopts::value<std::string>()->default_value(""))(
        "dump-dir,o", "Optional directory to dump input/output tensors for parity tests",
        cxxopts::value<std::string>()->default_value(""))("h,help", "Show help");
    return options;
}

/**
 * @brief 解析并校验命令行参数。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 解析后的参数。
 */
Arguments parseArguments(int argc, char *argv[])
{
    auto options = makeOptions(argv[0]);
    const auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << irt::model::ImageNetUtil::kDefaultImagePath.generic_string() << std::endl;
        std::cout << "Default labels: " << irt::model::ImageNetUtil::kDefaultLabelPath.generic_string() << std::endl;
        std::cout << "Supported models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            std::cout << ' ' << model_name;
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
    args.label_file   = result["label-file"].as<std::string>();
    args.dump_dir     = result["dump-dir"].as<std::string>();
    return args;
}

} // namespace

/**
 * @brief 运行单张图片的 ImageNet 分类推理，并打印 Top-3 结果。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
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

        fs::path project_root = irt::util::findProjectRoot(
            argv[0], {irt::model::ImageNetUtil::kDefaultImagePath, irt::model::ImageNetUtil::kDefaultLabelPath},
            __FILE__);
        fs::path img_path = args.image_path.empty() ? project_root / irt::model::ImageNetUtil::kDefaultImagePath
                                                    : args.image_path;
        fs::path label_file = args.label_file.empty() ? project_root / irt::model::ImageNetUtil::kDefaultLabelPath
                                                      : args.label_file;

        auto model = irt::model::CreateModel(args.model_name);
        if (!model)
        {
            std::cerr << "Failed to create model: " << args.model_name << std::endl;
            return -1;
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or Loading model..." << std::endl;
        model->buildOrLoad(args.weights_file.string());
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Loading image: " << img_path.generic_string() << std::endl;
        cv::Mat img = cv::imread(img_path.generic_string());
        if (img.empty())
        {
            std::cerr << "Failed to load image: " << img_path.generic_string() << std::endl;
            return -1;
        }

        const auto preprocess_start = Clock::now();
        const cv::Mat preprocessed = irt::model::ImageNetUtil::preprocess(img);
        const std::vector<float> input_data = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
        const auto preprocess_end = Clock::now();

        const auto &input_tensor_names  = model->modelConfig().inputTensorNames();
        const auto &output_tensor_names = model->modelConfig().outputTensorNames();
        if (input_tensor_names.empty() || output_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Model must expose at least one input and one output tensor");
        }

        const std::string &input_tensor_name = input_tensor_names.front();
        const nvinfer1::Dims input_dims      = model->tensorShape(input_tensor_name);
        const size_t         input_num_bytes = elementCount(input_dims) * elementSize(nvinfer1::DataType::kFLOAT);
        if (input_num_bytes != input_data.size() * sizeof(float))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Preprocessed input byte size does not match model input tensor");
        }

        const auto stream = model->resolveExecutionStream();

        void *d_input = nullptr;
        checkCuda(cudaMalloc(&d_input, input_num_bytes), "cudaMalloc(input)");
        checkCuda(cudaMemcpyAsync(d_input, input_data.data(), input_num_bytes, cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");

        std::vector<std::string> output_names;
        std::vector<nvinfer1::Dims> output_dims;
        std::vector<nvinfer1::DataType> output_types;
        std::vector<size_t> output_num_bytes;
        std::vector<void *> device_outputs;
        std::vector<std::vector<char>> host_outputs;

        output_names.reserve(output_tensor_names.size());
        output_dims.reserve(output_tensor_names.size());
        output_types.reserve(output_tensor_names.size());
        output_num_bytes.reserve(output_tensor_names.size());
        device_outputs.reserve(output_tensor_names.size());
        host_outputs.reserve(output_tensor_names.size());

        for (const auto &output_name : output_tensor_names)
        {
            const nvinfer1::Dims     dims      = model->tensorShape(output_name);
            const nvinfer1::DataType data_type = model->tensorDataType(output_name);
            const size_t             num_bytes = elementCount(dims) * elementSize(data_type);

            void *device_ptr = nullptr;
            checkCuda(cudaMalloc(&device_ptr, num_bytes), "cudaMalloc(output)");

            output_names.push_back(output_name);
            output_dims.push_back(dims);
            output_types.push_back(data_type);
            output_num_bytes.push_back(num_bytes);
            device_outputs.push_back(device_ptr);
            host_outputs.emplace_back(num_bytes);
        }

        std::vector<void *> buffers;
        buffers.reserve(1 + device_outputs.size());
        buffers.push_back(d_input);
        buffers.insert(buffers.end(), device_outputs.begin(), device_outputs.end());

        std::cout << "Running inference..." << std::endl;
        const auto infer_start = Clock::now();
        model->infer(buffers, stream, true);

        for (size_t i = 0; i < host_outputs.size(); ++i)
        {
            checkCuda(cudaMemcpyAsync(host_outputs[i].data(), device_outputs[i], output_num_bytes[i],
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync(D2H output)");
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inference)");
        const auto infer_end = Clock::now();

        const auto postprocess_start = Clock::now();
        std::vector<std::string> labels;
        bool                     has_labels = false;
        if (!label_file.empty() && fs::exists(label_file))
        {
            labels     = irt::model::readImagenetLabels(label_file.string());
            has_labels = true;
        }

        const std::string &primary_output_name = output_names.front();
        const size_t       primary_index       = 0;
        if (output_types[primary_index] != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Classification sample expects float32 logits output");
        }

        const size_t num_scores = elementCount(output_dims[primary_index]);
        const auto  *output_data = reinterpret_cast<const float *>(host_outputs[primary_index].data());

        std::vector<std::pair<float, int>> scores;
        scores.reserve(num_scores);
        for (size_t i = 0; i < num_scores; ++i)
        {
            scores.push_back({output_data[i], static_cast<int>(i)});
        }
        const size_t topk = std::min<size_t>(3, scores.size());
        std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(topk), scores.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        const auto postprocess_end = Clock::now();

        if (!args.dump_dir.empty())
        {
            fs::create_directories(args.dump_dir);
            writeBinaryFile(args.dump_dir / "input.bin", input_data.data(), input_num_bytes);

            std::ofstream manifest(args.dump_dir / "manifest.txt");
            if (!manifest)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file");
            }

            manifest << "version=1\n";
            manifest << "backend=inferrt_cpp\n";
            manifest << "model_name=" << args.model_name << "\n";
            manifest << "weights_file=" << fs::absolute(args.weights_file).generic_string() << "\n";
            manifest << "image_path=" << fs::absolute(img_path).generic_string() << "\n";
            manifest << "tensor|input|float32|" << dimsToCsv(input_dims) << "|input.bin\n";

            for (size_t i = 0; i < output_names.size(); ++i)
            {
                const std::string file_name = sanitizeFileStem(output_names[i]) + ".bin";
                writeBinaryFile(args.dump_dir / file_name, host_outputs[i].data(), output_num_bytes[i]);
                manifest << "tensor|" << output_names[i] << "|" << dataTypeToString(output_types[i]) << "|"
                         << dimsToCsv(output_dims[i]) << "|" << file_name << "\n";
            }

            std::cout << "Saved classification dump to: " << fs::absolute(args.dump_dir).string() << std::endl;
            std::cout << "Primary output tensor: " << primary_output_name << std::endl;
        }

        std::cout << "Timing: preprocess=" << elapsedMs(preprocess_start, preprocess_end)
                  << " ms, inference=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(postprocess_start, postprocess_end) << " ms" << std::endl;

        std::cout << "\nTop-" << topk << " predictions:" << std::endl;
        for (size_t i = 0; i < topk; ++i)
        {
            int   idx        = scores[i].second;
            float confidence = scores[i].first;
            if (has_labels)
            {
                std::cout << "top: " << (i + 1) << ", confidence: " << confidence << ", label[" << idx
                          << "]: " << labels[idx] << std::endl;
            }
            else
            {
                std::cout << "top: " << (i + 1) << ", confidence: " << confidence << ", label[" << idx << "]"
                          << std::endl;
            }
        }

        checkCuda(cudaFree(d_input), "cudaFree(input)");
        for (void *device_ptr : device_outputs)
        {
            checkCuda(cudaFree(device_ptr), "cudaFree(output)");
        }

        std::cout << "\nDone!" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const HelpRequested &)
    {
        return 0;
    }
}
