#include <cxxopts.hpp>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

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
};

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run InferRT classification models on a single image");
    options.positional_help("<model_name> <weights_file.wts> [image_path] [label_file]");
    options.add_options()
        ("model_name", "Built-in model name", cxxopts::value<std::string>())
        ("weights_file", "Weights file (.wts)", cxxopts::value<std::string>())
        ("image_path", "Input image path", cxxopts::value<std::string>()->default_value(""))
        ("label_file", "Imagenet label file", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Show help");
    options.parse_positional({"model_name", "weights_file", "image_path", "label_file"});
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

    if (!result.count("model_name") || !result.count("weights_file"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "model_name and weights_file are required");
    }

    Arguments args;
    args.model_name   = result["model_name"].as<std::string>();
    args.weights_file = result["weights_file"].as<std::string>();
    args.image_path   = result["image_path"].as<std::string>();
    args.label_file   = result["label_file"].as<std::string>();
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

        const cv::Mat           preprocessed = irt::model::ImageNetUtil::preprocess(img);
        const std::vector<float> input_data  = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);

        void *d_input  = nullptr;
        void *d_output = nullptr;

        size_t input_size  = 1 * 3 * 224 * 224 * sizeof(float);
        size_t output_size = 1 * 1000 * sizeof(float);

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);

        cudaMemcpy(d_input, input_data.data(), input_size, cudaMemcpyHostToDevice);

        std::vector<void *> buffers = {d_input, d_output};
        std::cout << "Running inference..." << std::endl;
        model->infer(buffers);

        std::vector<float> output_data(1000);
        cudaMemcpy(output_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);

        std::vector<std::string> labels;
        bool                     has_labels = false;
        if (!label_file.empty() && fs::exists(label_file))
        {
            labels     = irt::model::readImagenetLabels(label_file.string());
            has_labels = true;
        }

        std::vector<std::pair<float, int>> scores;
        scores.reserve(1000);
        for (int i = 0; i < 1000; ++i)
        {
            scores.push_back({output_data[i], i});
        }
        std::partial_sort(scores.begin(), scores.begin() + 3, scores.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::cout << "\nTop-3 predictions:" << std::endl;
        for (int i = 0; i < 3; ++i)
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

        cudaFree(d_input);
        cudaFree(d_output);

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
