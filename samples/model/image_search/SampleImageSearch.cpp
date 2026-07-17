#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ImageSearch.hpp>
#include <inferrt/model/IModel.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

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
 * @brief 图像检索 sample 的命令行参数集合。
 */
struct Arguments
{
    fs::path                         weights_file;
    fs::path                         gallery_dir;
    fs::path                         query_image;
    fs::path                         index_file;
    irt::features::ImageSearchConfig config;
    int                              top_k{irt::features::ImageSearch::kDefaultTopK};
    bool                             rebuild_index{false};
};

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

irt::features::ImageSearchFeatureNorm parseNorm(std::string value)
{
    value = toLower(std::move(value));
    if (value == "none")
    {
        return irt::features::ImageSearchFeatureNorm::None;
    }
    if (value == "l1")
    {
        return irt::features::ImageSearchFeatureNorm::L1;
    }
    if (value == "l2")
    {
        return irt::features::ImageSearchFeatureNorm::L2;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported norm mode: %s", value.c_str());
}

irt::features::ImageSearchPreprocessBackend parsePreprocessBackend(std::string value)
{
    value = toLower(std::move(value));
    if (value == "cpu")
    {
        return irt::features::ImageSearchPreprocessBackend::CPU;
    }
    if (value == "gpu")
    {
        return irt::features::ImageSearchPreprocessBackend::GPU;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported preprocess backend: %s", value.c_str());
}

irt::model::ModelBackend parseModelBackend(std::string value)
{
    value = toLower(std::move(value));
    if (value == "tensorrt" || value == "trt")
    {
        return irt::model::ModelBackend::TensorRT;
    }
    if (value == "openvino" || value == "ov")
    {
        return irt::model::ModelBackend::OpenVINO;
    }
    if (value == "onnxruntime" || value == "onnx" || value == "ort")
    {
        return irt::model::ModelBackend::ONNXRuntime;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model backend: %s", value.c_str());
}

irt::model::ModelDevice parseModelDevice(std::string value)
{
    value = toLower(std::move(value));
    if (value == "cpu")
    {
        return irt::model::ModelDevice::CPU;
    }
    if (value == "gpu" || value == "cuda")
    {
        return irt::model::ModelDevice::GPU;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model device: %s", value.c_str());
}

irt::features::ImageSearchFaissBackend parseFaissBackend(std::string value)
{
    value = toLower(std::move(value));
    if (value == "cpu")
    {
        return irt::features::ImageSearchFaissBackend::CPU;
    }
    if (value == "gpu")
    {
        return irt::features::ImageSearchFaissBackend::GPU;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported Faiss backend: %s", value.c_str());
}

irt::features::ImageSearchIndexStorage parseIndexStorage(std::string value)
{
    value = toLower(std::move(value));
    if (value == "ram")
    {
        return irt::features::ImageSearchIndexStorage::RAM;
    }
    if (value == "disk")
    {
        return irt::features::ImageSearchIndexStorage::Disk;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported index storage: %s", value.c_str());
}

const char *normName(irt::features::ImageSearchFeatureNorm norm)
{
    switch (norm)
    {
    case irt::features::ImageSearchFeatureNorm::None:
        return "none";
    case irt::features::ImageSearchFeatureNorm::L1:
        return "l1";
    case irt::features::ImageSearchFeatureNorm::L2:
        return "l2";
    }
    return "unknown";
}

const char *preprocessBackendName(irt::features::ImageSearchPreprocessBackend backend)
{
    switch (backend)
    {
    case irt::features::ImageSearchPreprocessBackend::CPU:
        return "cpu";
    case irt::features::ImageSearchPreprocessBackend::GPU:
        return "gpu";
    }
    return "unknown";
}

const char *faissBackendName(irt::features::ImageSearchFaissBackend backend)
{
    switch (backend)
    {
    case irt::features::ImageSearchFaissBackend::CPU:
        return "cpu";
    case irt::features::ImageSearchFaissBackend::GPU:
        return "gpu";
    }
    return "unknown";
}

const char *indexStorageName(irt::features::ImageSearchIndexStorage storage)
{
    switch (storage)
    {
    case irt::features::ImageSearchIndexStorage::RAM:
        return "ram";
    case irt::features::ImageSearchIndexStorage::Disk:
        return "disk";
    }
    return "unknown";
}

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Build and query a Faiss image retrieval index from InferRT features");
    options.add_options()("weights-file,w", "Weights/model file (.wts, .onnx or OpenVINO IR)",
                          cxxopts::value<std::string>())(
        "gallery-dir,g", "Gallery image directory", cxxopts::value<std::string>())("query-image,q", "Query image path",
                                                                                   cxxopts::value<std::string>())(
        "model,m", "Built-in model name",
        cxxopts::value<std::string>()->default_value(irt::features::ImageSearch::kDefaultModelName))(
        "feature,f", "Feature tensor name",
        cxxopts::value<std::string>()->default_value(irt::features::ImageSearch::kDefaultFeatureName))(
        "topk", "Top-k nearest results",
        cxxopts::value<int>()->default_value(std::to_string(irt::features::ImageSearch::kDefaultTopK)))(
        "index", "Faiss index path", cxxopts::value<std::string>()->default_value(""))(
        "norm", "Feature norm mode: none, l1, l2", cxxopts::value<std::string>()->default_value("l2"))(
        "backend", "Feature extraction backend: tensorrt, openvino, onnxruntime",
        cxxopts::value<std::string>()->default_value("tensorrt"))("device", "Feature extraction device: cpu, gpu",
                                                                  cxxopts::value<std::string>()->default_value("gpu"))(
        "device-id", "GPU device id (zero-based)", cxxopts::value<int>()->default_value("0"))(
        "preprocess-backend", "Preprocess backend: cpu, gpu", cxxopts::value<std::string>()->default_value("cpu"))(
        "faiss-backend", "Faiss backend: cpu, gpu", cxxopts::value<std::string>()->default_value("cpu"))(
        "index-storage", "Index storage for CPU Faiss search: ram, disk",
        cxxopts::value<std::string>()->default_value("ram"))(
        "model-batch-size", "Feature extraction model inference batch size",
        cxxopts::value<size_t>()->default_value(std::to_string(irt::features::kDefaultImageSearchModelBatchSize)))(
        "rebuild-index", "Force rebuild of the Faiss index")("h,help", "Show help");
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
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default model: " << irt::features::ImageSearch::kDefaultModelName << std::endl;
        std::cout << "Default feature tensor: " << irt::features::ImageSearch::kDefaultFeatureName << std::endl;
        std::cout << "Default top-k: " << irt::features::ImageSearch::kDefaultTopK << std::endl;
        std::cout << "Default config: --norm l2 --backend tensorrt --device gpu --device-id 0 --preprocess-backend cpu"
                  << " --faiss-backend cpu --index-storage ram --model-batch-size "
                  << irt::features::kDefaultImageSearchModelBatchSize << std::endl;
        std::cout << "If --index is omitted, the sample writes <gallery_dir>/<timestamp>.faiss" << std::endl;
        std::cout << "DINO feature hint: use x_norm_clstoken for compact image-level retrieval" << std::endl;
        std::cout << "Supported models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            std::cout << ' ' << model_name;
        }
        std::cout << std::endl;
        throw HelpRequested{};
    }

    if (!result.count("weights-file") || !result.count("gallery-dir") || !result.count("query-image"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--weights-file, --gallery-dir and --query-image are required");
    }

    Arguments args;
    args.weights_file                 = result["weights-file"].as<std::string>();
    args.gallery_dir                  = result["gallery-dir"].as<std::string>();
    args.query_image                  = result["query-image"].as<std::string>();
    args.index_file                   = result["index"].as<std::string>();
    args.top_k                        = result["topk"].as<int>();
    args.config.model_name            = result["model"].as<std::string>();
    args.config.feature_name          = result["feature"].as<std::string>();
    args.config.norm                  = parseNorm(result["norm"].as<std::string>());
    args.config.model_backend         = parseModelBackend(result["backend"].as<std::string>());
    args.config.model_device          = parseModelDevice(result["device"].as<std::string>());
    args.config.model_device_id       = result["device-id"].as<int>();
    args.config.preprocess_backend    = parsePreprocessBackend(result["preprocess-backend"].as<std::string>());
    args.config.faiss_backend         = parseFaissBackend(result["faiss-backend"].as<std::string>());
    args.config.index_storage         = parseIndexStorage(result["index-storage"].as<std::string>());
    args.config.model_batch_size      = result["model-batch-size"].as<size_t>();
    args.rebuild_index                = result.count("rebuild-index") > 0;

    if (args.top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }

    if (!irt::model::isSupportedModel(args.config.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s",
                             args.config.model_name.c_str());
    }

    return args;
}

} // namespace

/**
 * @brief 构建或加载图像检索索引，并返回查询图片的 Top-K 相似结果。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);

        irt::features::ImageSearch searcher(args.config);

        const auto build_start = Clock::now();
        auto       progress_callback
            = [last_width = size_t{0}](const irt::features::ImageSearchBuildProgress &progress) mutable
        {
            std::ostringstream line;
            line << "Index build [" << irt::features::imageSearchBuildStageName(progress.stage) << "]";
            if (progress.total_count > 0)
            {
                line << ": " << progress.processed_count << "/" << progress.total_count;
            }

            const auto text = line.str();
            std::cout << '\r' << text;
            if (last_width > text.size())
            {
                std::cout << std::string(last_width - text.size(), ' ');
            }
            std::cout << std::flush;
            last_width = std::max(last_width, text.size());
            if (progress.stage == irt::features::ImageSearchBuildStage::Finished)
            {
                std::cout << std::endl;
                last_width = 0;
            }
        };
        searcher.buildOrLoad(args.weights_file, args.gallery_dir, args.index_file, args.rebuild_index,
                             progress_callback);
        const auto build_end = Clock::now();

        const auto search_start = Clock::now();
        const auto results      = searcher.search(args.query_image, args.top_k);
        const auto search_end   = Clock::now();

        std::cout << "Query image: " << fs::absolute(args.query_image).string() << std::endl;
        std::cout << "Model: " << searcher.config().model_name << ", feature tensor: " << searcher.config().feature_name
                  << std::endl;
        std::cout << "Config: backend=" << modelBackendName(searcher.config().model_backend)
                  << ", device=" << modelDeviceName(searcher.config().model_device)
                  << ", device_id=" << searcher.config().model_device_id
                  << ", norm=" << normName(searcher.config().norm)
                  << ", preprocess=" << preprocessBackendName(searcher.config().preprocess_backend)
                  << ", faiss=" << faissBackendName(searcher.config().faiss_backend)
                  << ", index_storage=" << indexStorageName(searcher.config().index_storage)
                  << ", model_batch_size=" << searcher.config().model_batch_size << std::endl;
        std::cout << "Index: " << fs::absolute(searcher.indexPath()).string() << std::endl;
        std::cout << "Top " << results.size() << " similar images:" << std::endl;

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << (i + 1) << ". score=" << results[i].score << " image_id=" << results[i].image_id
                      << std::endl;
        }

        std::cout << "Timing: build_or_load=" << elapsedMs(build_start, build_end)
                  << " ms, search=" << elapsedMs(search_start, search_end) << " ms" << std::endl;

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
