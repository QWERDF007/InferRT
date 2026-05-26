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
    options.add_options()("weights-file,w", "Weights file (.wts)", cxxopts::value<std::string>())(
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
        "preprocess-backend", "Preprocess backend: cpu, gpu", cxxopts::value<std::string>()->default_value("cpu"))(
        "faiss-backend", "Faiss backend: cpu, gpu", cxxopts::value<std::string>()->default_value("cpu"))(
        "index-storage", "Index storage for CPU Faiss search: ram, disk",
        cxxopts::value<std::string>()->default_value("ram"))(
        "disk-build-batch-size", "Batch size used while building CPU disk indexes",
        cxxopts::value<size_t>()->default_value(std::to_string(irt::features::kDefaultImageSearchDiskBuildBatchSize)))(
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
        std::cout << "Default config: --norm l2 --preprocess-backend cpu --faiss-backend cpu --index-storage ram"
                  << " --disk-build-batch-size " << irt::features::kDefaultImageSearchDiskBuildBatchSize << std::endl;
        std::cout << "If --index is omitted, the sample uses <gallery_dir>/<model>_<feature>.faiss" << std::endl;
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
    args.config.preprocess_backend    = parsePreprocessBackend(result["preprocess-backend"].as<std::string>());
    args.config.faiss_backend         = parseFaissBackend(result["faiss-backend"].as<std::string>());
    args.config.index_storage         = parseIndexStorage(result["index-storage"].as<std::string>());
    args.config.disk_build_batch_size = result["disk-build-batch-size"].as<size_t>();
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
        searcher.buildOrLoad(args.weights_file, args.gallery_dir, args.index_file, args.rebuild_index);
        const auto build_end = Clock::now();

        const auto search_start = Clock::now();
        const auto results      = searcher.search(args.query_image, args.top_k);
        const auto search_end   = Clock::now();

        std::cout << "Query image: " << fs::absolute(args.query_image).string() << std::endl;
        std::cout << "Model: " << searcher.config().model_name << ", feature tensor: " << searcher.config().feature_name
                  << std::endl;
        std::cout << "Config: norm=" << normName(searcher.config().norm)
                  << ", preprocess=" << preprocessBackendName(searcher.config().preprocess_backend)
                  << ", faiss=" << faissBackendName(searcher.config().faiss_backend)
                  << ", index_storage=" << indexStorageName(searcher.config().index_storage)
                  << ", disk_build_batch_size=" << searcher.config().disk_build_batch_size << std::endl;
        std::cout << "Index: " << fs::absolute(searcher.indexPath()).string() << std::endl;
        std::cout << "Top " << results.size() << " similar images:" << std::endl;

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << (i + 1) << ". score=" << results[i].score << " image=" << results[i].image_path.string()
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
