#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ImageSearch.hpp>
#include <inferrt/model/IModel.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

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
    std::string model_name{irt::features::ImageSearch::kDefaultModelName};
    std::string feature_name{irt::features::ImageSearch::kDefaultFeatureName};
    fs::path    weights_file;
    fs::path    gallery_dir;
    fs::path    query_image;
    fs::path    index_file;
    int         top_k{irt::features::ImageSearch::kDefaultTopK};
    bool        rebuild_index{false};
};

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
        "model", "Built-in model name",
        cxxopts::value<std::string>()->default_value(irt::features::ImageSearch::kDefaultModelName))(
        "feature", "Feature tensor name",
        cxxopts::value<std::string>()->default_value(irt::features::ImageSearch::kDefaultFeatureName))(
        "topk", "Top-k nearest results",
        cxxopts::value<int>()->default_value(std::to_string(irt::features::ImageSearch::kDefaultTopK)))(
        "index", "Faiss index path", cxxopts::value<std::string>()->default_value(""))(
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
    args.model_name    = result["model"].as<std::string>();
    args.feature_name  = result["feature"].as<std::string>();
    args.weights_file  = result["weights-file"].as<std::string>();
    args.gallery_dir   = result["gallery-dir"].as<std::string>();
    args.query_image   = result["query-image"].as<std::string>();
    args.index_file    = result["index"].as<std::string>();
    args.top_k         = result["topk"].as<int>();
    args.rebuild_index = result.count("rebuild-index") > 0;

    if (args.top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }

    if (!irt::model::isSupportedModel(args.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", args.model_name.c_str());
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

        irt::features::ImageSearch searcher(args.model_name, args.feature_name);
        searcher.buildOrLoad(args.weights_file, args.gallery_dir, args.index_file, args.rebuild_index);
        const auto results = searcher.search(args.query_image, args.top_k);

        std::cout << "Query image: " << fs::absolute(args.query_image).string() << std::endl;
        std::cout << "Model: " << searcher.modelName() << ", feature tensor: " << searcher.featureName() << std::endl;
        std::cout << "Index: " << fs::absolute(searcher.indexPath()).string() << std::endl;
        std::cout << "Top " << results.size() << " similar images:" << std::endl;

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << (i + 1) << ". score=" << results[i].score << " image=" << results[i].image_path.string()
                      << std::endl;
        }

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
