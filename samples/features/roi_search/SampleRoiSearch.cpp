/**
 * @file SampleRoiSearch.cpp
 * @brief LabelMe ROI 特征检索示例，并输出各阶段耗时。
 */

#include <SampleSupport.hpp>

#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/RoiSearch.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/util/Timing.hpp>
#include <inferrt/util/String.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = irt::util::TimingClock;
using irt::samples::HelpRequested;
using irt::util::elapsedMs;
using irt::util::printTimingStats;
using irt::util::summarizeTimings;

constexpr const char *kDefaultModel   = "dinov3_vits16";
constexpr const char *kDefaultFeature = "x_norm_patchtokens";

struct Arguments
{
    fs::path    weights_file;
    fs::path    images_dir;
    fs::path    annotations_dir;
    fs::path    index_file{"build/roi_search/labelme.faiss"};
    std::string model_name{kDefaultModel};
    std::string feature_name{kDefaultFeature};
    std::string runtime{"tensorrt:0"};
    std::string precision{"fp16"};
    int         model_batch_size{8};
    int         pooled_height{7};
    int         pooled_width{7};
    int         sampling_ratio{-1};
    int         top_k{5};
    int         max_items{0};
    int         query_index{0};
    bool        rebuild_index{false};
    int         warmup{0};
    int         repeat{1};
};

struct LabelMeItem
{
    irt::features::RoiSearchItem item;
    std::string                   label;
};

struct BuildProfile
{
    using Stage = irt::features::ImageSearchBuildStage;

    Stage                              current{Stage::Unknown};
    Clock::time_point                  stage_begin{};
    std::map<Stage, double>             stage_ms;
    double                              total_ms{0.0};

    void update(const irt::features::ImageSearchBuildProgress &progress)
    {
        const auto now = Clock::now();
        if (current != progress.stage)
        {
            close(now);
            current    = progress.stage;
            stage_begin = now;
        }
    }

    void finish()
    {
        close(Clock::now());
    }

    void close(Clock::time_point now)
    {
        if (current != Stage::Unknown)
        {
            stage_ms[current] += elapsedMs(stage_begin, now);
        }
        current = Stage::Unknown;
    }
};

cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Build and query a LabelMe ROI feature search index");
    options.add_options()
        ("weights-file,w", "Weights file; TensorRT finds the same-name engine", cxxopts::value<std::string>())
        ("images-dir", "LabelMe images directory", cxxopts::value<std::string>())
        ("annotations-dir", "LabelMe annotations directory", cxxopts::value<std::string>())
        ("index", "Faiss index path", cxxopts::value<std::string>()->default_value("build/roi_search/labelme.faiss"))
        ("model,m", "Built-in model name", cxxopts::value<std::string>()->default_value(kDefaultModel))
        ("feature,f", "Spatial feature tensor", cxxopts::value<std::string>()->default_value(kDefaultFeature))
        ("runtime", "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id",
         cxxopts::value<std::string>()->default_value("tensorrt:0"))
        ("precision", "TensorRT model precision: fp32 or fp16",
         cxxopts::value<std::string>()->default_value("fp16"))
        ("model-batch-size", "Feature extraction batch size", cxxopts::value<int>()->default_value("8"))
        ("pooled-height", "ROIAlign output height", cxxopts::value<int>()->default_value("7"))
        ("pooled-width", "ROIAlign output width", cxxopts::value<int>()->default_value("7"))
        ("sampling-ratio", "ROIAlign sampling ratio (-1 is adaptive)", cxxopts::value<int>()->default_value("-1"))
        ("topk", "Number of query results", cxxopts::value<int>()->default_value("5"))
        ("max-items", "Maximum LabelMe ROIs to index; 0 means all", cxxopts::value<int>()->default_value("0"))
        ("query-index", "Gallery ROI index used as the query", cxxopts::value<int>()->default_value("0"))
        ("rebuild-index", "Force rebuilding the index")
        ("h,help", "Show help");
    irt::samples::addTimingOptions(options, "Timed ROI query iterations");
    return options;
}

Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "LabelMe JSON is parsed as YAML-compatible JSON; each shape becomes one ROI." << std::endl;
        throw HelpRequested{};
    }

    Arguments args;
    args.weights_file       = result["weights-file"].as<std::string>();
    args.images_dir         = result["images-dir"].as<std::string>();
    args.annotations_dir    = result["annotations-dir"].as<std::string>();
    args.index_file         = result["index"].as<std::string>();
    args.model_name         = result["model"].as<std::string>();
    args.feature_name       = result["feature"].as<std::string>();
    args.runtime            = result["runtime"].as<std::string>();
    args.precision          = result["precision"].as<std::string>();
    args.model_batch_size   = result["model-batch-size"].as<int>();
    args.pooled_height      = result["pooled-height"].as<int>();
    args.pooled_width       = result["pooled-width"].as<int>();
    args.sampling_ratio     = result["sampling-ratio"].as<int>();
    args.top_k              = result["topk"].as<int>();
    args.max_items          = result["max-items"].as<int>();
    args.query_index        = result["query-index"].as<int>();
    args.rebuild_index      = result.count("rebuild-index") > 0;
    const auto timing       = irt::samples::parseTimingOptions(result);
    args.warmup             = timing.warmup;
    args.repeat             = timing.repeat;

    if (args.weights_file.empty() || args.images_dir.empty() || args.annotations_dir.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--weights-file, --images-dir and --annotations-dir are required");
    }
    if (args.model_batch_size <= 0 || args.pooled_height <= 0 || args.pooled_width <= 0 || args.sampling_ratio < -1
        || args.top_k <= 0 || args.max_items < 0 || args.query_index < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI search numeric argument");
    }
    if (!irt::model::isSupportedModel(args.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", args.model_name.c_str());
    }
    args.precision = irt::util::toLower(irt::util::trim(std::move(args.precision)));
    if (args.precision != "fp32" && args.precision != "fp16")
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--precision must be fp32 or fp16");
    }
    return args;
}

double asFiniteDouble(const YAML::Node &node, const char *name)
{
    if (!node || !node.IsScalar())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "LabelMe %s must be a scalar", name);
    }
    const double value = node.as<double>();
    if (!std::isfinite(value))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "LabelMe %s must be finite", name);
    }
    return value;
}

fs::path resolveLabelMeImage(const fs::path &images_dir, const fs::path &annotation_path, const YAML::Node &root)
{
    if (const auto image_path = root["imagePath"]; image_path && image_path.IsScalar())
    {
        const fs::path candidate = images_dir / fs::path(image_path.as<std::string>()).filename();
        if (fs::is_regular_file(candidate))
        {
            return fs::absolute(candidate);
        }
    }

    const std::vector<std::string> extensions{".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    for (const auto &extension : extensions)
    {
        const fs::path candidate = images_dir / (annotation_path.stem().string() + extension);
        if (fs::is_regular_file(candidate))
        {
            return fs::absolute(candidate);
        }
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot find image for LabelMe annotation: %s",
                         annotation_path.string().c_str());
}

std::vector<LabelMeItem> loadLabelMeItems(const fs::path &images_dir, const fs::path &annotations_dir, int max_items)
{
    if (!fs::is_directory(images_dir) || !fs::is_directory(annotations_dir))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "LabelMe images/annotations directory is invalid");
    }

    std::vector<fs::path> annotation_paths;
    for (const auto &entry : fs::directory_iterator(annotations_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
        {
            annotation_paths.push_back(entry.path());
        }
    }
    std::sort(annotation_paths.begin(), annotation_paths.end());

    std::vector<LabelMeItem> items;
    int64_t                 next_id{0};
    for (const auto &annotation_path : annotation_paths)
    {
        const YAML::Node root = YAML::LoadFile(annotation_path.string());
        const auto        image_path = resolveLabelMeImage(images_dir, annotation_path, root);
        const auto        shapes      = root["shapes"];
        if (!shapes || !shapes.IsSequence())
        {
            continue;
        }

        for (const auto &shape : shapes)
        {
            const auto points = shape["points"];
            if (!points || !points.IsSequence() || points.size() < 2U)
            {
                continue;
            }
            double x1 = std::numeric_limits<double>::infinity();
            double y1 = std::numeric_limits<double>::infinity();
            double x2 = -std::numeric_limits<double>::infinity();
            double y2 = -std::numeric_limits<double>::infinity();
            for (const auto &point : points)
            {
                if (!point.IsSequence() || point.size() < 2U)
                {
                    continue;
                }
                const double x = asFiniteDouble(point[0], "point.x");
                const double y = asFiniteDouble(point[1], "point.y");
                x1             = std::min(x1, x);
                y1             = std::min(y1, y);
                x2             = std::max(x2, x);
                y2             = std::max(y2, y);
            }
            if (!(x2 > x1 && y2 > y1))
            {
                continue;
            }
            std::string label;
            if (const auto label_node = shape["label"]; label_node && label_node.IsScalar())
            {
                label = label_node.as<std::string>();
            }
            items.push_back({{next_id++, image_path,
                              {static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                               static_cast<float>(y2)}},
                             std::move(label)});
            if (max_items > 0 && static_cast<int>(items.size()) >= max_items)
            {
                return items;
            }
        }
    }

    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "No valid LabelMe polygon ROI was found");
    }
    return items;
}

const char *stageName(irt::features::ImageSearchBuildStage stage) noexcept
{
    return irt::features::imageSearchBuildStageName(stage);
}

void printBuildProfile(const BuildProfile &profile)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "timing (build): total=" << profile.total_ms << " ms\n";
    for (const auto stage : {irt::features::ImageSearchBuildStage::LoadingModel,
                             irt::features::ImageSearchBuildStage::ExtractingFeatures,
                             irt::features::ImageSearchBuildStage::BuildingIndex,
                             irt::features::ImageSearchBuildStage::LoadingIndex})
    {
        const auto it = profile.stage_ms.find(stage);
        if (it != profile.stage_ms.end())
        {
            std::cout << "  " << stageName(stage) << "=" << it->second << " ms";
            if (profile.total_ms > 0.0)
            {
                std::cout << " (" << (100.0 * it->second / profile.total_ms) << "%)";
            }
            std::cout << '\n';
        }
    }
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);

        const auto dataset_begin = Clock::now();
        const auto labelme_items = loadLabelMeItems(args.images_dir, args.annotations_dir, args.max_items);
        const auto dataset_ms    = elapsedMs(dataset_begin, Clock::now());
        if (args.query_index >= static_cast<int>(labelme_items.size()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--query-index is outside the ROI list");
        }

        irt::features::RoiSearchConfig config;
        config.model_name       = args.model_name;
        config.feature_name     = args.feature_name;
        config.model_runtime    = irt::model::ModelRuntime::parse(args.runtime);
        config.model_precision  = args.precision == "fp16" ? irt::model::ModelPrecision::FP16
                                                              : irt::model::ModelPrecision::FP32;
        config.model_batch_size = static_cast<size_t>(args.model_batch_size);
        config.pooled_height    = args.pooled_height;
        config.pooled_width     = args.pooled_width;
        config.sampling_ratio   = args.sampling_ratio;
        config.norm             = irt::features::ImageSearchFeatureNorm::L2;

        irt::features::RoiSearch searcher(config);
        BuildProfile            profile;
        const auto              build_begin = Clock::now();
        searcher.buildOrLoad(args.weights_file, [&]
        {
            std::vector<irt::features::RoiSearchItem> result;
            result.reserve(labelme_items.size());
            for (const auto &entry : labelme_items)
            {
                result.push_back(entry.item);
            }
            return result;
        }(), args.index_file, args.rebuild_index,
        [&](const irt::features::ImageSearchBuildProgress &progress)
        {
            profile.update(progress);
            if (progress.total_count > 0)
            {
                std::cout << "\rbuild " << stageName(progress.stage) << " " << progress.processed_count << "/"
                          << progress.total_count << std::flush;
            }
        });
        profile.finish();
        profile.total_ms = elapsedMs(build_begin, Clock::now());
        std::cout << "\n";

        const auto &query = labelme_items[static_cast<size_t>(args.query_index)];
        std::map<fs::path, size_t> unique_images;
        for (const auto &entry : labelme_items)
        {
            ++unique_images[entry.item.image_path];
        }
        std::vector<double> query_timings;
        query_timings.reserve(static_cast<size_t>(args.repeat));
        std::vector<irt::features::RoiSearchResult> results;
        for (int i = 0; i < args.warmup; ++i)
        {
            results = searcher.search(query.item.image_path, query.item.roi, args.top_k);
        }
        for (int i = 0; i < args.repeat; ++i)
        {
            const auto begin = Clock::now();
            results         = searcher.search(query.item.image_path, query.item.roi, args.top_k);
            query_timings.push_back(elapsedMs(begin, Clock::now()));
        }

        std::cout << "ROI items: " << labelme_items.size() << ", unique images: " << unique_images.size()
                  << ", query index: " << args.query_index << ", query label: " << query.label << "\n";
        std::cout << "query image: " << query.item.image_path.generic_string() << "\n";
        std::cout << "query roi: (" << query.item.roi.x1 << "," << query.item.roi.y1 << "," << query.item.roi.x2
                  << "," << query.item.roi.y2 << ")\n";
        std::cout << "index: " << fs::absolute(searcher.indexPath()).generic_string() << "\n";
        std::cout << "feature dim: " << searcher.featureDim() << "\n";
        std::cout << "top " << results.size() << ":\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << "  " << (i + 1) << ". score=" << results[i].score << " roi_id=" << results[i].roi_id;
            if (results[i].roi_id >= 0 && static_cast<size_t>(results[i].roi_id) < labelme_items.size())
            {
                const auto &hit = labelme_items[static_cast<size_t>(results[i].roi_id)];
                std::cout << " label=" << hit.label << " image=" << hit.item.image_path.filename().generic_string();
            }
            std::cout << '\n';
        }
        std::cout << "dataset parse: " << dataset_ms << " ms\n";
        printBuildProfile(profile);
        printTimingStats(std::cout, "query/RoiSearch::search", summarizeTimings(query_timings));
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
