/**
 * @file SampleDinoRegionSearch.cpp
 * @brief DINO 区域检索 CLI：build / search。
 *
 * stdout 只输出机器可读的 YAML（结果或报告），进度写 stderr；退出码遵循契约：
 * 0 完成、2 参数或 ROI 错误、3 权重/索引缺失或不兼容、4 建库含失败文件、5 查询未完成、6 内部错误。
 */

#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.hpp>
#include <inferrt/features/DinoRegionSearch.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Arguments
{
    std::string command{};
    std::string gallery{};
    std::string index{};
    std::string profile{};
    std::string query{};
    std::string request{};
    std::string requests{};
    std::string output{};
    std::string weights{};
    std::string backend{};
    std::string runtime{};
    std::string roi{};
    std::string polygon{};
    std::string scan_backend{"auto"};
    int         deadline_ms{-1};
    int         top_k{0};
    bool        include_self{false};
    bool        help{false};

    static double parseDouble(const std::string &text, const char *name);
};

void printUsage()
{
    std::cout << "inferrt_sample_dino_region_search <build|search|search-batch> [options]\n"
                 "  build --gallery <dir> --index <dir> --profile <profile.yaml>\n"
                 "  search --index <dir> --profile <profile.yaml>\n"
                 "         (--request <query.yaml> | --query <img> (--roi x0,y0,x1,y1 | --polygon x,y;x,y;x,y))\n"
                 "  search-batch --index <dir> --profile <profile.yaml> --requests <requests.yaml>\n"
                 "  --weights <path> --backend <tensorrt|onnxruntime|openvino> --device <cpu|gpu:0|tensorrt:0>\n"
                 "  --scan-backend <auto|cpu|cuda> --deadline-ms <n> --top-k <n> --include-self\n"
                 "  --output <result.yaml> --help\n";
}

std::vector<std::string> splitList(const std::string &text, const char separator)
{
    std::vector<std::string> values;
    std::string              current;
    std::istringstream       stream(text);
    while (std::getline(stream, current, separator))
    {
        if (!current.empty())
        {
            values.push_back(current);
        }
    }
    return values;
}

double Arguments::parseDouble(const std::string &text, const char *name)
{
    try
    {
        size_t position = 0;
        const auto value = std::stod(text, &position);
        if (position != text.size())
        {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    }
    catch (const std::exception &)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Option %s expects a number, got '%s'", name,
                             text.c_str());
    }
}

void printProgress(const std::string &stage, const size_t batch_index, const size_t processed,
                   const size_t total, const std::string &message)
{
    std::cerr << "[dino] " << stage;
    if (total > 0U)
    {
        std::cerr << " batch=" << batch_index << " (" << processed << "/" << total << ")";
    }
    if (!message.empty())
    {
        std::cerr << " " << message;
    }
    std::cerr << std::endl;
}

const char *buildStageName(const irt::features::DinoBuildStage stage)
{
    switch (stage)
    {
    case irt::features::DinoBuildStage::ScanningImages:  return "scanning_images";
    case irt::features::DinoBuildStage::LoadingModel:    return "loading_model";
    case irt::features::DinoBuildStage::ExtractingViews: return "extracting_views";
    case irt::features::DinoBuildStage::Quantizing:      return "quantizing";
    case irt::features::DinoBuildStage::WritingIndex:    return "writing_index";
    case irt::features::DinoBuildStage::Finalizing:      return "finalizing";
    default:                                            return "unknown";
    }
}

const char *searchStageName(const irt::features::DinoSearchStage stage)
{
    switch (stage)
    {
    case irt::features::DinoSearchStage::Decode:              return "decode";
    case irt::features::DinoSearchStage::QueryExtract:        return "query_extract";
    case irt::features::DinoSearchStage::RegionScan:          return "region_scan";
    case irt::features::DinoSearchStage::LocalScan:           return "local_scan";
    case irt::features::DinoSearchStage::LocalWindowRescore:  return "local_window_rescore";
    case irt::features::DinoSearchStage::Fusion:              return "fusion";
    case irt::features::DinoSearchStage::FineExtract:         return "fine_extract";
    case irt::features::DinoSearchStage::FineMatch:           return "fine_match";
    case irt::features::DinoSearchStage::Output:              return "output";
    default:                                                 return "unknown";
    }
}

std::string readTextFile(const fs::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open file: %s", path.string().c_str());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void writeTextFile(const fs::path &path, const std::string &content)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write file: %s", path.string().c_str());
    }
    stream << content;
}

std::vector<fs::path> collectGalleryImages(const fs::path &gallery_root)
{
    std::vector<fs::path> paths;
    if (!fs::exists(gallery_root))
    {
        return paths;
    }
    for (const auto &entry : fs::recursive_directory_iterator(gallery_root))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const auto ext = entry.path().extension().string();
        std::string lower = ext;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == ".jpg" || lower == ".jpeg" || lower == ".png" || lower == ".bmp" || lower == ".webp")
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end(), [](const fs::path &a, const fs::path &b) {
        return a.lexically_normal().generic_string() < b.lexically_normal().generic_string();
    });
    return paths;
}

irt::features::DinoRegionSearchConfig loadProfile(const Arguments &arguments)
{
    if (arguments.profile.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "This command requires --profile <profile.yaml>");
    }
    auto config = irt::features::dinoConfigFromYaml(readTextFile(fs::u8path(arguments.profile)));

    // 命令行覆盖只作用于显式给出的字段，避免“看起来生效但没有生效”的隐式行为。
    if (!arguments.weights.empty())
    {
        config.model.weights_file = fs::u8path(arguments.weights);
    }
    if (!arguments.runtime.empty())
    {
        config.runtime.model_runtime = irt::model::ModelRuntime(arguments.runtime);
    }
    if (!arguments.backend.empty())
    {
        const auto device = config.runtime.model_runtime.isCpu() ? "cpu" : std::to_string(config.runtime.model_runtime.deviceId());
        config.runtime.model_runtime = irt::model::ModelRuntime(arguments.backend + ":" + device);
    }
    if (arguments.scan_backend == "cpu")
    {
        config.runtime.scan_backend = irt::features::DinoScanBackend::Cpu;
    }
    else if (arguments.scan_backend == "cuda")
    {
        config.runtime.scan_backend = irt::features::DinoScanBackend::Cuda;
    }
    else if (arguments.scan_backend == "auto")
    {
        config.runtime.scan_backend = irt::features::DinoScanBackend::Auto;
    }
    config.validate();
    return config;
}


irt::features::DinoSearchRequest buildRequest(const Arguments &arguments, const irt::features::DinoRegionSearchConfig &config)
{
    if (!arguments.request.empty())
    {
        auto request = irt::features::dinoSearchRequestFromYaml(readTextFile(fs::u8path(arguments.request)));
        if (arguments.top_k > 0)
        {
            request.top_k = static_cast<size_t>(arguments.top_k);
        }
        if (arguments.include_self)
        {
            request.include_self = true;
        }
        if (request.preset_id.empty())
        {
            request.preset_id = config.preset_id;
        }
        if (arguments.deadline_ms > 0)
        {
            request.deadline_ms = arguments.deadline_ms;
        }
        if (!request.image_resolver && !arguments.gallery.empty() && fs::exists(fs::u8path(arguments.gallery)))
        {
            const auto gallery_images = collectGalleryImages(fs::u8path(arguments.gallery));
            request.image_resolver = [gallery_images](int64_t image_id) -> fs::path {
                if (image_id >= 0 && static_cast<size_t>(image_id) < gallery_images.size()) {
                    return gallery_images[static_cast<size_t>(image_id)];
                }
                return {};
            };
        }
        return request;
    }

    irt::features::DinoSearchRequest request;
    request.request_id   = "cli-query";
    if (arguments.deadline_ms > 0)
    {
        request.deadline_ms = arguments.deadline_ms;
    }
    request.query_path   = fs::u8path(arguments.query);
    request.preset_id    = config.preset_id;
    request.top_k        = arguments.top_k > 0 ? static_cast<size_t>(arguments.top_k) : 0U;
    request.include_self = arguments.include_self;
    if (!arguments.gallery.empty() && fs::exists(fs::u8path(arguments.gallery)))
    {
        const auto gallery_images = collectGalleryImages(fs::u8path(arguments.gallery));
        request.image_resolver = [gallery_images](int64_t image_id) -> fs::path {
            if (image_id >= 0 && static_cast<size_t>(image_id) < gallery_images.size()) {
                return gallery_images[static_cast<size_t>(image_id)];
            }
            return {};
        };
    }
    if (request.query_path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "search requires --request or --query");
    }

    if (!arguments.roi.empty())
    {
        const auto values = splitList(arguments.roi, ',');
        if (values.size() != 4U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--roi expects x0,y0,x1,y1");
        }
        request.roi.has_bbox = true;
        request.roi.bbox     = irt::features::DinoSearchRect{
            static_cast<float>(Arguments::parseDouble(values[0], "--roi")), static_cast<float>(Arguments::parseDouble(values[1], "--roi")),
            static_cast<float>(Arguments::parseDouble(values[2], "--roi")), static_cast<float>(Arguments::parseDouble(values[3], "--roi"))};
        return request;
    }
    if (!arguments.polygon.empty())
    {
        request.roi.has_polygon = true;
        for (const auto &vertex : splitList(arguments.polygon, ';'))
        {
            const auto values = splitList(vertex, ',');
            if (values.size() != 2U)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--polygon expects x,y;x,y;x,y");
            }
            request.roi.polygon.push_back(irt::features::DinoSearchPoint{
                static_cast<float>(Arguments::parseDouble(values[0], "--polygon")),
                static_cast<float>(Arguments::parseDouble(values[1], "--polygon"))});
        }
        return request;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "search requires --roi or --polygon");
}


int cliExitCode(const irt::Status status) noexcept
{
    switch (status)
    {
    case irt::Status::ERROR_INVALID_ARGUMENT:
    case irt::Status::INVALID_OPERATION:
        return 2;
    case irt::Status::ERROR_NOT_IMPLEMENTED:
    case irt::Status::NOT_READY:
    case irt::Status::ERROR_DEVICE:
        return 3;
    case irt::Status::ERROR_OUT_OF_MEMORY:
        return 4;
    default:
        return 6;
    }
}

void emitCliError(const Arguments &arguments, const std::string &message)
{
    irt::features::DinoSearchResponse response;
    response.request_id = "cli-query";
    response.status     = irt::features::DinoSearchStatus::Failed;
    response.decision   = irt::features::DinoSearchDecision::Error;
    response.message    = message;
    const auto yaml     = irt::features::dinoSearchResponseToYaml(response);
    std::cout << yaml;
    if (!arguments.output.empty())
    {
        try
        {
            writeTextFile(fs::u8path(arguments.output), yaml);
        }
        catch (const std::exception &error)
        {
            std::cerr << "[dino] failed to write error response: " << error.what() << std::endl;
        }
    }
}


} // namespace

int main(int argc, char **argv)
{
    Arguments arguments;
    try
    {
        cxxopts::Options options("inferrt_sample_dino_region_search",
                                 "Frozen DINO region search over unlabelled galleries");
        options.add_options()
            ("command", "build or search", cxxopts::value<std::string>(arguments.command))
            ("gallery", "gallery root", cxxopts::value<std::string>(arguments.gallery))
            ("index", "index root", cxxopts::value<std::string>(arguments.index))
            ("profile", "profile YAML", cxxopts::value<std::string>(arguments.profile))
            ("query", "query image", cxxopts::value<std::string>(arguments.query))
            ("request", "query request YAML", cxxopts::value<std::string>(arguments.request))
            ("requests", "batch request YAML sequence", cxxopts::value<std::string>(arguments.requests))
            ("output", "output file", cxxopts::value<std::string>(arguments.output))
            ("weights", "backbone weights", cxxopts::value<std::string>(arguments.weights))
            ("backend", "inference backend", cxxopts::value<std::string>(arguments.backend))
            ("device", "runtime target", cxxopts::value<std::string>(arguments.runtime))
            ("roi", "x0,y0,x1,y1", cxxopts::value<std::string>(arguments.roi))
            ("polygon", "x,y;x,y;x,y", cxxopts::value<std::string>(arguments.polygon))
            ("top-k", "top K", cxxopts::value<int>(arguments.top_k))
            ("scan-backend", "similarity backend: auto | cpu | cuda", cxxopts::value<std::string>(arguments.scan_backend))
            ("deadline-ms", "query deadline ms", cxxopts::value<int>(arguments.deadline_ms))
            ("include-self", "allow returning the query itself",
                cxxopts::value<bool>(arguments.include_self)->default_value("false")->implicit_value("true"))
            ("help", "print usage",
                cxxopts::value<bool>(arguments.help)->default_value("false")->implicit_value("true"));
        options.parse_positional({"command"});

        const auto result = options.parse(argc, argv);
        if (arguments.help || result.count("help") > 0U || argc < 2)
        {
            printUsage();
            return 0;
        }
        if (arguments.command != "build" && arguments.command != "search" && arguments.command != "search-batch")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported command: %s", arguments.command.c_str());
        }
        if (arguments.index.empty() || arguments.top_k < 0 || arguments.deadline_ms == 0 || arguments.deadline_ms < -1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Require --index and positive --top-k / --deadline-ms overrides");
        }
        if ((!arguments.request.empty() && (!arguments.query.empty() || !arguments.roi.empty() || !arguments.polygon.empty())) ||
            (!arguments.roi.empty() && !arguments.polygon.empty()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Use --request or --query with exactly one ROI form");
        }

        if (arguments.command == "build")
        {
            const auto config = loadProfile(arguments);
            if (arguments.gallery.empty())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "build requires --gallery <dir>");
            }
            const auto report = irt::features::DinoRegionSearch::build(
                fs::u8path(arguments.gallery), config, fs::u8path(arguments.index), [](const irt::features::DinoBuildProgress &progress)
                { printProgress(buildStageName(progress.stage), progress.batch_index, progress.processed_count, progress.total_count, progress.message); });
            const auto yaml = irt::features::dinoBuildReportToYaml(report);
            std::cout << yaml;
            if (arguments.output.empty() == false)
            {
                writeTextFile(fs::u8path(arguments.output), yaml);
            }
            return report.failed_image_count > 0U ? 4 : 0;
        }


        if (arguments.command == "search-batch")
        {
            if (arguments.requests.empty())
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "search-batch requires --requests");
            const auto config = loadProfile(arguments);
            auto requests = irt::features::dinoSearchRequestsFromYaml(readTextFile(fs::u8path(arguments.requests)));
            std::ofstream file;
            if (!arguments.output.empty()) {
                file.open(fs::u8path(arguments.output), std::ios::binary);
                if (!file) throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot write batch output");
            }
            int exit_code = 0;
            std::function<fs::path(int64_t)> batch_resolver;
            if (!arguments.gallery.empty() && fs::exists(fs::u8path(arguments.gallery))) {
                const auto gallery_images = collectGalleryImages(fs::u8path(arguments.gallery));
                batch_resolver = [gallery_images](int64_t image_id) -> fs::path {
                    if (image_id >= 0 && static_cast<size_t>(image_id) < gallery_images.size()) {
                        return gallery_images[static_cast<size_t>(image_id)];
                    }
                    return {};
                };
            }
            for (auto &request : requests) {
                if (arguments.top_k > 0) request.top_k = static_cast<size_t>(arguments.top_k);
                if (arguments.include_self) request.include_self = true;
                if (arguments.deadline_ms > 0) request.deadline_ms = arguments.deadline_ms;
                if (request.preset_id.empty()) request.preset_id = config.preset_id;
                if (!request.image_resolver && batch_resolver) request.image_resolver = batch_resolver;
                irt::features::DinoSearchResponse response;
                int item_code = 0;
                try {
                    response = irt::features::DinoRegionSearch::search(fs::u8path(arguments.index), request, config);
                    item_code = response.status == irt::features::DinoSearchStatus::Incomplete ? 5 :
                                response.status == irt::features::DinoSearchStatus::Failed ? 6 : 0;
                } catch (const irt::Exception &error) {
                    response.request_id = request.request_id;
                    response.status = irt::features::DinoSearchStatus::Failed;
                    response.decision = irt::features::DinoSearchDecision::Error;
                    response.message = error.msg();item_code = cliExitCode(error.code());
                } catch (const std::exception &error) {
                    response.request_id = request.request_id;
                    response.status = irt::features::DinoSearchStatus::Failed;
                    response.decision = irt::features::DinoSearchDecision::Error;
                    response.message = error.what();item_code = 6;
                }
                if (!exit_code && item_code) exit_code = item_code;
                const auto yaml = "---\n" + irt::features::dinoSearchResponseToYaml(response);
                std::cout << yaml;
                if (file.is_open()) {file << yaml;file.flush();if (!file) throw std::runtime_error("Batch output write failed");}
            }
            return exit_code;
        }

        if (arguments.command == "search")
        {
            const auto config   = loadProfile(arguments);
            const auto request  = buildRequest(arguments, config);
            const auto response = irt::features::DinoRegionSearch::search(
                fs::u8path(arguments.index), request, config, [](const irt::features::DinoSearchProgress &progress)
                { printProgress(searchStageName(progress.stage), progress.batch_index, progress.processed_count, progress.total_count, progress.message); });
            const auto yaml = irt::features::dinoSearchResponseToYaml(response);
            std::cout << yaml;
            if (!arguments.output.empty())
            {
                writeTextFile(fs::u8path(arguments.output), yaml);
            }
            if (response.status == irt::features::DinoSearchStatus::Incomplete)
            {
                return 5;
            }
            return 0;
        }


        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported command: %s",
                             arguments.command.c_str());
    }
    catch (const cxxopts::exceptions::exception &error)
    {
        emitCliError(arguments, error.what());
        return 2;
    }
    catch (const irt::Exception &error)
    {
        emitCliError(arguments, error.msg());
        return cliExitCode(error.code());
    }
    catch (const std::exception &error)
    {
        emitCliError(arguments, error.what());
        return 6;
    }
}
