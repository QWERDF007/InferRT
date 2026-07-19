#pragma once

#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/util/File.hpp>
#include <inferrt/util/String.hpp>
#include <inferrt/util/Timing.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace irt::samples {

struct HelpRequested
{
};

struct TimingOptions
{
    int warmup{0};
    int repeat{1};
};

inline void addTimingOptions(cxxopts::Options &options, const char *repeat_description)
{
    options.add_options()("warmup", "Warmup iterations before timing", cxxopts::value<int>()->default_value("0"))(
        "repeat", repeat_description, cxxopts::value<int>()->default_value("1"));
}

inline TimingOptions parseTimingOptions(const cxxopts::ParseResult &result)
{
    TimingOptions options{result["warmup"].as<int>(), result["repeat"].as<int>()};
    if (options.warmup < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--warmup must be >= 0");
    }
    if (options.repeat <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--repeat must be > 0");
    }
    return options;
}

inline std::vector<std::filesystem::path> parsePathList(std::string_view value)
{
    std::vector<std::filesystem::path> paths;
    for (auto &token : irt::util::split(value, ",;"))
    {
        paths.emplace_back(token);
    }
    return paths;
}

inline std::filesystem::path resolvePath(const std::filesystem::path &project_root,
                                          const std::filesystem::path &configured_path)
{
    return configured_path.is_absolute() ? configured_path : project_root / configured_path;
}

inline std::vector<std::filesystem::path> resolvePaths(const std::filesystem::path              &project_root,
                                                       const std::vector<std::filesystem::path> &configured_paths,
                                                       const std::filesystem::path              &default_path)
{
    if (configured_paths.empty())
    {
        return {resolvePath(project_root, default_path)};
    }

    std::vector<std::filesystem::path> paths;
    paths.reserve(configured_paths.size());
    for (const auto &path : configured_paths)
    {
        paths.push_back(resolvePath(project_root, path));
    }
    return paths;
}

inline std::vector<std::string> readLabelNames(const std::filesystem::path &label_file)
{
    if (label_file.empty() || !std::filesystem::exists(label_file))
    {
        std::cout << "Label file not found, class ids will be printed: " << label_file.generic_string() << std::endl;
        return {};
    }
    return irt::util::readNonEmptyLines(label_file);
}

} // namespace irt::samples
