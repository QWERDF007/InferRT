#include <inferrt/util/FileManifest.hpp>

#include <inferrt/core/Exception.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace irt::util {
namespace {

std::string normalizeExtension(std::string_view extension)
{
    if (extension.empty())
    {
        return {};
    }

    std::string normalized(extension);
    if (normalized.front() != '.')
    {
        normalized.insert(normalized.begin(), '.');
    }
    return normalized;
}

} // namespace

std::string timestampFileStem()
{
    const auto now       = std::chrono::system_clock::now();
    const auto seconds   = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto millis    = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();
    const auto time_tnow = std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &time_tnow);
#else
    localtime_r(&time_tnow, &local_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0') << millis;
    return stream.str();
}

fs::path manifestPathForDataFile(const fs::path &data_path)
{
    auto manifest_path = data_path;
    manifest_path.replace_extension(".manifest.yaml");
    return manifest_path;
}

fs::path ensureFileExtension(const fs::path &path, std::string_view expected_extension)
{
    const auto extension = normalizeExtension(expected_extension);
    if (path.empty() || extension.empty())
    {
        return path;
    }

    auto with_extension = path;
    if (with_extension.extension() != extension)
    {
        with_extension.replace_extension(extension);
    }
    return with_extension;
}

fs::path resolveOutputFilePath(const fs::path &requested_path, const fs::path &default_directory,
                               std::string_view expected_extension)
{
    const auto extension = normalizeExtension(expected_extension);
    std::error_code ec;
    const bool requested_is_directory = !requested_path.empty() && fs::is_directory(requested_path, ec);
    if (requested_path.empty() || !requested_path.has_filename() || requested_is_directory)
    {
        const fs::path directory = requested_path.empty()
                                     ? (default_directory.empty() ? fs::current_path() : default_directory)
                                     : requested_path;
        return directory / (timestampFileStem() + extension);
    }

    return ensureFileExtension(requested_path, extension);
}

fs::path deriveOutputFilePathFromSource(const fs::path &source_path, std::string_view expected_extension)
{
    const auto extension = normalizeExtension(expected_extension);
    if (source_path.empty() || !source_path.has_stem())
    {
        return fs::current_path() / (timestampFileStem() + extension);
    }

    auto output_path = source_path;
    output_path.replace_extension(extension);
    return output_path;
}

void writeYamlManifest(const fs::path &manifest_path, const ManifestEntries &entries)
{
    if (!manifest_path.parent_path().empty())
    {
        fs::create_directories(manifest_path.parent_path());
    }

    std::ofstream output(manifest_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file: %s",
                             manifest_path.string().c_str());
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    for (const auto &[key, value] : entries)
    {
        emitter << YAML::Key << key << YAML::Value << value;
    }
    emitter << YAML::EndMap;

    if (!emitter.good())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to serialize YAML manifest: %s",
                             manifest_path.string().c_str());
    }
    output << emitter.c_str() << '\n';
}

ManifestMap loadYamlManifest(const fs::path &manifest_path)
{
    std::error_code ec;
    if (!fs::exists(manifest_path, ec))
    {
        return {};
    }

    ManifestMap manifest;
    try
    {
        const YAML::Node root = YAML::LoadFile(manifest_path.string());
        if (!root || root.IsNull())
        {
            return {};
        }
        if (!root.IsMap())
        {
            return {};
        }

        for (const auto &entry : root)
        {
            if (!entry.first.IsScalar() || !entry.second.IsScalar())
            {
                continue;
            }
            manifest[entry.first.as<std::string>()] = entry.second.as<std::string>();
        }
    }
    catch (const YAML::Exception &)
    {
        return {};
    }
    return manifest;
}

std::string manifestValue(const ManifestMap &manifest, const std::string &key)
{
    const auto it = manifest.find(key);
    return it == manifest.end() ? std::string{} : it->second;
}

bool manifestHasValue(const ManifestMap &manifest, const std::string &key)
{
    return manifest.find(key) != manifest.end();
}

bool manifestValueEquals(const ManifestMap &manifest, const std::string &key, const std::string &expected)
{
    const auto it = manifest.find(key);
    return it != manifest.end() && it->second == expected;
}

bool manifestMatches(const ManifestMap &manifest, const ManifestEntries &expected)
{
    for (const auto &[key, value] : expected)
    {
        if (!manifestValueEquals(manifest, key, value))
        {
            return false;
        }
    }
    return true;
}

} // namespace irt::util
