#pragma once

#include <inferrt/util/Export.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::util {

using ManifestEntries = std::vector<std::pair<std::string, std::string>>;
using ManifestMap     = std::unordered_map<std::string, std::string>;

INFERRT_UTIL_API std::string timestampFileStem();

INFERRT_UTIL_API std::filesystem::path manifestPathForDataFile(const std::filesystem::path &data_path);

INFERRT_UTIL_API std::filesystem::path ensureFileExtension(const std::filesystem::path &path,
                                                           std::string_view expected_extension);

INFERRT_UTIL_API std::filesystem::path resolveOutputFilePath(const std::filesystem::path &requested_path,
                                                             const std::filesystem::path &default_directory,
                                                             std::string_view expected_extension);

INFERRT_UTIL_API std::filesystem::path deriveOutputFilePathFromSource(const std::filesystem::path &source_path,
                                                                      std::string_view expected_extension);

INFERRT_UTIL_API void writeYamlManifest(const std::filesystem::path &manifest_path,
                                        const ManifestEntries       &entries);

INFERRT_UTIL_API ManifestMap loadYamlManifest(const std::filesystem::path &manifest_path);

INFERRT_UTIL_API std::string manifestValue(const ManifestMap &manifest, const std::string &key);

INFERRT_UTIL_API bool manifestHasValue(const ManifestMap &manifest, const std::string &key);

INFERRT_UTIL_API bool manifestValueEquals(const ManifestMap &manifest, const std::string &key,
                                          const std::string &expected);

INFERRT_UTIL_API bool manifestMatches(const ManifestMap &manifest, const ManifestEntries &expected);

} // namespace irt::util
