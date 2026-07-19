#pragma once

#include <inferrt/util/Export.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace irt::util {

INFERRT_UTIL_API std::vector<std::string> readNonEmptyLines(const std::filesystem::path &file_path,
                                                            bool trim_lines = true);

INFERRT_UTIL_API void writeBinaryFile(const std::filesystem::path &file_path, const void *data, size_t num_bytes);

} // namespace irt::util
