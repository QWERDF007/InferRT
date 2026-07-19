#pragma once

#include <inferrt/util/Export.h>

#include <string>
#include <string_view>
#include <vector>

namespace irt::util {

INFERRT_UTIL_API std::string trim(std::string value);

INFERRT_UTIL_API std::string toLower(std::string value);

/**
 * @brief 按任意分隔字符拆分字符串，并忽略空项。
 */
INFERRT_UTIL_API std::vector<std::string> split(std::string_view value,
                                                std::string_view delimiters = ",");

/**
 * @brief 将字符串转换为适合文件名使用的 stem。
 */
INFERRT_UTIL_API std::string sanitizeFileStem(std::string_view value,
                                              std::string_view fallback = "file");

} // namespace irt::util
