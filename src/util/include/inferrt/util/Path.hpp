/**
 * @file Path.hpp
 *
 * @brief 路径相关工具函数声明
 */

#pragma once

#include <inferrt/util/Export.h>

#include <filesystem>
#include <initializer_list>

namespace irt::util {

/**
 * @brief 从多个候选起点向上查找项目根目录。
 *
 * @details 查找顺序依次为 `source_file` 所在目录、当前工作目录、可执行程序所在目录。
 * 对每个起点会逐级向父目录回溯；当某个目录同时包含 `required_paths` 中指定的全部相对路径时，
 * 该目录即被视为项目根目录并返回。
 *
 * @param program_name 可执行程序路径，通常传入 `argv[0]`。
 * @param required_paths 用于判定项目根目录必须存在的相对路径集合。
 * @param source_file 源文件路径，通常传入 `__FILE__`，可为空。
 * @return 查找到的项目根目录；若未找到，则返回当前工作目录。
 */
INFERRT_UTIL_API std::filesystem::path findProjectRoot(const char                                  *program_name,
                                                       std::initializer_list<std::filesystem::path> required_paths,
                                                       const char *source_file = nullptr);

} // namespace irt::util
