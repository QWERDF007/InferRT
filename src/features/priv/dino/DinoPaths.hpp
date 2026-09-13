#pragma once

/**
 * @file DinoPaths.hpp
 * @brief 文件系统路径与契约文本之间的唯一转换入口。
 *
 * 对外契约（profile / request / response / manifest）里的路径一律是 UTF-8 文本；
 * 而 ``std::filesystem::path`` 在本机是宽字符，二者的隐式转换在非 ASCII 路径上会既丢信息
 * 又产出非法 JSON。因此路径一旦离开 ``fs::path`` 必须经过这里，反向解析同理。
 */

#include <inferrt/features/Export.h>

#include <filesystem>
#include <string>

namespace irt::features::priv {

/** @brief 把路径编码为 UTF-8 文本（统一用 ``/`` 分隔，便于跨平台比较）。 */
INFERRT_FEATURES_API std::string dinoPathToUtf8(const std::filesystem::path &path);

/** @brief 按 UTF-8 解析路径文本；非 ASCII 路径在 Windows 上会走宽字符接口。 */
INFERRT_FEATURES_API std::filesystem::path dinoPathFromUtf8(const std::string &text);

} // namespace irt::features::priv
