/**
 * @file DinoPaths.cpp
 * @brief 路径与 UTF-8 文本互转实现。
 */

#include "DinoPaths.hpp"

namespace irt::features::priv {

std::string dinoPathToUtf8(const std::filesystem::path &path)
{
    const auto generic = path.generic_u8string();
    return std::string(generic.begin(), generic.end());
}

std::filesystem::path dinoPathFromUtf8(const std::string &text)
{
    std::u8string utf8;
    utf8.reserve(text.size());
    for (const char value : text)
    {
        utf8.push_back(static_cast<char8_t>(static_cast<unsigned char>(value)));
    }
    return std::filesystem::path(utf8);
}

} // namespace irt::features::priv
