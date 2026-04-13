#pragma once

#include <inferrt/core/Export.h>

#include <string>

namespace irt {

// 获取完整版本信息字符串，包含版本号、分支、提交哈希和构建时间
INFERRT_CORE_API std::string GetFullVersionString();

// 获取版本号字符串，格式如 "0.0.1-beta"
INFERRT_CORE_API std::string GetVersionString();

// 获取 Git 分支名称
INFERRT_CORE_API std::string GetBranchString();

// 获取 Git 提交哈希值
INFERRT_CORE_API std::string GetCommitHashString();

// 获取构建时间字符串
INFERRT_CORE_API std::string GetBuildTimeString();

} // namespace irt
