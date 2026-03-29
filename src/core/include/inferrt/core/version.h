#pragma once

#include "export.h"

#include <string>

namespace inferrt::core {

INFERRT_CORE_API std::string GetFullVersionString();
INFERRT_CORE_API std::string GetVersionString();
INFERRT_CORE_API std::string GetBranchString();
INFERRT_CORE_API std::string GetCommitHashString();

} // namespace inferrt::core
