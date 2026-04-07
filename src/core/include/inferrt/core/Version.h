#pragma once

#include "Export.h"

#include <string>

namespace irt::core {

INFERRT_CORE_API std::string GetFullVersionString();
INFERRT_CORE_API std::string GetVersionString();
INFERRT_CORE_API std::string GetBranchString();
INFERRT_CORE_API std::string GetCommitHashString();
INFERRT_CORE_API std::string GetBuildTimeString();

} // namespace irt::core
