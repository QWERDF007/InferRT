#include <inferrt/core/VersionDef.h>
#include <inferrt/core/detail/VersionUtils.h>
#include <inferrt/core/version.h>

#include <string>

namespace inferrt::core {

std::string GetVersionString()
{
    return INFERRT_VERSION_STRING;
}

std::string GetBranchString()
{
    return INFERRT_BRANCH;
}

std::string GetCommitHashString()
{
    return INFERRT_COMMIT;
}

std::string GetFullVersionString()
{
    std::string s = INFERRT_VERSION_STRING;
    s += " - [";
    s += INFERRT_BRANCH;
    s += "] - (";
    s += INFERRT_COMMIT;
    s += ")";
    return s;
}

std::string GetBuildTimeString()
{
    return INFERRT_BUILD_TIME;
}

} // namespace inferrt::core
