#include <cuda_runtime.h>
#include <inferrt/util/CheckError.hpp>

#include <cstdarg>
#include <regex>
#include <sstream>

namespace irt::util {

static std::string_view GetFunctionName(const std::string_view &stmt)
{
    static std::regex rgx("^([A-Za-z0-9_]+)\\(.*$");

    std::match_results<std::string_view::const_iterator> match;
    if (regex_match(stmt.begin(), stmt.end(), match, rgx))
    {
        // Construct string_view from iterator and length
        return std::string_view(&(*match[1].first), std::distance(match[1].first, match[1].second));
    }
    else
    {
        return "";
    }
}

namespace detail {

const char *GetCheckMessage(char *buf, int bufsize)
{
    // NVCV_ASSERT(buf != nullptr);
    (void)buf;
    (void)bufsize;

    return "";
}

char *GetCheckMessage(char *buf, int bufsize, const char *fmt, ...)
{
    // NVCV_ASSERT(buf != nullptr);
    // NVCV_ASSERT(fmt != nullptr);

    va_list va;
    va_start(va, fmt);

    vsnprintf(buf, bufsize - 1, fmt, va);

    va_end(va);

    return buf;
}

std::string FormatErrorMessage(const std::string_view &errname, const std::string_view &callstr,
                               const std::string_view &msg)
{
    // TODO: avoid heap memory allocation here
    std::ostringstream ss;
    ss << "[ ";

    // 如果有完整的调用语句，直接打印完整语句（包括参数）
    if (!callstr.empty())
    {
        ss << callstr << " - ";
    }

    ss << errname << " ]: ";
    if (!msg.empty())
    {
        ss << msg;
    }

    return ss.str();
}

} // namespace detail

IRTStatus TranslateError(cudaError_t err)
{
    switch (err)
    {
    case cudaErrorMemoryAllocation:
        return IRT_ERROR_OUT_OF_MEMORY;

    case cudaErrorNotReady:
        return IRT_ERROR_NOT_READY;

    case cudaErrorInvalidValue:
        return IRT_ERROR_INVALID_ARGUMENT;

    default:
        return IRT_ERROR_INTERNAL;
    }
}

void PreprocessError(cudaError_t err)
{
    (void)err;
    // consume the error
    cudaGetLastError();
}

const char *ToString(cudaError_t err, const char **perrdescr)
{
    if (perrdescr != nullptr)
    {
        *perrdescr = cudaGetErrorString(err);
    }

    return cudaGetErrorName(err);
}

} // namespace irt::util