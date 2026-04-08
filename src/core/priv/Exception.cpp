#include "Exception.hpp"

#include "Status.hpp"

#include <cstdio>
#include <string>

namespace irt::core::priv {

Exception::Exception(IRTStatus code)
    : Exception(code, "%s", "")
{
}

Exception::Exception(IRTStatus code, const char *fmt, va_list va)
    : code_(code)
{
    snprintf(buffer_, sizeof(buffer_) - 1, "%s", GetName(code));

    if (fmt != nullptr)
    {
        size_t len = std::char_traits<char>::length(buffer_);
        snprintf(buffer_ + len, sizeof(buffer_) - len - 1, ": ");

        len = std::char_traits<char>::length(buffer_);
        vsnprintf(buffer_ + len, sizeof(buffer_) - len - 1, fmt, va);
    }
}

Exception::Exception(IRTStatus code, const char *fmt, ...)
    : code_(code)
{
    va_list va;
    va_start(va, fmt);

    snprintf(buffer_, sizeof(buffer_) - 1, "%s", GetName(code));

    if (fmt != nullptr)
    {
        size_t len = std::char_traits<char>::length(buffer_);
        snprintf(buffer_ + len, sizeof(buffer_) - len - 1, ": ");

        len = std::char_traits<char>::length(buffer_);
        vsnprintf(buffer_ + len, sizeof(buffer_) - len - 1, fmt, va);
    }

    va_end(va);
}

IRTStatus Exception::code() const
{
    return code_;
}

const char *Exception::msg() const
{
    // Only return the message part
    const char *out = strchr(buffer_, ':');
    // NVCV_ASSERT(out != nullptr);

    return out += 2; // skip ': '
}

const char *Exception::what() const noexcept
{
    return buffer_;
}

} // namespace irt::core::priv