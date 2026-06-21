#include "Exception.hpp"

#include "Status.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace irt::core::priv {

Exception::Exception(IRTStatus code)
    : code_(code)
{
    snprintf(buffer_, sizeof(buffer_), "%s", GetName(code));
}

Exception::Exception(IRTStatus code, const char *fmt, va_list va)
    : code_(code)
{
    int written = snprintf(buffer_, sizeof(buffer_), "%s", GetName(code));

    if (fmt != nullptr && fmt[0] != '\0' && written > 0 && written < sizeof(buffer_))
    {
        int remain = sizeof(buffer_) - written;
        int used   = snprintf(buffer_ + written, remain, ": ");

        if (used > 0 && used < remain)
            vsnprintf(buffer_ + written + used, remain - used, fmt, va);
    }
}

Exception::Exception(IRTStatus code, const char *fmt, ...)
    : code_(code)
{
    int written = snprintf(buffer_, sizeof(buffer_), "%s", GetName(code));

    if (fmt && fmt[0] != '\0' && written > 0 && written < sizeof(buffer_))
    {
        int remain = sizeof(buffer_) - written;

        int used = snprintf(buffer_ + written, remain, ": ");
        if (used > 0 && used < remain)
        {
            va_list va;
            va_start(va, fmt);

            vsnprintf(buffer_ + written + used, remain - used, fmt, va);

            va_end(va);
        }
    }
}

IRTStatus Exception::code() const
{
    return code_;
}

const char *Exception::msg() const
{
    // Only return the message part
    const char *out = strchr(buffer_, ':');

    if (out != nullptr)
        return out + 2; // skip ': '
    else
        return ""; // 没有 msg 部分, 返回空
}

const char *Exception::what() const noexcept
{
    return buffer_;
}

} // namespace irt::core::priv
