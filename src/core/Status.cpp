#pragma once

#include "priv/Status.hpp"

#include "priv/Exception.hpp"

#include <inferrt/core/Status.h>

namespace irt::core {

const char *StatusGetName(IRTStatus code)
{
    return priv::GetName(code);
}

IRTStatus GetLastError()
{
    return priv::GetLastThreadError();
}

IRTStatus GetLastErrorMessage(char *msg, int32_t len)
{
    return priv::GetLastThreadError(msg, len);
}

IRTStatus PeekAtLastError()
{
    return priv::PeekAtLastThreadError();
}

IRTStatus PeekAtLastErrorMessage(char *msg, int32_t len)
{
    return priv::PeekAtLastThreadError(msg, len);
}

void SetThreadStatus(IRTStatus status, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    IRTStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });

    va_end(va);
}

void SetThreadStatusVarArgList(IRTStatus status, const char *fmt, va_list va)
{
    IRTStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });
}

} // namespace irt::core