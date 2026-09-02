#include "priv/Status.hpp"
#include "priv/TLS.hpp"

#include <inferrt/core/Status.h>
#include <inferrt/core/Status.hpp>

#include <cstdarg>
#include <cstdio>

namespace irt {

const char *StatusGetName(IRTStatus code)
{
    return core::priv::GetName(code);
}

IRTStatus GetLastError()
{
    return core::priv::GetLastThreadError();
}

IRTStatus GetLastErrorMessage(char *msg, int32_t len)
{
    return core::priv::GetLastThreadError(msg, len);
}

IRTStatus PeekAtLastError()
{
    return core::priv::PeekAtLastThreadError();
}

IRTStatus PeekAtLastErrorMessage(char *msg, int32_t len)
{
    return core::priv::PeekAtLastThreadError(msg, len);
}

void SetThreadStatus(IRTStatus status, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    SetThreadStatusVarArgList(status, fmt, va);
    va_end(va);
}

void SetThreadStatusVarArgList(IRTStatus status, const char *fmt, va_list va)
{
    core::priv::CoreTLS &tls           = core::priv::GetCoreTLS();
    tls.last_error_status              = status;
    const int error_msg_len            = static_cast<int>(sizeof(tls.last_error_message)) - 1;
    if (fmt != nullptr)
    {
        vsnprintf(tls.last_error_message, sizeof(tls.last_error_message), fmt, va);
    }
    else
    {
        tls.last_error_message[0] = '\0';
    }
    tls.last_error_message[error_msg_len] = '\0';
}

} // namespace irt