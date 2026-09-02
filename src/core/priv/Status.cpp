#include "Status.hpp"
#include "TLS.hpp"

#include <inferrt/core/Exception.hpp>

#include <cstdio>
#include <new>
#include <stdexcept>

namespace irt::core::priv {

void SetThreadError(std::exception_ptr e)
{
    irt::SetThreadError(e);
}

IRTStatus PeekAtLastThreadError(char *msg, int32_t len) noexcept
{
    CoreTLS &tls = GetCoreTLS();
    if (msg && len > 0)
    {
        snprintf(msg, static_cast<size_t>(len), "%s", tls.last_error_message);
        msg[len - 1] = '\0';
    }
    return tls.last_error_status;
}

IRTStatus PeekAtLastThreadError() noexcept
{
    return PeekAtLastThreadError(nullptr, 0);
}

IRTStatus GetLastThreadError(char *msg, int32_t len) noexcept
{
    IRTStatus status = PeekAtLastThreadError(msg, len);
    SetThreadError(std::exception_ptr{});
    return status;
}

IRTStatus GetLastThreadError() noexcept
{
    return GetLastThreadError(nullptr, 0);
}

const char *GetName(IRTStatus code)
{
#define CASE(ERR) \
    case ERR:     \
        return #ERR

    switch (code)
    {
        CASE(IRT_SUCCESS);
        CASE(IRT_ERROR_NOT_IMPLEMENTED);
        CASE(IRT_ERROR_INVALID_ARGUMENT);
        CASE(IRT_ERROR_INVALID_OPERATION);
        CASE(IRT_ERROR_DEVICE);
        CASE(IRT_ERROR_NOT_READY);
        CASE(IRT_ERROR_OUT_OF_MEMORY);
        CASE(IRT_ERROR_INTERNAL);
        CASE(IRT_ERROR_UNKNOWN);
    }

    return "Unknown error";
#undef CASE
}

} // namespace irt::core::priv