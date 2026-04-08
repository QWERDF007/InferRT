#include "Status.hpp"

#include "Exception.hpp"
#include "TLS.hpp"

#include <iostream>

namespace irt::core::priv {

void SetThreadError(std::exception_ptr e)
{
    CoreTLS &tls = GetCoreTLS();

    const int error_msg_len = sizeof(tls.last_error_message) - 1;

    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            tls.last_error_status = IRT_SUCCESS;
            snprintf(tls.last_error_message, error_msg_len, "success");
        }
    }
    catch (const Exception &e)
    {
        tls.last_error_status = static_cast<IRTStatus>(e.code());
        snprintf(tls.last_error_message, error_msg_len, "%s", e.msg());
    }
    catch (const std::invalid_argument &e)
    {
        tls.last_error_status = IRT_ERROR_INVALID_ARGUMENT;
        snprintf(tls.last_error_message, error_msg_len, "%s", e.what());
    }
    catch (const std::bad_alloc &)
    {
        tls.last_error_status = IRT_ERROR_OUT_OF_MEMORY;
        snprintf(tls.last_error_message, error_msg_len, "Not enough space for resource allocation");
    }
    catch (const std::exception &e)
    {
        tls.last_error_status = IRT_ERROR_INTERNAL;
        snprintf(tls.last_error_message, error_msg_len, "%s", e.what());
    }
    catch (...)
    {
        tls.last_error_status = IRT_ERROR_INTERNAL;
        snprintf(tls.last_error_message, error_msg_len, "Unexpected error");
    }

    tls.last_error_message[error_msg_len] = '\0'; // Make sure it's null-terminated
}

IRTStatus PeekAtLastThreadError(char *msg, int32_t len) noexcept
{
    CoreTLS &tls = GetCoreTLS();
    if (msg && len > 0)
        snprintf(msg, len, "%s", tls.last_error_message);
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

    // written this way, without a default case,
    // the compiler can warn us if we forgot to add a new error here.
    switch (code)
    {
        CASE(IRT_SUCCESS);
        CASE(IRT_ERROR_NOT_IMPLEMENTED);
        CASE(IRT_ERROR_INVALID_ARGUMENT);
        CASE(IRT_ERROR_INVALID_OPERATION);
        CASE(IRT_ERROR_DEVICE);
        CASE(IRT_ERROR_NOT_READY);
        CASE(IRT_ERROR_OUT_OF_MEMORY);
        CASE(IRT_ERROR_UNKNOWN);
    }

    // Status not found?
    return "Unknown error";
#undef CASE
}

} // namespace irt::core::priv