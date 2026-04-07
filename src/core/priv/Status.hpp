#pragma once

#include <inferrt/core/Status.hpp>

#include <exception>

namespace irt::core::priv {

void SetThreadError(std::exception_ptr e);

IRTStatus PeekAtLastThreadError(char *msg, int32_t len) noexcept;
IRTStatus GetLastThreadError(char *msg, int32_t len) noexcept;

IRTStatus PeekAtLastThreadError() noexcept;
IRTStatus GetLastThreadError() noexcept;

const char *GetName(IRTStatus code);

template<class F>
IRTStatus ProtectCall(F &&fn)
{
    try
    {
        fn();
        return IRT_SUCCESS;
    }
    catch (...)
    {
        // 设置 tls 中的错误码和错误信息
        SetThreadError(std::current_exception());
        return PeekAtLastThreadError();
    }
}

} // namespace irt::core::priv