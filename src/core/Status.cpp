/**
 * @file Status.cpp
 *
 * @brief 转发设置线程状态到内部实现 priv:: 的相关函数中
 */

#pragma once

#include "priv/Status.hpp"

#include "priv/Exception.hpp"

#include <inferrt/core/Status.h>

#include <iostream>

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

/**
 * @note \ref priv::ProtectCall 会在捕获异常时设置线程状态，包括错误码和错误信息
 */
void SetThreadStatus(IRTStatus status, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    IRTStatus ret = core::priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw core::priv::Exception(status, fmt, va);
            }
            else
            {
                throw core::priv::Exception(status);
            }
        });
    (void)ret;
    va_end(va);
}

void SetThreadStatusVarArgList(IRTStatus status, const char *fmt, va_list va)
{
    IRTStatus ret = core::priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw core::priv::Exception(status, fmt, va);
            }
            else
            {
                throw core::priv::Exception(status);
            }
        });
    (void)ret;
}

} // namespace irt