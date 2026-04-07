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

/**
 * @note \ref priv::ProtectCall 会在捕获异常时设置线程状态，包括错误码和错误信息
 */
void SetThreadStatus(IRTStatus status, const char *fmt, ...)
{
    std::cout << __FUNCTION__ << " " << __LINE__ << " " << status << std::endl;
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
    std::cout << __FUNCTION__ << " " << __LINE__ << " " << status << std::endl;
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