#pragma once

#include <inferrt/core/Status.h>

#include <exception>

namespace irt::core::priv {

/**
 * @brief 设置当前线程的错误异常
 * @param e 异常指针
 * 
 * 将异常信息存储到线程局部存储（TLS）中，用于后续错误查询
 */
void SetThreadError(std::exception_ptr e);

/**
 * @brief 查看当前线程的最后一个错误（不清除）
 * @param msg 用于存储错误消息的缓冲区
 * @param len 缓冲区长度
 * @return IRTStatus 错误状态码
 * 
 * 查询线程局部存储中的错误信息，但不清除它
 */
IRTStatus PeekAtLastThreadError(char *msg, int32_t len) noexcept;

/**
 * @brief 获取并清除当前线程的最后一个错误
 * @param msg 用于存储错误消息的缓冲区
 * @param len 缓冲区长度
 * @return IRTStatus 错误状态码
 * 
 * 查询线程局部存储中的错误信息，并清除它
 */
IRTStatus GetLastThreadError(char *msg, int32_t len) noexcept;

/**
 * @brief 查看当前线程的最后一个错误码（不清除，不获取消息）
 * @return IRTStatus 错误状态码
 */
IRTStatus PeekAtLastThreadError() noexcept;

/**
 * @brief 获取并清除当前线程的最后一个错误码（不获取消息）
 * @return IRTStatus 错误状态码
 */
IRTStatus GetLastThreadError() noexcept;

/**
 * @brief 获取错误状态码对应的名称字符串
 * @param code 错误状态码
 * @return 错误码名称的 C 字符串指针
 */
const char *GetName(IRTStatus code);

/**
 * @brief 保护性调用函数，捕获异常并转换为错误码
 * @tparam F 可调用对象类型
 * @param fn 要执行的函数或 lambda
 * @return IRTStatus 执行结果状态码，成功返回 IRT_SUCCESS，失败返回对应错误码
 * 
 * 该模板函数用于包装可能抛出异常的代码，将异常转换为 IRTStatus 错误码，
 * 并将异常信息存储到线程局部存储中
 */
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