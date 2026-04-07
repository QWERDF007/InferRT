#pragma once

#include "Status.hpp"

#include <cassert>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace irt::core {

/**
 * @brief InferRT 异常类
 *
 * 该类继承自 std::exception，用于表示 InferRT 库中的异常情况。
 * 它封装了错误状态码和格式化的错误消息，并自动设置线程局部存储的错误状态。
 */
class Exception : public std::exception
{
public:
    /**
     * @brief 构造一个异常对象
     *
     * @param code 错误状态码
     * @param fmt 格式化字符串（printf 风格），可选
     * @param ... 可变参数列表，用于格式化消息
     *
     * @note 该构造函数会自动设置当前线程的错误状态
     * @note 在 GCC 编译器下，会进行 printf 格式检查
     */
    explicit Exception(Status code, const char *fmt = nullptr, ...)
#if __GNUC__
        __attribute__((format(printf, 3, 4)))
#endif
        : code_(code)
    {
        va_list va;
        va_start(va, fmt);
        SetThreadStatusVarArgList(static_cast<IRTStatus>(code), fmt, va);
        va_end(va);

        va_start(va, fmt);
        doSetMessage(fmt, va);
        va_end(va);
    }

    /**
     * @brief 获取异常的状态码
     *
     * @return 错误状态码
     */
    Status code() const
    {
        return code_;
    }

    /**
     * @brief 获取异常的错误消息
     *
     * @return 指向错误消息字符串的指针（不包含状态码前缀）
     */
    const char *msg() const
    {
        return msg_;
    }

    /**
     * @brief 获取完整的异常消息
     *
     * 该函数重写了标准异常类的 what() 方法。
     * 返回的消息格式为："状态码名称: 错误消息"
     *
     * @return 完整的错误消息字符串
     */
    const char *what() const noexcept override
    {
        return msg_buffer_;
    }

private:
    Status      code_; ///< 错误状态码
    const char *msg_;  ///< 指向消息缓冲区中实际消息部分的指针

    // 消息缓冲区大小计算：
    // MAX_STATUS_MESSAGE_LENGTH: 最大消息长度
    // 64: 状态枚举字符串表示的最大长度
    // 2: 分隔符 ": " 的长度
    char msg_buffer_[MAX_STATUS_MESSAGE_LENGTH + 64 + 2];

    /**
     * @brief 设置格式化的错误消息
     *
     * 该函数将状态码名称和格式化的消息组合到消息缓冲区中。
     * 消息格式为："状态码名称: 格式化消息"
     *
     * @param fmt 格式化字符串
     * @param va 可变参数列表
     */
    void doSetMessage(const char *fmt, va_list va)
    {
        int buflen   = sizeof(msg_buffer_);
        int nwritten = snprintf(msg_buffer_, buflen, "%s: ", StatusGetName(code_));

        // 检查是否有足够的空间写入消息
        if (nwritten < buflen)
        {
            buflen -= nwritten;
            msg_ = msg_buffer_ + nwritten;
            vsnprintf(msg_buffer_ + nwritten, buflen, fmt, va);
        }

        // 确保字符串以 null 结尾
        msg_buffer_[sizeof(msg_buffer_) - 1] = '\0';
    }
};

/**
 * @brief 根据捕获的异常设置线程错误状态
 *
 * 该函数尝试重新抛出给定的异常，并根据异常类型设置当前线程的相应错误状态。
 *
 * @param e 捕获的异常指针，如果为空则设置成功状态
 */
inline void SetThreadError(std::exception_ptr e)
{
    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            SetThreadStatus(IRT_SUCCESS, nullptr);
        }
    }
    catch (const Exception &e) // InferRT 自定义异常，使用其状态码和消息
    {
        SetThreadStatus(static_cast<IRTStatus>(e.code()), "%s", e.msg());
    }
    catch (const std::invalid_argument &e) // 无效参数异常
    {
        SetThreadStatus(IRT_ERROR_INVALID_ARGUMENT, "%s", e.what());
    }
    catch (const std::bad_alloc &) // 内存分配失败异常
    {
        SetThreadStatus(IRT_ERROR_OUT_OF_MEMORY, "Not enough space for resource allocation");
    }
    catch (const std::exception &e) // 其他标准异常
    {
        SetThreadStatus(IRT_ERROR_INTERNAL, "%s", e.what());
    }
    catch (...) // 未知异常
    {
        SetThreadStatus(IRT_ERROR_INTERNAL, "Unexpected error");
    }
}

/**
 * @brief 安全地执行函数，捕获并设置可能产生的异常
 *
 * 该函数作为包装器，用于安全地执行给定的函数或 lambda 表达式。
 * 如果函数抛出任何异常，异常会被捕获，并通过 SetThreadError 函数
 * 设置当前线程的错误状态。
 *
 * 这是 C API 边界的关键函数，用于将 C++ 异常转换为 C 风格的错误码。
 *
 * @tparam F 函数或 lambda 表达式的类型
 * @param fn 要执行的函数或 lambda 表达式
 * @return 如果 fn 执行成功返回 IRT_SUCCESS，否则返回捕获的异常对应的错误码
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
        // 捕获所有异常并设置线程错误状态
        SetThreadError(std::current_exception());
        return PeekAtLastError();
    }
}

} // namespace irt::core