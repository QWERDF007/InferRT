#pragma once

#include <inferrt/core/Status.h>

#include <exception>

namespace irt::core::priv {

/**
 * @brief 内部异常类，用于封装 InferRT 错误状态码和错误消息
 * 
 * 该类继承自 std::exception，提供了格式化错误消息的能力，
 * 并将 IRTStatus 错误码与异常机制结合使用。
 */
class Exception : public std::exception
{
public:
    /**
     * @brief 使用 va_list 构造异常
     * @param code 错误状态码
     * @param fmt 格式化字符串（printf 风格）
     * @param va 可变参数列表
     */
    explicit Exception(IRTStatus code, const char *fmt, va_list va);

    /**
     * @brief 使用可变参数构造异常
     * @param code 错误状态码
     * @param fmt 格式化字符串（printf 风格）
     * @param ... 可变参数
     */
    explicit Exception(IRTStatus code, const char *fmt, ...)
#if __GNUC__
        // first argument is actually 'this'
        __attribute__((format(printf, 3, 4)));
#else
        ;
#endif

    /**
     * @brief 仅使用错误码构造异常
     * @param code 错误状态码
     */
    explicit Exception(IRTStatus code);

    /**
     * @brief 获取错误状态码
     * @return IRTStatus 错误码
     */
    IRTStatus code() const;

    /**
     * @brief 获取错误消息字符串
     * @return 错误消息的 C 字符串指针
     */
    const char *msg() const;

    /**
     * @brief 获取异常描述（重写 std::exception::what）
     * @return 异常描述的 C 字符串指针
     */
    const char *what() const noexcept override;

private:
    IRTStatus code_ = IRT_ERROR_INTERNAL;                        // 错误状态码，默认为内部错误
    char      buffer_[IRT_MAX_STATUS_MESSAGE_LENGTH + 64 + 2]{}; // 错误消息缓冲区
};

} // namespace irt::core::priv