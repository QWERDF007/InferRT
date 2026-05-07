#pragma once

#include <inferrt/core/Export.h>
#include <stdarg.h>
#include <stdint.h>

/**
 * @brief 状态码
 */
typedef enum
{
    IRT_SUCCESS = 0,             /**< 运算成功. */
    IRT_ERROR_NOT_IMPLEMENTED,   /**< 没有实现. */
    IRT_ERROR_INVALID_ARGUMENT,  /**< 无效参数. */
    IRT_ERROR_INVALID_OPERATION, /**< 无效运算. */
    IRT_ERROR_DEVICE,            /**< 设备后端错误. */
    IRT_ERROR_NOT_READY,         /**< 运算未完成, 稍后再试. */
    IRT_ERROR_OUT_OF_MEMORY,     /**< 内存不足. */
    IRT_ERROR_INTERNAL,          /**< 内部, 未指定错误. */
    IRT_ERROR_UNKNOWN            /**< 未知错误. */
} IRTStatus;

namespace irt {

/**
 * @brief 状态消息的最大长度（字节）
 *
 * 这是 \ref GetLastErrorMessage 和 \ref PeekAtLastErrorMessage 函数
 * 写入状态消息输出缓冲区的最大字节数，包括末尾的 '\0' 终止符。
 */
#define IRT_MAX_STATUS_MESSAGE_LENGTH (256)

/**
 * @brief 返回状态码的字符串表示
 *
 * @param [in] status 要获取字符串表示的状态码
 *
 * @return 状态码的字符串表示
 *
 * @note 返回的字符串在同一调用线程的下次调用之前保持有效
 * @note 返回的指针不应被释放
 */
INFERRT_CORE_API const char *StatusGetName(IRTStatus status);

/**
 * @brief 返回并重置当前线程中最后一次失败的 InferRT 函数调用的错误状态
 *
 * 再次调用此函数将返回 \ref IRT_SUCCESS，因为线程特定的状态已被重置。
 * 此操作不会影响其他线程中的状态。
 *
 * @return 当前线程中最后一次失败的 InferRT 函数调用的状态码
 */
INFERRT_CORE_API IRTStatus GetLastError();

/**
 * @brief 返回并重置当前线程中最后一次失败的 InferRT 函数调用的错误状态码和消息
 *
 * 再次调用此函数将返回 \ref IRT_SUCCESS，因为线程特定的状态已被重置。
 * 此操作不会影响其他线程中的状态。
 *
 * 保证消息长度不会超过 \ref IRT_MAX_STATUS_MESSAGE_LENGTH 字节（包括 '\0' 终止符）。
 *
 * @param[out] msg 指向用于写入状态消息的内存的指针。
 *                 如果为 NULL，则不返回消息。
 * @param[in] len msg 缓冲区的大小（字节）。
 *                如果小于零，则 len 被视为 0。
 *
 * @return 当前线程中最后一次失败的 InferRT 函数调用的状态码
 */
INFERRT_CORE_API IRTStatus GetLastErrorMessage(char *msg, int32_t len);

/**
 * @brief 返回当前线程中最后一次失败的 InferRT 函数调用的错误状态
 *
 * 当前线程的内部状态码和消息不会被重置。
 *
 * @return 当前线程中最后一次失败的 InferRT 函数调用的状态码
 */
INFERRT_CORE_API IRTStatus PeekAtLastError();

/**
 * @brief 返回当前线程中最后一次失败的 InferRT 函数调用的状态码和消息
 *
 * 当前线程的内部状态码和消息不会被重置。
 *
 * 保证消息长度不会超过 IRT_MAX_STATUS_MESSAGE_LENGTH 字节（包括 '\0' 终止符）。
 *
 * @param[out] msg 指向用于写入状态消息的内存的指针。如果为 NULL，则不返回消息。
 * @param[in] len msg 缓冲区的大小（字节）。如果小于零，则 len 被视为 0。
 *
 * @return 当前线程中最后一次失败的 InferRT 函数调用的状态码
 */
INFERRT_CORE_API IRTStatus PeekAtLastErrorMessage(char *msg, int32_t len);

/**
 * @brief 设置当前线程的内部状态
 *
 * 此函数供 InferRT 扩展和/或语言绑定使用，以便将其状态处理与 C API 无缝集成。
 *
 * @param[in] status 要设置的状态码
 * @param[in] fmt 与状态码关联的状态消息（printf 风格的格式字符串）。如果不需要自定义错误消息，则传递 NULL。
 * @param[in] ... 可变参数列表，用于格式化消息
 *
 * @note 在 GCC 编译器下，会进行 printf 格式检查
 */
INFERRT_CORE_API void SetThreadStatus(IRTStatus status, const char *fmt, ...)
#if __GNUC__
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/**
 * @brief 使用 va_list 设置当前线程的内部状态
 *
 * 此函数供 InferRT 扩展和/或语言绑定使用，以便将其状态处理
 * 与 C API 无缝集成。这是 SetThreadStatus 的 va_list 版本。
 *
 * @param[in] status 要设置的状态码
 * @param[in] fmt 与状态码关联的状态消息（printf 风格的格式字符串）。如果不需要自定义错误消息，则传递 NULL。
 * @param[in] va 可变参数列表，用于格式化消息
 *
 * @note 此函数通常在包装函数内部使用，用于转发可变参数
 */
INFERRT_CORE_API void SetThreadStatusVarArgList(IRTStatus status, const char *fmt, va_list va);

} // namespace irt