/**
 * @file CheckError.hpp
 *
 * @brief 可扩展的错误处理框架, 将各种错误码转换成异常或者日志消息
 *
 * @details 本文件提供了一个统一的错误检查和处理机制，支持：
 *          - 将不同类型的错误码（如 CUDA 错误）转换为统一的状态码
 *          - 通过宏自动检查函数调用结果并抛出异常或记录日志
 *          - 可扩展的错误类型支持，通过模板特化添加新的错误类型
 *
 * @note  对于自定义错误类型，需要定义以下函数：
 *        * inline bool CheckSucceeded(ErrorType err) - 判断错误码是否表示成功
 *        * const char *ToString(ErrorType err, const char **perrdescr=nullptr) - 将错误码转换为字符串
 *        * IRTStatus TranslateError(ErrorType err) - 将错误码转换为统一的 IRTStatus
 *        * void PreprocessError(ErrorType err) - 错误预处理（可选）
 */

#pragma once

#include "Assert.h"

#if INFERRT_HAS_CUDA
#include <driver_types.h> // for cudaError
#endif
#include <inferrt/core/Exception.hpp>
#include <inferrt/util/Export.h>
#include <iostream>

namespace irt::util {

namespace detail {
/**
 * @brief 获取检查消息（无格式化）
 * @param buf 缓冲区指针
 * @param buflen 缓冲区长度
 * @return 返回消息字符串指针
 */
INFERRT_UTIL_API const char *GetCheckMessage(char *buf, int buflen);

/**
 * @brief 获取格式化的检查消息
 * @param buf 缓冲区指针
 * @param buflen 缓冲区长度
 * @param fmt 格式化字符串（printf 风格）
 * @param ... 可变参数列表
 * @return 返回格式化后的消息字符串指针
 */
INFERRT_UTIL_API char *GetCheckMessage(char *buf, int buflen, const char *fmt, ...);

/**
 * @brief 格式化错误消息
 * @param errname 错误名称（如 "cudaErrorInvalidValue"）
 * @param callstr 调用语句字符串（如 "cudaMalloc(&ptr, size)"）
 * @param msg 附加消息
 * @return 返回格式化后的完整错误消息
 */
INFERRT_UTIL_API std::string FormatErrorMessage(const std::string_view &errname, const std::string_view &callstr,
                                                const std::string_view &msg);
} // namespace detail

#if INFERRT_HAS_CUDA
// ============================================================================
// CUDA 错误处理特化
// ============================================================================

/**
 * @brief 检查 CUDA 错误码是否表示成功
 * @param err CUDA 错误码
 * @return 如果错误码为 cudaSuccess 返回 true，否则返回 false
 */
inline bool CheckSucceeded(cudaError_t err)
{
    return err == cudaSuccess;
}

/**
 * @brief 将 CUDA 错误码转换为 InferRT 统一状态码
 * @param err CUDA 错误码
 * @return 对应的 IRTStatus 状态码
 */
INFERRT_UTIL_API IRTStatus TranslateError(cudaError_t err);

/**
 * @brief 将 CUDA 错误码转换为字符串描述
 * @param err CUDA 错误码
 * @param perrdescr 可选的输出参数，用于接收详细错误描述指针
 * @return 返回错误名称字符串（如 "cudaErrorInvalidValue"）
 */
INFERRT_UTIL_API const char *ToString(cudaError_t err, const char **perrdescr = nullptr);

/**
 * @brief CUDA 错误预处理函数
 * @param err CUDA 错误码
 * @details 在错误检查之前调用，可用于清除 CUDA 错误状态等操作
 */
INFERRT_UTIL_API void PreprocessError(cudaError_t err);
#endif

// ============================================================================
// 默认错误处理实现（模板）
// ============================================================================
// 这些模板函数为未特化的错误类型提供默认行为
// 对于自定义错误类型，应该提供特化版本以实现正确的错误处理

/**
 * @brief 检查错误码是否表示成功（默认实现）
 * @tparam T 错误码类型
 * @param err 错误码
 * @return 如果错误码等于 IRT_SUCCESS 返回 true，否则返回 false
 * @note 对于自定义错误类型，应该提供特化版本
 */
template<class T>
inline bool CheckSucceeded(T err)
{
    return err == IRT_SUCCESS;
}

/**
 * @brief 将错误码转换为 InferRT 统一状态码（默认实现）
 * @tparam T 错误码类型
 * @param err 错误码
 * @return 成功返回 IRT_SUCCESS，失败返回 IRT_ERROR_INTERNAL
 * @note 对于自定义错误类型，应该提供特化版本以实现更精确的错误映射
 */
template<class T>
IRTStatus TranslateError(T err)
{
    if (CheckSucceeded(err))
    {
        return IRT_SUCCESS;
    }
    else
    {
        return IRT_ERROR_INTERNAL;
    }
}

/**
 * @brief 错误预处理函数（默认实现）
 * @tparam T 错误码类型
 * @param err 错误码
 * @note 默认实现为空操作，对于需要预处理的错误类型应提供特化版本
 */
template<class T>
inline void PreprocessError(T err)
{
}

/**
 * @brief 将错误码转换为字符串描述（默认实现）
 * @tparam T 错误码类型
 * @param err 错误码
 * @param perrdescr 可选的输出参数，用于接收详细错误描述指针
 * @return 返回空字符串
 * @note 对于自定义错误类型，应该提供特化版本以返回有意义的错误描述
 */
template<class T>
inline const char *ToString(T err, const char **perrdescr = nullptr)
{
    (void)err;
    (void)perrdescr;
    return "";
}

namespace detail {

/**
 * @brief 抛出异常的内部实现
 * @tparam T 错误码类型
 * @param error 错误码
 * @param file 源文件名（可为 nullptr）
 * @param line 源文件行号
 * @param stmt 导致错误的语句字符串
 * @param errmsg 附加错误消息
 * @throws Exception 总是抛出包含错误信息的异常
 * @details 根据是否提供源文件信息，构造包含不同详细程度的异常消息
 */
template<class T>
void DoThrow(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
    // 检查是否可以暴露源文件信息
    if (file != nullptr)
    {
        // 包含文件名和行号的异常消息
        throw Exception(static_cast<Status>(TranslateError(error)), "%s:%d %s", file, line,
                        FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
    else
    {
        // 不包含源文件信息的异常消息
        throw Exception(static_cast<Status>(TranslateError(error)), "%s",
                        FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
}

/**
 * @brief 记录错误日志的内部实现
 * @tparam T 错误码类型
 * @param error 错误码
 * @param file 源文件名（可为 nullptr）
 * @param line 源文件行号
 * @param stmt 导致错误的语句字符串
 * @param errmsg 附加错误消息
 * @details 将错误信息输出到 std::cerr，如果提供了源文件信息则一并输出
 * @todo 替换为真正的日志系统
 */
template<class T>
void DoLog(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
    // TODO: 替换为真正的日志设施

    // 检查是否可以暴露源文件信息
    if (file != nullptr)
    {
        // 输出文件名和行号
        std::cerr << file << ":" << line << ' ';
    }
    // 输出格式化的错误消息
    std::cerr << FormatErrorMessage(ToString(error), stmt, errmsg);
}

} // namespace detail

/**
 * @brief 检查语句执行结果，失败时抛出异常
 * 
 * @param STMT 要执行并检查的语句（返回错误码）
 * @param ... 可选的格式化消息参数（printf 风格）
 * 
 * @details 这个宏会：
 *          1. 执行 STMT 语句并获取返回的错误码
 *          2. 调用 PreprocessError 进行预处理
 *          3. 使用 CheckSucceeded 检查是否成功
 *          4. 如果失败，抛出包含详细信息的 Exception 异常
 * 
 * @note 使用 lambda 表达式包装以避免宏展开问题
 * 
 * @example
 * IRT_CHECK_THROW(cudaMalloc(&ptr, size), "Failed to allocate %zu bytes", size);
 */
#define IRT_CHECK_THROW(STMT, ...)                                                                               \
    [&]()                                                                                                        \
    {                                                                                                            \
        using ::irt::util::PreprocessError;                                                                      \
        using ::irt::util::CheckSucceeded;                                                                       \
        auto status = (STMT);                                                                                    \
        PreprocessError(status);                                                                                 \
        if (!CheckSucceeded(status))                                                                             \
        {                                                                                                        \
            char buf[IRT_MAX_STATUS_MESSAGE_LENGTH];                                                             \
            ::irt::util::detail::DoThrow(status, IRT_SOURCE_FILE_NAME, IRT_SOURCE_FILE_LINENO,                   \
                                         IRT_OPTIONAL_STRINGIFY(STMT),                                           \
                                         ::irt::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
        }                                                                                                        \
    }()

/**
 * @brief 检查语句执行结果，失败时记录日志
 * 
 * @param STMT 要执行并检查的语句（返回错误码）
 * @param ... 可选的格式化消息参数（printf 风格）
 * 
 * @return bool 成功返回 true，失败返回 false
 * 
 * @details 这个宏会：
 *          1. 执行 STMT 语句并获取返回的错误码
 *          2. 调用 PreprocessError 进行预处理
 *          3. 使用 CheckSucceeded 检查是否成功
 *          4. 如果失败，记录错误日志到 std::cerr 并返回 false
 *          5. 如果成功，返回 true
 * 
 * @note 使用 lambda 表达式包装以避免宏展开问题
 * @note 与 IRT_CHECK_THROW 不同，此宏不会抛出异常，而是返回布尔值
 * 
 * @example
 * if (!IRT_CHECK_LOG(cudaMalloc(&ptr, size), "Failed to allocate %zu bytes", size)) {
 *     // 处理错误
 * }
 */
#define IRT_CHECK_LOG(STMT, ...)                                                                               \
    [&]()                                                                                                      \
    {                                                                                                          \
        using ::irt::util::PreprocessError;                                                                    \
        using ::irt::util::CheckSucceeded;                                                                     \
        auto status = (STMT);                                                                                  \
        PreprocessError(status);                                                                               \
        if (!CheckSucceeded(status))                                                                           \
        {                                                                                                      \
            char buf[IRT_MAX_STATUS_MESSAGE_LENGTH];                                                           \
            ::irt::util::detail::DoLog(status, IRT_SOURCE_FILE_NAME, IRT_SOURCE_FILE_LINENO,                   \
                                       IRT_OPTIONAL_STRINGIFY(STMT),                                           \
                                       ::irt::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
            return false;                                                                                      \
        }                                                                                                      \
        else                                                                                                   \
        {                                                                                                      \
            return true;                                                                                       \
        }                                                                                                      \
    }()

} // namespace irt::util
