#pragma once

#include "Status.hpp"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace irt {

/**
 * @brief InferRT 异常类（具有完整值语义，消息基于 std::string）
 */
class Exception : public std::exception
{
public:
    explicit Exception(Status code, const char *fmt = nullptr, ...)
#if __GNUC__
        __attribute__((format(printf, 3, 4)))
#endif
        : code_(code)
    {
        if (fmt != nullptr)
        {
            va_list va;
            va_start(va, fmt);
            char buf[IRT_MAX_STATUS_MESSAGE_LENGTH];
            vsnprintf(buf, sizeof(buf), fmt, va);
            buf[sizeof(buf) - 1] = '\0';
            msg_ = buf;
            va_end(va);
        }
        updateFullWhat();
    }

    explicit Exception(Status code, std::string msg)
        : code_(code)
        , msg_(std::move(msg))
    {
        updateFullWhat();
    }

    Exception(const Exception &other)
        : std::exception(other)
        , code_(other.code_)
        , msg_(other.msg_)
        , full_what_(other.full_what_)
    {
    }

    Exception &operator=(const Exception &other)
    {
        if (this != &other)
        {
            std::exception::operator=(other);
            code_      = other.code_;
            msg_       = other.msg_;
            full_what_ = other.full_what_;
        }
        return *this;
    }

    Exception(Exception &&other) noexcept
        : std::exception(std::move(other))
        , code_(other.code_)
        , msg_(std::move(other.msg_))
        , full_what_(std::move(other.full_what_))
    {
    }

    Exception &operator=(Exception &&other) noexcept
    {
        if (this != &other)
        {
            std::exception::operator=(std::move(other));
            code_      = other.code_;
            msg_       = std::move(other.msg_);
            full_what_ = std::move(other.full_what_);
        }
        return *this;
    }

    [[nodiscard]] Status code() const noexcept
    {
        return code_;
    }

    [[nodiscard]] const char *msg() const noexcept
    {
        return msg_.c_str();
    }

    [[nodiscard]] const char *what() const noexcept override
    {
        return full_what_.c_str();
    }

private:
    Status      code_{Status::ERROR_INTERNAL};
    std::string msg_;
    std::string full_what_;

    void updateFullWhat()
    {
        const char *name = StatusGetName(code_);
        if (name == nullptr)
        {
            name = "ERROR_UNKNOWN";
        }
        if (msg_.empty())
        {
            full_what_ = name;
        }
        else
        {
            full_what_ = std::string(name) + ": " + msg_;
        }
    }
};

/**
 * @brief 根据捕获的异常设置线程错误状态
 */
inline void SetThreadError(std::exception_ptr e)
{
    try
    {
        if (e)
        {
            std::rethrow_exception(e);
        }
        else
        {
            SetThreadStatus(IRT_SUCCESS, "success");
        }
    }
    catch (const Exception &ex)
    {
        SetThreadStatus(static_cast<IRTStatus>(ex.code()), "%s", ex.msg());
    }
    catch (const std::invalid_argument &ex)
    {
        SetThreadStatus(IRT_ERROR_INVALID_ARGUMENT, "%s", ex.what());
    }
    catch (const std::bad_alloc &)
    {
        SetThreadStatus(IRT_ERROR_OUT_OF_MEMORY, "Not enough space for resource allocation");
    }
    catch (const std::exception &ex)
    {
        SetThreadStatus(IRT_ERROR_INTERNAL, "%s", ex.what());
    }
    catch (...)
    {
        SetThreadStatus(IRT_ERROR_INTERNAL, "Unexpected error");
    }
}

/**
 * @brief 安全地执行函数，捕获并设置可能产生的异常（用于 C ABI 边界）
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
        SetThreadError(std::current_exception());
        return PeekAtLastError();
    }
}

} // namespace irt