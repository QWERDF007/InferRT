#pragma once

#include <inferrt/core/Status.h>

#include <exception>

namespace irt::core::priv {

class Exception : public std::exception
{
public:
    explicit Exception(IRTStatus code, const char *fmt, va_list va);

    explicit Exception(IRTStatus code, const char *fmt, ...)
#if __GNUC__
        // first argument is actually 'this'
        __attribute__((format(printf, 3, 4)));
#else
        ;
#endif

    explicit Exception(IRTStatus code);

    IRTStatus   code() const;
    const char *msg() const;

    const char *what() const noexcept override;

private:
    IRTStatus code_ = IRT_ERROR_INTERNAL;
    char      buffer_[IRT_MAX_STATUS_MESSAGE_LENGTH + 64 + 2]{};
};

} // namespace irt::core::priv