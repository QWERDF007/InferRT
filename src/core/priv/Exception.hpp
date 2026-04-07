#pragma once

#include <inferrt/core/Status.h>

#include <strstream>

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

    template<class T>
    Exception &&operator<<(const T &v) &&
    {
        // TODO: must avoid allocating memory from heap, can't use ostringstream
        std::ostream ss(&strbuf_);
        ss << v << std::flush;
        return std::move(*this);
    }

private:
    IRTStatus code_ = IRT_ERROR_INTERNAL;
    char      buffer_[MAX_STATUS_MESSAGE_LENGTH + 64 + 2]{};

    class StrBuffer : public std::strstreambuf
    {
    public:
        using std::strstreambuf::seekpos;
        using std::strstreambuf::strstreambuf;
    };

    StrBuffer strbuf_;
};

} // namespace irt::core::priv