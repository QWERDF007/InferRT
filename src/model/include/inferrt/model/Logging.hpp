#pragma once

#include <inferrt/model/Export.h>

#include <string>
#include <utility>

namespace irt::model {

/** Backend-neutral logging severity used by public model APIs. */
enum class LogLevel
{
    InternalError,
    Error,
    Warning,
    Info,
    Verbose,
};

/** Small sink-independent logger contract for public consumers. */
class INFERRT_MODEL_API Logger
{
public:
    explicit Logger(std::string name = {}, LogLevel level = LogLevel::Warning) noexcept
        : name_(std::move(name)), level_(level)
    {
    }

    [[nodiscard]] const std::string &name() const noexcept { return name_; }
    [[nodiscard]] LogLevel level() const noexcept { return level_; }
    void setLevel(LogLevel level) noexcept { level_ = level; }

private:
    std::string name_;
    LogLevel    level_;
};

} // namespace irt::model
