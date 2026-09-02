#pragma once

#include <inferrt/core/Status.h>

#include <cstdint>
#include <exception>

namespace irt::core::priv {

void SetThreadError(std::exception_ptr e);

IRTStatus PeekAtLastThreadError(char *msg, int32_t len) noexcept;
IRTStatus GetLastThreadError(char *msg, int32_t len) noexcept;
IRTStatus PeekAtLastThreadError() noexcept;
IRTStatus GetLastThreadError() noexcept;
const char *GetName(IRTStatus code);

} // namespace irt::core::priv