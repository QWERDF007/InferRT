#pragma once

#include <cstdint>

namespace irt::model {

/** Opaque execution-stream handle used by backend-neutral model APIs. */
using StreamHandle = std::uintptr_t;

} // namespace irt::model
