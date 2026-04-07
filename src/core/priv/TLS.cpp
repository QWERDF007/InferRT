#include "TLS.hpp"

namespace irt::core::priv {

// 使用 C++11 thread_local 关键字
// 每个线程拥有独立的 CoreTLS 实例
namespace {
thread_local CoreTLS s_TLS;
}

CoreTLS &GetCoreTLS() noexcept
{
    return s_TLS;
}

} // namespace irt::core::priv