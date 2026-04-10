#pragma once

#include <inferrt/core/Status.h>

namespace irt::core::priv {

/**
 * @brief 错误状态码和消息
 * @note 该结构体为 TLS 数据，每个线程拥有独立实例
 */
struct CoreTLS
{
    IRTStatus last_error_status;
    char      last_error_message[IRT_MAX_STATUS_MESSAGE_LENGTH];
};

/**
 * @brief 获取当前线程的 TLS 实例
 */
CoreTLS &GetCoreTLS() noexcept;

} // namespace irt::core::priv