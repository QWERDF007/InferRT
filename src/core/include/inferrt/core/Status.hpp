#pragma once

#include "Status.h"

#include <cstdint>

namespace irt::core {

/**
 * @brief 运算状态码枚举类
 * 
 * @note 此枚举类的值与 IRTStatus 中定义的宏一一对应, 见 \ref IRTStatus
 * @see IRTStatus
 */
enum class Status : int8_t
{
    SUCCESS                 = IRT_SUCCESS,
    ERROR_NOT_IMPLEMENTED   = IRT_ERROR_NOT_IMPLEMENTED,
    ERROR_INVALID_ARGUMENT  = IRT_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_OPERATION = IRT_ERROR_INVALID_OPERATION,
    ERROR_DEVICE            = IRT_ERROR_DEVICE,
    ERROR_NOT_READY         = IRT_ERROR_NOT_READY,
    ERROR_OUT_OF_MEMORY     = IRT_ERROR_OUT_OF_MEMORY,
    ERROR_INTERNAL          = IRT_ERROR_INTERNAL,
    ERROR_UNKNOWN           = IRT_ERROR_UNKNOWN
};

/**
 * @brief 返回状态码的字符串表示
 */
inline const char *StatusGetName(Status status)
{
    return StatusGetName(static_cast<IRTStatus>(status));
}

} // namespace irt::core