/**
 * @file IOperator.hpp
 *
 * @brief 定义算子接口的公共 C++ 接口。
 */

#pragma once

#include <inferrt/cvcuda/Export.h>

#include <memory>

namespace irt::cvcuda::priv {
class IOperatorImpl;
}

typedef irt::cvcuda::priv::IOperatorImpl *OperatorHandle;

namespace irt::cvcuda {
class INFERRT_CVCUDA_API IOperator
{
public:
    virtual ~IOperator() = default;

    virtual OperatorHandle handle() const noexcept = 0;
};

} // namespace irt::cvcuda