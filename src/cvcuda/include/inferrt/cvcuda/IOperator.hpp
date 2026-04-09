/**
 * @file IOperator.hpp
 *
 * @brief 定义算子接口的公共 C++ 接口。
 */

#pragma once

#include <memory>

namespace irt::cvcuda::priv {
class IOperatorImpl;
}

namespace irt::cvcuda {

class IOperator
{
public:
    virtual ~IOperator() = default;
};

} // namespace irt::cvcuda