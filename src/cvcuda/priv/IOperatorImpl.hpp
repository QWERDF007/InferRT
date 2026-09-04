#pragma once

#include <inferrt/cvcuda/IOperator.hpp>
#include <inferrt/core/Version.h>

namespace irt::cvcuda::priv {

class IOperatorImpl : public irt::cvcuda::OperatorImplementation
{
public:
    virtual ~IOperatorImpl() = default;
};

} // namespace irt::cvcuda::priv
