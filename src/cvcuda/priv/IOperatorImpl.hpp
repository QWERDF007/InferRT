#pragma once

#include <inferrt/core/Version.h>

namespace irt::cvcuda::priv {

class IOperatorImpl
{
public:
    virtual ~IOperatorImpl() = default;
};

} // namespace irt::cvcuda::priv