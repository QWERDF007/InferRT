#pragma once

#include <inferrt/model/Export.h>

namespace irt::model {

class INFERRT_MODEL_API IModel
{
public:
    virtual ~IModel() = default;
};

} // namespace irt::model