#pragma once

#include "IModel.hpp"

#include <string>

namespace irt::model {

INFERRT_MODEL_API IModel *CreateModel(const std::string &name);

} // namespace irt::model