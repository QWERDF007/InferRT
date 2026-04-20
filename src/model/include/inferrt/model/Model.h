#pragma once

#include "IModel.hpp"

#include <inferrt/core/Status.h>

#include <string>

namespace irt::model {

IModel *CreateModel(const std::string &name);

} // namespace irt::model