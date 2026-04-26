#pragma once

#include "IModel.hpp"

#include <memory>
#include <string>

namespace irt::model {

using ModelCreator = std::unique_ptr<priv::IModelImpl> (*)();

class INFERRT_MODEL_API ModelRegistrar
{
public:
    ModelRegistrar(const std::string &name, ModelCreator creator);
};

INFERRT_MODEL_API bool RegisterModel(const std::string &name, ModelCreator creator);

INFERRT_MODEL_API std::unique_ptr<IModel> CreateModel(const std::string &name, std::unique_ptr<IModelConfig> config
                                                                               = std::make_unique<IModelConfig>());

} // namespace irt::model
