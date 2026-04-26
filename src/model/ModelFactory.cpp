#include <inferrt/model/ModelFactory.h>

#include "priv/IModelImpl.hpp"

#include <algorithm>
#include <cctype>
#include <map>

namespace irt::model {

namespace {

using ModelRegistry = std::map<std::string, ModelCreator>;

ModelRegistry &GetModelRegistry()
{
    static ModelRegistry registry;
    return registry;
}

} // namespace

ModelRegistrar::ModelRegistrar(const std::string &name, ModelCreator creator)
{
    RegisterModel(name, creator);
}

bool RegisterModel(const std::string &name, ModelCreator creator)
{
    return GetModelRegistry().emplace(name, creator).second;
}

std::unique_ptr<IModel> CreateModel(const std::string &name, std::unique_ptr<IModelConfig> config)
{
    std::string normalized_name = name;
    std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    const auto it = GetModelRegistry().find(normalized_name);
    if (it == GetModelRegistry().end())
    {
        return nullptr;
    }

    auto model = std::make_unique<IModel>(it->second());
    model->setModelConfig(std::move(config));
    return model;
}

} // namespace irt::model
