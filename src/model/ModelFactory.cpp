#include <inferrt/model/ModelFactory.h>

#include "priv/IModelImpl.hpp"

#include <algorithm>
#include <cctype>
#include <map>

namespace irt::model {

namespace {

using ModelRegistry = std::map<std::string, ModelCreator>;

/**
 * @brief 获取全局模型注册表单例。
 * @return 注册表引用。
 */
ModelRegistry &GetModelRegistry()
{
    static ModelRegistry registry;
    return registry;
}

} // namespace

/**
 * @brief 构造时立即完成模型注册。
 * @param name 模型名称。
 * @param creator 模型创建函数。
 */
ModelRegistrar::ModelRegistrar(const std::string &name, ModelCreator creator)
{
    RegisterModel(name, creator);
}

/**
 * @brief 向全局注册表写入模型创建器。
 * @param name 模型名称。
 * @param creator 模型创建函数。
 * @return 若名称未冲突则返回 `true`。
 */
bool RegisterModel(const std::string &name, ModelCreator creator)
{
    return GetModelRegistry().emplace(name, creator).second;
}

/**
 * @brief 根据名称创建模型并注入配置。
 * @param name 模型名称。
 * @param config 模型配置对象。
 * @return 成功时返回模型对象，失败时返回空指针。
 */
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
