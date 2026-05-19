#include "priv/IModelImpl.hpp"

#include <inferrt/model/ModelFactory.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <vector>

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

/**
 * @brief 将模型名称归一化为小写形式。
 * @param name 原始模型名称。
 * @return 归一化后的模型名称，用于注册表查找与写入。
 */
std::string normalizeModelName(const std::string &name)
{
    std::string normalized_name = name;
    std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized_name;
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
    return GetModelRegistry().emplace(normalizeModelName(name), creator).second;
}

/**
 * @brief 判断指定模型名称是否已注册。
 * @param name 待查询的模型名称。
 * @return 若名称存在于全局注册表中则返回 true，否则返回 false。
 */
bool isSupportedModel(const std::string &name)
{
    return GetModelRegistry().find(normalizeModelName(name)) != GetModelRegistry().end();
}

/**
 * @brief 获取当前全局注册表中的全部模型名称。
 * @return 按注册表遍历顺序返回模型名称列表。
 */
std::vector<std::string> getRegisteredModelNames()
{
    std::vector<std::string> names;
    names.reserve(GetModelRegistry().size());
    for (const auto &[name, creator] : GetModelRegistry())
    {
        (void)creator;
        names.push_back(name);
    }
    return names;
}

/**
 * @brief 根据名称创建模型并注入配置。
 * @param name 模型名称。
 * @param config 模型配置对象。
 * @return 成功时返回模型对象，失败时返回空指针。
 */
std::unique_ptr<IModel> CreateModel(const std::string &name, std::unique_ptr<IModelConfig> config)
{
    const auto it = GetModelRegistry().find(normalizeModelName(name));
    if (it == GetModelRegistry().end())
    {
        return nullptr;
    }

    auto model = std::make_unique<IModel>(it->second());
    model->setModelConfig(std::move(config));
    return model;
}

} // namespace irt::model
