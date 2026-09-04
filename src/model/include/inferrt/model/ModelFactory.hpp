#pragma once

#include "IModel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 公共模型对象的创建函数类型。
 *
 * 具体模型实现只在库内通过注册宏包装为 `IModel`，注册表不暴露私有
 * 实现类型或后端句柄。
 */
using ModelCreator = std::unique_ptr<IModel> (*)();

/**
 * @brief 模型注册器。
 *
 * 该类通常配合静态对象或注册宏使用，在程序启动时将模型名称与创建函数加入全局工厂表。
 */
class INFERRT_MODEL_API ModelRegistrar
{
public:
    /**
     * @brief 注册一个模型创建器。
     * @param name 模型注册名称。
     * @param creator 模型创建函数。
     */
    ModelRegistrar(const std::string &name, ModelCreator creator);
};

/**
 * @brief 向全局模型注册表注册模型。
 * @param name 模型注册名称。
 * @param creator 模型创建函数。
 * @return 注册成功返回 true；名称重复或创建函数无效时返回 false。
 */
INFERRT_MODEL_API bool RegisterModel(const std::string &name, ModelCreator creator);

/**
 * @brief 查询模型名称是否已注册。
 * @param name 模型名称，查找时会进行大小写归一化。
 * @return 已注册返回 true，否则返回 false。
 */
INFERRT_MODEL_API bool isSupportedModel(const std::string &name);

/**
 * @brief 获取当前全局模型注册表中的模型名称。
 * @return 已注册模型名称列表。
 */
INFERRT_MODEL_API std::vector<std::string> getRegisteredModelNames();

/**
 * @brief 根据名称创建模型对象。
 * @param name 模型名称，查找时会进行大小写归一化。
 * @param config 模型初始配置；为空时使用默认配置。
 * @return 成功时返回模型对象，失败时返回 nullptr。
 */
INFERRT_MODEL_API std::unique_ptr<IModel> CreateModel(const std::string &name, std::unique_ptr<IModelConfig> config
                                                                               = std::make_unique<IModelConfig>());

} // namespace irt::model
