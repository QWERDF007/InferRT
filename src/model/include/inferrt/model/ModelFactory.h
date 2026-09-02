#pragma once

#include "ModelFactory.hpp"

/**
 * @brief 注册模型实现类到全局模型工厂。
 *
 * MODEL_CLASS 必须位于 irt::model 命名空间内，继承自模型内部实现基类，
 * 并提供静态成员函数 key() 作为注册名称。该宏通常放在模型实现的 .cpp 文件中。
 *
 * @param MODEL_CLASS 模型实现类名。
 */
#define INFERRT_REGISTER_MODEL(MODEL_CLASS)                                                                        \
    namespace {                                                                                                    \
    std::unique_ptr<::irt::model::IModel> Create##MODEL_CLASS()                                                    \
    {                                                                                                              \
        return ::irt::model::IModel::fromImplementation(                                                          \
            std::make_unique<::irt::model::MODEL_CLASS>());                                                       \
    }                                                                                                              \
    [[maybe_unused]] const ::irt::model::ModelRegistrar registered_##MODEL_CLASS(::irt::model::MODEL_CLASS::key(), \
                                                                                 &Create##MODEL_CLASS);            \
    }
