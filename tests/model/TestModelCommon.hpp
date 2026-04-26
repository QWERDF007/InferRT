#pragma once

#include <gtest/gtest.h>

#include <inferrt/model/IModel.h>

#include "../../src/model/priv/IModelImpl.hpp"

#include <array>
#include <memory>
#include <vector>

namespace test::model {

/**
 * @brief 已注册模型的测试样例
 *
 * key            : 注册表中使用的小写 key
 * mixed_case_key : 用于验证大小写不敏感查找
 * display_name   : 实例方法 name() 返回的展示名
 */
struct RegisteredModelCase
{
    const char *key;
    const char *mixed_case_key;
    const char *display_name;
};

/**
 * @brief 一个仅用于注册表测试的最小假模型
 *
 * 这个模型不会真正构图或推理，只用于验证 RegisterModel / CreateModel
 * 的注册与创建流程是否正常工作。
 */
class DummyModel final : public irt::model::priv::IModelImpl
{
public:
    static constexpr const char *key() noexcept
    {
        return "dummy_model";
    }

    std::string name() const noexcept override
    {
        return "DummyModel";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *, const irt::model::WeightsMap &) override
    {
    }

    void infer(const std::vector<void *> &) override
    {
    }
};

inline std::unique_ptr<irt::model::priv::IModelImpl> CreateDummyModel()
{
    return std::make_unique<DummyModel>();
}

/**
 * @brief 当前仓库中内置注册的模型清单
 */
inline constexpr std::array<RegisteredModelCase, 6> kRegisteredModels = {{
    {"alexnet", "AlexNet", "AlexNet"},
    {"resnet18", "ResNet18", "ResNet18"},
    {"resnet34", "ResNet34", "ResNet34"},
    {"resnet50", "ResNet50", "ResNet50"},
    {"resnet101", "ResNet101", "ResNet101"},
    {"resnet152", "ResNet152", "ResNet152"},
}};

/**
 * @brief 对所有已注册模型做参数化测试
 */
class RegisteredModelsTest : public ::testing::TestWithParam<RegisteredModelCase>
{
};

} // namespace test::model
