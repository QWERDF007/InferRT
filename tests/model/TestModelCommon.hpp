#pragma once

#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.hpp>
#include <inferrt/model/IModel.h>

#include <array>
#include <utility>
#include <vector>

namespace test::model {

/**
 * @brief 已注册模型的测试样例。
 *
 * `key` 为注册表中的标准 key，`mixed_case_key` 用于验证大小写不敏感查找，
 * `display_name` 则对应模型实例 `name()` 的期望返回值。
 */
struct RegisteredModelCase
{
    const char *key;
    const char *mixed_case_key;
    const char *display_name;
};

/**
 * @brief 当前仓库中内置注册的模型清单。
 */
inline constexpr std::array<RegisteredModelCase, 17> kRegisteredModels = {{
    {"onnx", "ONNX", "ONNX"},
    {"alexnet", "AlexNet", "AlexNet"},
    {"googlenet", "GoogLeNet", "GoogLeNet"},
    {"mobilenet_v2", "MobileNet_V2", "MobileNetV2"},
    {"mobilenet_v3_large", "MobileNet_V3_Large", "MobileNetV3Large"},
    {"mobilenet_v3_small", "MobileNet_V3_Small", "MobileNetV3Small"},
    {"vgg11", "VGG11", "VGG11"},
    {"vgg13", "VGG13", "VGG13"},
    {"vgg16", "VGG16", "VGG16"},
    {"vgg19", "VGG19", "VGG19"},
    {"resnet18", "ResNet18", "ResNet18"},
    {"resnet34", "ResNet34", "ResNet34"},
    {"resnet50", "ResNet50", "ResNet50"},
    {"resnet101", "ResNet101", "ResNet101"},
    {"resnet152", "ResNet152", "ResNet152"},
    {"wide_resnet50_2", "Wide_ResNet50_2", "WideResNet50_2"},
    {"wide_resnet101_2", "Wide_ResNet101_2", "WideResNet101_2"},
}};

/**
 * @brief 面向全部已注册模型的参数化测试基类。
 */
class RegisteredModelsTest : public ::testing::TestWithParam<RegisteredModelCase>
{
};

/**
 * @brief 构造用于未初始化 runtime 测试的空 buffer 列表。
 * @param count buffer 数量。
 * @return 指针均为空的 buffer 列表。
 */
inline std::vector<void *> MakeNullBuffers(size_t count)
{
    return std::vector<void *>(count, nullptr);
}

/**
 * @brief 断言指定调用抛出 InferRT 异常且错误码符合预期。
 *
 * 该辅助函数用于减少测试中重复的 `try/catch` 样板代码，同时保留错误码校验。
 *
 * @tparam Fn 可调用对象类型。
 * @param fn 待执行的调用。
 * @param expected_code 期望的 InferRT 错误码。
 */
template <typename Fn>
void ExpectIrtExceptionCode(Fn &&fn, irt::Status expected_code)
{
    try
    {
        std::forward<Fn>(fn)();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), expected_code);
    }
}

} // namespace test::model
