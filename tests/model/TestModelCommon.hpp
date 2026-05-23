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
inline constexpr std::array<RegisteredModelCase, 38> kRegisteredModels = {{
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
    {"vit", "ViT", "ViTBasePatch16_224"},
    {"vit_tiny_patch16_224", "ViT_Tiny_Patch16_224", "ViTTinyPatch16_224"},
    {"vit_tiny_patch16_384", "ViT_Tiny_Patch16_384", "ViTTinyPatch16_384"},
    {"vit_small_patch32_224", "ViT_Small_Patch32_224", "ViTSmallPatch32_224"},
    {"vit_small_patch32_384", "ViT_Small_Patch32_384", "ViTSmallPatch32_384"},
    {"vit_small_patch16_224", "ViT_Small_Patch16_224", "ViTSmallPatch16_224"},
    {"vit_small_patch16_384", "ViT_Small_Patch16_384", "ViTSmallPatch16_384"},
    {"vit_small_patch8_224", "ViT_Small_Patch8_224", "ViTSmallPatch8_224"},
    {"vit_base_patch32_224", "ViT_Base_Patch32_224", "ViTBasePatch32_224"},
    {"vit_base_patch32_384", "ViT_Base_Patch32_384", "ViTBasePatch32_384"},
    {"vit_base_patch16_224", "ViT_Base_Patch16_224", "ViTBasePatch16_224"},
    {"vit_base_patch16_384", "ViT_Base_Patch16_384", "ViTBasePatch16_384"},
    {"vit_base_patch8_224", "ViT_Base_Patch8_224", "ViTBasePatch8_224"},
    {"vit_large_patch32_224", "ViT_Large_Patch32_224", "ViTLargePatch32_224"},
    {"vit_large_patch32_384", "ViT_Large_Patch32_384", "ViTLargePatch32_384"},
    {"vit_large_patch16_224", "ViT_Large_Patch16_224", "ViTLargePatch16_224"},
    {"vit_large_patch16_384", "ViT_Large_Patch16_384", "ViTLargePatch16_384"},
    {"vit_large_patch14_224", "ViT_Large_Patch14_224", "ViTLargePatch14_224"},
    {"vit_huge_patch14_224", "ViT_Huge_Patch14_224", "ViTHugePatch14_224"},
    {"vit_giant_patch14_224", "ViT_Giant_Patch14_224", "ViTGiantPatch14_224"},
    {"vit_gigantic_patch14_224", "ViT_Gigantic_Patch14_224", "ViTGiganticPatch14_224"},
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
