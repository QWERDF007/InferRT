#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/IModel.h>

#include <vector>

/**
 * @brief AlexNet 配置特征张量但未初始化特征执行上下文时，forwardFeatures 应抛出 ERROR_INVALID_OPERATION。
 */
TEST(AlexNetFeatureInferTest, ForwardFeaturesWithoutFeatureContextThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1"});

    auto model = irt::model::CreateModel("alexnet", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->forwardFeatures(buffers); }, irt::Exception);

    try
    {
        model->forwardFeatures(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief AlexNet 处于 featureOnly 模式但未初始化执行上下文时，forwardFeatures 应抛出 ERROR_INVALID_OPERATION。
 */
TEST(AlexNetFeatureInferTest, ForwardFeaturesWithFeatureOnlyConfigWithoutContextThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("alexnet", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->forwardFeatures(buffers); }, irt::Exception);

    try
    {
        model->forwardFeatures(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief AlexNet 处于 featureOnly 模式但未初始化执行上下文时，infer 也应被运行时保护拦下。
 */
TEST(AlexNetFeatureInferTest, InferWithFeatureOnlyConfigWithoutContextStillThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("alexnet", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);

    try
    {
        model->infer(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief ResNet 配置特征张量但未初始化特征执行上下文时，forwardFeatures 应抛出 ERROR_INVALID_OPERATION。
 */
TEST(ResNetFeatureInferTest, ForwardFeaturesWithoutFeatureContextThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer1"});

    auto model = irt::model::CreateModel("resnet18", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->forwardFeatures(buffers); }, irt::Exception);

    try
    {
        model->forwardFeatures(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief 当前实现中，ResNet 在特征 context 缺失时会先于 buffer 数量检查失败。
 */
TEST(ResNetFeatureInferTest, ForwardFeaturesWithWrongBufferCountStillThrowsWhenContextIsMissing)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer1"});

    auto model = irt::model::CreateModel("resnet18", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(1, nullptr);
    EXPECT_THROW({ model->forwardFeatures(buffers); }, irt::Exception);
}

/**
 * @brief ResNet 处于 featureOnly 模式但未初始化执行上下文时，forwardFeatures 应抛出 ERROR_INVALID_OPERATION。
 */
TEST(ResNetFeatureInferTest, ForwardFeaturesWithFeatureOnlyConfigWithoutContextThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer1"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("resnet18", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->forwardFeatures(buffers); }, irt::Exception);

    try
    {
        model->forwardFeatures(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief ResNet 处于 featureOnly 模式但未初始化执行上下文时，infer 也应被运行时保护拦下。
 */
TEST(ResNetFeatureInferTest, InferWithFeatureOnlyConfigWithoutContextStillThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer1"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("resnet18", std::move(config));
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);

    try
    {
        model->infer(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}
