#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/IModel.h>

#include <vector>

/**
 * @brief AlexNet 在执行上下文未初始化时调用 infer，应抛出 ERROR_INVALID_OPERATION
 */
TEST(AlexNetInferTest, InferWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("alexnet");
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
 * @brief 当前实现中，AlexNet 在 context 缺失时会先于 buffer 数量检查失败
 *
 * 这条测试用于固定当前行为，避免后续重构时无意改掉异常顺序。
 */
TEST(AlexNetInferTest, InferWithWrongBufferCountStillThrowsWhenContextIsMissing)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(1, nullptr);
    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);
}

/**
 * @brief ResNet 在执行上下文未初始化时调用 infer，应抛出 ERROR_INVALID_OPERATION
 */
TEST(ResNetInferTest, InferWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("resnet50");
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
 * @brief 当前实现中，ResNet 在 context 缺失时同样会先于 buffer 数量检查失败
 */
TEST(ResNetInferTest, InferWithWrongBufferCountStillThrowsWhenContextIsMissing)
{
    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(1, nullptr);
    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);
}

TEST(WideResNetInferTest, InferWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("wide_resnet50_2");
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

TEST(WideResNetInferTest, InferWithWrongBufferCountStillThrowsWhenContextIsMissing)
{
    auto model = irt::model::CreateModel("wide_resnet101_2");
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(1, nullptr);
    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);
}
