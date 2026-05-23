#include "TestModelCommon.hpp"

using test::model::ExpectIrtExceptionCode;
using test::model::MakeNullBuffers;
using test::model::RegisteredModelsTest;
using test::model::kRegisteredModels;

namespace {

/**
 * @brief 模型主推理入口的参数化测试基类。
 */
class ModelInferRegisteredModelsTest : public RegisteredModelsTest
{
};

} // namespace

INSTANTIATE_TEST_SUITE_P(KnownModels, ModelInferRegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 所有已注册模型在执行上下文未初始化时调用 infer 都应抛出非法操作异常。
 */
TEST_P(ModelInferRegisteredModelsTest, InferWithoutContextThrowsInvalidOperation)
{
    const auto &param = GetParam();
    auto        model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->infer(MakeNullBuffers(2)); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 未初始化 context 时，infer 应先报运行时未就绪，而不是进入 buffer 数量校验。
 *
 * 该测试固定当前错误顺序，避免后续重构时把未构建模型误报为参数数量错误。
 */
TEST_P(ModelInferRegisteredModelsTest, InferWithWrongBufferCountStillFailsBeforeBufferValidation)
{
    const auto &param = GetParam();
    auto        model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->infer(MakeNullBuffers(1)); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 未初始化 runtime 且无外部 stream 时，resolveExecutionStream 应返回空指针。
 */
TEST_P(ModelInferRegisteredModelsTest, ResolveExecutionStreamWithoutRuntimeReturnsNullptr)
{
    const auto &param = GetParam();
    auto        model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->resolveExecutionStream(), nullptr);
}

/**
 * @brief 单次传入的 stream 覆盖参数应优先于模型内部状态。
 */
TEST_P(ModelInferRegisteredModelsTest, ResolveExecutionStreamPrefersCallOverride)
{
    const auto &param = GetParam();
    auto        model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);

    const cudaStream_t override_stream = reinterpret_cast<cudaStream_t>(0x1234);

    EXPECT_EQ(model->resolveExecutionStream(override_stream), override_stream);
}

/**
 * @brief 显式设置和清除外部 stream 应影响后续默认执行 stream 的解析结果。
 */
TEST_P(ModelInferRegisteredModelsTest, SetAndClearStreamAffectsResolvedDefaultStream)
{
    const auto &param = GetParam();
    auto        model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);

    const cudaStream_t external_stream = reinterpret_cast<cudaStream_t>(0x5678);

    model->setStream(external_stream);
    EXPECT_EQ(model->resolveExecutionStream(), external_stream);

    model->clearStream();
    EXPECT_EQ(model->resolveExecutionStream(), nullptr);
}
