#include "TestModelCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>

#include "priv/BackendRuntime.hpp"
#include "priv/IModelImpl.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <stdexcept>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#ifndef INFERRT_BUILD_ONNX
#define INFERRT_BUILD_ONNX 0
#endif

using test::model::ExpectIrtExceptionCode;
using test::model::kRegisteredModels;
using test::model::RegisteredModelsTest;

namespace {

struct FakeBackendState
{
    std::atomic_bool fail_next_load{false};
    std::atomic_bool fail_next_io_names{false};
    std::atomic_bool fail_next_shape{false};
    std::atomic_bool fail_next_data_type{false};
    std::atomic_bool fail_next_execute{false};
    std::atomic_bool fail_next_create{false};
    std::atomic_int  load_calls{0};
    std::atomic_int  io_names_calls{0};
    std::atomic_int  shape_calls{0};
    std::atomic_int  data_type_calls{0};
    std::atomic_int  infer_calls{0};
    std::vector<std::string> last_execute_names;
};

FakeBackendState *g_fake_backend_state = nullptr;

class SessionDescriptorTestSession final : public irt::ITensorRuntimeSession
{
public:
    explicit SessionDescriptorTestSession(FakeBackendState &state) : state_(state) {}

    irt::Shape tensorShape(const std::string &tensor_name) const override
    {
        if (tensor_name != "input" && tensor_name != "output")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown session tensor: %s",
                                 tensor_name.c_str());
        }
        return shape_;
    }

    irt::TensorDataType tensorDataType(const std::string &tensor_name) const override
    {
        if (tensor_name != "input" && tensor_name != "output")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown session tensor: %s",
                                 tensor_name.c_str());
        }
        return irt::TensorDataType::F32;
    }

    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override
    {
        if (tensor_name != "input")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown session input: %s",
                                 tensor_name.c_str());
        }
        shape_ = shape;
    }

    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions = {}) override
    {
        state_.last_execute_names.clear();
        for (const auto &buffer : buffers)
        {
            state_.last_execute_names.push_back(buffer.tensor_name);
        }
        ++state_.infer_calls;
    }

private:
    FakeBackendState &state_;
    irt::Shape        shape_{2, 3, 2, 2};
};

class TransactionTestBackend final : public irt::model::IBackendRuntime
{
public:
    explicit TransactionTestBackend(FakeBackendState &state) : state_(state) {}

    irt::model::ModelRuntime::Backend backend() const noexcept override
    {
        return irt::model::ModelRuntime::Backend::ONNXRuntime;
    }

    void load(const std::string &, const irt::model::IModelConfig &, const std::string &) override
    {
        ++state_.load_calls;
        if (state_.fail_next_load.exchange(false))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "injected backend load failure");
        }
    }

    std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const override
    {
        ++state_.io_names_calls;
        if (state_.fail_next_io_names.exchange(false))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "injected backend I/O enumeration failure");
        }
        if (mode == irt::TensorIOMode::Input)
        {
            return {"input"};
        }
        return {"output"};
    }

    irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const noexcept override
    {
        (void)mode;
        return irt::MemoryKind::HOST;
    }

    irt::Shape tensorShape(const std::string &) const override
    {
        ++state_.shape_calls;
        if (state_.fail_next_shape.exchange(false))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "injected backend shape failure");
        }
        return irt::Shape{1, 3, 2, 2};
    }

    irt::TensorDataType tensorDataType(const std::string &) const override
    {
        ++state_.data_type_calls;
        if (state_.fail_next_data_type.exchange(false))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "injected backend data type failure");
        }
        return irt::TensorDataType::F32;
    }

    void setTensorShape(const std::string &, const irt::Shape &) override {}

    std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override
    {
        return std::make_unique<SessionDescriptorTestSession>(state_);
    }

    void executeNormalized(std::span<const irt::BufferView> buffers, irt::ExecuteOptions = {}) override
    {
        if (state_.fail_next_execute.exchange(false))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "injected backend execute failure");
        }
        state_.last_execute_names.clear();
        for (const auto &buffer : buffers)
        {
            state_.last_execute_names.push_back(buffer.tensor_name);
        }
        ++state_.infer_calls;
    }

    void save(const std::string &) const override {}
    void setStream(std::uintptr_t stream) override { external_stream_ = stream; }
    void clearStream() override { external_stream_ = 0; }
    std::uintptr_t resolveExecutionStream(std::uintptr_t stream_override = 0) override
    {
        return stream_override != 0 ? stream_override : external_stream_;
    }
    irt::model::LogLevel logLevel() const noexcept override { return log_level_; }
    void setLogLevel(irt::model::LogLevel level) override { log_level_ = level; }

private:
    FakeBackendState &state_;
    std::uintptr_t    external_stream_{0};
    irt::model::LogLevel log_level_{irt::model::LogLevel::Warning};
};

std::unique_ptr<irt::model::IBackendRuntime> transactionBackendFactory(
    irt::model::ModelRuntime::Backend backend)
{
    if ((backend != irt::model::ModelRuntime::Backend::ONNXRuntime
         && backend != irt::model::ModelRuntime::Backend::TensorRT)
        || g_fake_backend_state == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unexpected backend in transaction test");
    }
    if (g_fake_backend_state->fail_next_create.exchange(false))
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "injected backend creation failure");
    }
    return std::make_unique<TransactionTestBackend>(*g_fake_backend_state);
}

std::vector<irt::BufferView> MakeTransactionExecutionBuffers(irt::model::IModel &model,
                                                             std::vector<float> &input_storage,
                                                             std::vector<float> &output_storage)
{
    const auto input_infos  = model.inputs();
    const auto output_infos = model.outputs();
    if (input_infos.size() != 1 || output_infos.size() != 1)
    {
        throw std::runtime_error("transaction backend must expose one input and one output");
    }
    input_storage.resize(input_infos.front().desc.elementCount(), 0.0F);
    output_storage.resize(output_infos.front().desc.elementCount(), 0.0F);
    return {
        {input_storage.data(), input_infos.front().desc, input_infos.front().desc.byteSize(), 1,
         input_infos.front().desc.byteSize(), input_infos.front().name},
        {output_storage.data(), output_infos.front().desc, output_infos.front().desc.byteSize(), 1,
         output_infos.front().desc.byteSize(), output_infos.front().name},
    };
}

TEST(BackendRuntimeContractTest, ImplementsTheSharedCoreExecutionContract)
{
    static_assert(std::is_base_of_v<irt::IExecutableModel, irt::model::IBackendRuntime>);
    static_assert(std::is_base_of_v<irt::IExecutableModel, irt::model::IModel::Implementation>);
    static_assert(std::is_base_of_v<irt::IExecutableModel, irt::model::priv::IModelImpl>);

    FakeBackendState state;
    TransactionTestBackend backend(state);
    irt::IExecutableModel &executable = backend;

    const auto input_infos  = executable.inputs();
    const auto output_infos = executable.outputs();
    ASSERT_EQ(input_infos.size(), 1U);
    ASSERT_EQ(output_infos.size(), 1U);
    EXPECT_EQ(input_infos.front().name, "input");
    EXPECT_EQ(input_infos.front().mode, irt::TensorIOMode::Input);
    EXPECT_EQ(input_infos.front().desc.data_type, irt::TensorDataType::F32);
    EXPECT_EQ(input_infos.front().desc.memory_kind, irt::MemoryKind::HOST);
    EXPECT_EQ(input_infos.front().desc.shape, irt::Shape({1, 3, 2, 2}));
    EXPECT_EQ(output_infos.front().name, "output");
    EXPECT_EQ(output_infos.front().mode, irt::TensorIOMode::Output);
    EXPECT_EQ(output_infos.front().desc.data_type, irt::TensorDataType::F32);

    const auto capabilities = executable.capabilities();
    EXPECT_FALSE(capabilities.supports_dynamic_batch);
    EXPECT_FALSE(capabilities.supports_feature_outputs);
    EXPECT_EQ(capabilities.fixed_batch_size, 1);

    EXPECT_NO_THROW(executable.setInputShape("input", irt::Shape{1, 3, 2, 2}));
}

TEST(BackendRuntimeContractTest, DirectExecutionCanonicalizesNamedBuffers)
{
    FakeBackendState state;
    TransactionTestBackend backend(state);

    const auto input_info  = backend.inputs().front();
    const auto output_info = backend.outputs().front();
    std::vector<float> input(input_info.desc.elementCount(), 1.0F);
    std::vector<float> output(output_info.desc.elementCount(), 0.0F);
    const irt::BufferView input_buffer{input.data(), input_info.desc, input_info.desc.byteSize(), 1,
                                       input_info.desc.byteSize(), input_info.name};
    const irt::BufferView output_buffer{output.data(), output_info.desc, output_info.desc.byteSize(), 1,
                                        output_info.desc.byteSize(), output_info.name};

    const std::vector<irt::BufferView> caller_order{output_buffer, input_buffer};
    ASSERT_NO_THROW(backend.execute(caller_order));
    EXPECT_EQ(state.last_execute_names, (std::vector<std::string>{"input", "output"}));
}

TEST(BackendRuntimeContractTest, SessionExecutionUsesTheSessionTensorDescriptors)
{
    FakeBackendState state;
    TransactionTestBackend backend(state);
    auto session = backend.createSession();

    const irt::TensorDesc session_desc{irt::TensorDataType::F32, irt::TensorLayout::NCHW, irt::MemoryKind::HOST,
                                       irt::Shape{2, 3, 2, 2}};
    std::vector<float> input(session_desc.elementCount(), 1.0F);
    std::vector<float> output(session_desc.elementCount(), 0.0F);
    const std::vector<irt::BufferView> buffers{
        {output.data(), session_desc, session_desc.byteSize(), 1, session_desc.byteSize(), "output"},
        {input.data(), session_desc, session_desc.byteSize(), 1, session_desc.byteSize(), "input"},
    };

    ASSERT_NO_THROW(backend.executeSession(*session, buffers));
    EXPECT_EQ(state.last_execute_names, (std::vector<std::string>{"input", "output"}));
}

TEST(IModelExecutionContractTest, CanonicalizesNamedBuffersIndependentOfCallerOrder)
{
    FakeBackendState state;
    g_fake_backend_state = &state;
    irt::model::SetBackendRuntimeFactoryOverride(&transactionBackendFactory);

    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));
    ASSERT_NO_THROW(model->setModelConfig(std::move(config)));
    ASSERT_NO_THROW(model->load("contract.onnx"));

    const auto input_info  = model->inputs().front();
    const auto output_info = model->outputs().front();
    std::vector<float> input(input_info.desc.elementCount(), 1.0F);
    std::vector<float> output(output_info.desc.elementCount(), 0.0F);
    irt::BufferView input_buffer{input.data(), input_info.desc, input_info.desc.byteSize(), 1,
                                 input_info.desc.byteSize(), input_info.name};
    irt::BufferView output_buffer{output.data(), output_info.desc, output_info.desc.byteSize(), 1,
                                  output_info.desc.byteSize(), output_info.name};

    std::vector<irt::BufferView> caller_order{output_buffer, input_buffer};
    ASSERT_NO_THROW(model->execute(caller_order));
    EXPECT_EQ(state.last_execute_names, (std::vector<std::string>{"input", "output"}));

    model.reset();
    g_fake_backend_state = nullptr;
    irt::model::SetBackendRuntimeFactoryOverride(nullptr);
}

/**
 * @brief 模型生命周期与默认配置的参数化测试基类。
 */
class ModelLifecycleRegisteredModelsTest : public RegisteredModelsTest
{
};

/**
 * @brief 判断模型 key 是否使用 ONNX 文件作为权重入口。
 * @param key 模型注册 key。
 * @return ONNX 通用模型或 ONNX-backed 模型族返回 true。
 */
bool IsOnnxBackedModelKey(const char *key)
{
    const std::string key_value = key ? key : "";
    return key_value == "onnx";
}

/**
 * @brief 使用新配置替换模型的输入/输出张量名称。
 * @param model 待更新的模型实例。
 * @param input_names 新输入张量名称列表。
 * @param output_names 新输出张量名称列表。
 */
void SetModelTensorNames(irt::model::IModel &model, std::vector<std::string> input_names,
                         std::vector<std::string> output_names)
{
    const auto &config     = model.modelConfig();
    auto        new_config = std::make_unique<irt::model::IModelConfig>();
    new_config->setNumClasses(config.numClasses());
    new_config->setInputShapes(config.inputShapes());
    new_config->setInputTensorNames(std::move(input_names));
    new_config->setOutputTensorNames(std::move(output_names));
    new_config->setFeatureTensorNames(config.featureTensorNames());
    new_config->setFeatureOnly(config.featureOnly());
    if (config.dynamicBatch())
    {
        new_config->setDynamicBatchRange(config.minBatchSize(), config.optBatchSize(), config.maxBatchSize());
    }
    new_config->setRuntime(config.runtime());
    model.setModelConfig(std::move(new_config));
}

/**
 * @brief 仅替换模型输入张量名称，其他配置保持不变。
 * @param model 待更新的模型实例。
 * @param input_names 新输入张量名称列表。
 */
void SetModelInputTensorNames(irt::model::IModel &model, std::vector<std::string> input_names)
{
    SetModelTensorNames(model, std::move(input_names), model.modelConfig().outputTensorNames());
}

/**
 * @brief 仅替换模型输出张量名称，其他配置保持不变。
 * @param model 待更新的模型实例。
 * @param output_names 新输出张量名称列表。
 */
void SetModelOutputTensorNames(irt::model::IModel &model, std::vector<std::string> output_names)
{
    SetModelTensorNames(model, model.modelConfig().inputTensorNames(), std::move(output_names));
}

std::filesystem::path FindResNet18Engine()
{
    for (const auto &candidate : {std::filesystem::path{"F:/models/resnet/resnet18-f37072fd/resnet18.engine"},
                                  std::filesystem::path{"assets/models/resnet/resnet18.engine"}})
    {
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }
    return {};
}

class DeviceAllocation final
{
public:
    explicit DeviceAllocation(const size_t bytes)
    {
        if (cudaMalloc(&data_, bytes) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed in model binding test");
        }
    }

    ~DeviceAllocation()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
        }
    }

    DeviceAllocation(const DeviceAllocation &)            = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;

    void *data() const noexcept
    {
        return data_;
    }

private:
    void *data_{nullptr};
};

std::vector<irt::BufferView> MakeDeviceBindings(const std::vector<irt::TensorInfo> &inputs,
                                                const std::vector<irt::TensorInfo> &outputs,
                                                std::vector<std::unique_ptr<DeviceAllocation>> &storage)
{
    std::vector<irt::BufferView> buffers;
    buffers.reserve(inputs.size() + outputs.size());
    const auto append = [&](const irt::TensorInfo &info)
    {
        const auto bytes = info.desc.byteSize();
        storage.push_back(std::make_unique<DeviceAllocation>(bytes));
        buffers.push_back(irt::BufferView{storage.back()->data(), info.desc, bytes, 1, bytes, info.name});
    };
    for (const auto &info : inputs)
    {
        append(info);
    }
    for (const auto &info : outputs)
    {
        append(info);
    }
    return buffers;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(KnownModels, ModelLifecycleRegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 所有内置模型都应使用统一的默认权重扩展名和引擎扩展名。
 */
TEST_P(ModelLifecycleRegisteredModelsTest, DefaultExtensionsMatchExpectedValues)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    const std::string expected_wts_extension = IsOnnxBackedModelKey(param.key) ? ".onnx" : ".wts";
    EXPECT_EQ(model->wtsExtension(), expected_wts_extension);
    EXPECT_EQ(model->engineExtension(), ".engine");
}

/**
 * @brief 所有内置模型的默认日志级别都应为 WARNING。
 */
TEST_P(ModelLifecycleRegisteredModelsTest, DefaultLogLevelIsWarning)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->logLevel(), irt::model::LogLevel::Warning);
}

/**
 * @brief 多次设置日志级别后，当前值应始终与最后一次设置保持一致。
 */
TEST(IModelDefaultPropertiesTest, SetLogLevelTakesEffect)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(irt::model::LogLevel::Verbose);
    EXPECT_EQ(model->logLevel(), irt::model::LogLevel::Verbose);

    model->setLogLevel(irt::model::LogLevel::Error);
    EXPECT_EQ(model->logLevel(), irt::model::LogLevel::Error);

    model->setLogLevel(irt::model::LogLevel::Info);
    EXPECT_EQ(model->logLevel(), irt::model::LogLevel::Info);
}

/**
 * @brief 通过 IModel 接口设置输入/输出张量名称后，应能原样读回配置值。
 */
TEST(IModelConfigTest, TensorNamesCanBeConfiguredThroughModelApi)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    const std::vector<std::string> input_names{"image", "aux"};
    const std::vector<std::string> output_names{"logits", "scores"};

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShapes({irt::Shape{1, 3, 224, 224}, irt::Shape{1, 3, 224, 224}});
    config->setInputTensorNames(input_names);
    config->setOutputTensorNames(output_names);
    model->setModelConfig(std::move(config));

    EXPECT_EQ(model->modelConfig().inputTensorNames(), input_names);
    EXPECT_EQ(model->modelConfig().outputTensorNames(), output_names);
}

/**
 * @brief IModelConfig 默认值应与 ImageNet 分类模型约定保持一致。
 */
TEST(IModelConfigTest, DefaultConfigMatchesImageNetClassificationContract)
{
    const irt::model::IModelConfig config;

    EXPECT_EQ(config.numClasses(), 1000);
    EXPECT_EQ(config.inputTensorNames(), (std::vector<std::string>{"input"}));
    EXPECT_EQ(config.outputTensorNames(), (std::vector<std::string>{"output"}));
    EXPECT_TRUE(config.featureTensorNames().empty());
    EXPECT_FALSE(config.featureOnly());
    EXPECT_EQ(config.runtime(), irt::model::ModelRuntime{});
    EXPECT_FALSE(config.dynamicBatch());
    EXPECT_EQ(config.minBatchSize(), 1);
    EXPECT_EQ(config.optBatchSize(), 1);
    EXPECT_EQ(config.maxBatchSize(), 1);
    ASSERT_EQ(config.inputShapes().size(), 1U);
    EXPECT_EQ(config.inputShape().rank(), 4);
    EXPECT_EQ(config.inputShape()[0], 1);
    EXPECT_EQ(config.inputShape()[1], 3);
    EXPECT_EQ(config.inputShape()[2], 224);
    EXPECT_EQ(config.inputShape()[3], 224);
}

/**
 * @brief IModelConfig setter 应能覆盖类别数、输入形状和特征配置。
 */
TEST(IModelConfigTest, SettersUpdateAllPublicConfigFields)
{
    irt::model::IModelConfig config;
    config.setNumClasses(7);
    config.setInputShape(irt::Shape{2, 3, 32, 32});
    config.setInputTensorNames({"image"});
    config.setOutputTensorNames({"logits", "aux"});
    config.setFeatureTensorNames({"layer1", "layer2"});
    config.setFeatureOnly(true);
    config.setDynamicBatchRange(1, 2, 4);
    config.setRuntime(irt::model::ModelRuntime::parse("onnxruntime:2"));

    EXPECT_EQ(config.numClasses(), 7);
    EXPECT_EQ(config.inputTensorNames(), (std::vector<std::string>{"image"}));
    EXPECT_EQ(config.outputTensorNames(), (std::vector<std::string>{"logits", "aux"}));
    EXPECT_EQ(config.featureTensorNames(), (std::vector<std::string>{"layer1", "layer2"}));
    EXPECT_TRUE(config.featureOnly());
    EXPECT_TRUE(config.dynamicBatch());
    EXPECT_EQ(config.minBatchSize(), 1);
    EXPECT_EQ(config.optBatchSize(), 2);
    EXPECT_EQ(config.maxBatchSize(), 4);
    EXPECT_EQ(config.runtime().backend(), irt::model::ModelRuntime::Backend::ONNXRuntime);
    EXPECT_EQ(config.runtime().device(), irt::model::ModelRuntime::Device::GPU);
    EXPECT_EQ(config.runtime().deviceId(), 2);
    EXPECT_EQ(config.inputShape()[0], 2);
    EXPECT_EQ(config.inputShape()[2], 32);
}

/**
 * @brief 动态 batch 支持按模型声明，尚未改造的手写网络应明确拒绝。
 */
TEST(IModelConfigTest, DynamicBatchSupportIsModelScoped)
{
    auto supported_config = std::make_unique<irt::model::IModelConfig>();
    supported_config->setDynamicBatchRange(1, 2, 4);
    auto resnet = irt::model::CreateModel("resnet18", std::move(supported_config));
    ASSERT_NE(resnet, nullptr);
    EXPECT_TRUE(resnet->modelConfig().dynamicBatch());
    EXPECT_EQ(resnet->modelConfig().maxBatchSize(), 4);

    auto unsupported_config = std::make_unique<irt::model::IModelConfig>();
    unsupported_config->setDynamicBatchRange(1, 2, 2);
    ExpectIrtExceptionCode([&] { irt::model::CreateModel("alexnet", std::move(unsupported_config)); },
                           irt::Status::ERROR_NOT_IMPLEMENTED);
}

TEST(IModelConfigTest, SetModelConfigRejectsUnsupportedDynamicBatchBeforeCommit)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);
    ASSERT_FALSE(model->modelConfig().dynamicBatch());

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setDynamicBatchRange(1, 2, 2);

    ExpectIrtExceptionCode([&] { model->setModelConfig(std::move(config)); }, irt::Status::ERROR_NOT_IMPLEMENTED);

    EXPECT_FALSE(model->modelConfig().dynamicBatch());
    EXPECT_EQ(model->modelConfig().maxBatchSize(), 1);
}

/**
 * @brief CreateModel 应保留调用方传入的后端和设备配置。
 */
TEST(IModelConfigTest, CreateModelPreservesBackendAndDeviceFromConfig)
{
#if !INFERRT_BUILD_ONNX
    GTEST_SKIP() << "ONNX Runtime backend was disabled at CMake configure time";
#endif

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:1"));

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->runtime(), irt::model::ModelRuntime::parse("onnxruntime:1"));
    EXPECT_EQ(model->modelConfig().runtime(), irt::model::ModelRuntime::parse("onnxruntime:1"));
}

TEST(ModelRuntimeTest, ParsesDeviceShorthandsAndBackendTargets)
{
    const std::vector<std::pair<std::string, std::string>> cases{
        {"cpu", "onnxruntime:cpu"},
        {"gpu:0", "tensorrt:0"},
        {"gpu:3", "tensorrt:3"},
        {"cuda:2", "tensorrt:2"},
        {"tensorrt:1", "tensorrt:1"},
        {"onnx:cpu", "onnxruntime:cpu"},
        {"openvino:4", "openvino:4"},
    };

    for (const auto &[specification, expected] : cases)
    {
        EXPECT_EQ(irt::model::ModelRuntime::parse(specification).toString(), expected) << specification;
    }
}

TEST(ModelRuntimeTest, RejectsInvalidSpecifications)
{
    for (const auto &specification : {std::string{}, std::string{"gpu:x"}, std::string{"cuda:-1"},
                                      std::string{"unknown:cpu"}, std::string{"tensorrt:cpu"}})
    {
        EXPECT_THROW(irt::model::ModelRuntime::parse(specification), irt::Exception) << specification;
    }
}

/**
 * @brief setInputShapes 应替换完整输入形状列表，并保持 inputShape 指向第一个输入。
 */
TEST(IModelConfigTest, SetInputShapesReplacesShapeListAndUpdatesFirstShape)
{
    irt::model::IModelConfig config;
    config.setInputShapes({
        irt::Shape{1, 3, 64, 64},
        irt::Shape{1, 1, 16, 16}
    });

    ASSERT_EQ(config.inputShapes().size(), 2U);
    EXPECT_EQ(config.inputShape()[2], 64);
    EXPECT_EQ(config.inputShapes()[1][1], 1);
}

/**
 * @brief setModelConfig 传入空指针时应恢复默认配置并清理旧的自定义值。
 */
TEST(IModelConfigTest, SetNullModelConfigResetsToDefaultConfig)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(12);
    config->setInputTensorNames({"custom_input"});
    config->setFeatureTensorNames({"feature"});
    config->setOutputTensorNames({"feature"});
    config->setFeatureOnly(true);
    model->setModelConfig(std::move(config));

    model->setModelConfig(nullptr);

    EXPECT_EQ(model->modelConfig().numClasses(), 1000);
    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"input"}));
    EXPECT_FALSE(model->modelConfig().featureOnly());
}

/**
 * @brief 通过 CreateModel 传入的自定义 config，应保留其中的输入/输出张量名称。
 */
TEST(IModelConfigTest, CreateModelPreservesCustomTensorNamesFromConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShapes({irt::Shape{1, 3, 224, 224}, irt::Shape{1, 3, 224, 224}});
    config->setInputTensorNames({"custom_input_0", "custom_input_1"});
    config->setOutputTensorNames({"custom_output_0", "custom_output_1"});

    auto model = irt::model::CreateModel("alexnet", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"custom_input_0", "custom_input_1"}));
    EXPECT_EQ(model->modelConfig().outputTensorNames(),
              (std::vector<std::string>{"custom_output_0", "custom_output_1"}));
}

/**
 * @brief featureOnly 配置应保留层 key、导出张量名及 featureOnly 标志（层 key 与导出名可不同）。
 */
TEST(IModelConfigTest, FeatureOnlyConfigPreservesLayerKeysAndOutputNames)
{
    auto via_model_api = irt::model::CreateModel("resnet18");
    ASSERT_NE(via_model_api, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(via_model_api->modelConfig().numClasses());
    config->setInputShapes(via_model_api->modelConfig().inputShapes());
    config->setInputTensorNames(via_model_api->modelConfig().inputTensorNames());
    config->setFeatureTensorNames({"layer1", "layer4"});
    config->setOutputTensorNames({"feat_low", "feat_high"});
    config->setFeatureOnly(true);
    via_model_api->setModelConfig(std::move(config));

    EXPECT_TRUE(via_model_api->modelConfig().featureOnly());
    EXPECT_EQ(via_model_api->modelConfig().featureTensorNames(), (std::vector<std::string>{"layer1", "layer4"}));
    EXPECT_EQ(via_model_api->modelConfig().outputTensorNames(), (std::vector<std::string>{"feat_low", "feat_high"}));

    auto create_config = std::make_unique<irt::model::IModelConfig>();
    create_config->setFeatureTensorNames({"layer2"});
    create_config->setOutputTensorNames({"layer2"});
    create_config->setFeatureOnly(true);

    auto via_create = irt::model::CreateModel("resnet50", std::move(create_config));
    ASSERT_NE(via_create, nullptr);
    EXPECT_TRUE(via_create->modelConfig().featureOnly());
    EXPECT_EQ(via_create->modelConfig().featureTensorNames(), (std::vector<std::string>{"layer2"}));
    EXPECT_EQ(via_create->modelConfig().outputTensorNames(), (std::vector<std::string>{"layer2"}));
}

/**
 * @brief 在 logger 尚未初始化时设置日志级别，也应正确保存在模型内部。
 */
TEST(IModelLogLevelTest, SetLogLevelBeforeLoggerInitPersistsValue)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(irt::model::LogLevel::Verbose);
    EXPECT_EQ(model->logLevel(), irt::model::LogLevel::Verbose);
}

/**
 * @brief 未构建 engine 时调用 save，应抛出 ERROR_INVALID_OPERATION。
 */
TEST(IModelSaveTest, SaveWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->save("output.engine"); }, irt::Status::INVALID_OPERATION);
}

/**
 * @brief 加载不存在的 engine 文件时，应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelLoadTest, LoadNonExistentFileThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->load("/non/existent/path/model.engine"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 使用不存在的权重文件构建模型时，应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称列表为空时，配置方法应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithEmptyInputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelInputTensorNames(*model, {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称列表为空时，配置方法应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithEmptyOutputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelOutputTensorNames(*model, {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称中包含空字符串时，配置方法应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyInputTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelInputTensorNames(*model, {""}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称中包含空字符串时，配置方法应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyOutputTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelOutputTensorNames(*model, {""}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当特征张量名称中包含空字符串时，配置方法应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyFeatureTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    ExpectIrtExceptionCode([&] { config->setFeatureTensorNames({""}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief featureOnly 时若输出张量名数量与特征层 key 不一致，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithMismatchedOutputTensorCountInFeatureOnlyModeThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1", "pool3"});
    config->setOutputTensorNames({"feat_only_one"});
    config->setFeatureOnly(true);
    ExpectIrtExceptionCode([&] { model->setModelConfig(std::move(config)); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当启用 featureOnly 但未提供特征张量名称时，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithFeatureOnlyAndNoFeatureTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureOnly(true);
    config->setOutputTensorNames({"layer1"});
    ExpectIrtExceptionCode([&] { model->setModelConfig(std::move(config)); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief buildOrLoad 在权重文件不存在时，也应沿用同样的错误码约定。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称列表为空时，配置方法也应沿用相同的校验规则。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithEmptyInputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelInputTensorNames(*model, {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称列表为空时，配置方法也应沿用相同的校验规则。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithEmptyOutputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { SetModelOutputTensorNames(*model, {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief featureOnly 时若输出张量名数量与特征层 key 不一致，buildOrLoad 应拒绝该非法配置。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithMismatchedOutputTensorCountInFeatureOnlyModeThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1", "pool3"});
    config->setOutputTensorNames({"feat_only_one"});
    config->setFeatureOnly(true);
    ExpectIrtExceptionCode([&] { model->setModelConfig(std::move(config)); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当启用 featureOnly 但未提供特征张量名称时，buildOrLoad 应拒绝该非法配置。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithFeatureOnlyAndNoFeatureTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureOnly(true);
    config->setOutputTensorNames({"layer1"});
    ExpectIrtExceptionCode([&] { model->setModelConfig(std::move(config)); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当加载不存在或非法的 engine 文件失败时，模型原有配置与状态必须完整保留（强异常安全保证）。
 */
TEST(IModelBuildOrLoadTest, StrongExceptionSafetyOnFailedLoadPreservesConfig)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(100);
    config->setInputShapes({irt::Shape{1, 3, 224, 224}});
    model->setModelConfig(std::move(config));

    EXPECT_EQ(model->modelConfig().numClasses(), 100);
    EXPECT_EQ(model->modelConfig().inputShapes().size(), 1);

    ExpectIrtExceptionCode([&] { model->load("/invalid/path/nonexistent.engine"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    // 验证失败后旧配置依然完整保留
    EXPECT_EQ(model->modelConfig().numClasses(), 100);
    EXPECT_EQ(model->modelConfig().inputShapes().size(), 1);
    EXPECT_EQ(model->name(), "AlexNet");
}

/**
 * @brief 当从权重文件构建失败时，模型原有配置与状态必须完整保留（强异常安全保证）。
 */
TEST(IModelBuildOrLoadTest, StrongExceptionSafetyOnFailedBuildPreservesConfig)
{
    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(42);
    config->setInputShapes({irt::Shape{1, 3, 224, 224}});
    model->setModelConfig(std::move(config));

    EXPECT_EQ(model->modelConfig().numClasses(), 42);

    ExpectIrtExceptionCode([&] { model->build("/invalid/path/nonexistent.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(model->modelConfig().numClasses(), 42);
    EXPECT_EQ(model->modelConfig().inputShapes().size(), 1);
    EXPECT_EQ(model->name(), "ResNet18");
}

TEST(IModelBuildOrLoadTest, FailedBackendReloadKeepsPreviousRuntimeExecutable)
{
    FakeBackendState state;
    g_fake_backend_state = &state;

    irt::model::SetBackendRuntimeFactoryOverride(&transactionBackendFactory);
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));
    model->setModelConfig(std::move(config));

    EXPECT_NO_THROW(model->load("initial.onnx"));
    std::vector<float> input_storage;
    std::vector<float> output_storage;
    auto              execution_buffers = MakeTransactionExecutionBuffers(*model, input_storage, output_storage);
    EXPECT_NO_THROW(model->infer(execution_buffers));
    EXPECT_EQ(state.load_calls.load(), 1);
    EXPECT_EQ(state.infer_calls.load(), 1);

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_EQ(input_infos.size(), 1U);
    ASSERT_EQ(output_infos.size(), 1U);
    EXPECT_EQ(input_infos.front().name, "input");
    EXPECT_EQ(input_infos.front().mode, irt::TensorIOMode::Input);
    EXPECT_EQ(input_infos.front().desc.shape, irt::Shape({1, 3, 2, 2}));
    EXPECT_EQ(input_infos.front().desc.memory_kind, irt::MemoryKind::HOST);
    EXPECT_EQ(output_infos.front().mode, irt::TensorIOMode::Output);
    EXPECT_EQ(output_infos.front().desc.memory_kind, irt::MemoryKind::HOST);

    EXPECT_NO_THROW(model->execute(execution_buffers));
    EXPECT_EQ(state.infer_calls.load(), 2);

    state.fail_next_load = true;
    try
    {
        model->load("replacement.onnx");
        FAIL() << "Expected injected backend load failure";
    }
    catch (const irt::Exception &exception)
    {
        EXPECT_EQ(exception.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }

    EXPECT_EQ(model->runtime().backend(), irt::model::ModelRuntime::Backend::ONNXRuntime);
    EXPECT_NO_THROW(model->infer(execution_buffers));
    EXPECT_EQ(state.load_calls.load(), 2);
    EXPECT_EQ(state.infer_calls.load(), 3);

    model.reset();
    g_fake_backend_state = nullptr;
    irt::model::SetBackendRuntimeFactoryOverride(nullptr);
}

TEST(IModelBuildOrLoadTest, BackendMetadataFailuresKeepPreviousRuntimeExecutable)
{
    FakeBackendState state;
    g_fake_backend_state = &state;
    irt::model::SetBackendRuntimeFactoryOverride(&transactionBackendFactory);

    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));
    model->setModelConfig(std::move(config));
    ASSERT_NO_THROW(model->load("initial.onnx"));
    std::vector<float> input_storage;
    std::vector<float> output_storage;
    const auto        execution_buffers = MakeTransactionExecutionBuffers(*model, input_storage, output_storage);
    ASSERT_NO_THROW(model->infer(execution_buffers));

    const auto old_config = model->modelConfig();
    const auto old_input_shape = old_config.inputShapes().front();
    const auto old_load_calls = state.load_calls.load();

    for (const auto failure : {0, 1, 2})
    {
        if (failure == 0)
        {
            state.fail_next_io_names = true;
        }
        else if (failure == 1)
        {
            state.fail_next_shape = true;
        }
        else
        {
            state.fail_next_data_type = true;
        }

        EXPECT_THROW(model->load("replacement.onnx"), irt::Exception);
        EXPECT_EQ(model->modelConfig().runtime(), old_config.runtime());
        EXPECT_EQ(model->modelConfig().inputShapes().front(), old_input_shape);
        EXPECT_NO_THROW(model->infer(execution_buffers));
    }

    EXPECT_EQ(state.load_calls.load(), old_load_calls + 3);
    EXPECT_GE(state.io_names_calls.load(), 2);
    EXPECT_GE(state.shape_calls.load(), 2);
    EXPECT_GE(state.data_type_calls.load(), 2);

    model.reset();
    g_fake_backend_state = nullptr;
    irt::model::SetBackendRuntimeFactoryOverride(nullptr);
}

TEST(IModelBuildOrLoadTest, BackendCreationFailureKeepsPreviousRuntimeExecutable)
{
    FakeBackendState state;
    g_fake_backend_state = &state;
    irt::model::SetBackendRuntimeFactoryOverride(&transactionBackendFactory);

    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));
    model->setModelConfig(std::move(config));
    ASSERT_NO_THROW(model->load("initial.onnx"));
    std::vector<float> input_storage;
    std::vector<float> output_storage;
    const auto        execution_buffers = MakeTransactionExecutionBuffers(*model, input_storage, output_storage);
    ASSERT_NO_THROW(model->infer(execution_buffers));

    state.fail_next_create = true;
    auto replacement = std::make_unique<irt::model::IModelConfig>(model->modelConfig());
    EXPECT_THROW(model->setModelConfig(std::move(replacement)), irt::Exception);
    EXPECT_EQ(model->runtime().backend(), irt::model::ModelRuntime::Backend::ONNXRuntime);
    EXPECT_NO_THROW(model->infer(execution_buffers));

    model.reset();
    g_fake_backend_state = nullptr;
    irt::model::SetBackendRuntimeFactoryOverride(nullptr);
}

#if INFERRT_BUILD_ONNX
/**
 * @brief 当 ONNX/OpenVINO 后端加载失败时，旧配置与状态亦需完整保留。
 */
TEST(IModelBuildOrLoadTest, StrongExceptionSafetyOnFailedBackendLoadPreservesConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(10);
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->modelConfig().numClasses(), 10);

    ExpectIrtExceptionCode([&] { model->build("/invalid/path/nonexistent.onnx"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(model->modelConfig().numClasses(), 10);
    EXPECT_EQ(model->runtime().backend(), irt::model::ModelRuntime::Backend::ONNXRuntime);
}
#endif

TEST(IModelConfigValidationTest, SetInputShapesRejectsEmptyOrNonPositiveDimensions)
{
    irt::model::IModelConfig config;
    EXPECT_THROW(config.setInputShapes({}), irt::Exception);
    EXPECT_THROW(config.setInputShapes({irt::Shape{0, 3, 224, 224}}), irt::Exception);
    EXPECT_THROW(config.setInputShapes({irt::Shape{1, -1, 224, 224}}), irt::Exception);
    EXPECT_THROW(config.setInputShape(irt::Shape{-1, 3, 224, 224}), irt::Exception);
    EXPECT_THROW(config.setNumClasses(0), irt::Exception);
    EXPECT_THROW(config.setNumClasses(-5), irt::Exception);
}

TEST(IModelLifecycleTest, InvalidHandleRejectsExecutionCapabilitiesQuery)
{
    irt::model::IModel model;

    EXPECT_FALSE(model.isValid());
    ExpectIrtExceptionCode([&] { (void)model.capabilities(); }, irt::Status::INVALID_OPERATION);
}

TEST(IModelLifecycleTest, StrongExceptionSafetyOnFailedLoadPreservesConfig)
{
    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(100);
    config->setInputShapes({irt::Shape{1, 3, 256, 256}});
    model->setModelConfig(std::move(config));

    EXPECT_EQ(model->modelConfig().numClasses(), 100);

    ExpectIrtExceptionCode([&] { model->load("/invalid/path/nonexistent.engine"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    // 强异常安全：加载失败后配置必须保持不变
    EXPECT_EQ(model->modelConfig().numClasses(), 100);
    EXPECT_EQ(model->modelConfig().inputShapes().front()[2], 256);
}

TEST(IModelLifecycleTest, SetModelConfigRejectsNullOrInvalidTensorNames)
{
    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);

    // 空名称输入张量列表应抛出异常
    EXPECT_THROW(SetModelInputTensorNames(*model, {""}), irt::Exception);

    // 空名称输出张量列表应抛出异常
    EXPECT_THROW(SetModelOutputTensorNames(*model, {""}), irt::Exception);
}

TEST(IModelLifecycleTest, StrongExceptionSafetyOnInvalidConfigRejectionPreservesOldState)
{
    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);

    auto valid_config = std::make_unique<irt::model::IModelConfig>();
    valid_config->setNumClasses(80);
    valid_config->setInputShapes({irt::Shape{1, 3, 320, 320}});
    model->setModelConfig(std::move(valid_config));

    EXPECT_EQ(model->modelConfig().numClasses(), 80);
    EXPECT_EQ(model->modelConfig().inputShapes().front()[2], 320);

    // 尝试传入非法配置
    auto bad_config = std::make_unique<irt::model::IModelConfig>();
    EXPECT_THROW(bad_config->setNumClasses(-1), irt::Exception);

    // 旧配置与模型能力完整保留
    EXPECT_EQ(model->modelConfig().numClasses(), 80);
    EXPECT_EQ(model->modelConfig().inputShapes().front()[2], 320);
    EXPECT_EQ(model->name(), "ResNet18");
}

TEST(IModelLifecycleTest, StrongExceptionSafetyOnRuntimeReloadFailurePreservesActiveRuntime)
{
    const auto engine_path = FindResNet18Engine();

    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available to test live runtime reload failure";
    }

    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);
    EXPECT_NO_THROW(model->load(engine_path.string()));
    EXPECT_TRUE(model->isValid());

    const int old_classes = model->modelConfig().numClasses();
    const auto old_runtime = model->runtime();

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());
    for (const auto &info : input_infos)
    {
        EXPECT_EQ(info.desc.memory_kind, irt::MemoryKind::DEVICE);
    }
    for (const auto &info : output_infos)
    {
        EXPECT_EQ(info.desc.memory_kind, irt::MemoryKind::DEVICE);
    }

    // 尝试加载一个不存在的文件，预期抛出异常
    EXPECT_THROW(model->load("non_existent_engine_file.engine"), irt::Exception);

    // 强异常安全保证：旧运行时、旧配置和有效性依然完整保留
    EXPECT_TRUE(model->isValid());
    EXPECT_EQ(model->modelConfig().numClasses(), old_classes);
    EXPECT_EQ(model->runtime().backend(), old_runtime.backend());
}

TEST(TensorRTBackendLifecycleTest, FailedReloadPreservesActiveEngineAndExecution)
{
    const auto engine_path = FindResNet18Engine();
    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available to test direct TensorRT backend reload";
    }

    auto backend = irt::model::CreateBackendRuntime(irt::model::ModelRuntime::Backend::TensorRT);
    ASSERT_NE(backend, nullptr);
    irt::model::IModelConfig config;
    ASSERT_NO_THROW(backend->load(engine_path.string(), config, "TensorRTBackendLifecycleTest"));

    const auto input_infos  = backend->inputs();
    const auto output_infos = backend->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());
    const auto input_names  = backend->ioTensorNames(irt::TensorIOMode::Input);
    const auto output_names = backend->ioTensorNames(irt::TensorIOMode::Output);
    const auto input_shape  = backend->tensorShape(input_names.front());
    const auto output_shape = backend->tensorShape(output_names.front());
    const auto input_type   = backend->tensorDataType(input_names.front());
    const auto output_type  = backend->tensorDataType(output_names.front());

    std::vector<std::unique_ptr<DeviceAllocation>> storage;
    auto buffers = MakeDeviceBindings(input_infos, output_infos, storage);
    ASSERT_NO_THROW(backend->execute(buffers));

    test::model::TempWeightsFile invalid_engine("inferrt_invalid_engine_");
    EXPECT_THROW(backend->load(invalid_engine.path().string(), config, "TensorRTBackendLifecycleTest"),
                 irt::Exception);

    EXPECT_EQ(backend->ioTensorNames(irt::TensorIOMode::Input), input_names);
    EXPECT_EQ(backend->ioTensorNames(irt::TensorIOMode::Output), output_names);
    EXPECT_EQ(backend->tensorShape(input_names.front()), input_shape);
    EXPECT_EQ(backend->tensorShape(output_names.front()), output_shape);
    EXPECT_EQ(backend->tensorDataType(input_names.front()), input_type);
    EXPECT_EQ(backend->tensorDataType(output_names.front()), output_type);
    EXPECT_NO_THROW(backend->execute(buffers));
}

TEST(IModelTensorRTBindingTest, RejectsTypedBufferWithWrongMemoryKind)
{
    const auto engine_path = FindResNet18Engine();
    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available for TensorRT binding validation";
    }

    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);
    ASSERT_NO_THROW(model->load(engine_path.string()));

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());
    std::vector<std::unique_ptr<DeviceAllocation>> storage;
    auto buffers = MakeDeviceBindings(input_infos, output_infos, storage);
    buffers.front().desc.memory_kind = irt::MemoryKind::HOST;

    ExpectIrtExceptionCode([&] { model->infer(buffers); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(IModelTensorRTBindingTest, RejectsTypedBufferWithWrongDataType)
{
    const auto engine_path = FindResNet18Engine();
    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available for TensorRT binding validation";
    }

    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);
    ASSERT_NO_THROW(model->load(engine_path.string()));

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());
    std::vector<std::unique_ptr<DeviceAllocation>> storage;
    auto buffers = MakeDeviceBindings(input_infos, output_infos, storage);
    buffers.front().desc.data_type = irt::TensorDataType::F16;
    buffers.front().bytes_per_request = buffers.front().desc.byteSize();
    buffers.front().capacity_bytes     = buffers.front().bytes_per_request;

    ExpectIrtExceptionCode([&] { model->infer(buffers); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(IModelTensorRTBindingTest, RejectsTypedBufferWithWrongShape)
{
    const auto engine_path = FindResNet18Engine();
    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available for TensorRT binding validation";
    }

    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);
    ASSERT_NO_THROW(model->load(engine_path.string()));

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());
    ASSERT_EQ(input_infos.front().desc.shape, irt::Shape({1, 3, 224, 224}));
    std::vector<std::unique_ptr<DeviceAllocation>> storage;
    auto buffers = MakeDeviceBindings(input_infos, output_infos, storage);
    buffers.front().desc.shape = irt::Shape{1, 3, 112, 448};

    ExpectIrtExceptionCode([&] { model->infer(buffers); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(IModelTensorRTBindingTest, AcceptsOpaqueByteViewsWithValidDeviceCapacity)
{
    const auto engine_path = FindResNet18Engine();
    if (engine_path.empty())
    {
        GTEST_SKIP() << "No ResNet18 engine available for TensorRT byte-view validation";
    }

    auto model = irt::model::CreateModel("resnet18");
    ASSERT_NE(model, nullptr);
    ASSERT_NO_THROW(model->load(engine_path.string()));

    const auto input_infos  = model->inputs();
    const auto output_infos = model->outputs();
    ASSERT_FALSE(input_infos.empty());
    ASSERT_FALSE(output_infos.empty());

    std::vector<std::unique_ptr<DeviceAllocation>> storage;
    std::vector<irt::BufferView>                    buffers;
    buffers.reserve(input_infos.size() + output_infos.size());
    const auto append = [&](const irt::TensorInfo &info)
    {
        const auto bytes = info.desc.byteSize();
        storage.push_back(std::make_unique<DeviceAllocation>(bytes));
        buffers.push_back(irt::BufferView::fromBytes(storage.back()->data(), bytes, irt::MemoryKind::DEVICE,
                                                     info.name));
    };
    for (const auto &info : input_infos)
    {
        append(info);
    }
    for (const auto &info : output_infos)
    {
        append(info);
    }

    EXPECT_NO_THROW(model->infer(buffers));
}
