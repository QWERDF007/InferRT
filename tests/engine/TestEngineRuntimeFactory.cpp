#include "../../src/engine/priv/EngineRuntimePlan.hpp"
#include "../../src/engine/priv/EngineTestHooks.hpp"
#include "../../src/model/priv/BackendRuntime.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/InferenceEngine.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

namespace {

using irt::engine::EngineConfig;
using irt::engine::PipelinePlan;
using irt::engine::TensorDataType;
using irt::engine::TensorLayout;
using irt::engine::MemoryKind;

class FakeSession final : public irt::ITensorRuntimeSession
{
public:
    irt::Shape tensorShape(const std::string &) const override
    {
        return shape_;
    }

    irt::TensorDataType tensorDataType(const std::string &) const override
    {
        return irt::TensorDataType::F32;
    }

    void setTensorShape(const std::string &, const irt::Shape &shape) override
    {
        shape_ = shape;
    }

    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options) override
    {
        ASSERT_EQ(buffers.size(), 2U);
        ASSERT_NE(buffers[0].data, nullptr);
        ASSERT_NE(buffers[1].data, nullptr);
        ASSERT_EQ(buffers[0].bytes_per_request, buffers[1].bytes_per_request);
        ASSERT_EQ(cudaMemcpyAsync(buffers[1].data, buffers[0].data, buffers[0].bytes_per_request,
                                  cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(options.stream)),
                  cudaSuccess);
    }

private:
    irt::Shape shape_{1, 3, 2, 2};
};

class FakeRuntimePlan final : public irt::engine::priv::IEngineRuntimePlan
{
public:
    std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override
    {
        return std::make_unique<FakeSession>();
    }

    void executeSession(irt::ITensorRuntimeSession &session, std::span<const irt::BufferView> buffers,
                        irt::ExecuteOptions options) const override
    {
        auto normalized = irt::normalizeExecutionBuffers(buffers, input_infos_, output_infos_);
        session.execute(normalized, options);
    }

    std::vector<irt::TensorInfo> inputs() const override
    {
        return input_infos_;
    }

    std::vector<irt::TensorInfo> outputs() const override
    {
        return output_infos_;
    }

    int fixedBatchSize() const noexcept override
    {
        return 0;
    }

private:
    std::vector<irt::TensorInfo> input_infos_{
        {"image",
         {irt::TensorDataType::F32, irt::TensorLayout::Opaque, irt::MemoryKind::DEVICE, irt::Shape{1, 3, 2, 2}},
         irt::TensorIOMode::Input}};
    std::vector<irt::TensorInfo> output_infos_{
        {"output",
         {irt::TensorDataType::F32, irt::TensorLayout::Opaque, irt::MemoryKind::DEVICE, irt::Shape{1, 3, 2, 2}},
         irt::TensorIOMode::Output}};
};

std::atomic<int> g_factory_calls{0};
std::atomic<int> g_fail_on_call{0};

std::shared_ptr<irt::engine::priv::IEngineRuntimePlan> fakeRuntimeFactory(
    const EngineConfig &, int, const PipelinePlan *)
{
    const int call = g_factory_calls.fetch_add(1) + 1;
    if (g_fail_on_call.load() == call)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "injected runtime plan factory failure at call %d", call);
    }
    return std::make_shared<FakeRuntimePlan>();
}

class RuntimeFactoryGuard final
{
public:
    explicit RuntimeFactoryGuard(const int fail_on_call)
    {
        g_factory_calls = 0;
        g_fail_on_call  = fail_on_call;
        irt::engine::priv::SetEngineRuntimePlanFactoryOverride(&fakeRuntimeFactory);
    }

    ~RuntimeFactoryGuard()
    {
        irt::engine::priv::SetEngineRuntimePlanFactoryOverride(nullptr);
        g_fail_on_call = 0;
    }
};

class RuntimeBackendFactoryGuard final
{
public:
    explicit RuntimeBackendFactoryGuard(std::unique_ptr<irt::model::IBackendRuntime> (*factory)(
        irt::model::ModelRuntime::Backend))
    {
        irt::model::SetBackendRuntimeFactoryOverride(factory);
    }

    ~RuntimeBackendFactoryGuard()
    {
        irt::model::SetBackendRuntimeFactoryOverride(nullptr);
    }
};

struct RuntimeShapeFixture
{
    bool dynamic_batch{false};
    int  fixed_batch{1};
};

RuntimeShapeFixture *g_runtime_shape_fixture = nullptr;

class ShapeContractBackend final : public irt::model::IBackendRuntime
{
public:
    explicit ShapeContractBackend(const RuntimeShapeFixture &fixture) : fixture_(fixture) {}

    irt::model::ModelRuntime::Backend backend() const noexcept override
    {
        return irt::model::ModelRuntime::Backend::TensorRT;
    }

    void load(const std::string &, const irt::model::IModelConfig &, const std::string &) override {}

    std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const override
    {
        return mode == irt::TensorIOMode::Input ? std::vector<std::string>{"input"}
                                                : std::vector<std::string>{"output"};
    }

    irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const noexcept override
    {
        (void)mode;
        return irt::MemoryKind::DEVICE;
    }

    irt::Shape tensorShape(const std::string &name) const override
    {
        if (name == "input")
        {
            return irt::Shape{fixture_.fixed_batch, 3, 2, 2};
        }
        if (name == "output")
        {
            return irt::Shape{fixture_.fixed_batch, 3, 2, 2};
        }
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown fixture tensor: %s", name.c_str());
    }

    bool isInputBatchDynamic(const std::string &name) const override
    {
        if (name != "input")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown fixture input: %s", name.c_str());
        }
        return fixture_.dynamic_batch;
    }

    irt::TensorDataType tensorDataType(const std::string &name) const override
    {
        if (name != "input" && name != "output")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unknown fixture tensor: %s", name.c_str());
        }
        return irt::TensorDataType::F32;
    }

    void setTensorShape(const std::string &, const irt::Shape &) override {}

    void execute(std::span<const irt::BufferView>, irt::ExecuteOptions) override {}

private:
    const RuntimeShapeFixture &fixture_;
};

std::unique_ptr<irt::model::IBackendRuntime> shapeContractBackendFactory(
    const irt::model::ModelRuntime::Backend backend)
{
    if (backend != irt::model::ModelRuntime::Backend::TensorRT || g_runtime_shape_fixture == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "unexpected backend in shape contract test");
    }
    return std::make_unique<ShapeContractBackend>(*g_runtime_shape_fixture);
}

EngineConfig makeRuntimePlanConfig(const int min_batch, const int opt_batch, const int max_batch)
{
    EngineConfig config;
    config.model_name         = "shape_contract_engine";
    config.engine_file        = "shape_contract.engine";
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size      = min_batch;
    config.opt_batch_size      = opt_batch;
    config.max_batch_size      = max_batch;
    return config;
}

std::shared_ptr<const PipelinePlan> makePipeline()
{
    auto registry = std::make_shared<irt::engine::OperatorRegistry>();
    irt::engine::registerBuiltinOperators(*registry);

    irt::engine::TensorDesc host{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3};
    irt::engine::TensorDesc device{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3};
    irt::engine::PipelineBuilder builder;
    builder.addTensor("image_host", host)
        .addTensor("image_device", device)
        .addNode("cpu.image_to_tensor",
                  {{},
                   {"image_host"},
                   irt::engine::CpuImageToTensorOptions{[]
                                                        {
                                                            irt::PreprocessSpec spec;
                                                            spec.input_width    = 2;
                                                            spec.input_height   = 2;
                                                            spec.input_channels = 3;
                                                            spec.mean           = {0.0F, 0.0F, 0.0F};
                                                            spec.stddev         = {1.0F, 1.0F, 1.0F};
                                                            return spec;
                                                        }()}})
        .addNode("cuda.upload", {{"image_host"}, {"image_device"}, {}})
        .bindModelInput("image_device", "image");
    return builder.build(std::move(registry));
}

EngineConfig makeConfig()
{
    EngineConfig config;
    config.model_name              = "fake_runtime_engine";
    config.engine_file             = "unused.engine";
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size          = 1;
    config.opt_batch_size          = 1;
    config.max_batch_size          = 1;
    config.execution_slots         = 1;
    config.cpu_preprocess_workers  = 1;
    config.cpu_postprocess_workers = 1;
    return config;
}

bool hasCudaDevice()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

TEST(EngineRuntimePlanContractTest, ExposesTheSharedCoreTensorDescriptorContract)
{
    static_assert(std::is_base_of_v<irt::IExecutionDescriptor, irt::engine::priv::IEngineRuntimePlan>);
    static_assert(std::is_base_of_v<irt::IExecutionPlan, irt::engine::priv::IEngineRuntimePlan>);

    RuntimeFactoryGuard guard(0);
    auto plan = irt::engine::priv::CreateEngineRuntimePlan(makeConfig(), 0, nullptr);
    const auto input_infos  = plan->inputs();
    const auto output_infos = plan->outputs();

    ASSERT_EQ(input_infos.size(), 1U);
    ASSERT_EQ(output_infos.size(), 1U);
    EXPECT_EQ(input_infos.front().name, "image");
    EXPECT_EQ(input_infos.front().mode, irt::TensorIOMode::Input);
    EXPECT_EQ(input_infos.front().desc.data_type, irt::TensorDataType::F32);
    EXPECT_EQ(input_infos.front().desc.memory_kind, irt::MemoryKind::DEVICE);
    EXPECT_EQ(output_infos.front().name, "output");
    EXPECT_EQ(output_infos.front().mode, irt::TensorIOMode::Output);
    EXPECT_EQ(output_infos.front().desc.data_type, irt::TensorDataType::F32);
    EXPECT_EQ(output_infos.front().desc.memory_kind, irt::MemoryKind::DEVICE);
}

TEST(EngineRuntimeFactoryTest, ValidFakeSessionTraversesProductionExecutionChain)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    RuntimeFactoryGuard guard(0);
    auto engine = irt::engine::InferenceEngine::create(makeConfig(), makePipeline());
    ASSERT_NO_THROW(engine->start());

    const auto result = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(7, 8, 9))).get();
    ASSERT_TRUE(result.outputs.contains("output"));
    EXPECT_EQ(result.outputs.at("output").size(), 12U);
    engine->shutdown();
}

TEST(EngineRuntimeFactoryTest, FactoryFailureRollsBackBeforeAnySlotIsPublished)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    auto config          = makeConfig();
    config.device_ids    = {0, 1};
    config.execution_slots = 2;
    RuntimeFactoryGuard guard(2);
    auto engine = irt::engine::InferenceEngine::create(std::move(config), makePipeline());

    EXPECT_THROW(engine->start(), irt::Exception);
    EXPECT_EQ(g_factory_calls.load(), 2);
    EXPECT_EQ(engine->state(), irt::engine::EngineState::Failed);
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.queued_requests, 0U);
    EXPECT_EQ(metrics.inflight_batches, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.pinned_memory_capacity_bytes, 0U);
    EXPECT_EQ(metrics.device_arena_bytes, 0U);
    ASSERT_FALSE(engine->faults().empty());
    EXPECT_NE(engine->faults().back().message.find("factory failure"), std::string::npos);
}

TEST(TensorRTRuntimePlanShapeTest, DetectsFixedBatchFromStaticEngineMetadata)
{
    RuntimeShapeFixture fixture;
    g_runtime_shape_fixture = &fixture;
    RuntimeBackendFactoryGuard guard(&shapeContractBackendFactory);

    auto plan = irt::engine::priv::CreateEngineRuntimePlan(makeRuntimePlanConfig(1, 1, 1), 0, nullptr);
    EXPECT_EQ(plan->fixedBatchSize(), 1);

    g_runtime_shape_fixture = nullptr;
}

TEST(TensorRTRuntimePlanShapeTest, RejectsDynamicConfigForStaticEngineAtStartup)
{
    RuntimeShapeFixture fixture;
    g_runtime_shape_fixture = &fixture;
    RuntimeBackendFactoryGuard guard(&shapeContractBackendFactory);

    try
    {
        const auto plan = irt::engine::priv::CreateEngineRuntimePlan(makeRuntimePlanConfig(1, 4, 8), 0, nullptr);
        (void)plan;
        FAIL() << "A dynamic batch range must be rejected for a static engine";
    }
    catch (const irt::Exception &error)
    {
        EXPECT_EQ(error.code(), irt::Status::ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::string(error.msg()).find("static TensorRT engine"), std::string::npos);
    }

    g_runtime_shape_fixture = nullptr;
}

TEST(TensorRTRuntimePlanShapeTest, RejectsFixedConfigThatDoesNotMatchStaticEngine)
{
    RuntimeShapeFixture fixture;
    fixture.fixed_batch       = 1;
    g_runtime_shape_fixture   = &fixture;
    RuntimeBackendFactoryGuard guard(&shapeContractBackendFactory);

    try
    {
        const auto plan = irt::engine::priv::CreateEngineRuntimePlan(makeRuntimePlanConfig(2, 2, 2), 0, nullptr);
        (void)plan;
        FAIL() << "A fixed batch mismatch must be rejected at startup";
    }
    catch (const irt::Exception &error)
    {
        EXPECT_EQ(error.code(), irt::Status::ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::string(error.msg()).find("does not match static TensorRT engine"), std::string::npos);
    }

    g_runtime_shape_fixture = nullptr;
}

TEST(TensorRTRuntimePlanShapeTest, LeavesDynamicEngineWithoutFixedBatch)
{
    RuntimeShapeFixture fixture;
    fixture.dynamic_batch      = true;
    g_runtime_shape_fixture    = &fixture;
    RuntimeBackendFactoryGuard guard(&shapeContractBackendFactory);

    const auto plan = irt::engine::priv::CreateEngineRuntimePlan(makeRuntimePlanConfig(1, 4, 8), 0, nullptr);
    EXPECT_EQ(plan->fixedBatchSize(), 0);

    g_runtime_shape_fixture = nullptr;
}

} // namespace
