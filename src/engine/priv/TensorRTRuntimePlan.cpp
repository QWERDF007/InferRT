#include "TensorRTRuntimePlan.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/Pipeline.hpp>
#include <inferrt/model/BackendRuntime.hpp>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/ModelRuntime.hpp>

#include <algorithm>
#include <filesystem>
#include <limits>

#include <atomic>

namespace irt::engine::priv {
namespace {

std::atomic<EngineRuntimePlanFactory> g_factory_override{nullptr};

irt::Shape pipelineInputShape(const EngineConfig &config, const irt::TensorDesc &desc)
{
    if (desc.shape.rank() == 4)
    {
        auto shape = desc.shape;
        shape[0]   = config.max_batch_size;
        return shape;
    }
    if (desc.layout != irt::TensorLayout::NCHW || !desc.hasGeometry())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT model input descriptor must be NCHW with positive dimensions");
    }
    return irt::Shape{config.max_batch_size, desc.channels(), desc.height(), desc.width()};
}

irt::model::IModelConfig makeBackendConfig(const EngineConfig &config, const int device_id,
                                           const PipelinePlan *pipeline)
{
    irt::model::IModelConfig backend_config;
    if (pipeline != nullptr && !pipeline->modelInputs().empty())
    {
        std::vector<irt::Shape>       input_shapes;
        std::vector<std::string>      input_names;
        input_shapes.reserve(pipeline->modelInputs().size());
        input_names.reserve(pipeline->modelInputs().size());
        for (const auto &binding : pipeline->modelInputs())
        {
            const auto tensor_it = pipeline->tensors().find(binding.tensor_name);
            if (tensor_it == pipeline->tensors().end())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Pipeline model input tensor is not declared: %s", binding.tensor_name.c_str());
            }
            input_shapes.push_back(pipelineInputShape(config, tensor_it->second));
            input_names.push_back(binding.engine_tensor_name.empty() ? "input" : binding.engine_tensor_name);
        }
        backend_config.setInputTensorNames(std::move(input_names));
        backend_config.setInputShapes(std::move(input_shapes));
    }
    else
    {
        const auto &preprocess = config.preprocessSpec();
        backend_config.setInputShape(
            irt::Shape{config.max_batch_size, preprocess.input_channels, preprocess.input_height,
                       preprocess.input_width});
    }
    backend_config.setRuntime(irt::model::ModelRuntime{irt::model::ModelRuntime::Backend::TensorRT,
                                                       irt::model::ModelRuntime::Device::GPU, device_id});
    if (config.min_batch_size != config.max_batch_size)
    {
        backend_config.setDynamicBatchRange(config.min_batch_size, config.opt_batch_size, config.max_batch_size);
    }
    backend_config.setFeatureOnly(config.feature_only);
    if (!config.output_tensor_names.empty())
    {
        backend_config.setOutputTensorNames(config.output_tensor_names);
    }
    if (!config.feature_tensor_names.empty())
    {
        backend_config.setFeatureTensorNames(config.feature_tensor_names);
    }
    return backend_config;
}

void validateFloat32(const std::vector<irt::TensorInfo> &infos, const char *kind)
{
    for (const auto &info : infos)
    {
        if (info.desc.data_type != irt::TensorDataType::F32)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "Engine %s tensor %s must be float32", kind, info.name.c_str());
        }
    }
}

} // namespace

std::shared_ptr<IEngineRuntimePlan> CreateEngineRuntimePlan(const EngineConfig &config, const int device_id,
                                                            const PipelinePlan *pipeline)
{
    if (const auto factory = g_factory_override.load(std::memory_order_acquire))
    {
        auto plan = factory(config, device_id, pipeline);
        if (!plan)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Engine runtime-plan factory returned nullptr");
        }
        return plan;
    }
    return std::make_shared<TensorRTRuntimePlan>(config, device_id, pipeline);
}

void SetEngineRuntimePlanFactoryOverride(const EngineRuntimePlanFactory factory) noexcept
{
    g_factory_override.store(factory, std::memory_order_release);
}

TensorRTRuntimePlan::TensorRTRuntimePlan(const EngineConfig &config, const int device_id,
                                         const PipelinePlan *pipeline)
{
    auto backend = irt::model::CreateBackendRuntime(irt::model::ModelRuntime::Backend::TensorRT);
    if (!backend)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to create TensorRT backend runtime");
    }

    const auto backend_config = makeBackendConfig(config, device_id, pipeline);
    backend->load(config.engine_file.string(), backend_config, config.model_name);
    inputs_  = backend->inputs();
    outputs_ = backend->outputs();
    if (inputs_.empty() || outputs_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT engine must expose at least one input and one output");
    }
    validateFloat32(inputs_, "input");
    validateFloat32(outputs_, "output");

    // Keep the engine metadata as the source of truth.  A dynamic profile is
    // resolved to its configured shape during load, so ``tensorShape`` alone
    // cannot distinguish it from a genuinely static engine.
    bool has_dynamic_batch = false;
    for (const auto &info : inputs_)
    {
        const auto &name  = info.name;
        const auto  shape = backend->tensorShape(name);
        if (shape.rank() == 0 || shape[0] <= 0)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION,
                                 "TensorRT input shape is not concrete: %s", name.c_str());
        }

        if (backend->isInputBatchDynamic(name))
        {
            has_dynamic_batch = true;
            continue;
        }

        if (shape[0] > static_cast<int64_t>(std::numeric_limits<int>::max()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "TensorRT fixed batch size exceeds int range");
        }
        const int static_batch = static_cast<int>(shape[0]);
        if (fixed_batch_size_ != 0 && fixed_batch_size_ != static_batch)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "TensorRT inputs must use the same fixed batch size");
        }
        if (config.min_batch_size != config.max_batch_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured dynamic batch range [%d, %d, %d] cannot be used with static TensorRT "
                                 "engine input %s (fixed batch %d)",
                                 config.min_batch_size, config.opt_batch_size, config.max_batch_size, name.c_str(),
                                 static_batch);
        }
        if (config.min_batch_size != static_batch)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured fixed batch %d does not match static TensorRT engine input %s "
                                 "(fixed batch %d)",
                                 config.min_batch_size, name.c_str(), static_batch);
        }
        fixed_batch_size_ = static_batch;
    }
    if (has_dynamic_batch)
    {
        fixed_batch_size_ = 0;
    }
    for (const auto &name : config.output_tensor_names)
    {
        if (std::none_of(outputs_.begin(), outputs_.end(), [&name](const irt::TensorInfo &info)
                         { return info.name == name; }))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured output tensor does not exist in TensorRT engine: %s", name.c_str());
        }
    }
    backend_ = std::move(backend);
}

TensorRTRuntimePlan::~TensorRTRuntimePlan() = default;

std::unique_ptr<irt::ITensorRuntimeSession> TensorRTRuntimePlan::createSession() const
{
    if (!backend_)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "TensorRT backend is not initialized");
    }
    return backend_->createSession();
}

void TensorRTRuntimePlan::executeSession(irt::ITensorRuntimeSession &session,
                                          const std::span<const irt::BufferView> buffers,
                                          const irt::ExecuteOptions options) const
{
    auto normalized = irt::normalizeExecutionBuffers(buffers, inputs_, outputs_);
    session.execute(normalized, options);
}

std::vector<irt::TensorInfo> TensorRTRuntimePlan::inputs() const
{
    return inputs_;
}

std::vector<irt::TensorInfo> TensorRTRuntimePlan::outputs() const
{
    return outputs_;
}

int TensorRTRuntimePlan::fixedBatchSize() const noexcept
{
    return fixed_batch_size_;
}

} // namespace irt::engine::priv
