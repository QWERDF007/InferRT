#include "BackendRuntime.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <fstream>

namespace irt::model::priv {

namespace {

bool HasTensor(const nvinfer1::ICudaEngine *engine, const std::string &tensor_name)
{
    if (!engine)
    {
        return false;
    }

    const int32_t num_io_tensors = engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (name && tensor_name == name)
        {
            return true;
        }
    }

    return false;
}

void EnsureInternalStream(TRTParams &params)
{
    if (params.stream)
    {
        return;
    }

    params.stream = MakeCudaStream();
    if (!params.stream)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create CUDA stream");
    }
}

void SaveEngineToFile(const std::string &engine_file, const std::shared_ptr<nvinfer1::ICudaEngine> &engine)
{
    if (!engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized, cannot save");
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
    if (!serialized)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to serialize engine");
    }

    std::ofstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open file for writing: %s",
                             engine_file.c_str());
    }

    file.write(static_cast<const char *>(serialized->data()), serialized->size());
    file.close();
}

} // namespace

ModelBackend TensorRTBackend::backend() const noexcept
{
    return ModelBackend::TensorRT;
}

void TensorRTBackend::load(const std::string &engine_file, const IModelConfig &config, const std::string &model_name)
{
    (void)config;
    params_.context.reset();
    params_.engine.reset();

    if (params_.logger == nullptr)
    {
        initLogger(model_name);
    }

    LOG_INFO(*params_.logger) << "Loading TensorRT engine from: " << engine_file << std::endl;

    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open engine file: %s", engine_file.c_str());
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*params_.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    params_.engine
        = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), file_size),
                                                 [](nvinfer1::ICudaEngine *engine) { delete engine; });
    if (!params_.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    params_.context.reset(params_.engine->createExecutionContext());
    if (!params_.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }
}

void TensorRTBackend::save(const std::string &engine_file) const
{
    SaveEngineToFile(engine_file, params_.engine);
}

std::vector<std::string> TensorRTBackend::ioTensorNames(nvinfer1::TensorIOMode mode) const
{
    if (!params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    std::vector<std::string> names;
    const int32_t            num_io_tensors = params_.engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char *tensor_name = params_.engine->getIOTensorName(i);
        if (params_.engine->getTensorIOMode(tensor_name) == mode)
        {
            names.emplace_back(tensor_name);
        }
    }

    return names;
}

nvinfer1::Dims TensorRTBackend::tensorShape(const std::string &tensor_name) const
{
    if (params_.context && HasTensor(params_.engine.get(), tensor_name))
    {
        return params_.context->getTensorShape(tensor_name.c_str());
    }

    if (params_.engine && HasTensor(params_.engine.get(), tensor_name))
    {
        return params_.engine->getTensorShape(tensor_name.c_str());
    }

    if (!params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

nvinfer1::DataType TensorRTBackend::tensorDataType(const std::string &tensor_name) const
{
    if (params_.engine && HasTensor(params_.engine.get(), tensor_name))
    {
        return params_.engine->getTensorDataType(tensor_name.c_str());
    }

    if (!params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

void TensorRTBackend::setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims)
{
    if (!params_.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    if (!params_.context->setInputShape(tensor_name.c_str(), dims))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to set input tensor shape: %s",
                             tensor_name.c_str());
    }
}

void TensorRTBackend::infer(const std::vector<void *> &buffers)
{
    execute(buffers, nullptr, false);
}

void TensorRTBackend::setStream(cudaStream_t stream)
{
    if (!stream)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "stream must not be null; use clearStream to reset");
    }

    params_.external_stream = stream;
}

void TensorRTBackend::clearStream()
{
    params_.external_stream = nullptr;
}

cudaStream_t TensorRTBackend::resolveExecutionStream(cudaStream_t stream_override)
{
    if (stream_override)
    {
        return stream_override;
    }

    if (params_.external_stream)
    {
        return params_.external_stream;
    }

    if (params_.stream)
    {
        return *params_.stream;
    }

    if (params_.context)
    {
        EnsureInternalStream(params_);
        return *params_.stream;
    }

    return nullptr;
}

nvinfer1::ILogger::Severity TensorRTBackend::logLevel() const noexcept
{
    return params_.log_level;
}

void TensorRTBackend::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    log_level_        = severity;
    params_.log_level = severity;
    if (params_.logger)
    {
        params_.logger->setReportableSeverity(severity);
    }
}

TensorRTBackend *TensorRTBackend::asTensorRT() noexcept
{
    return this;
}

const TensorRTBackend *TensorRTBackend::asTensorRT() const noexcept
{
    return this;
}

TRTParams &TensorRTBackend::params() noexcept
{
    return params_;
}

const TRTParams &TensorRTBackend::params() const noexcept
{
    return params_;
}

void TensorRTBackend::initLogger(const std::string &model_name)
{
    params_.logger = std::make_shared<Logger>(model_name, logLevel());
}

void TensorRTBackend::buildFromNetwork(const std::string &source_file, const std::string &model_name,
                                       NetworkBuildFn build_fn)
{
    using namespace nvinfer1;
    if (params_.logger == nullptr)
    {
        initLogger(model_name);
    }

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*params_.logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    const auto flags   = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto       network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create NetworkDefinition");
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create BuilderConfig");
    }

    LOG_INFO(*params_.logger) << "Building TensorRT network from: " << source_file << std::endl;
    build_fn(network.get());

    auto buffer = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!buffer)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to build serialized network");
    }

    auto runtime = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(*params_.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    params_.engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer->data(), buffer->size()),
                                                  [](ICudaEngine *engine) { delete engine; });
    if (!params_.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    params_.context.reset(params_.engine->createExecutionContext());
    if (!params_.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }
}

void TensorRTBackend::execute(const std::vector<void *> &buffers, cudaStream_t stream_override, bool non_blocking)
{
    bindTensorAddresses(buffers);

    const auto stream = resolveExecutionStream(stream_override);
    if (!params_.context->enqueueV3(stream))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute TensorRT enqueue");
    }

    if (non_blocking)
    {
        return;
    }

    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to synchronize CUDA stream: %s",
                             cudaGetErrorString(status));
    }
}

void TensorRTBackend::setFeatureOnly(bool feature_only) noexcept
{
    params_.feature_only = feature_only;
}

void TensorRTBackend::bindTensorAddresses(const std::vector<void *> &buffers)
{
    if (!params_.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    const auto  input_names        = ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    const auto  output_names       = ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
    const auto *output_description = params_.feature_only ? "feature outputs" : "outputs";
    const auto  expected           = input_names.size() + output_names.size();
    if (buffers.size() != expected)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Expected %zu buffers (%zu inputs and %zu %s), got %zu",
                             expected, input_names.size(), output_names.size(), output_description, buffers.size());
    }

    size_t buffer_index = 0;
    for (const auto &input_name : input_names)
    {
        if (!params_.context->setTensorAddress(input_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set input tensor address: %s", input_name.c_str());
        }
    }

    for (const auto &output_name : output_names)
    {
        if (!params_.context->setTensorAddress(output_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set output tensor address: %s",
                                 output_name.c_str());
        }
    }
}

} // namespace irt::model::priv
