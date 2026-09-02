#include "TensorRTBackend.hpp"
#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include "TRTUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <utility>

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
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized, cannot save");
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

bool HasDynamicDimension(const nvinfer1::Dims &dims)
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return true;
        }
    }
    return false;
}

size_t ResolveConfiguredInputIndex(const IModelConfig &config, const std::string &tensor_name, size_t fallback_index)
{
    const auto &input_names = config.inputTensorNames();
    const auto  it          = std::find(input_names.begin(), input_names.end(), tensor_name);
    if (it != input_names.end())
    {
        return static_cast<size_t>(std::distance(input_names.begin(), it));
    }

    if (fallback_index < config.inputShapes().size())
    {
        return fallback_index;
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Cannot resolve configured shape for input tensor: %s",
                         tensor_name.c_str());
}

nvinfer1::Dims MakeProfileDims(nvinfer1::Dims dims, int batch)
{
    if (dims.nbDims <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Dynamic batch requires an explicit batch dimension");
    }
    dims.d[0] = batch;
    return dims;
}

irt::Shape DimsToCoreShape(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Invalid TensorRT rank for tensor: %s", tensor_name.c_str());
    }

    std::vector<int64_t> values;
    values.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        values.push_back(static_cast<int64_t>(dims.d[index]));
    }
    return irt::Shape{std::move(values)};
}

irt::TensorDataType ToCoreDataType(const nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kUINT8:
        return irt::TensorDataType::U8;
    case nvinfer1::DataType::kINT8:
        return irt::TensorDataType::I8;
    case nvinfer1::DataType::kHALF:
        return irt::TensorDataType::F16;
    case nvinfer1::DataType::kFLOAT:
        return irt::TensorDataType::F32;
    case nvinfer1::DataType::kINT32:
        return irt::TensorDataType::I32;
    case nvinfer1::DataType::kINT64:
        return irt::TensorDataType::I64;
    case nvinfer1::DataType::kBOOL:
        return irt::TensorDataType::Bool;
    default:
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported TensorRT tensor data type");
    }
}

std::shared_ptr<TRTLogger> ResolveLogger(const TRTParams &params, const std::string &model_name)
{
    if (params.logger)
    {
        return params.logger;
    }
    return std::make_shared<TRTLogger>(model_name, params.log_level);
}

nvinfer1::TensorIOMode ToTensorRtIOMode(const irt::TensorIOMode mode)
{
    return mode == irt::TensorIOMode::Input ? nvinfer1::TensorIOMode::kINPUT
                                             : nvinfer1::TensorIOMode::kOUTPUT;
}

nvinfer1::ILogger::Severity ToTensorRtSeverity(const LogLevel level)
{
    switch (level)
    {
    case LogLevel::InternalError:
        return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    case LogLevel::Error:
        return nvinfer1::ILogger::Severity::kERROR;
    case LogLevel::Warning:
        return nvinfer1::ILogger::Severity::kWARNING;
    case LogLevel::Info:
        return nvinfer1::ILogger::Severity::kINFO;
    case LogLevel::Verbose:
        return nvinfer1::ILogger::Severity::kVERBOSE;
    }
    return nvinfer1::ILogger::Severity::kWARNING;
}

LogLevel FromTensorRtSeverity(const nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return LogLevel::InternalError;
    case nvinfer1::ILogger::Severity::kERROR:
        return LogLevel::Error;
    case nvinfer1::ILogger::Severity::kWARNING:
        return LogLevel::Warning;
    case nvinfer1::ILogger::Severity::kINFO:
        return LogLevel::Info;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return LogLevel::Verbose;
    }
    return LogLevel::Warning;
}

nvinfer1::Dims MakeProfileDims(const irt::Shape &shape, int batch)
{
    auto dims = ShapeToDims(shape);
    return MakeProfileDims(dims, batch);
}

void ConfigureDynamicBatchProfile(nvinfer1::IBuilder &builder, nvinfer1::IBuilderConfig &builder_config,
                                  nvinfer1::INetworkDefinition &network, const IModelConfig &config)
{
    if (!config.dynamicBatch())
    {
        return;
    }

    auto *profile = builder.createOptimizationProfile();
    if (!profile)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create TensorRT optimization profile");
    }

    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto *input = network.getInput(i);
        if (!input)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "TensorRT network input at index %d is null", i);
        }

        const auto  input_name  = std::string(input->getName());
        const auto  shape_index = ResolveConfiguredInputIndex(config, input_name, static_cast<size_t>(i));
        const auto &base_shape  = config.inputShapes().at(shape_index);

        const auto min_dims = MakeProfileDims(ShapeToDims(base_shape), config.minBatchSize());
        const auto opt_dims = MakeProfileDims(ShapeToDims(base_shape), config.optBatchSize());
        const auto max_dims = MakeProfileDims(ShapeToDims(base_shape), config.maxBatchSize());

        if (!profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims)
            || !profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims)
            || !profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Failed to set dynamic batch profile for input tensor: %s", input_name.c_str());
        }
    }

    if (!profile->isValid())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT dynamic batch profile is invalid, min=%d opt=%d max=%d", config.minBatchSize(),
                             config.optBatchSize(), config.maxBatchSize());
    }

    if (builder_config.addOptimizationProfile(profile) < 0)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add TensorRT optimization profile");
    }
}

void SetConfiguredInputShapes(nvinfer1::ICudaEngine &engine, nvinfer1::IExecutionContext &context,
                              const IModelConfig &config)
{
    size_t input_index = 0;
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
    {
        const char *tensor_name = engine.getIOTensorName(i);
        if (!tensor_name || engine.getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT)
        {
            continue;
        }

        const auto engine_dims = engine.getTensorShape(tensor_name);
        if (!HasDynamicDimension(engine_dims))
        {
            ++input_index;
            continue;
        }

        const auto shape_index = ResolveConfiguredInputIndex(config, tensor_name, input_index);
        auto       dims        = ShapeToDims(config.inputShapes().at(shape_index));
        if (config.dynamicBatch())
        {
            dims.d[0] = config.optBatchSize();
        }

        if (!context.setInputShape(tensor_name, dims))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Failed to set default dynamic input shape for tensor: %s", tensor_name);
        }
        ++input_index;
    }
}

bool IsOpaqueByteView(const irt::BufferView &buffer)
{
    if (buffer.desc.layout != irt::TensorLayout::Opaque || buffer.desc.data_type != irt::TensorDataType::U8
        || buffer.desc.shape.rank() != 1 || buffer.capacity_batch != 1 || buffer.bytes_per_request == 0
        || buffer.bytes_per_request > static_cast<size_t>((std::numeric_limits<int64_t>::max)()))
    {
        return false;
    }

    return buffer.desc.shape[0] == static_cast<int64_t>(buffer.bytes_per_request);
}

bool MatchesRuntimeShape(const irt::Shape &buffer_shape, const irt::Shape &runtime_shape)
{
    if (buffer_shape == runtime_shape)
    {
        return true;
    }

    // EngineSlot keeps opaque model outputs flattened per request.  Preserve
    // that representation while still checking the runtime element count.
    if (runtime_shape.rank() > 1 && buffer_shape.rank() == 1 && runtime_shape[0] > 0)
    {
        const auto runtime_batch = irt::checkedInt64ToSize(runtime_shape[0], "TensorRT runtime batch");
        const auto runtime_elements = runtime_shape.elementCount();
        if (runtime_elements % runtime_batch == 0
            && buffer_shape[0] == irt::checkedSizeToInt64(runtime_elements / runtime_batch,
                                                          "TensorRT per-request element count"))
        {
            return true;
        }
    }

    if (runtime_shape.rank() <= 1 || buffer_shape.rank() + 1 != runtime_shape.rank())
    {
        return false;
    }

    for (size_t index = 0; index < buffer_shape.rank(); ++index)
    {
        if (buffer_shape[index] != runtime_shape[index + 1])
        {
            return false;
        }
    }
    return true;
}

void ValidateTensorBuffer(const irt::BufferView &buffer, const std::string &tensor_name,
                          const nvinfer1::IExecutionContext &context, const nvinfer1::ICudaEngine &engine,
                          const char *tensor_kind)
{
    if (buffer.data == nullptr || buffer.bytes_per_request == 0 || buffer.capacity_batch <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer is incomplete: %s", tensor_kind,
                             tensor_name.c_str());
    }
    if (!buffer.tensor_name.empty() && buffer.tensor_name != tensor_name)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer name mismatch: expected %s, got %s",
                             tensor_kind, tensor_name.c_str(), buffer.tensor_name.c_str());
    }
    if (buffer.desc.memory_kind != irt::MemoryKind::DEVICE)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer must use device memory: %s", tensor_kind,
                             tensor_name.c_str());
    }

    const auto expected_type = ToCoreDataType(engine.getTensorDataType(tensor_name.c_str()));
    const auto runtime_dims = context.getTensorShape(tensor_name.c_str());
    const auto runtime_shape = DimsToCoreShape(runtime_dims, tensor_name);
    const auto runtime_elements = TensorElementCount(runtime_dims, tensor_name);
    const auto available_bytes = buffer.byteSize();
    const auto required_bytes =
        irt::checkedSizeMul(runtime_elements, irt::dataTypeSize(expected_type), "TensorRT runtime tensor bytes");

    if (!IsOpaqueByteView(buffer))
    {
        if (buffer.desc.data_type != expected_type)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "%s buffer data type mismatch: %s, expected %s", tensor_kind, tensor_name.c_str(),
                                 irt::dataTypeToString(expected_type).data());
        }
        if (!MatchesRuntimeShape(buffer.desc.shape, runtime_shape))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer shape mismatch: %s", tensor_kind,
                                 tensor_name.c_str());
        }
    }

    if (available_bytes < required_bytes)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "%s buffer is too small: %s, required=%zu, available=%zu", tensor_kind,
                             tensor_name.c_str(), required_bytes, available_bytes);
    }
}

void bindTensorAddresses(nvinfer1::IExecutionContext &context, const nvinfer1::ICudaEngine &engine,
                         std::span<const irt::BufferView> buffers, const bool feature_only)
{
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (int32_t index = 0; index < engine.getNbIOTensors(); ++index)
    {
        const char *name = engine.getIOTensorName(index);
        if (name == nullptr)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "TensorRT engine has an unnamed I/O tensor");
        }
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_names.emplace_back(name);
        }
        else if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output_names.emplace_back(name);
        }
    }

    const size_t expected = input_names.size() + output_names.size();
    if (buffers.size() != expected)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Expected %zu buffers (%zu inputs and %zu %s), got %zu", expected, input_names.size(),
                             output_names.size(), feature_only ? "feature outputs" : "outputs", buffers.size());
    }

    size_t buffer_index = 0;
    for (const auto &name : input_names)
    {
        const auto &buffer = buffers[buffer_index++];
        ValidateTensorBuffer(buffer, name, context, engine, "Input");
        if (!context.setTensorAddress(name.c_str(), buffer.data))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set input tensor address: %s", name.c_str());
        }
    }
    for (const auto &name : output_names)
    {
        const auto &buffer = buffers[buffer_index++];
        ValidateTensorBuffer(buffer, name, context, engine, "Output");
        if (!context.setTensorAddress(name.c_str(), buffer.data))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set output tensor address: %s", name.c_str());
        }
    }
}

class TensorRTSession final : public irt::ITensorRuntimeSession
{
public:
    TensorRTSession(std::shared_ptr<nvinfer1::ICudaEngine> engine, std::unique_ptr<nvinfer1::IExecutionContext> context,
                    const int device_id)
        : engine_(std::move(engine)), context_(std::move(context)), device_id_(device_id)
    {
        if (!engine_ || !context_)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "TensorRT session requires an engine and execution context");
        }
    }

    irt::Shape tensorShape(const std::string &tensor_name) const override
    {
        if (!HasTensor(engine_.get(), tensor_name))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
        }
        return DimsToCoreShape(context_->getTensorShape(tensor_name.c_str()), tensor_name);
    }

    irt::TensorDataType tensorDataType(const std::string &tensor_name) const override
    {
        if (!HasTensor(engine_.get(), tensor_name))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
        }
        return ToCoreDataType(engine_->getTensorDataType(tensor_name.c_str()));
    }

    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override
    {
        if (!HasTensor(engine_.get(), tensor_name))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
        }
        if (engine_->getTensorIOMode(tensor_name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor is not an input: %s", tensor_name.c_str());
        }
        const auto declared_shape = DimsToCoreShape(engine_->getTensorShape(tensor_name.c_str()), tensor_name);
        irt::validateRuntimeShape(declared_shape, shape, tensor_name);
        const auto dims = ShapeToDims(shape);
        if (!context_->setInputShape(tensor_name.c_str(), dims))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to set input tensor shape: %s",
                                 tensor_name.c_str());
        }
    }

    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options) override
    {
        setCudaDevice(device_id_);
        bindTensorAddresses(*context_, *engine_, buffers, false);
        const auto stream = reinterpret_cast<cudaStream_t>(options.stream);
        if (!context_->enqueueV3(stream))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute TensorRT enqueue");
        }
        if (!options.non_blocking)
        {
            const auto status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess)
            {
                throw irt::Exception(Status::ERROR_INTERNAL, "Failed to synchronize CUDA stream: %s",
                                     cudaGetErrorString(status));
            }
        }
    }

private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    int device_id_{0};
};

} // namespace

ModelRuntime::Backend TensorRTBackend::backend() const noexcept
{
    return ModelRuntime::Backend::TensorRT;
}

void TensorRTBackend::load(const std::string &engine_file, const IModelConfig &config, const std::string &model_name)
{
    const int requested_device_id = config.runtime().deviceId();
    setCudaDevice(requested_device_id);

    // Keep the active engine, context, stream, and logger untouched until the
    // complete replacement has been loaded and validated.
    auto new_logger = ResolveLogger(params_, model_name);
    LOG_INFO(*new_logger) << "Loading TensorRT engine from: " << engine_file << std::endl;

    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open engine file: %s", engine_file.c_str());
    }

    file.seekg(0, std::ios::end);
    const auto file_size = file.tellg();
    if (file_size <= 0
        || static_cast<std::uintmax_t>(file_size) > static_cast<std::uintmax_t>((std::numeric_limits<size_t>::max)())
        || static_cast<std::uintmax_t>(file_size)
               > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "TensorRT engine file is empty or too large: %s",
                             engine_file.c_str());
    }
    file.seekg(0, std::ios::beg);
    if (!file)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to seek engine file: %s", engine_file.c_str());
    }

    const auto byte_count = static_cast<size_t>(file_size);
    std::vector<char> engine_data(byte_count);
    file.read(engine_data.data(), static_cast<std::streamsize>(byte_count));
    if (file.gcount() != static_cast<std::streamsize>(byte_count))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to read TensorRT engine file: %s",
                             engine_file.c_str());
    }
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*new_logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    auto new_engine
        = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), byte_count),
                                                 [](nvinfer1::ICudaEngine *engine) { delete engine; });
    if (!new_engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    auto new_context = std::unique_ptr<nvinfer1::IExecutionContext>(new_engine->createExecutionContext());
    if (!new_context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    SetConfiguredInputShapes(*new_engine, *new_context, config);

    params_.context = std::move(new_context);
    params_.engine  = std::move(new_engine);
    params_.logger  = std::move(new_logger);
    device_id_      = requested_device_id;
}

void TensorRTBackend::save(const std::string &engine_file) const
{
    SaveEngineToFile(engine_file, params_.engine);
}

std::vector<std::string> TensorRTBackend::ioTensorNames(const irt::TensorIOMode mode) const
{
    if (!params_.engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized");
    }

    std::vector<std::string> names;
    const int32_t            num_io_tensors = params_.engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char *tensor_name = params_.engine->getIOTensorName(i);
        if (params_.engine->getTensorIOMode(tensor_name) == ToTensorRtIOMode(mode))
        {
            names.emplace_back(tensor_name);
        }
    }

    return names;
}

irt::MemoryKind TensorRTBackend::ioMemoryKind(const irt::TensorIOMode mode) const noexcept
{
    (void)mode;
    return irt::MemoryKind::DEVICE;
}

irt::Shape TensorRTBackend::tensorShape(const std::string &tensor_name) const
{
    if (params_.context && HasTensor(params_.engine.get(), tensor_name))
    {
        return DimsToCoreShape(params_.context->getTensorShape(tensor_name.c_str()), tensor_name);
    }

    if (params_.engine && HasTensor(params_.engine.get(), tensor_name))
    {
        return DimsToCoreShape(params_.engine->getTensorShape(tensor_name.c_str()), tensor_name);
    }

    if (!params_.engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

bool TensorRTBackend::isInputBatchDynamic(const std::string &tensor_name) const
{
    if (!params_.engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized");
    }
    if (!HasTensor(params_.engine.get(), tensor_name))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }
    if (params_.engine->getTensorIOMode(tensor_name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor is not an input: %s", tensor_name.c_str());
    }

    const auto dims = params_.engine->getTensorShape(tensor_name.c_str());
    return dims.nbDims > 0 && dims.d[0] < 0;
}

irt::TensorDataType TensorRTBackend::tensorDataType(const std::string &tensor_name) const
{
    if (params_.engine && HasTensor(params_.engine.get(), tensor_name))
    {
        return ToCoreDataType(params_.engine->getTensorDataType(tensor_name.c_str()));
    }

    if (!params_.engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

void TensorRTBackend::setTensorShape(const std::string &tensor_name, const irt::Shape &shape)
{
    if (!params_.context)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Execution context is not initialized");
    }

    if (!params_.engine || !HasTensor(params_.engine.get(), tensor_name)
        || params_.engine->getTensorIOMode(tensor_name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor is not an input: %s", tensor_name.c_str());
    }
    const auto declared_shape = DimsToCoreShape(params_.engine->getTensorShape(tensor_name.c_str()), tensor_name);
    irt::validateRuntimeShape(declared_shape, shape, tensor_name);
    const auto dims = ShapeToDims(shape);
    if (!params_.context->setInputShape(tensor_name.c_str(), dims))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to set input tensor shape: %s",
                             tensor_name.c_str());
    }
}

void TensorRTBackend::execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options)
{
    setCudaDevice(device_id_);
    if (!params_.context)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Execution context is not initialized");
    }
    bindTensorAddresses(*params_.context, *params_.engine, buffers, params_.feature_only);

    const auto stream = resolveExecutionStream(options.stream);
    options.stream    = stream;
    if (!params_.context->enqueueV3(reinterpret_cast<cudaStream_t>(stream)))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute TensorRT enqueue");
    }

    if (!options.non_blocking)
    {
        const auto status = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        if (status != cudaSuccess)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to synchronize CUDA stream: %s",
                                 cudaGetErrorString(status));
        }
    }
}

std::unique_ptr<irt::ITensorRuntimeSession> TensorRTBackend::createSession() const
{
    if (!params_.engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Engine is not initialized");
    }
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(params_.engine->createExecutionContext());
    if (!context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create TensorRT execution context");
    }
    return std::make_unique<TensorRTSession>(params_.engine, std::move(context), device_id_);
}

void TensorRTBackend::setStream(const std::uintptr_t stream)
{
    if (stream == 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "stream must not be null; use clearStream to reset");
    }

    params_.external_stream = reinterpret_cast<cudaStream_t>(stream);
}

void TensorRTBackend::clearStream()
{
    params_.external_stream = nullptr;
}

std::uintptr_t TensorRTBackend::resolveExecutionStream(const std::uintptr_t stream_override)
{
    setCudaDevice(device_id_);

    if (stream_override != 0)
    {
        return stream_override;
    }

    if (params_.external_stream)
    {
        return reinterpret_cast<std::uintptr_t>(params_.external_stream);
    }

    if (params_.stream)
    {
        return reinterpret_cast<std::uintptr_t>(*params_.stream);
    }

    if (params_.context)
    {
        EnsureInternalStream(params_);
        return reinterpret_cast<std::uintptr_t>(*params_.stream);
    }

    return 0;
}

LogLevel TensorRTBackend::logLevel() const noexcept
{
    return FromTensorRtSeverity(params_.log_level);
}

void TensorRTBackend::setLogLevel(const LogLevel level)
{
    log_level_        = level;
    params_.log_level = ToTensorRtSeverity(level);
    if (params_.logger)
    {
        params_.logger->setReportableSeverity(params_.log_level);
    }
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
    params_.logger = std::make_shared<TRTLogger>(model_name, ToTensorRtSeverity(logLevel()));
}

void TensorRTBackend::buildFromNetwork(const std::string &source_file, const std::string &model_name,
                                       const IModelConfig &model_config, NetworkBuildFn build_fn)
{
    using namespace nvinfer1;
    const int requested_device_id = model_config.runtime().deviceId();
    setCudaDevice(requested_device_id);
    auto new_logger = ResolveLogger(params_, model_name);

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*new_logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    // FP32 使用 strong typing，确保现有网络的 float32 类型推导保持不变。
    // TensorRT 10.x 已将 kFP16 标记为 deprecated，但对于非 strong-typed
    // 网络它仍是启用 FP16 layer tactic 的兼容入口；模型 I/O 仍保持 float32。
    const bool use_fp16 = model_config.precision() == ModelPrecision::FP16;
    const auto flags    = use_fp16 ? 0U : 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto       network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create NetworkDefinition");
    }

    auto builder_config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!builder_config)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create BuilderConfig");
    }

    if (use_fp16)
    {
        builder_config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        // Keep the FP32 reference path IEEE-accurate. TensorRT enables TF32
        // tactics by default, which changes long attention/matmul chains and
        // can move model parity outputs beyond the intended FP32 tolerance.
        builder_config->clearFlag(BuilderFlag::kTF32);
    }

    LOG_INFO(*new_logger) << "Building TensorRT network from: " << source_file << std::endl;
    build_fn(network.get());

    ConfigureDynamicBatchProfile(*builder, *builder_config, *network, model_config);

    auto buffer = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *builder_config));
    if (!buffer)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to build serialized network");
    }

    auto runtime = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(*new_logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    auto new_engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer->data(), buffer->size()),
                                                   [](ICudaEngine *engine) { delete engine; });
    if (!new_engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    auto new_context = std::unique_ptr<IExecutionContext>(new_engine->createExecutionContext());
    if (!new_context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    SetConfiguredInputShapes(*new_engine, *new_context, model_config);

    params_.context = std::move(new_context);
    params_.engine  = std::move(new_engine);
    params_.logger  = std::move(new_logger);
    device_id_      = requested_device_id;
}

void TensorRTBackend::setFeatureOnly(bool feature_only) noexcept
{
    params_.feature_only = feature_only;
}

} // namespace irt::model::priv
