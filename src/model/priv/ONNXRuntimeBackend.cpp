#include <inferrt/model/BackendRuntime.hpp>
#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::model::priv {

namespace {

struct TensorInfo
{
    irt::Shape                shape;
    irt::Shape                declared_shape;
    ONNXTensorElementDataType element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    bool                      dynamic_batch{false};
};

irt::TensorDataType OrtTypeToCore(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return irt::TensorDataType::F32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return irt::TensorDataType::F16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return irt::TensorDataType::I8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return irt::TensorDataType::U8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return irt::TensorDataType::I32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return irt::TensorDataType::I64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return irt::TensorDataType::Bool;
    default:
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported ONNX Runtime tensor element type: %d",
                             static_cast<int>(type));
    }
}

TensorInfo ReadTensorInfo(const Ort::TypeInfo &type_info)
{
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto dims        = tensor_info.GetShape();
    const bool dynamic_batch = !dims.empty() && dims.front() < 0;
    irt::Shape  shape{std::move(dims)};
    return TensorInfo{shape, shape, tensor_info.GetElementType(), dynamic_batch};
}

Ort::SessionOptions MakeSessionOptions(const bool use_gpu, const int device_id)
{
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_gpu)
    {
        OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
        const auto release_cuda_options = [](OrtCUDAProviderOptionsV2 *options) {
            Ort::GetApi().ReleaseCUDAProviderOptions(options);
        };
        const std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release_cuda_options)> cuda_options_guard{
            cuda_options, release_cuda_options};

        const std::string device = std::to_string(device_id);
        const char       *keys[]   = {"device_id", "use_tf32"};
        const char       *values[] = {device.c_str(), "0"};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys, values, 2));
        session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    }
    return session_options;
}

std::unique_ptr<Ort::Session> CreateOrtSession(const std::shared_ptr<Ort::Env> &env,
                                               const std::filesystem::path &model_path, const bool use_gpu,
                                               const int device_id)
{
    if (!env)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime environment is not initialized");
    }
    auto session_options = MakeSessionOptions(use_gpu, device_id);
    return std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
}

void ValidateOrtBuffer(const irt::BufferView &view, const TensorInfo &info, const std::string &tensor_name,
                       const char *tensor_kind)
{
    if (view.data == nullptr || view.bytes_per_request == 0 || view.capacity_batch <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer is incomplete: %s", tensor_kind,
                             tensor_name.c_str());
    }

    (void)view.byteSize();
    const auto bytes = irt::checkedSizeMul(info.shape.elementCount(), irt::dataTypeSize(OrtTypeToCore(info.element_type)),
                                           "ONNX Runtime tensor bytes");
    if (view.bytes_per_request < bytes)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer is too small: %s", tensor_kind,
                             tensor_name.c_str());
    }
}

void ExecuteOrt(Ort::Session &session, const std::vector<std::string> &input_names,
                const std::vector<std::string> &output_names, const std::unordered_map<std::string, TensorInfo> &input_info,
                const std::unordered_map<std::string, TensorInfo> &output_info,
                const std::span<const irt::BufferView> buffers, const irt::ExecuteOptions options)
{
    const size_t expected = input_names.size() + output_names.size();
    if (buffers.size() != expected)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Expected %zu host buffers (%zu inputs and %zu outputs), got %zu", expected,
                             input_names.size(), output_names.size(), buffers.size());
    }

    try
    {
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value>   input_values;
        std::vector<Ort::Value>   output_values;
        std::vector<const char *> input_name_ptrs;
        std::vector<const char *> output_name_ptrs;
        input_values.reserve(input_names.size());
        output_values.reserve(output_names.size());
        input_name_ptrs.reserve(input_names.size());
        output_name_ptrs.reserve(output_names.size());

        size_t buffer_index = 0;
        for (const auto &input_name : input_names)
        {
            const auto &view = buffers[buffer_index++];
            const auto &info = input_info.at(input_name);
            ValidateOrtBuffer(view, info, input_name, "Input");
            const auto shape = info.shape.dims;
            const auto bytes = irt::checkedSizeMul(info.shape.elementCount(),
                                                   irt::dataTypeSize(OrtTypeToCore(info.element_type)),
                                                   "ONNX input tensor bytes");
            input_values.emplace_back(Ort::Value::CreateTensor(memory_info, view.data, bytes, shape.data(), shape.size(),
                                                               info.element_type));
            input_name_ptrs.push_back(input_name.c_str());
        }

        for (const auto &output_name : output_names)
        {
            const auto &view = buffers[buffer_index++];
            const auto &info = output_info.at(output_name);
            ValidateOrtBuffer(view, info, output_name, "Output");
            const auto shape = info.shape.dims;
            const auto bytes = irt::checkedSizeMul(info.shape.elementCount(),
                                                   irt::dataTypeSize(OrtTypeToCore(info.element_type)),
                                                   "ONNX output tensor bytes");
            output_values.emplace_back(Ort::Value::CreateTensor(memory_info, view.data, bytes, shape.data(), shape.size(),
                                                                info.element_type));
            output_name_ptrs.push_back(output_name.c_str());
        }

        session.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(), input_values.size(),
                    output_name_ptrs.data(), output_values.data(), output_values.size());
        (void)options;
    }
    catch (const irt::Exception &)
    {
        throw;
    }
    catch (const Ort::Exception &e)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime inference failed: %s", e.what());
    }
    catch (const std::exception &e)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime inference failed: %s", e.what());
    }
}

class ONNXRuntimeSession final : public irt::ITensorRuntimeSession
{
public:
    ONNXRuntimeSession(std::shared_ptr<Ort::Env> env, std::unique_ptr<Ort::Session> session,
                       std::vector<std::string> input_names, std::vector<std::string> output_names,
                       std::unordered_map<std::string, TensorInfo> input_info,
                       std::unordered_map<std::string, TensorInfo> output_info)
        : env_(std::move(env))
        , session_(std::move(session))
        , input_names_(std::move(input_names))
        , output_names_(std::move(output_names))
        , input_info_(std::move(input_info))
        , output_info_(std::move(output_info))
    {
        if (!env_ || !session_)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime session is not initialized");
        }
    }

    irt::Shape tensorShape(const std::string &tensor_name) const override
    {
        const auto input_it = input_info_.find(tensor_name);
        if (input_it != input_info_.end())
        {
            return input_it->second.shape;
        }
        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return output_it->second.shape;
        }
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    irt::TensorDataType tensorDataType(const std::string &tensor_name) const override
    {
        const auto input_it = input_info_.find(tensor_name);
        if (input_it != input_info_.end())
        {
            return OrtTypeToCore(input_it->second.element_type);
        }
        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OrtTypeToCore(output_it->second.element_type);
        }
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override
    {
        const auto it = input_info_.find(tensor_name);
        if (it == input_info_.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input tensor not found: %s", tensor_name.c_str());
        }
        irt::validateRuntimeShape(it->second.declared_shape, shape, tensor_name);
        input_info_[tensor_name].shape = shape;
        PropagateResolvedBatchDimToDynamicOutputs(output_info_, shape);
    }

    void execute(const std::span<const irt::BufferView> buffers, const irt::ExecuteOptions options) override
    {
        ExecuteOrt(*session_, input_names_, output_names_, input_info_, output_info_, buffers, options);
    }

private:
    std::shared_ptr<Ort::Env>                      env_;
    std::unique_ptr<Ort::Session>                 session_;
    std::vector<std::string>                      input_names_;
    std::vector<std::string>                      output_names_;
    std::unordered_map<std::string, TensorInfo>   input_info_;
    std::unordered_map<std::string, TensorInfo>   output_info_;
};

class ONNXRuntimeBackend final : public irt::model::IBackendRuntime
{
public:
    ModelRuntime::Backend backend() const noexcept override
    {
        return ModelRuntime::Backend::ONNXRuntime;
    }

    void load(const std::string &model_file, const IModelConfig &config, const std::string &model_name) override
    {
        std::filesystem::path model_path(model_file);
        if (!std::filesystem::exists(model_path))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open ONNX model file: %s",
                                 model_file.c_str());
        }

        try
        {
            auto new_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());
            auto new_session = CreateOrtSession(new_env, model_path, config.runtime().isGpu(),
                                                config.runtime().deviceId());
            std::vector<std::string>                    new_input_names;
            std::vector<std::string>                    new_output_names;
            std::unordered_map<std::string, TensorInfo> new_input_info;
            std::unordered_map<std::string, TensorInfo> new_output_info;
            refreshMetadata(*new_session, config, new_input_names, new_output_names, new_input_info, new_output_info);

            env_.swap(new_env);
            session_.swap(new_session);
            model_path_.swap(model_path);
            use_gpu_   = config.runtime().isGpu();
            device_id_ = config.runtime().deviceId();
            input_names_.swap(new_input_names);
            output_names_.swap(new_output_names);
            input_info_.swap(new_input_info);
            output_info_.swap(new_output_info);
        }
        catch (const irt::Exception &)
        {
            throw;
        }
        catch (const Ort::Exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime error while loading %s: %s", model_file.c_str(),
                                 e.what());
        }
        catch (const std::exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime error while loading %s: %s", model_file.c_str(),
                                 e.what());
        }
    }

    std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const override
    {
        ensureLoaded();
        return mode == irt::TensorIOMode::Input ? input_names_ : output_names_;
    }

    irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const noexcept override
    {
        (void)mode;
        return irt::MemoryKind::HOST;
    }

    bool isInputBatchDynamic(const std::string &tensor_name) const override
    {
        ensureLoaded();
        const auto it = input_info_.find(tensor_name);
        if (it == input_info_.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input tensor not found: %s", tensor_name.c_str());
        }
        return it->second.dynamic_batch;
    }

    irt::Shape tensorShape(const std::string &tensor_name) const override
    {
        ensureLoaded();
        const auto input_it = input_info_.find(tensor_name);
        if (input_it != input_info_.end())
        {
            return input_it->second.shape;
        }

        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return output_it->second.shape;
        }

        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    irt::TensorDataType tensorDataType(const std::string &tensor_name) const override
    {
        ensureLoaded();
        const auto input_it = input_info_.find(tensor_name);
        if (input_it != input_info_.end())
        {
            return OrtTypeToCore(input_it->second.element_type);
        }

        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OrtTypeToCore(output_it->second.element_type);
        }

        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override
    {
        ensureLoaded();
        const auto it = input_info_.find(tensor_name);
        if (it == input_info_.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input tensor not found: %s", tensor_name.c_str());
        }

        irt::validateRuntimeShape(it->second.declared_shape, shape, tensor_name);
        input_info_[tensor_name].shape = shape;
        PropagateResolvedBatchDimToDynamicOutputs(output_info_, shape);
    }

    void executeNormalized(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options) override
    {
        ensureLoaded();
        ExecuteOrt(*session_, input_names_, output_names_, input_info_, output_info_, buffers, options);
    }

    std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override
    {
        ensureLoaded();
        try
        {
            auto session = CreateOrtSession(env_, model_path_, use_gpu_, device_id_);
            return std::make_unique<ONNXRuntimeSession>(env_, std::move(session), input_names_, output_names_,
                                                        input_info_, output_info_);
        }
        catch (const irt::Exception &)
        {
            throw;
        }
        catch (const Ort::Exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime session creation failed: %s", e.what());
        }
        catch (const std::exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime session creation failed: %s", e.what());
        }
    }

private:
    void ensureLoaded() const
    {
        if (!env_ || !session_)
        {
            throw irt::Exception(Status::INVALID_OPERATION, "ONNX Runtime session is not initialized");
        }
    }

    static void refreshMetadata(Ort::Session &session, const IModelConfig &config,
                                std::vector<std::string> &input_names, std::vector<std::string> &output_names,
                                std::unordered_map<std::string, TensorInfo> &input_info,
                                std::unordered_map<std::string, TensorInfo> &output_info)
    {
        input_names.clear();
        output_names.clear();
        input_info.clear();
        output_info.clear();

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t                     input_count = session.GetInputCount();
        for (size_t i = 0; i < input_count; ++i)
        {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names.emplace_back(name.get());
            input_info[input_names.back()] = ReadTensorInfo(session.GetInputTypeInfo(i));
        }

        std::vector<std::string> graph_output_names;
        const size_t             output_count = session.GetOutputCount();
        graph_output_names.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i)
        {
            auto name = session.GetOutputNameAllocated(i, allocator);
            graph_output_names.emplace_back(name.get());
            output_info[graph_output_names.back()] = ReadTensorInfo(session.GetOutputTypeInfo(i));
        }

        const bool use_graph_outputs = IsDefaultOutputConfig(config);
        output_names                 = use_graph_outputs ? graph_output_names : config.outputTensorNames();
        for (const auto &output_name : output_names)
        {
            if (output_info.find(output_name) == output_info.end())
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Requested output tensor is not an ONNX graph output: %s", output_name.c_str());
            }
        }

        const auto &configured_inputs = config.inputTensorNames();
        if (configured_inputs != std::vector<std::string>{"input"})
        {
            for (const auto &input_name : configured_inputs)
            {
                if (input_info.find(input_name) == input_info.end())
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                         "Configured input tensor is not an ONNX graph input: %s", input_name.c_str());
                }
            }
            input_names = configured_inputs;
        }
    }

    std::shared_ptr<Ort::Env>      env_;
    std::unique_ptr<Ort::Session>  session_;
    std::filesystem::path          model_path_;
    bool                            use_gpu_{false};
    int                             device_id_{0};

    std::vector<std::string>                    input_names_;
    std::vector<std::string>                    output_names_;
    std::unordered_map<std::string, TensorInfo> input_info_;
    std::unordered_map<std::string, TensorInfo> output_info_;
};

} // namespace

std::unique_ptr<irt::model::IBackendRuntime> CreateONNXRuntimeBackend()
{
    return std::make_unique<ONNXRuntimeBackend>();
}

} // namespace irt::model::priv
