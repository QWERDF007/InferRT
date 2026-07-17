#include "BackendRuntime.hpp"
#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace irt::model::priv {

namespace {

struct TensorInfo
{
    nvinfer1::Dims            shape{};
    ONNXTensorElementDataType element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    bool                      dynamic_batch{false};
};

nvinfer1::DataType OrtTypeToTrt(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return nvinfer1::DataType::kFLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return nvinfer1::DataType::kHALF;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return nvinfer1::DataType::kINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return nvinfer1::DataType::kUINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return nvinfer1::DataType::kINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return nvinfer1::DataType::kINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return nvinfer1::DataType::kBOOL;
    default:
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported ONNX Runtime tensor element type: %d",
                             static_cast<int>(type));
    }
}

TensorInfo ReadTensorInfo(const Ort::TypeInfo &type_info)
{
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto dims        = Int64ShapeToDims(tensor_info.GetShape());
    return TensorInfo{dims, tensor_info.GetElementType(), dims.nbDims > 0 && dims.d[0] < 0};
}

class ONNXRuntimeBackend final : public IBackendRuntime
{
public:
    ModelRuntime::Backend backend() const noexcept override
    {
        return ModelRuntime::Backend::ONNXRuntime;
    }

    void load(const std::string &model_file, const IModelConfig &config, const std::string &model_name) override
    {
        const std::filesystem::path model_path(model_file);
        if (!std::filesystem::exists(model_path))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open ONNX model file: %s",
                                 model_file.c_str());
        }

        try
        {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());

            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            if (config.runtime().isGpu())
            {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = config.runtime().deviceId();
                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }

            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
            refreshMetadata(config);
        }
        catch (const Ort::Exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime error while loading %s: %s", model_file.c_str(),
                                 e.what());
        }
    }

    std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const override
    {
        ensureLoaded();
        return mode == nvinfer1::TensorIOMode::kINPUT ? input_names_ : output_names_;
    }

    nvinfer1::Dims tensorShape(const std::string &tensor_name) const override
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

    nvinfer1::DataType tensorDataType(const std::string &tensor_name) const override
    {
        ensureLoaded();
        const auto input_it = input_info_.find(tensor_name);
        if (input_it != input_info_.end())
        {
            return OrtTypeToTrt(input_it->second.element_type);
        }

        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OrtTypeToTrt(output_it->second.element_type);
        }

        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) override
    {
        ensureLoaded();
        const auto it = input_info_.find(tensor_name);
        if (it == input_info_.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input tensor not found: %s", tensor_name.c_str());
        }

        input_info_[tensor_name].shape = dims;
        PropagateResolvedBatchDimToDynamicOutputs(output_info_, dims);
    }

    void infer(const std::vector<void *> &buffers) override
    {
        ensureLoaded();

        const size_t expected = input_names_.size() + output_names_.size();
        if (buffers.size() != expected)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Expected %zu host buffers (%zu inputs and %zu outputs), got %zu", expected,
                                 input_names_.size(), output_names_.size(), buffers.size());
        }

        try
        {
            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<Ort::Value>   input_values;
            std::vector<Ort::Value>   output_values;
            std::vector<const char *> input_name_ptrs;
            std::vector<const char *> output_name_ptrs;
            input_values.reserve(input_names_.size());
            output_values.reserve(output_names_.size());
            input_name_ptrs.reserve(input_names_.size());
            output_name_ptrs.reserve(output_names_.size());

            size_t buffer_index = 0;
            for (const auto &input_name : input_names_)
            {
                void *buffer = buffers[buffer_index++];
                if (!buffer)
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input buffer is null: %s",
                                         input_name.c_str());
                }

                const auto &info  = input_info_.at(input_name);
                const auto  shape = DimsToInt64Shape(info.shape, input_name);
                const auto  bytes
                    = TensorElementCount(info.shape, input_name) * elementSize(OrtTypeToTrt(info.element_type));
                input_values.emplace_back(Ort::Value::CreateTensor(memory_info, buffer, bytes, shape.data(),
                                                                   shape.size(), info.element_type));
                input_name_ptrs.push_back(input_name.c_str());
            }

            for (const auto &output_name : output_names_)
            {
                void *buffer = buffers[buffer_index++];
                if (!buffer)
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Output buffer is null: %s",
                                         output_name.c_str());
                }

                const auto &info  = output_info_.at(output_name);
                const auto  shape = DimsToInt64Shape(info.shape, output_name);
                const auto  bytes
                    = TensorElementCount(info.shape, output_name) * elementSize(OrtTypeToTrt(info.element_type));
                output_values.emplace_back(Ort::Value::CreateTensor(memory_info, buffer, bytes, shape.data(),
                                                                    shape.size(), info.element_type));
                output_name_ptrs.push_back(output_name.c_str());
            }

            session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(), input_values.size(),
                          output_name_ptrs.data(), output_values.data(), output_values.size());
        }
        catch (const Ort::Exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "ONNX Runtime inference failed: %s", e.what());
        }
    }

private:
    void ensureLoaded() const
    {
        if (!session_)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "ONNX Runtime session is not initialized");
        }
    }

    void refreshMetadata(const IModelConfig &config)
    {
        input_names_.clear();
        output_names_.clear();
        input_info_.clear();
        output_info_.clear();

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t                     input_count = session_->GetInputCount();
        for (size_t i = 0; i < input_count; ++i)
        {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name.get());
            input_info_[input_names_.back()] = ReadTensorInfo(session_->GetInputTypeInfo(i));
        }

        std::vector<std::string> graph_output_names;
        const size_t             output_count = session_->GetOutputCount();
        graph_output_names.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i)
        {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            graph_output_names.emplace_back(name.get());
            output_info_[graph_output_names.back()] = ReadTensorInfo(session_->GetOutputTypeInfo(i));
        }

        const bool use_graph_outputs = IsDefaultOutputConfig(config);
        output_names_                = use_graph_outputs ? graph_output_names : config.outputTensorNames();
        for (const auto &output_name : output_names_)
        {
            if (output_info_.find(output_name) == output_info_.end())
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
                if (input_info_.find(input_name) == input_info_.end())
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                         "Configured input tensor is not an ONNX graph input: %s", input_name.c_str());
                }
            }
            input_names_ = configured_inputs;
        }
    }

    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string>                    input_names_;
    std::vector<std::string>                    output_names_;
    std::unordered_map<std::string, TensorInfo> input_info_;
    std::unordered_map<std::string, TensorInfo> output_info_;
};

} // namespace

std::unique_ptr<IBackendRuntime> CreateONNXRuntimeBackend()
{
    return std::make_unique<ONNXRuntimeBackend>();
}

} // namespace irt::model::priv
