#include <inferrt/model/BackendRuntime.hpp>
#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include <openvino/openvino.hpp>

#include <filesystem>
#include <limits>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::model::priv {

namespace {

struct TensorInfo
{
    irt::Shape        shape;
    irt::Shape        declared_shape;
    ov::element::Type element_type{};
    bool              dynamic_batch{false};
};

std::string DeviceName(const ModelRuntime &runtime)
{
    if (runtime.isCpu())
    {
        return "CPU";
    }
    return runtime.deviceId() == 0 ? "GPU" : "GPU." + std::to_string(runtime.deviceId());
}

irt::TensorDataType OvTypeToCore(const ov::element::Type &type)
{
    if (type == ov::element::f32)
    {
        return irt::TensorDataType::F32;
    }
    if (type == ov::element::f16)
    {
        return irt::TensorDataType::F16;
    }
    if (type == ov::element::i8)
    {
        return irt::TensorDataType::I8;
    }
    if (type == ov::element::u8)
    {
        return irt::TensorDataType::U8;
    }
    if (type == ov::element::i32)
    {
        return irt::TensorDataType::I32;
    }
    if (type == ov::element::i64)
    {
        return irt::TensorDataType::I64;
    }
    if (type == ov::element::boolean)
    {
        return irt::TensorDataType::Bool;
    }

    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported OpenVINO element type: %s",
                         type.get_type_name().c_str());
}

irt::Shape PartialShapeToCoreShape(const ov::PartialShape &shape)
{
    if (shape.rank().is_dynamic())
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Dynamic-rank OpenVINO tensors are not supported");
    }
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i].is_dynamic())
        {
            dims.push_back(-1);
            continue;
        }
        const auto value = shape[i].get_length();
        if (value > std::numeric_limits<int64_t>::max())
        {
            throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Tensor dimension exceeds int64: %lld",
                                 static_cast<long long>(value));
        }
        dims.push_back(static_cast<int64_t>(value));
    }
    return irt::Shape{std::move(dims)};
}

ov::Shape CoreShapeToSizeTShape(const irt::Shape &shape, const std::string &tensor_name)
{
    if (shape.empty())
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Tensor shape is empty: %s", tensor_name.c_str());
    }
    ov::Shape result;
    result.reserve(shape.rank());
    for (size_t index = 0; index < shape.rank(); ++index)
    {
        const auto dimension = shape[index];
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max())
        {
            throw irt::Exception(Status::INVALID_OPERATION,
                                 "Tensor shape is not fully resolved: %s dim[%zu]=%lld", tensor_name.c_str(), index,
                                 static_cast<long long>(dimension));
        }
        result.push_back(static_cast<size_t>(dimension));
    }
    return result;
}

std::string PortName(const ov::Output<const ov::Node> &port)
{
    const auto &tensor = port.get_tensor();
    if (!tensor.get_names().empty())
    {
        return tensor.get_any_name();
    }
    return port.get_node()->get_friendly_name();
}

void ValidateOpenVINOBuffer(const irt::BufferView &view, const TensorInfo &info, const std::string &tensor_name,
                            const char *tensor_kind)
{
    if (view.data == nullptr || view.bytes_per_request == 0 || view.capacity_batch <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer is incomplete: %s", tensor_kind,
                             tensor_name.c_str());
    }

    (void)view.byteSize();
    const auto bytes = irt::checkedSizeMul(info.shape.elementCount(), irt::dataTypeSize(OvTypeToCore(info.element_type)),
                                           "OpenVINO tensor bytes");
    if (view.bytes_per_request < bytes)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s buffer is too small: %s", tensor_kind,
                             tensor_name.c_str());
    }
}

void ExecuteOpenVINO(ov::InferRequest &infer_request, const std::vector<std::string> &input_names,
                     const std::vector<std::string> &output_names,
                     const std::unordered_map<std::string, TensorInfo> &input_info,
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
        std::vector<ov::Tensor> tensors;
        tensors.reserve(expected);

        size_t buffer_index = 0;
        for (const auto &input_name : input_names)
        {
            const auto &view = buffers[buffer_index++];
            const auto &info = input_info.at(input_name);
            ValidateOpenVINOBuffer(view, info, input_name, "Input");
            const auto shape = CoreShapeToSizeTShape(info.shape, input_name);
            tensors.emplace_back(info.element_type, shape, view.data);
            infer_request.set_tensor(input_name, tensors.back());
        }

        for (const auto &output_name : output_names)
        {
            const auto &view = buffers[buffer_index++];
            const auto &info = output_info.at(output_name);
            ValidateOpenVINOBuffer(view, info, output_name, "Output");
            const auto shape = CoreShapeToSizeTShape(info.shape, output_name);
            tensors.emplace_back(info.element_type, shape, view.data);
            infer_request.set_tensor(output_name, tensors.back());
        }

        infer_request.infer();
        (void)options;
    }
    catch (const irt::Exception &)
    {
        throw;
    }
    catch (const std::exception &e)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "OpenVINO inference failed: %s", e.what());
    }
}

class OpenVINOSession final : public irt::ITensorRuntimeSession
{
public:
    OpenVINOSession(ov::CompiledModel compiled_model, std::vector<std::string> input_names,
                    std::vector<std::string> output_names, std::unordered_map<std::string, TensorInfo> input_info,
                    std::unordered_map<std::string, TensorInfo> output_info)
        : compiled_model_(std::move(compiled_model))
        , infer_request_(compiled_model_.create_infer_request())
        , input_names_(std::move(input_names))
        , output_names_(std::move(output_names))
        , input_info_(std::move(input_info))
        , output_info_(std::move(output_info))
    {
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
            return OvTypeToCore(input_it->second.element_type);
        }
        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OvTypeToCore(output_it->second.element_type);
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
        ExecuteOpenVINO(infer_request_, input_names_, output_names_, input_info_, output_info_, buffers, options);
    }

private:
    ov::CompiledModel                         compiled_model_;
    ov::InferRequest                          infer_request_;
    std::vector<std::string>                  input_names_;
    std::vector<std::string>                  output_names_;
    std::unordered_map<std::string, TensorInfo> input_info_;
    std::unordered_map<std::string, TensorInfo> output_info_;
};

class OpenVINOBackend final : public irt::model::IBackendRuntime
{
public:
    ModelRuntime::Backend backend() const noexcept override
    {
        return ModelRuntime::Backend::OpenVINO;
    }

    void load(const std::string &model_file, const IModelConfig &config, const std::string &model_name) override
    {
        (void)model_name;
        const std::filesystem::path model_path(model_file);
        if (!std::filesystem::exists(model_path))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open OpenVINO model file: %s",
                                 model_file.c_str());
        }

        try
        {
            auto new_model = core_.read_model(model_path);
            auto new_compiled_model = core_.compile_model(new_model, DeviceName(config.runtime()),
                                                          ov::hint::inference_precision(ov::element::f32));
            auto new_infer_request = new_compiled_model.create_infer_request();
            std::vector<std::string>                    new_input_names;
            std::vector<std::string>                    new_output_names;
            std::unordered_map<std::string, TensorInfo> new_input_info;
            std::unordered_map<std::string, TensorInfo> new_output_info;
            refreshMetadata(new_compiled_model, config, new_input_names, new_output_names, new_input_info,
                            new_output_info);

            model_.swap(new_model);
            std::swap(compiled_model_, new_compiled_model);
            std::swap(infer_request_, new_infer_request);
            input_names_.swap(new_input_names);
            output_names_.swap(new_output_names);
            input_info_.swap(new_input_info);
            output_info_.swap(new_output_info);
        }
        catch (const irt::Exception &)
        {
            throw;
        }
        catch (const std::exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "OpenVINO error while loading %s: %s", model_file.c_str(),
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
            return OvTypeToCore(input_it->second.element_type);
        }

        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OvTypeToCore(output_it->second.element_type);
        }

        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override
    {
        ensureLoaded();
        if (input_info_.find(tensor_name) == input_info_.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input tensor not found: %s", tensor_name.c_str());
        }
        const auto &info = input_info_.at(tensor_name);
        irt::validateRuntimeShape(info.declared_shape, shape, tensor_name);
        input_info_[tensor_name].shape = shape;
        PropagateResolvedBatchDimToDynamicOutputs(output_info_, shape);
    }

    void executeNormalized(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options) override
    {
        ensureLoaded();
        ExecuteOpenVINO(infer_request_, input_names_, output_names_, input_info_, output_info_, buffers, options);
    }

    std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override
    {
        ensureLoaded();
        try
        {
            return std::make_unique<OpenVINOSession>(compiled_model_, input_names_, output_names_, input_info_,
                                                     output_info_);
        }
        catch (const irt::Exception &)
        {
            throw;
        }
        catch (const std::exception &e)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "OpenVINO session creation failed: %s", e.what());
        }
    }

private:
    void ensureLoaded() const
    {
        if (!model_)
        {
            throw irt::Exception(Status::INVALID_OPERATION, "OpenVINO model is not initialized");
        }
    }

    static void refreshMetadata(const ov::CompiledModel &compiled_model, const IModelConfig &config,
                                std::vector<std::string> &input_names, std::vector<std::string> &output_names,
                                std::unordered_map<std::string, TensorInfo> &input_info,
                                std::unordered_map<std::string, TensorInfo> &output_info)
    {
        input_names.clear();
        output_names.clear();
        input_info.clear();
        output_info.clear();

        const auto inputs  = compiled_model.inputs();
        const auto outputs = compiled_model.outputs();

        for (const auto &input : inputs)
        {
            auto name = PortName(input);
            input_names.push_back(name);
            auto dims         = PartialShapeToCoreShape(input.get_partial_shape());
            input_info[name]  = TensorInfo{dims, dims, input.get_element_type(),
                                          !dims.empty() && dims.dims.front() < 0};
        }

        std::vector<std::string> graph_output_names;
        for (const auto &output : outputs)
        {
            auto name = PortName(output);
            graph_output_names.push_back(name);
            auto dims          = PartialShapeToCoreShape(output.get_partial_shape());
            output_info[name]  = TensorInfo{dims, dims, output.get_element_type(),
                                           !dims.empty() && dims.dims.front() < 0};
        }

        const auto &configured_inputs = config.inputTensorNames();
        if (configured_inputs != std::vector<std::string>{"input"})
        {
            for (const auto &input_name : configured_inputs)
            {
                if (input_info.find(input_name) == input_info.end())
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                         "Configured input tensor is not an OpenVINO model input: %s",
                                         input_name.c_str());
                }
            }
            input_names = configured_inputs;
        }

        output_names = IsDefaultOutputConfig(config) ? graph_output_names : config.outputTensorNames();
        for (const auto &output_name : output_names)
        {
            if (output_info.find(output_name) == output_info.end())
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Requested output tensor is not an OpenVINO model output: %s",
                                     output_name.c_str());
            }
        }
    }

    ov::Core                   core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel          compiled_model_;
    ov::InferRequest           infer_request_;

    std::vector<std::string>                    input_names_;
    std::vector<std::string>                    output_names_;
    std::unordered_map<std::string, TensorInfo> input_info_;
    std::unordered_map<std::string, TensorInfo> output_info_;
};

} // namespace

std::unique_ptr<irt::model::IBackendRuntime> CreateOpenVINOBackend()
{
    return std::make_unique<OpenVINOBackend>();
}

} // namespace irt::model::priv
