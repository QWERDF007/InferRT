#include "BackendRuntime.hpp"
#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include <openvino/openvino.hpp>

#include <filesystem>
#include <limits>
#include <unordered_map>

namespace irt::model::priv {

namespace {

struct TensorInfo
{
    nvinfer1::Dims    shape{};
    ov::element::Type element_type{};
    bool              dynamic_batch{false};
};

std::string DeviceName(ModelDevice device)
{
    return device == ModelDevice::GPU ? "GPU" : "CPU";
}

nvinfer1::DataType OvTypeToTrt(const ov::element::Type &type)
{
    if (type == ov::element::f32)
    {
        return nvinfer1::DataType::kFLOAT;
    }
    if (type == ov::element::f16)
    {
        return nvinfer1::DataType::kHALF;
    }
    if (type == ov::element::i8)
    {
        return nvinfer1::DataType::kINT8;
    }
    if (type == ov::element::u8)
    {
        return nvinfer1::DataType::kUINT8;
    }
    if (type == ov::element::i32)
    {
        return nvinfer1::DataType::kINT32;
    }
    if (type == ov::element::i64)
    {
        return nvinfer1::DataType::kINT64;
    }
    if (type == ov::element::boolean)
    {
        return nvinfer1::DataType::kBOOL;
    }

    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported OpenVINO element type: %s",
                         type.get_type_name().c_str());
}

nvinfer1::Dims PartialShapeToDims(const ov::PartialShape &shape)
{
    if (shape.rank().is_dynamic())
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Dynamic-rank OpenVINO tensors are not supported");
    }
    if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Tensor rank %zu exceeds TensorRT Dims capacity",
                             shape.size());
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i].is_dynamic())
        {
            dims.d[i] = -1;
            continue;
        }
        const auto value = shape[i].get_length();
        if (value > std::numeric_limits<int32_t>::max())
        {
            throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Tensor dimension exceeds int32: %lld",
                                 static_cast<long long>(value));
        }
        dims.d[i] = static_cast<int32_t>(value);
    }
    return dims;
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

class OpenVINOBackend final : public IBackendRuntime
{
public:
    ModelBackend backend() const noexcept override
    {
        return ModelBackend::OpenVINO;
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
            model_          = core_.read_model(model_path);
            compiled_model_ = core_.compile_model(model_, DeviceName(config.device()));
            infer_request_  = compiled_model_.create_infer_request();
            refreshMetadata(config);
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
            return OvTypeToTrt(input_it->second.element_type);
        }

        const auto output_it = output_info_.find(tensor_name);
        if (output_it != output_info_.end())
        {
            return OvTypeToTrt(output_it->second.element_type);
        }

        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
    }

    void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) override
    {
        ensureLoaded();
        if (input_info_.find(tensor_name) == input_info_.end())
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
            std::vector<ov::Tensor> tensors;
            tensors.reserve(expected);

            size_t buffer_index = 0;
            for (const auto &input_name : input_names_)
            {
                void *buffer = buffers[buffer_index++];
                if (!buffer)
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input buffer is null: %s",
                                         input_name.c_str());
                }
                const auto     &info  = input_info_.at(input_name);
                const ov::Shape shape = DimsToSizeTShape(info.shape, input_name);
                tensors.emplace_back(info.element_type, shape, buffer);
                infer_request_.set_tensor(input_name, tensors.back());
            }

            for (const auto &output_name : output_names_)
            {
                void *buffer = buffers[buffer_index++];
                if (!buffer)
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Output buffer is null: %s",
                                         output_name.c_str());
                }
                const auto     &info  = output_info_.at(output_name);
                const ov::Shape shape = DimsToSizeTShape(info.shape, output_name);
                tensors.emplace_back(info.element_type, shape, buffer);
                infer_request_.set_tensor(output_name, tensors.back());
            }

            infer_request_.infer();
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

private:
    void ensureLoaded() const
    {
        if (!model_)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "OpenVINO model is not initialized");
        }
    }

    void refreshMetadata(const IModelConfig &config)
    {
        input_names_.clear();
        output_names_.clear();
        input_info_.clear();
        output_info_.clear();

        const auto inputs  = compiled_model_.inputs();
        const auto outputs = compiled_model_.outputs();

        for (const auto &input : inputs)
        {
            auto name = PortName(input);
            input_names_.push_back(name);
            auto dims         = PartialShapeToDims(input.get_partial_shape());
            input_info_[name] = TensorInfo{dims, input.get_element_type(), dims.nbDims > 0 && dims.d[0] < 0};
        }

        std::vector<std::string> graph_output_names;
        for (const auto &output : outputs)
        {
            auto name = PortName(output);
            graph_output_names.push_back(name);
            auto dims          = PartialShapeToDims(output.get_partial_shape());
            output_info_[name] = TensorInfo{dims, output.get_element_type(), dims.nbDims > 0 && dims.d[0] < 0};
        }

        const auto &configured_inputs = config.inputTensorNames();
        if (configured_inputs != std::vector<std::string>{"input"})
        {
            for (const auto &input_name : configured_inputs)
            {
                if (input_info_.find(input_name) == input_info_.end())
                {
                    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                         "Configured input tensor is not an OpenVINO model input: %s",
                                         input_name.c_str());
                }
            }
            input_names_ = configured_inputs;
        }

        output_names_ = IsDefaultOutputConfig(config) ? graph_output_names : config.outputTensorNames();
        for (const auto &output_name : output_names_)
        {
            if (output_info_.find(output_name) == output_info_.end())
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

std::unique_ptr<IBackendRuntime> CreateOpenVINOBackend()
{
    return std::make_unique<OpenVINOBackend>();
}

} // namespace irt::model::priv
