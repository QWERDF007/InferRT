#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/ModelFactory.hpp>
#include <inferrt/util/CheckError.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

/**
 * @brief 将 TensorRT 维度对象转换为 Python 友好的整型数组。
 * @param dims TensorRT 维度对象。
 * @return 维度数组，顺序与 TensorRT 保持一致。
 */
std::vector<int64_t> dimsToVector(const nvinfer1::Dims &dims)
{
    std::vector<int64_t> values;
    values.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        values.push_back(dims.d[i]);
    }
    return values;
}

/**
 * @brief 将 Python 侧维度数组转换为 TensorRT 维度对象。
 * @param shape Python 侧维度数组。
 * @return TensorRT 维度对象。
 */
nvinfer1::Dims vectorToDims(const std::vector<int64_t> &shape)
{
    constexpr size_t kMaxDims = 8;
    if (shape.size() > kMaxDims)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Too many dimensions: %zu", shape.size());
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = static_cast<int32_t>(shape[i]);
    }
    return dims;
}

/**
 * @brief 克隆一份模型配置，避免 Python 对象生命周期影响 C++ 模型实例。
 * @param config 待克隆的配置对象。
 * @return 独立的新配置对象。
 */
std::unique_ptr<irt::model::IModelConfig> cloneModelConfig(const irt::model::IModelConfig &config)
{
    auto cloned = std::make_unique<irt::model::IModelConfig>();
    cloned->setNumClasses(config.numClasses());
    cloned->setInputShapes(config.inputShapes());
    cloned->setInputTensorNames(config.inputTensorNames());
    cloned->setOutputTensorNames(config.outputTensorNames());
    cloned->setFeatureTensorNames(config.featureTensorNames());
    cloned->setFeatureOutputTensorNames(config.featureOutputTensorNames());
    cloned->setFeatureOnly(config.featureOnly());
    return cloned;
}

/**
 * @brief 返回 TensorRT 数据类型对应的字节数。
 * @param data_type TensorRT 数据类型。
 * @return 单个元素字节数。
 */
size_t dataTypeSize(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8:
        return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    default:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported TensorRT data type");
    }
}

/**
 * @brief 将 TensorRT 数据类型转换为可读字符串。
 * @param data_type TensorRT 数据类型。
 * @return 类型名称。
 */
std::string dataTypeToString(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return "float32";
    case nvinfer1::DataType::kHALF:
        return "float16";
    case nvinfer1::DataType::kINT8:
        return "int8";
    case nvinfer1::DataType::kINT32:
        return "int32";
    case nvinfer1::DataType::kBOOL:
        return "bool";
    default:
        return "unknown";
    }
}

/**
 * @brief 将 TensorRT 数据类型转换为 NumPy dtype。
 * @param data_type TensorRT 数据类型。
 * @return 对应的 NumPy dtype。
 */
py::dtype dataTypeToPyDType(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return py::dtype::of<float>();
    case nvinfer1::DataType::kHALF:
        return py::dtype("float16");
    case nvinfer1::DataType::kINT8:
        return py::dtype::of<int8_t>();
    case nvinfer1::DataType::kINT32:
        return py::dtype::of<int32_t>();
    case nvinfer1::DataType::kBOOL:
        return py::dtype::of<bool>();
    default:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported TensorRT data type");
    }
}

/**
 * @brief 计算 TensorRT 维度对应的元素总数。
 * @param dims TensorRT 维度。
 * @param tensor_name 张量名称，仅用于报错。
 * @return 元素总数。
 */
size_t tensorElementCount(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    if (dims.nbDims <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Tensor has invalid rank: %s", tensor_name.c_str());
    }

    size_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                 "Tensor shape is not fully resolved, tensor=%s dim[%d]=%d", tensor_name.c_str(), i,
                                 dims.d[i]);
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

/**
 * @brief 将任意 Python 输入规整为指定 dtype 的连续 NumPy 数组。
 * @param input Python 输入对象。
 * @param data_type 目标 TensorRT 数据类型。
 * @return 连续 NumPy 数组。
 */
py::array asContiguousArray(const py::handle &input, nvinfer1::DataType data_type)
{
    static py::object numpy = py::module_::import("numpy");
    return py::array(numpy.attr("ascontiguousarray")(input, dataTypeToPyDType(data_type)));
}

bool isCContiguous(const py::array &array)
{
    return (array.flags() & py::array::c_style) != 0;
}

/**
 * @brief 管理一次推理所需的设备端缓冲区。
 */
class DeviceBuffer
{
public:
    /**
     * @brief 构造指定大小的设备缓冲区。
     * @param num_bytes 申请的字节数。
     */
    explicit DeviceBuffer(size_t num_bytes)
        : size_(num_bytes)
    {
        IRT_CHECK_THROW(cudaMalloc(&data_, size_), "Failed to allocate %zu bytes on device", size_);
    }

    /** 
     * @brief 析构时自动释放显存。 
     **/
    ~DeviceBuffer()
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    /**
     * @brief 移动构造。
     * @param other 被移动对象。
     */
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : data_(other.data_)
        , size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief 移动赋值。
     * @param other 被移动对象。
     * @return 当前对象。
     */
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        if (data_ != nullptr)
        {
            cudaFree(data_);
        }
        data_       = other.data_;
        size_       = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    /**
     * @brief 返回设备指针。
     * @return 设备端地址。
     */
    void *data() const noexcept
    {
        return data_;
    }

    /**
     * @brief 返回缓冲区大小。
     * @return 字节数。
     */
    size_t size() const noexcept
    {
        return size_;
    }

private:
    void  *data_{nullptr};
    size_t size_{0};
};

/**
 * @brief 保存一次张量搬运所需的宿主侧和设备侧对象。
 */
struct TensorBinding
{
    std::string  name;
    py::array    host_array;
    DeviceBuffer device_buffer;

    /**
     * @brief 构造张量绑定对象。
     * @param tensor_name 张量名称。
     * @param array 宿主侧 NumPy 数组。
     */
    TensorBinding(std::string tensor_name, py::array array)
        : name(std::move(tensor_name))
        , host_array(std::move(array))
        , device_buffer(static_cast<size_t>(host_array.nbytes()))
    {
    }
};

/**
 * @brief 将多个输出张量打包成 Python 返回值。
 * @param tensors 输出张量列表。
 * @return 单输出返回 `numpy.ndarray`，多输出返回 `dict[str, numpy.ndarray]`。
 */
py::object packOutputs(const std::vector<TensorBinding> &tensors)
{
    if (tensors.size() == 1)
    {
        return tensors.front().host_array;
    }

    py::dict result;
    for (const auto &tensor : tensors)
    {
        result[py::str(tensor.name)] = tensor.host_array;
    }
    return std::move(result);
}

/**
 * @brief 绑定后的 Python 模型包装器。
 *
 * 该类在 `irt::model::IModel` 之上补充了 NumPy 输入输出适配，
 * 让 Python 侧可以直接用 `numpy.ndarray` 调用推理与特征前向。
 */
class PyModel
{
public:
    /**
     * @brief 使用已有的 C++ 模型对象构造 Python 包装器。
     * @param model C++ 模型对象。
     */
    explicit PyModel(std::unique_ptr<irt::model::IModel> model)
        : model_(std::move(model))
    {
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model instance must not be null");
        }
    }

    /**
     * @brief 返回模型名称。
     * @return 模型名称。
     */
    std::string name() const noexcept
    {
        return model_->name();
    }

    /**
     * @brief 返回输入张量名称列表。
     * @return 输入张量名称列表。
     */
    std::vector<std::string> inputTensorNames() const
    {
        return model_->modelConfig().inputTensorNames();
    }

    /**
     * @brief 返回主输出张量名称列表。
     * @return 主输出张量名称列表。
     */
    std::vector<std::string> outputTensorNames() const
    {
        return model_->modelConfig().outputTensorNames();
    }

    /**
     * @brief 返回特征输出张量名称列表。
     * @return 特征输出张量名称列表。
     */
    std::vector<std::string> featureOutputTensorNames() const
    {
        const auto &config_names = model_->modelConfig().featureOutputTensorNames();
        if (!config_names.empty())
        {
            return config_names;
        }
        return model_->modelConfig().featureTensorNames();
    }

    /**
     * @brief 构建 TensorRT engine。
     * @param weights_file 权重文件路径。
     */
    void build(const std::string &weights_file)
    {
        model_->build(weights_file);
    }

    /**
     * @brief 加载 TensorRT engine。
     * @param engine_file engine 文件路径。
     */
    void load(const std::string &engine_file)
    {
        model_->load(engine_file);
    }

    /**
     * @brief 保存 TensorRT engine。
     * @param engine_file 目标文件路径。
     */
    void save(const std::string &engine_file)
    {
        model_->save(engine_file);
    }

    /**
     * @brief 优先加载已有 engine，不存在时从权重构建。
     * @param weights_file 权重文件路径。
     */
    void buildOrLoad(const std::string &weights_file)
    {
        model_->buildOrLoad(weights_file);
    }

    /**
     * @brief 查询指定张量的运行时形状。
     * @param tensor_name 张量名称。
     * @return 形状数组。
     */
    std::vector<int64_t> tensorShape(const std::string &tensor_name) const
    {
        return dimsToVector(model_->tensorShape(tensor_name));
    }

    /**
     * @brief 查询指定张量的数据类型。
     * @param tensor_name 张量名称。
     * @return NumPy 风格的类型名称。
     */
    std::string tensorDType(const std::string &tensor_name) const
    {
        return dataTypeToString(model_->tensorDataType(tensor_name));
    }

    /**
     * @brief 显式设置输入张量运行时形状。
     * @param tensor_name 张量名称。
     * @param shape 目标形状。
     */
    void setTensorShape(const std::string &tensor_name, const std::vector<int64_t> &shape)
    {
        model_->setTensorShape(tensor_name, vectorToDims(shape));
    }

    /**
     * @brief 设置 TensorRT 日志级别。
     * @param severity 日志级别。
     */
    void setLogLevel(nvinfer1::ILogger::Severity severity)
    {
        model_->setLogLevel(severity);
    }

    /**
     * @brief 获取当前 TensorRT 日志级别。
     * @return 日志级别。
     */
    nvinfer1::ILogger::Severity logLevel() const
    {
        return model_->logLevel();
    }

    /**
     * @brief 执行主推理。
     * @param inputs 输入张量；单输入可直接传 `numpy.ndarray`，多输入可传 `list/tuple/dict`。
     * @return 单输出返回 `numpy.ndarray`，多输出返回 `dict[str, numpy.ndarray]`。
     */
    py::object infer(const py::object &inputs, const py::object &outputs)
    {
        return execute(inputs, outputs, outputTensorNames(),
                       [this](const std::vector<void *> &buffers) { model_->infer(buffers); });
    }

    /**
     * @brief 执行特征前向。
     * @param inputs 输入张量；单输入可直接传 `numpy.ndarray`，多输入可传 `list/tuple/dict`。
     * @return 单输出返回 `numpy.ndarray`，多输出返回 `dict[str, numpy.ndarray]`。
     */
    py::object forwardFeatures(const py::object &inputs)
    {
        const auto feature_names = featureOutputTensorNames();
        if (feature_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "No feature tensors are configured");
        }

        return execute(inputs, py::none(), feature_names,
                       [this](const std::vector<void *> &buffers) { model_->forwardFeatures(buffers); });
    }

private:
    /**
     * @brief 将 Python 输入对象规整为有序的输入张量数组列表。
     * @param inputs Python 输入对象。
     * @return 与配置输入顺序一致的数组列表。
     */
    std::vector<py::handle> normalizeTensorObjects(const py::object &tensors, const std::vector<std::string> &tensor_names,
                                                   const char *tensor_kind) const
    {
        if (tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model has no configured %s", tensor_kind);
        }

        if (tensor_names.size() == 1 && py::isinstance<py::array>(tensors))
        {
            return {tensors};
        }

        if (py::isinstance<py::dict>(tensors))
        {
            py::dict                mapping = py::reinterpret_borrow<py::dict>(tensors);
            std::vector<py::handle> values;
            values.reserve(tensor_names.size());
            for (const auto &tensor_name : tensor_names)
            {
                if (!mapping.contains(py::str(tensor_name)))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Missing %s tensor: %s", tensor_kind,
                                         tensor_name.c_str());
                }
                values.push_back(mapping[py::str(tensor_name)]);
            }
            return values;
        }

        if (py::isinstance<py::sequence>(tensors))
        {
            py::sequence sequence = py::reinterpret_borrow<py::sequence>(tensors);
            if (static_cast<size_t>(py::len(sequence)) != tensor_names.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Expected %zu %s tensors, got %zd",
                                     tensor_names.size(), tensor_kind, py::len(sequence));
            }

            std::vector<py::handle> values;
            values.reserve(tensor_names.size());
            for (const auto &item : sequence)
            {
                values.push_back(item);
            }
            return values;
        }

        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "%s tensors must be numpy.ndarray, sequence, or dict[str, numpy.ndarray]", tensor_kind);
    }

    std::vector<py::handle> normalizeInputs(const py::object &inputs) const
    {
        return normalizeTensorObjects(inputs, inputTensorNames(), "input");
    }

    std::vector<py::handle> normalizeOutputs(const py::object &outputs,
                                             const std::vector<std::string> &output_names) const
    {
        return normalizeTensorObjects(outputs, output_names, "output");
    }

    /**
     * @brief 准备输入张量，并将运行时形状同步到模型上下文。
     * @param inputs Python 输入对象。
     * @return 已规整好的输入张量绑定列表。
     */
    std::vector<TensorBinding> prepareInputBindings(const py::object &inputs)
    {
        const auto input_names  = inputTensorNames();
        const auto input_values = normalizeInputs(inputs);

        std::vector<TensorBinding> bindings;
        bindings.reserve(input_names.size());

        for (size_t i = 0; i < input_names.size(); ++i)
        {
            const auto &input_name = input_names[i];
            const auto  data_type  = model_->tensorDataType(input_name);
            auto        array      = asContiguousArray(input_values[i], data_type);

            std::vector<int64_t> shape;
            shape.reserve(static_cast<size_t>(array.ndim()));
            for (py::ssize_t dim_index = 0; dim_index < array.ndim(); ++dim_index)
            {
                shape.push_back(array.shape(dim_index));
            }

            model_->setTensorShape(input_name, vectorToDims(shape));
            bindings.emplace_back(input_name, std::move(array));
        }

        return bindings;
    }

    /**
     * @brief 根据运行时张量信息创建输出数组和设备缓冲区。
     * @param tensor_names 待创建的输出张量名称列表。
     * @return 输出张量绑定列表。
     */
    std::vector<TensorBinding> prepareOutputBindings(const py::object &outputs, const std::vector<std::string> &tensor_names)
    {
        if (outputs.is_none())
        {
            std::vector<TensorBinding> bindings;
            bindings.reserve(tensor_names.size());

            for (const auto &tensor_name : tensor_names)
            {
                const auto dims        = model_->tensorShape(tensor_name);
                const auto data_type   = model_->tensorDataType(tensor_name);
                const auto element_num = tensorElementCount(dims, tensor_name);
                const auto dtype       = dataTypeToPyDType(data_type);
                const auto shape       = dimsToVector(dims);

                std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
                py::array                array(dtype, py_shape);

                if (static_cast<size_t>(array.nbytes()) != element_num * dataTypeSize(data_type))
                {
                    throw irt::Exception(irt::Status::ERROR_INTERNAL, "Unexpected numpy buffer size for tensor: %s",
                                         tensor_name.c_str());
                }

                bindings.emplace_back(tensor_name, std::move(array));
            }

            return bindings;
        }

        const auto output_values = normalizeOutputs(outputs, tensor_names);
        std::vector<TensorBinding> bindings;
        bindings.reserve(tensor_names.size());

        for (size_t i = 0; i < tensor_names.size(); ++i)
        {
            const auto &tensor_name = tensor_names[i];
            const auto dims        = model_->tensorShape(tensor_name);
            const auto data_type   = model_->tensorDataType(tensor_name);
            const auto element_num = tensorElementCount(dims, tensor_name);
            const auto dtype       = dataTypeToPyDType(data_type);
            const auto shape       = dimsToVector(dims);
            if (!py::isinstance<py::array>(output_values[i]))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be numpy.ndarray: %s",
                                     tensor_name.c_str());
            }

            py::array array = py::reinterpret_borrow<py::array>(output_values[i]);
            if (!array.writeable())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor is not writable: %s",
                                     tensor_name.c_str());
            }
            if (!isCContiguous(array))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Output tensor must be C-contiguous: %s", tensor_name.c_str());
            }
            if (!array.dtype().equal(dtype))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Output tensor dtype mismatch: %s, expected=%s", tensor_name.c_str(),
                                     dataTypeToString(data_type).c_str());
            }
            if (static_cast<size_t>(array.ndim()) != shape.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Output tensor rank mismatch: %s, expected=%zu, got=%zd", tensor_name.c_str(),
                                     shape.size(), array.ndim());
            }
            for (size_t dim_index = 0; dim_index < shape.size(); ++dim_index)
            {
                if (array.shape(static_cast<py::ssize_t>(dim_index)) != shape[dim_index])
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Output tensor shape mismatch: %s, dim[%zu] expected=%lld, got=%zd",
                                         tensor_name.c_str(), dim_index, static_cast<long long>(shape[dim_index]),
                                         array.shape(static_cast<py::ssize_t>(dim_index)));
                }
            }
            if (static_cast<size_t>(array.nbytes()) != element_num * dataTypeSize(data_type))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected numpy buffer size for tensor: %s",
                                     tensor_name.c_str());
            }

            bindings.emplace_back(tensor_name, std::move(array));
        }

        return bindings;
    }

    /**
     * @brief 将输入输出张量拼装成 InferRT 所需的缓冲区指针数组。
     * @param inputs 输入张量绑定列表。
     * @param outputs 输出张量绑定列表。
     * @return 指针数组。
     */
    std::vector<void *> buildBufferList(const std::vector<TensorBinding> &inputs,
                                        const std::vector<TensorBinding> &outputs) const
    {
        std::vector<void *> buffers;
        buffers.reserve(inputs.size() + outputs.size());
        for (const auto &input : inputs)
        {
            buffers.push_back(input.device_buffer.data());
        }
        for (const auto &output : outputs)
        {
            buffers.push_back(output.device_buffer.data());
        }
        return buffers;
    }

    /**
     * @brief 将宿主侧输入拷贝到设备侧。
     * @param inputs 输入张量绑定列表。
     */
    void copyInputsToDevice(const std::vector<TensorBinding> &inputs) const
    {
        for (const auto &input : inputs)
        {
            IRT_CHECK_THROW(cudaMemcpy(input.device_buffer.data(), input.host_array.data(), input.device_buffer.size(),
                                       cudaMemcpyHostToDevice),
                            "Failed to copy input tensor to device: %s", input.name.c_str());
        }
    }

    /**
     * @brief 将设备侧输出拷贝回宿主侧。
     * @param outputs 输出张量绑定列表。
     */
    void copyOutputsToHost(std::vector<TensorBinding> &outputs) const
    {
        for (auto &output : outputs)
        {
            IRT_CHECK_THROW(cudaMemcpy(output.host_array.mutable_data(), output.device_buffer.data(),
                                       output.device_buffer.size(), cudaMemcpyDeviceToHost),
                            "Failed to copy output tensor to host: %s", output.name.c_str());
        }
    }

    /**
     * @brief 执行一次通用的 NumPy <-> TensorRT 前向。
     * @param inputs Python 输入对象。
     * @param output_names 输出张量名称列表。
     * @param forward_fn 真正执行前向的函数。
     * @return Python 返回值。
     */
    py::object execute(const py::object &inputs, const py::object &outputs, const std::vector<std::string> &output_names,
                       const std::function<void(const std::vector<void *> &)> &forward_fn)
    {
        auto input_bindings  = prepareInputBindings(inputs);
        auto output_bindings = prepareOutputBindings(outputs, output_names);
        auto buffers         = buildBufferList(input_bindings, output_bindings);

        {
            py::gil_scoped_release release;
            copyInputsToDevice(input_bindings);
            forward_fn(buffers);
            copyOutputsToHost(output_bindings);
        }

        return packOutputs(output_bindings);
    }

private:
    std::unique_ptr<irt::model::IModel> model_;
};

} // namespace

/**
 * @brief InferRT 模型 Python 绑定模块。
 * @param m Python 模块对象。
 */
PYBIND11_MODULE(inferrt_model_py, m)
{
    m.doc() = "InferRT model bindings powered by pybind11";

    py::register_exception<irt::Exception>(m, "InferRTError");

    py::enum_<nvinfer1::ILogger::Severity>(m, "LogLevel", "TensorRT 日志级别。")
        .value("INTERNAL_ERROR", nvinfer1::ILogger::Severity::kINTERNAL_ERROR)
        .value("ERROR", nvinfer1::ILogger::Severity::kERROR)
        .value("WARNING", nvinfer1::ILogger::Severity::kWARNING)
        .value("INFO", nvinfer1::ILogger::Severity::kINFO)
        .value("VERBOSE", nvinfer1::ILogger::Severity::kVERBOSE);

    py::class_<irt::model::IModelConfig>(m, "ModelConfig", "InferRT 模型配置对象。")
        .def(py::init<>(), "构造默认的 ImageNet 分类配置。")
        .def_property("num_classes", &irt::model::IModelConfig::numClasses, &irt::model::IModelConfig::setNumClasses,
                      "类别数。")
        .def_property(
            "input_shape", [](const irt::model::IModelConfig &self) { return dimsToVector(self.inputShape()); },
            [](irt::model::IModelConfig &self, const std::vector<int64_t> &shape)
            {
                if (shape.size() != 4)
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "input_shape must contain exactly 4 dimensions");
                }
                self.setInputShape(nvinfer1::Dims4{static_cast<int32_t>(shape.at(0)), static_cast<int32_t>(shape.at(1)),
                                                   static_cast<int32_t>(shape.at(2)),
                                                   static_cast<int32_t>(shape.at(3))});
            },
            "第一个输入张量形状，按 NCHW 顺序。")
        .def_property(
            "input_shapes",
            [](const irt::model::IModelConfig &self)
            {
                std::vector<std::vector<int64_t>> shapes;
                for (const auto &dims : self.inputShapes())
                {
                    shapes.push_back(dimsToVector(dims));
                }
                return shapes;
            },
            [](irt::model::IModelConfig &self, const std::vector<std::vector<int64_t>> &shapes)
            {
                std::vector<nvinfer1::Dims4> dims_list;
                dims_list.reserve(shapes.size());
                for (const auto &shape : shapes)
                {
                    if (shape.size() != 4)
                    {
                        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                             "Each input shape must contain 4 dimensions");
                    }
                    dims_list.push_back(nvinfer1::Dims4{static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[1]),
                                                        static_cast<int32_t>(shape[2]),
                                                        static_cast<int32_t>(shape[3])});
                }
                self.setInputShapes(std::move(dims_list));
            },
            "所有输入张量形状列表。")
        .def_property("input_tensor_names", &irt::model::IModelConfig::inputTensorNames,
                      &irt::model::IModelConfig::setInputTensorNames, "输入张量名称列表。")
        .def_property("output_tensor_names", &irt::model::IModelConfig::outputTensorNames,
                      &irt::model::IModelConfig::setOutputTensorNames, "主输出张量名称列表。")
        .def_property("feature_tensor_names", &irt::model::IModelConfig::featureTensorNames,
                      &irt::model::IModelConfig::setFeatureTensorNames, "中间特征层 key 列表。")
        .def_property("feature_output_tensor_names", &irt::model::IModelConfig::featureOutputTensorNames,
                      &irt::model::IModelConfig::setFeatureOutputTensorNames, "特征输出张量名称列表。")
        .def_property("feature_only", &irt::model::IModelConfig::featureOnly, &irt::model::IModelConfig::setFeatureOnly,
                      "是否仅构建特征提取网络。");

    py::class_<PyModel>(m, "Model", "InferRT Python 模型包装器。")
        .def("name", &PyModel::name, "返回模型名称。")
        .def("build", &PyModel::build, py::arg("weights_file"), "从权重文件构建 TensorRT engine。")
        .def("load", &PyModel::load, py::arg("engine_file"), "从 engine 文件加载模型。")
        .def("save", &PyModel::save, py::arg("engine_file"), "将当前 engine 序列化到磁盘。")
        .def("build_or_load", &PyModel::buildOrLoad, py::arg("weights_file"),
             "优先加载已有 engine，不存在时从权重文件构建。")
        .def("infer", &PyModel::infer, py::arg("inputs"), py::arg("outputs"),
             "执行一次主推理，由 Python 传入输入和输出 ndarray、sequence 或 dict。")
        .def("forward_features", &PyModel::forwardFeatures, py::arg("inputs"),
             "执行一次特征前向，输入支持 ndarray、sequence 或 dict。")
        .def("input_tensor_names", &PyModel::inputTensorNames, "返回输入张量名称列表。")
        .def("output_tensor_names", &PyModel::outputTensorNames, "返回主输出张量名称列表。")
        .def("feature_output_tensor_names", &PyModel::featureOutputTensorNames, "返回特征输出张量名称列表。")
        .def("tensor_shape", &PyModel::tensorShape, py::arg("tensor_name"), "返回指定张量的运行时形状。")
        .def("tensor_dtype", &PyModel::tensorDType, py::arg("tensor_name"), "返回指定张量的数据类型名称。")
        .def("set_tensor_shape", &PyModel::setTensorShape, py::arg("tensor_name"), py::arg("shape"),
             "设置输入张量的运行时形状。")
        .def_property("log_level", &PyModel::logLevel, &PyModel::setLogLevel, "TensorRT 日志级别。");

    m.def(
        "create_model",
        [](const std::string &name, const py::object &config_object)
        {
            std::unique_ptr<irt::model::IModelConfig> config;
            if (config_object.is_none())
            {
                config = std::make_unique<irt::model::IModelConfig>();
            }
            else
            {
                const auto &config_ref = config_object.cast<const irt::model::IModelConfig &>();
                config                 = cloneModelConfig(config_ref);
            }

            return PyModel(irt::model::CreateModel(name, std::move(config)));
        },
        py::arg("name"), py::arg("config") = py::none(), "根据注册名称创建模型对象。");

    m.def("is_supported_model", &irt::model::isSupportedModel, py::arg("name"), "查询模型名称是否已注册。");
    m.def("get_registered_model_names", &irt::model::getRegisteredModelNames, "返回当前所有已注册模型名称。");
}
