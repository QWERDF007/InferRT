/**
 * @file InferRTModelPybind.cpp
 * @brief InferRT 模型的 pybind11 Python 绑定实现。
 *
 * 提供 NumPy 与 DLPack 张量到 TensorRT 推理管线的桥接，包括模型创建、
 * engine 构建/加载、主推理、特征前向及张量形状/类型查询等能力。
 */

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/ModelFactory.hpp>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/CheckError.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

/** @brief 匿名命名空间，封装仅在本翻译单元内使用的绑定辅助逻辑。 */
namespace {

using irt::model::dataTypeSize;
using irt::model::dataTypeToString;

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
        if (shape[i] < std::numeric_limits<int32_t>::min() || shape[i] > std::numeric_limits<int32_t>::max())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Dimension out of int32 range: dim[%zu]=%lld", i,
                                 static_cast<long long>(shape[i]));
        }
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
    cloned->setFeatureOnly(config.featureOnly());
    return cloned;
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
 * @brief 带溢出检查的 size_t 乘法。
 * @param lhs 左操作数。
 * @param rhs 右操作数。
 * @param tensor_name 张量名称，仅用于溢出报错信息。
 * @return 乘积 `lhs * rhs`。
 * @throws irt::Exception 乘积超过 `size_t` 可表示范围时抛出。
 */
size_t multiplyChecked(size_t lhs, size_t rhs, const std::string &tensor_name)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Tensor size overflow: %s", tensor_name.c_str());
    }
    return lhs * rhs;
}

/**
 * @brief 根据形状向量计算张量元素总数。
 * @param shape 形状数组，各维须为非负整数。
 * @param tensor_name 张量名称，仅用于报错信息。
 * @return 元素总数。
 * @throws irt::Exception 形状为空、含动态维（负值）或乘积溢出时抛出。
 */
size_t tensorElementCount(const std::vector<int64_t> &shape, const std::string &tensor_name)
{
    if (shape.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Tensor has invalid rank: %s", tensor_name.c_str());
    }

    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                 "Tensor shape is not fully resolved, tensor=%s dim[%zu]=%lld", tensor_name.c_str(), i,
                                 static_cast<long long>(shape[i]));
        }
        count = multiplyChecked(count, static_cast<size_t>(shape[i]), tensor_name);
    }
    return count;
}

/**
 * @brief 根据 TensorRT 维度对象计算张量元素总数。
 * @param dims TensorRT 维度。
 * @param tensor_name 张量名称，仅用于报错信息。
 * @return 元素总数。
 */
size_t tensorElementCount(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    return tensorElementCount(dimsToVector(dims), tensor_name);
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

/**
 * @brief 将 NumPy 数组的 shape 提取为 int64 向量。
 * @param array NumPy 数组。
 * @return 与 `array.shape` 一致的维度列表。
 */
std::vector<int64_t> arrayShapeToVector(const py::array &array)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(array.ndim()));
    for (py::ssize_t i = 0; i < array.ndim(); ++i)
    {
        shape.push_back(array.shape(i));
    }
    return shape;
}

/**
 * @brief 判断 NumPy 数组是否为 C 连续（行主序）布局。
 * @param array NumPy 数组。
 * @return 若为 C 连续则返回 true。
 */
bool isCContiguous(const py::array &array)
{
    return (array.flags() & py::array::c_style) != 0;
}

/**
 * @brief 检测 Python 对象是否实现 `__dlpack__` 协议。
 * @param object 待检测对象。
 * @return 若存在 `__dlpack__` 属性则返回 true。
 */
bool hasDLPackProtocol(const py::handle &object)
{
    return PyObject_HasAttrString(object.ptr(), "__dlpack__") == 1;
}

/**
 * @brief 调用对象的 `__dlpack__()` 并返回 DLPack 胶囊。
 * 
 *        PyCapsule 的设计理念是：把一个 C 语言的 void* 指针“装”进去，外面包一层不透明的壳。
 * @param object 实现 DLPack 协议的 Python 张量对象。
 * @param stream_ptr 可选 CUDA stream 指针；非 0 时作为 `stream` 关键字传入。
 * @return 包含 `DLManagedTensor` 的 PyCapsule 对象。
 */
py::object callDLPack(const py::handle &object, uintptr_t stream_ptr)
{
    py::object method = py::reinterpret_steal<py::object>(PyObject_GetAttrString(object.ptr(), "__dlpack__"));
    if (!method)
    {
        throw py::error_already_set();
    }

    py::tuple args = py::tuple();
    py::dict  kwargs;
    if (stream_ptr != 0)
    {
        kwargs[py::str("stream")] = py::int_(stream_ptr);
    }

    PyObject *result = PyObject_Call(method.ptr(), args.ptr(), kwargs.ptr());
    if (result == nullptr)
    {
        throw py::error_already_set();
    }

    return py::reinterpret_steal<py::object>(result);
}

/**
 * @brief 从 DLPack 胶囊中取出 `DLManagedTensor` 指针（不消费胶囊所有权）。
 * @param capsule `__dlpack__()` 返回的 PyCapsule。
 * @return 托管张量指针；胶囊名须为 `"dltensor"`。
 * @throws irt::Exception 对象不是合法 DLPack 胶囊时抛出。
 */
DLManagedTensor *consumeDLPackCapsulePointer(const py::handle &capsule)
{
    if (!PyCapsule_CheckExact(capsule.ptr()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "DLPack provider must return a PyCapsule from __dlpack__()");
    }

    auto *managed_tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (managed_tensor == nullptr)
    {
        throw py::error_already_set();
    }

    return managed_tensor;
}

/**
 * @brief 将 DLPack 数据类型映射为 TensorRT 数据类型。
 * @param dtype DLPack `DLDataType` 描述符。
 * @return 对应的 TensorRT 类型。
 * @throws irt::Exception 不支持的 code/bits/lanes 组合时抛出。
 */
nvinfer1::DataType dlDataTypeToTrt(const DLDataType &dtype)
{
    if (dtype.lanes != 1)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported DLPack lanes=%u",
                             static_cast<unsigned>(dtype.lanes));
    }

    switch (dtype.code)
    {
    case kDLFloat:
        if (dtype.bits == 16)
        {
            return nvinfer1::DataType::kHALF;
        }
        if (dtype.bits == 32)
        {
            return nvinfer1::DataType::kFLOAT;
        }
        break;
    case kDLInt:
        if (dtype.bits == 8)
        {
            return nvinfer1::DataType::kINT8;
        }
        if (dtype.bits == 32)
        {
            return nvinfer1::DataType::kINT32;
        }
        break;
    case kDLBool:
        if (dtype.bits == 8)
        {
            return nvinfer1::DataType::kBOOL;
        }
        break;
    default:
        break;
    }

    throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported DLPack dtype: code=%u bits=%u lanes=%u",
                         static_cast<unsigned>(dtype.code), static_cast<unsigned>(dtype.bits),
                         static_cast<unsigned>(dtype.lanes));
}

/**
 * @brief 将 DLPack 张量的 shape 转为 int64 向量。
 * @param tensor DLPack `DLTensor`。
 * @return 各维大小列表。
 * @throws irt::Exception 存在负维度时抛出。
 */
std::vector<int64_t> dlShapeToVector(const DLTensor &tensor)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor.ndim));
    for (int32_t i = 0; i < tensor.ndim; ++i)
    {
        if (tensor.shape[i] < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "DLPack tensor has negative dimension at %d", i);
        }
        shape.push_back(tensor.shape[i]);
    }
    return shape;
}

/**
 * @brief 判断 DLPack 张量是否为紧凑的行主序（末维 stride 为 1）布局。
 * @param tensor DLPack `DLTensor`。
 * @return 无 stride 信息、秩 ≤ 1 或各维 stride 符合行主序时返回 true。
 */
bool isCompactRowMajor(const DLTensor &tensor)
{
    if (tensor.strides == nullptr || tensor.ndim <= 1)
    {
        return true;
    }

    int64_t expected_stride = 1;
    for (int32_t i = tensor.ndim - 1; i >= 0; --i)
    {
        if (tensor.shape[i] == 0)
        {
            return true;
        }
        if (tensor.shape[i] != 1 && tensor.strides[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= tensor.shape[i];
    }
    return true;
}

/**
 * @brief 判断 DLPack 设备是否为主机内存。
 * @param device DLPack 设备描述符。
 * @return CPU 或 CUDA 固定主机内存时返回 true。
 */
bool dlDeviceIsHost(const DLDevice &device)
{
    return device.device_type == kDLCPU || device.device_type == kDLCUDAHost;
}

/**
 * @brief 判断 DLPack 设备是否为 CUDA 设备内存。
 * @param device DLPack 设备描述符。
 * @return CUDA 或 CUDA 托管内存时返回 true。
 */
bool dlDeviceIsCuda(const DLDevice &device)
{
    return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

/**
 * @brief 查询当前 CUDA 上下文绑定的设备编号。
 * @return 设备 ID。
 */
int currentCudaDevice()
{
    int device = 0;
    IRT_CHECK_THROW(cudaGetDevice(&device), "Failed to query current CUDA device");
    return device;
}

/**
 * @brief 管理一次推理所需的设备端缓冲区。
 */
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    /**
     * @brief 构造指定大小的设备缓冲区。
     * @param num_bytes 申请的字节数。
     */
    explicit DeviceBuffer(size_t num_bytes)
        : size_(num_bytes)
    {
        if (size_ > 0)
        {
            IRT_CHECK_THROW(cudaMalloc(&data_, size_), "Failed to allocate %zu bytes on device", size_);
        }
    }

    /**
     * @brief 析构时自动释放显存。
     */
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
    void  *data_{nullptr}; ///< 设备端缓冲区指针，未分配时为 nullptr。
    size_t size_{0};       ///< 缓冲区字节数。
};

/**
 * @brief 保存一次 NumPy 推理所需的宿主侧数组与设备侧缓冲区。
 */
struct TensorBinding
{
    std::string  name;          ///< 张量名称，与 engine 绑定名一致。
    py::array    host_array;    ///< 宿主侧 NumPy 数组（输入/输出数据）。
    DeviceBuffer device_buffer; ///< 与 host_array 等长的设备端 staging 缓冲区。

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
 * @brief 将 Python 侧单个张量对象与其配置名称配对。
 */
struct NamedTensorObject
{
    std::string name;   ///< 张量名称。
    py::handle  object; ///< Python 张量对象（ndarray 或 DLPack 提供者）。
};

/**
 * @brief 保存一次 DLPack / NumPy 混合推理所需的张量绑定状态。
 *
 * 支持直接使用 CUDA 设备指针，或在主机内存与 staging 缓冲区之间按需拷贝。
 */
struct TensorBindingV2
{
    std::string  name;                        ///< 张量名称。
    py::object   object;                      ///< 原始 Python 张量对象，用于返回值透传。
    py::object   dlpack_capsule;              ///< `__dlpack__()` 返回的胶囊；NumPy 路径为空。
    py::array    host_array;                  ///< NumPy 路径下的宿主数组。
    void        *host_ptr{nullptr};           ///< 主机侧数据指针（NumPy 或 DLPack 主机张量）。
    void        *device_ptr{nullptr};         ///< 送入 TensorRT 的设备指针。
    DeviceBuffer staging_buffer;              ///< 主机张量对应的设备 staging 区。
    size_t       num_bytes{0};                ///< 张量数据区字节数。
    bool         copy_input_to_device{false}; ///< 推理前是否执行 H2D。
    bool         copy_output_to_host{false};  ///< 推理后是否执行 D2H。
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
     * @brief 执行主推理（NumPy 路径）。
     * @param inputs 输入张量；单输入可直接传 `numpy.ndarray`，多输入可传 `list/tuple/dict`。
     * @param outputs 输出张量；`None` 时由绑定层自动分配，否则须为 ndarray/sequence/dict。
     * @param non_blocking 是否以非阻塞方式提交 CUDA 拷贝与 enqueue。
     * @return `outputs` 为 `None` 时返回新分配的 ndarray 或 dict；否则与 `outputs` 同构。
     */
    py::object infer(const py::object &inputs, const py::object &outputs, bool non_blocking = false)
    {
        const auto stream = model_->resolveExecutionStream();
        return execute(
            inputs, outputs, outputTensorNames(), [this, stream, non_blocking](const std::vector<void *> &buffers)
            { model_->infer(buffers, stream, non_blocking); }, stream);
    }

    /**
     * @brief 使用 DLPack 张量执行主推理，可直接复用外部设备内存。
     * @param inputs 输入张量；支持单张量、sequence 或 dict[str, tensor]。
     * @param outputs 输出张量；必须由调用方提供，支持单张量、sequence 或 dict[str, tensor]。
     * @param stream_ptr 可选 CUDA stream 指针；同时传给 `__dlpack__(stream=...)` 与 TensorRT enqueue。
     * @param non_blocking 是否以非阻塞方式提交 CUDA 拷贝与 enqueue。
     * @return 原样返回 `outputs`。
     */
    py::object inferV2(const py::object &inputs, const py::object &outputs, const py::object &stream_ptr,
                       bool non_blocking = false)
    {
        const auto stream_value = parseStreamPtr(stream_ptr);
        return executeV2(inputs, outputs, outputTensorNames(), stream_value,
                         [this, stream_value, non_blocking](const std::vector<void *> &buffers)
                         { model_->infer(buffers, reinterpret_cast<cudaStream_t>(stream_value), non_blocking); });
    }

    /**
     * @brief 执行特征前向（NumPy 路径，输出由绑定层自动分配）。
     * @param inputs 输入张量；单输入可直接传 `numpy.ndarray`，多输入可传 `list/tuple/dict`。
     * @param non_blocking 是否以非阻塞方式提交 CUDA 拷贝与 enqueue。
     * @return 单输出返回 `numpy.ndarray`，多输出返回 `dict[str, numpy.ndarray]`。
     */
    py::object forwardFeatures(const py::object &inputs, bool non_blocking = false)
    {
        const auto output_names = outputTensorNames();
        if (output_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "No output tensors are configured");
        }

        const auto stream = model_->resolveExecutionStream();
        return execute(
            inputs, py::none(), output_names, [this, stream, non_blocking](const std::vector<void *> &buffers)
            { model_->forwardFeatures(buffers, stream, non_blocking); }, stream);
    }

private:
    /**
     * @brief 将 Python 输入对象规整为有序的输入张量数组列表。
     * @param tensors Python 输入对象。
     * @param tensor_names 张量名称列表。
     * @param tensor_kind 张量类型描述（用于错误信息）。
     * @param allow_any_single 是否允许任意单个对象（V2 模式），false 则仅允许 numpy.ndarray。
     * @return 与配置输入顺序一致的数组列表。
     */
    std::vector<NamedTensorObject> normalizeTensorObjects(const py::object               &tensors,
                                                          const std::vector<std::string> &tensor_names,
                                                          const char *tensor_kind, bool allow_any_single = false) const
    {
        if (tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model has no configured %s", tensor_kind);
        }

        if (tensor_names.size() == 1)
        {
            if (py::isinstance<py::array>(tensors))
            {
                return {
                    {tensor_names.front(), tensors}
                };
            }
            if (allow_any_single && !py::isinstance<py::dict>(tensors) && !py::isinstance<py::sequence>(tensors))
            {
                return {
                    {tensor_names.front(), tensors}
                };
            }
        }

        if (py::isinstance<py::dict>(tensors))
        {
            py::dict                       mapping = py::reinterpret_borrow<py::dict>(tensors);
            std::vector<NamedTensorObject> values;
            values.reserve(tensor_names.size());

            for (const auto &item : mapping)
            {
                if (!py::isinstance<py::str>(item.first))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s tensor dict keys must be str",
                                         tensor_kind);
                }

                if (!allow_any_single)
                {
                    const auto tensor_name = item.first.cast<std::string>();
                    if (std::find(tensor_names.begin(), tensor_names.end(), tensor_name) == tensor_names.end())
                    {
                        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected %s tensor: %s",
                                             tensor_kind, tensor_name.c_str());
                    }
                }
            }

            for (const auto &tensor_name : tensor_names)
            {
                if (!mapping.contains(py::str(tensor_name)))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Missing %s tensor: %s", tensor_kind,
                                         tensor_name.c_str());
                }
                values.push_back({tensor_name, mapping[py::str(tensor_name)]});
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

            std::vector<NamedTensorObject> values;
            values.reserve(tensor_names.size());
            size_t index = 0;
            for (const auto &item : sequence)
            {
                values.push_back({tensor_names[index++], item});
            }
            return values;
        }

        const char *expected_types = allow_any_single ? "a tensor object, sequence, or dict[str, tensor]"
                                                      : "numpy.ndarray, sequence, or dict[str, numpy.ndarray]";
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s tensors must be %s", tensor_kind, expected_types);
    }

    /**
     * @brief 校验调用方提供的输出 NumPy 数组是否满足 engine 要求。
     * @param tensor_name 张量名称。
     * @param array 输出数组。
     * @param expected_dtype 期望的 TensorRT 数据类型。
     * @param expected_shape 期望的运行时形状。
     * @throws irt::Exception 不可写、非 C 连续、dtype/shape/大小不匹配时抛出。
     */
    void validateOutputArray(const std::string &tensor_name, const py::array &array, nvinfer1::DataType expected_dtype,
                             const std::vector<int64_t> &expected_shape) const
    {
        if (!array.writeable())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor is not writable: %s",
                                 tensor_name.c_str());
        }
        if (!isCContiguous(array))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be C-contiguous: %s",
                                 tensor_name.c_str());
        }
        if (!array.dtype().equal(dataTypeToPyDType(expected_dtype)))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor dtype mismatch: %s, expected=%s",
                                 tensor_name.c_str(), dataTypeToString(expected_dtype).c_str());
        }
        if (static_cast<size_t>(array.ndim()) != expected_shape.size())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Output tensor rank mismatch: %s, expected=%zu, got=%zd", tensor_name.c_str(),
                                 expected_shape.size(), array.ndim());
        }
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            if (array.shape(static_cast<py::ssize_t>(i)) != expected_shape[i])
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Output tensor shape mismatch: %s, dim[%zu] expected=%lld, got=%zd",
                                     tensor_name.c_str(), i, static_cast<long long>(expected_shape[i]),
                                     array.shape(static_cast<py::ssize_t>(i)));
            }
        }
        if (static_cast<size_t>(array.nbytes())
            != tensorElementCount(expected_shape, tensor_name) * dataTypeSize(expected_dtype))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected numpy buffer size for tensor: %s",
                                 tensor_name.c_str());
        }
    }

    /**
     * @brief 将 Python 传入的 stream 参数解析为整数指针。
     * @param stream_ptr `None` 或 `int` 形式的 CUDA stream 地址。
     * @return stream 指针值；`None` 时返回 0。
     */
    uintptr_t parseStreamPtr(const py::object &stream_ptr) const
    {
        if (stream_ptr.is_none())
        {
            return 0;
        }

        return stream_ptr.cast<uintptr_t>();
    }

    /**
     * @brief 准备输入张量，并将运行时形状同步到模型上下文。
     * @param inputs Python 输入对象。
     * @return 已规整好的输入张量绑定列表。
     */
    std::vector<TensorBinding> prepareInputBindings(const py::object &inputs)
    {
        const auto input_values = normalizeTensorObjects(inputs, inputTensorNames(), "input");

        std::vector<TensorBinding> bindings;
        bindings.reserve(input_values.size());

        for (const auto &input_value : input_values)
        {
            const auto &input_name = input_value.name;
            const auto  data_type  = model_->tensorDataType(input_name);
            auto        array      = asContiguousArray(input_value.object, data_type);

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
     * @brief 根据运行时张量信息准备输出绑定。
     * @param outputs Python 输出对象；`None` 时自动分配 ndarray。
     * @param tensor_names 待绑定的输出张量名称列表。
     * @return 输出张量绑定列表。
     */
    std::vector<TensorBinding> prepareOutputBindings(const py::object               &outputs,
                                                     const std::vector<std::string> &tensor_names)
    {
        std::vector<TensorBinding> bindings;
        bindings.reserve(tensor_names.size());

        if (outputs.is_none())
        {
            for (const auto &tensor_name : tensor_names)
            {
                const auto dims      = model_->tensorShape(tensor_name);
                const auto data_type = model_->tensorDataType(tensor_name);
                const auto shape     = dimsToVector(dims);

                std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
                py::array                array(dataTypeToPyDType(data_type), py_shape);
                if (static_cast<size_t>(array.nbytes())
                    != tensorElementCount(shape, tensor_name) * dataTypeSize(data_type))
                {
                    throw irt::Exception(irt::Status::ERROR_INTERNAL, "Unexpected numpy buffer size for tensor: %s",
                                         tensor_name.c_str());
                }

                bindings.emplace_back(tensor_name, std::move(array));
            }
            return bindings;
        }

        const auto output_values = normalizeTensorObjects(outputs, tensor_names, "output");
        for (const auto &output_value : output_values)
        {
            const auto &tensor_name = output_value.name;
            if (!py::isinstance<py::array>(output_value.object))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Output tensor must be numpy.ndarray: %s",
                                     tensor_name.c_str());
            }

            py::array  array     = py::reinterpret_borrow<py::array>(output_value.object);
            const auto data_type = model_->tensorDataType(tensor_name);
            const auto shape     = dimsToVector(model_->tensorShape(tensor_name));
            validateOutputArray(tensor_name, array, data_type, shape);
            bindings.emplace_back(tensor_name, std::move(array));
        }

        return bindings;
    }

    /**
     * @brief 校验 DLPack 张量的 dtype、秩、布局与可选的期望形状。
     * @param tensor_name 张量名称。
     * @param tensor DLPack `DLTensor` 视图。
     * @param expected_dtype 期望的 TensorRT 数据类型。
     * @param expected_shape 期望形状；为 nullptr 时仅校验 dtype 与布局。
     */
    void validateDLPackTensor(const std::string &tensor_name, const DLTensor &tensor, nvinfer1::DataType expected_dtype,
                              const std::vector<int64_t> *expected_shape) const
    {
        const auto actual_dtype = dlDataTypeToTrt(tensor.dtype);
        if (actual_dtype != expected_dtype)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor dtype mismatch: %s, expected=%s",
                                 tensor_name.c_str(), dataTypeToString(expected_dtype).c_str());
        }

        if (tensor.ndim < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor rank is invalid: %s",
                                 tensor_name.c_str());
        }
        if (tensor.ndim > 0 && tensor.shape == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor shape pointer is null: %s",
                                 tensor_name.c_str());
        }
        const auto actual_shape = dlShapeToVector(tensor);
        if (tensor.ndim > 0 && tensor.strides != nullptr && !isCompactRowMajor(tensor))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor must be compact row-major for InferRT: %s", tensor_name.c_str());
        }
        if (actual_shape.size() > 8)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Too many dimensions: %zu", actual_shape.size());
        }
        if (expected_shape != nullptr)
        {
            if (actual_shape.size() != expected_shape->size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Tensor rank mismatch: %s, expected=%zu, got=%zu", tensor_name.c_str(),
                                     expected_shape->size(), actual_shape.size());
            }

            for (size_t i = 0; i < actual_shape.size(); ++i)
            {
                if (actual_shape[i] != expected_shape->at(i))
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                         "Tensor shape mismatch: %s, dim[%zu] expected=%lld, got=%lld",
                                         tensor_name.c_str(), i, static_cast<long long>(expected_shape->at(i)),
                                         static_cast<long long>(actual_shape[i]));
                }
            }
        }
    }

    /**
     * @brief 根据 DLPack 张量 dtype 与 shape 计算数据区字节数。
     * @param tensor_name 张量名称，用于报错。
     * @param tensor DLPack `DLTensor`。
     * @return 数据区总字节数（含各维乘积）。
     */
    size_t dlTensorNumBytes(const std::string &tensor_name, const DLTensor &tensor) const
    {
        if (tensor.dtype.lanes != 1)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported DLPack lanes=%u for tensor: %s",
                                 static_cast<unsigned>(tensor.dtype.lanes), tensor_name.c_str());
        }
        if (tensor.dtype.bits % 8 != 0)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "Unsupported non-byte-aligned DLPack dtype bits=%u for tensor: %s",
                                 static_cast<unsigned>(tensor.dtype.bits), tensor_name.c_str());
        }

        size_t num_bytes = static_cast<size_t>(tensor.dtype.bits / 8);
        for (int32_t i = 0; i < tensor.ndim; ++i)
        {
            if (tensor.shape[i] < 0)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "DLPack tensor has negative dimension at %d for tensor: %s", i,
                                     tensor_name.c_str());
            }
            num_bytes = multiplyChecked(num_bytes, static_cast<size_t>(tensor.shape[i]), tensor_name);
        }
        return num_bytes;
    }

    /**
     * @brief 为单个张量构建 V2 绑定（NumPy 或 DLPack）。
     * @param tensor_name 张量名称。
     * @param object Python 张量对象。
     * @param expected_dtype 期望数据类型。
     * @param expected_shape 输出张量的期望形状；输入可为 nullptr。
     * @param stream_ptr 传给 `__dlpack__(stream=...)` 的 CUDA stream 指针。
     * @param is_input true 表示输入张量，false 表示输出张量。
     * @return 填充好的 `TensorBindingV2`。
     */
    TensorBindingV2 makeDLPackBinding(const std::string &tensor_name, const py::handle &object,
                                      nvinfer1::DataType expected_dtype, const std::vector<int64_t> *expected_shape,
                                      uintptr_t stream_ptr, bool is_input)
    {
        TensorBindingV2 binding;
        binding.name   = tensor_name;
        binding.object = py::reinterpret_borrow<py::object>(object);

        if (py::isinstance<py::array>(object))
        {
            py::array array = py::reinterpret_borrow<py::array>(object);
            if (is_input)
            {
                array = asContiguousArray(array, expected_dtype);
            }
            else
            {
                if (expected_shape == nullptr)
                {
                    throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                         "Expected output shape must be available for tensor: %s", tensor_name.c_str());
                }
                validateOutputArray(tensor_name, array, expected_dtype, *expected_shape);
            }

            binding.host_array           = std::move(array);
            binding.host_ptr             = binding.host_array.mutable_data();
            binding.num_bytes            = static_cast<size_t>(binding.host_array.nbytes());
            binding.staging_buffer       = DeviceBuffer(binding.num_bytes);
            binding.device_ptr           = binding.staging_buffer.data();
            binding.copy_input_to_device = is_input;
            binding.copy_output_to_host  = !is_input;
            return binding;
        }

        if (!hasDLPackProtocol(object))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor object must be numpy.ndarray or implement __dlpack__(): %s",
                                 tensor_name.c_str());
        }

        binding.dlpack_capsule     = callDLPack(object, stream_ptr);
        auto       *managed_tensor = consumeDLPackCapsulePointer(binding.dlpack_capsule);
        const auto &tensor         = managed_tensor->dl_tensor;

        validateDLPackTensor(tensor_name, tensor, expected_dtype, expected_shape);
        binding.num_bytes = dlTensorNumBytes(tensor_name, tensor);

        auto *base_ptr = static_cast<char *>(tensor.data);
        if (base_ptr == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor data pointer is null: %s",
                                 tensor_name.c_str());
        }

        void *tensor_ptr = base_ptr + tensor.byte_offset;
        if (dlDeviceIsCuda(tensor.device))
        {
            if (tensor.device.device_id != currentCudaDevice())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Tensor CUDA device mismatch: %s, expected=%d, got=%d", tensor_name.c_str(),
                                     currentCudaDevice(), tensor.device.device_id);
            }
            binding.device_ptr = tensor_ptr;
            return binding;
        }

        if (dlDeviceIsHost(tensor.device))
        {
            binding.host_ptr             = tensor_ptr;
            binding.staging_buffer       = DeviceBuffer(binding.num_bytes);
            binding.device_ptr           = binding.staging_buffer.data();
            binding.copy_input_to_device = is_input;
            binding.copy_output_to_host  = !is_input;
            return binding;
        }

        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported DLPack device type=%d for tensor: %s",
                             static_cast<int>(tensor.device.device_type), tensor_name.c_str());
    }

    /**
     * @brief 准备 V2 输入绑定，并将运行时形状写入模型上下文。
     * @param inputs Python 输入（单张量、sequence 或 dict）。
     * @param stream_ptr 可选 CUDA stream 指针。
     * @return 输入绑定列表，顺序与 `inputTensorNames()` 一致。
     */
    std::vector<TensorBindingV2> prepareInputBindingsV2(const py::object &inputs, uintptr_t stream_ptr)
    {
        const auto input_values = normalizeTensorObjects(inputs, inputTensorNames(), "input", true);

        std::vector<TensorBindingV2> bindings;
        bindings.reserve(input_values.size());

        for (const auto &input_value : input_values)
        {
            const auto     &input_name = input_value.name;
            const auto      data_type  = model_->tensorDataType(input_name);
            TensorBindingV2 binding
                = makeDLPackBinding(input_name, input_value.object, data_type, nullptr, stream_ptr, true);

            std::vector<int64_t> shape;
            if (py::isinstance<py::array>(binding.object))
            {
                shape.reserve(static_cast<size_t>(binding.host_array.ndim()));
                for (py::ssize_t i = 0; i < binding.host_array.ndim(); ++i)
                {
                    shape.push_back(binding.host_array.shape(i));
                }
            }
            else
            {
                const auto *managed_tensor = consumeDLPackCapsulePointer(binding.dlpack_capsule);
                shape                      = dlShapeToVector(managed_tensor->dl_tensor);
            }

            model_->setTensorShape(input_name, vectorToDims(shape));
            bindings.push_back(std::move(binding));
        }

        return bindings;
    }

    /**
     * @brief 准备 V2 输出绑定；输出必须由调用方显式提供。
     * @param outputs Python 输出张量集合。
     * @param tensor_names 待绑定的输出张量名称列表。
     * @param stream_ptr 可选 CUDA stream 指针。
     * @return 输出绑定列表。
     * @throws irt::Exception `outputs` 为 `None` 时抛出。
     */
    std::vector<TensorBindingV2> prepareOutputBindingsV2(const py::object               &outputs,
                                                         const std::vector<std::string> &tensor_names,
                                                         uintptr_t                       stream_ptr)
    {
        if (outputs.is_none())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "infer_v2 requires caller-provided output tensors");
        }

        const auto                   output_values = normalizeTensorObjects(outputs, tensor_names, "output", true);
        std::vector<TensorBindingV2> bindings;
        bindings.reserve(output_values.size());

        for (const auto &output_value : output_values)
        {
            const auto &tensor_name = output_value.name;
            const auto  dims        = model_->tensorShape(tensor_name);
            const auto  data_type   = model_->tensorDataType(tensor_name);
            const auto  shape       = dimsToVector(dims);

            bindings.push_back(
                makeDLPackBinding(tensor_name, output_value.object, data_type, &shape, stream_ptr, false));
        }

        return bindings;
    }

    /**
     * @brief 将输入输出张量拼装成 InferRT 所需的缓冲区指针数组。
     * @tparam BindingType `TensorBinding` 或 `TensorBindingV2`。
     * @param inputs 输入张量绑定列表。
     * @param outputs 输出张量绑定列表。
     * @return 先输入后输出的 `void*` 指针数组，顺序与 engine 绑定一致。
     */
    template<typename BindingType>
    std::vector<void *> buildBufferList(const std::vector<BindingType> &inputs,
                                        const std::vector<BindingType> &outputs) const
    {
        std::vector<void *> buffers;
        buffers.reserve(inputs.size() + outputs.size());
        for (const auto &input : inputs)
        {
            if constexpr (std::is_same_v<BindingType, TensorBinding>)
            {
                buffers.push_back(input.device_buffer.data());
            }
            else
            {
                buffers.push_back(input.device_ptr);
            }
        }
        for (const auto &output : outputs)
        {
            if constexpr (std::is_same_v<BindingType, TensorBinding>)
            {
                buffers.push_back(output.device_buffer.data());
            }
            else
            {
                buffers.push_back(output.device_ptr);
            }
        }
        return buffers;
    }

    /**
     * @brief 将宿主侧输入拷贝到设备侧（按需跳过已在 GPU 的 V2 输入）。
     * @tparam BindingType `TensorBinding` 或 `TensorBindingV2`。
     * @param inputs 输入张量绑定列表。
     * @param stream 异步拷贝使用的 CUDA stream，可为 nullptr。
     */
    template<typename BindingType>
    void copyInputsToDevice(const std::vector<BindingType> &inputs, cudaStream_t stream = nullptr) const
    {
        for (const auto &input : inputs)
        {
            if constexpr (std::is_same_v<BindingType, TensorBindingV2>)
            {
                if (!input.copy_input_to_device)
                {
                    continue;
                }
                IRT_CHECK_THROW(
                    cudaMemcpyAsync(input.device_ptr, input.host_ptr, input.num_bytes, cudaMemcpyHostToDevice, stream),
                    "Failed to copy input tensor to device: %s", input.name.c_str());
            }
            else
            {
                IRT_CHECK_THROW(cudaMemcpyAsync(input.device_buffer.data(), input.host_array.data(),
                                                input.device_buffer.size(), cudaMemcpyHostToDevice, stream),
                                "Failed to copy input tensor to device: %s", input.name.c_str());
            }
        }
    }

    /**
     * @brief 将设备侧输出拷贝回宿主侧（按需跳过结果留在 GPU 的 V2 输出）。
     * @tparam BindingType `TensorBinding` 或 `TensorBindingV2`。
     * @param outputs 输出张量绑定列表。
     * @param stream 异步拷贝使用的 CUDA stream，可为 nullptr。
     */
    template<typename BindingType>
    void copyOutputsToHost(std::vector<BindingType> &outputs, cudaStream_t stream = nullptr) const
    {
        for (auto &output : outputs)
        {
            if constexpr (std::is_same_v<BindingType, TensorBindingV2>)
            {
                if (!output.copy_output_to_host)
                {
                    continue;
                }
                IRT_CHECK_THROW(cudaMemcpyAsync(output.host_ptr, output.device_ptr, output.num_bytes,
                                                cudaMemcpyDeviceToHost, stream),
                                "Failed to copy output tensor to host: %s", output.name.c_str());
            }
            else
            {
                IRT_CHECK_THROW(cudaMemcpyAsync(output.host_array.mutable_data(), output.device_buffer.data(),
                                                output.device_buffer.size(), cudaMemcpyDeviceToHost, stream),
                                "Failed to copy output tensor to host: %s", output.name.c_str());
            }
        }
    }

    /**
     * @brief 执行一次通用的 NumPy ↔ TensorRT 前向。
     * @param inputs Python 输入对象。
     * @param outputs Python 输出对象，可为 `None`。
     * @param output_names 输出张量名称列表。
     * @param forward_fn 真正执行前向的函数（如 `IModel::infer`）。
     * @param stream 用于 H2D/D2H 与同步的 CUDA stream。
     * @return 打包后的输出 ndarray 或 dict。
     */
    py::object execute(const py::object &inputs, const py::object &outputs,
                       const std::vector<std::string>                         &output_names,
                       const std::function<void(const std::vector<void *> &)> &forward_fn, cudaStream_t stream)
    {
        auto input_bindings  = prepareInputBindings(inputs);
        auto output_bindings = prepareOutputBindings(outputs, output_names);
        auto buffers         = buildBufferList(input_bindings, output_bindings);

        {
            py::gil_scoped_release release;
            copyInputsToDevice(input_bindings, stream);
            forward_fn(buffers);
            copyOutputsToHost(output_bindings, stream);
            IRT_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");
        }

        return packOutputs(output_bindings);
    }

    /**
     * @brief 执行一次 DLPack / 可选 stream 的通用前向。
     * @param inputs Python 输入对象。
     * @param outputs 调用方提供的输出张量（原样返回）。
     * @param output_names 输出张量名称列表。
     * @param stream_value CUDA stream 指针整数值，0 表示默认流。
     * @param forward_fn 实际调用 `IModel::infer` 或 `forwardFeatures` 的回调。
     * @return 与传入的 `outputs` 相同的 Python 对象。
     */
    py::object executeV2(const py::object &inputs, const py::object &outputs,
                         const std::vector<std::string> &output_names, uintptr_t stream_value,
                         const std::function<void(const std::vector<void *> &)> &forward_fn)
    {
        auto input_bindings  = prepareInputBindingsV2(inputs, stream_value);
        auto output_bindings = prepareOutputBindingsV2(outputs, output_names, stream_value);
        auto buffers         = buildBufferList(input_bindings, output_bindings);
        auto stream          = reinterpret_cast<cudaStream_t>(stream_value);

        {
            py::gil_scoped_release release;
            copyInputsToDevice(input_bindings, stream);
            forward_fn(buffers);
            copyOutputsToHost(output_bindings, stream);
            IRT_CHECK_THROW(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");
        }

        return py::reinterpret_borrow<py::object>(outputs);
    }

    std::unique_ptr<irt::model::IModel> model_; ///< 底层 C++ 模型实例。
};

} // namespace

/**
 * @brief InferRT 模型 Python 绑定模块。
 * @param m Python 模块对象。
 */
PYBIND11_MODULE(inferrt_model_py, m)
{
    m.doc() = "InferRT 模型 Python 扩展模块（pybind11）。";

    /** Python 侧 InferRT 异常类型，对应 C++ `irt::Exception`。 */
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
        .def_property("feature_only", &irt::model::IModelConfig::featureOnly, &irt::model::IModelConfig::setFeatureOnly,
                      "是否仅构建特征提取网络。");

    py::class_<PyModel>(m, "Model", "InferRT Python 模型包装器。")
        .def("name", &PyModel::name, "返回模型名称。")
        .def("build", &PyModel::build, py::arg("weights_file"), "从权重文件构建 TensorRT engine。")
        .def("load", &PyModel::load, py::arg("engine_file"), "从 engine 文件加载模型。")
        .def("save", &PyModel::save, py::arg("engine_file"), "将当前 engine 序列化到磁盘。")
        .def("build_or_load", &PyModel::buildOrLoad, py::arg("weights_file"),
             "优先加载已有 engine，不存在时从权重文件构建。")
        .def("infer", &PyModel::infer, py::arg("inputs"), py::arg("outputs"), py::arg("non_blocking") = false,
             "执行一次主推理，由 Python 传入输入和输出 ndarray、sequence 或 dict。")
        .def("infer_v2", &PyModel::inferV2, py::arg("inputs"), py::arg("outputs"), py::arg("stream_ptr") = py::none(),
             py::arg("non_blocking") = false,
             "使用 DLPack 张量执行主推理；输出须由调用方提供，可选传入 CUDA stream 指针。")
        .def("forward_features", &PyModel::forwardFeatures, py::arg("inputs"), py::arg("non_blocking") = false,
             "执行一次特征前向，输入支持 ndarray、sequence 或 dict。")
        .def("input_tensor_names", &PyModel::inputTensorNames, "返回输入张量名称列表。")
        .def("output_tensor_names", &PyModel::outputTensorNames, "返回输出张量名称列表（分类 logits 或特征导出名）。")
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
