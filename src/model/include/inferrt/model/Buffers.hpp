#pragma once

#include <NvInfer.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Export.h>
#include <inferrt/model/Utils.hpp>

#include <cstddef>

namespace irt::model {

/**
 * @brief CUDA 设备内存分配器。
 */
class INFERRT_MODEL_API DeviceAllocator
{
public:
    /**
     * @brief 分配指定字节数的设备内存。
     * @param ptr 输出参数，接收设备指针。
     * @param num_bytes 需要分配的字节数。
     * @return 分配成功返回 true，否则返回 false。
     */
    bool operator()(void **ptr, size_t num_bytes) const noexcept;
};

/**
 * @brief CUDA 设备内存释放器。
 */
class INFERRT_MODEL_API DeviceFree
{
public:
    /**
     * @brief 释放设备内存。
     * @param ptr 待释放的设备指针；允许为 nullptr。
     */
    void operator()(void *ptr) const noexcept;
};

/**
 * @brief 主机内存分配器。
 *
 * 当前实现使用标准 `malloc`，用于与 TensorRT sample 的 HostBuffer 语义保持一致。
 */
class INFERRT_MODEL_API HostAllocator
{
public:
    /**
     * @brief 分配指定字节数的主机内存。
     * @param ptr 输出参数，接收主机指针。
     * @param num_bytes 需要分配的字节数。
     * @return 分配成功返回 true，否则返回 false。
     */
    bool operator()(void **ptr, size_t num_bytes) const noexcept;
};

/**
 * @brief 主机内存释放器。
 */
class INFERRT_MODEL_API HostFree
{
public:
    /**
     * @brief 释放主机内存。
     * @param ptr 待释放的主机指针；允许为 nullptr。
     */
    void operator()(void *ptr) const noexcept;
};

/**
 * @brief 通用缓冲区 RAII 模板。
 *
 * 该模板参考 TensorRT `samples/common/buffers.h` 中的 GenericBuffer 设计，
 * 通过分配器与释放器模板参数抽象不同内存位置。缓冲区以“元素个数 + TensorRT 数据类型”
 * 描述大小，并在内部转换为字节数执行真实分配。
 *
 * @tparam AllocFunc 分配器类型，需实现 `bool operator()(void **, size_t)`。
 * @tparam FreeFunc 释放器类型，需实现 `void operator()(void *)`。
 */
template<typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    /**
     * @brief 构造空缓冲区。
     * @param data_type 缓冲区元素类型，默认 float32。
     */
    explicit GenericBuffer(nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT)
        : data_type_(data_type)
    {
    }

    /**
     * @brief 构造并按元素数量分配缓冲区。
     * @param size 元素数量，而不是字节数。
     * @param data_type 缓冲区元素类型。
     */
    GenericBuffer(size_t size, nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT)
        : data_type_(data_type)
    {
        resize(size);
    }

    /**
     * @brief 析构并释放已持有的内存。
     */
    ~GenericBuffer()
    {
        reset();
    }

    /** @brief 禁止拷贝构造，避免重复释放底层内存。 */
    GenericBuffer(const GenericBuffer &) = delete;

    /** @brief 禁止拷贝赋值，避免重复释放底层内存。 */
    GenericBuffer &operator=(const GenericBuffer &) = delete;

    /**
     * @brief 移动构造，转移底层内存所有权。
     * @param other 被移动对象。
     */
    GenericBuffer(GenericBuffer &&other) noexcept
        : size_(other.size_)
        , capacity_(other.capacity_)
        , data_type_(other.data_type_)
        , data_(other.data_)
    {
        other.size_      = 0;
        other.capacity_  = 0;
        other.data_type_ = nvinfer1::DataType::kFLOAT;
        other.data_      = nullptr;
    }

    /**
     * @brief 移动赋值，释放当前内存后接管另一个缓冲区。
     * @param other 被移动对象。
     * @return 当前对象引用。
     */
    GenericBuffer &operator=(GenericBuffer &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            size_            = other.size_;
            capacity_        = other.capacity_;
            data_type_       = other.data_type_;
            data_            = other.data_;
            other.size_      = 0;
            other.capacity_  = 0;
            other.data_type_ = nvinfer1::DataType::kFLOAT;
            other.data_      = nullptr;
        }
        return *this;
    }

    /**
     * @brief 获取底层内存指针。
     * @return 内存指针；空缓冲区返回 nullptr。
     */
    void *data() noexcept
    {
        return data_;
    }

    /**
     * @brief 获取底层只读内存指针。
     * @return 只读内存指针；空缓冲区返回 nullptr。
     */
    const void *data() const noexcept
    {
        return data_;
    }

    /**
     * @brief 获取底层内存指针。
     *
     * 该函数用于兼容需要 `get()` 命名的调用点，新代码优先使用 `data()`。
     *
     * @return 内存指针；空缓冲区返回 nullptr。
     */
    void *get() noexcept
    {
        return data();
    }

    /**
     * @brief 获取底层只读内存指针。
     * @return 只读内存指针；空缓冲区返回 nullptr。
     */
    const void *get() const noexcept
    {
        return data();
    }

    /**
     * @brief 获取当前逻辑元素数量。
     * @return 元素数量，而不是字节数。
     */
    size_t size() const noexcept
    {
        return size_;
    }

    /**
     * @brief 获取当前容量。
     * @return 以元素数量表示的容量。
     */
    size_t capacity() const noexcept
    {
        return capacity_;
    }

    /**
     * @brief 判断缓冲区是否为空。
     * @return 未分配内存或逻辑大小为 0 时返回 true。
     */
    bool empty() const noexcept
    {
        return data_ == nullptr || size_ == 0;
    }

    /**
     * @brief 获取缓冲区元素类型。
     * @return TensorRT 数据类型。
     */
    nvinfer1::DataType dataType() const noexcept
    {
        return data_type_;
    }

    /**
     * @brief 获取逻辑大小对应的字节数。
     * @return `size() * elementSize(dataType())`。
     */
    size_t sizeBytes() const
    {
        return size_ * elementSize(data_type_);
    }

    /**
     * @brief 获取逻辑大小对应的字节数。
     * @return 与 `sizeBytes()` 相同。
     */
    size_t nbBytes() const
    {
        return sizeBytes();
    }

    /**
     * @brief 按元素数量调整逻辑大小。
     *
     * 当新大小不超过当前容量时不会重新分配；超过容量时会申请新内存并释放旧内存。
     *
     * @param new_size 新元素数量。
     */
    void resize(size_t new_size)
    {
        resize(new_size, data_type_);
    }

    /**
     * @brief 按元素数量和数据类型调整逻辑大小。
     *
     * 若数据类型发生变化，会重新计算所需字节数；当当前容量不足时重新分配。
     *
     * @param new_size 新元素数量。
     * @param data_type 新元素类型。
     */
    void resize(size_t new_size, nvinfer1::DataType data_type)
    {
        const bool type_changed = data_type_ != data_type;
        if (new_size == 0)
        {
            if (type_changed)
            {
                reset();
            }
            data_type_ = data_type;
            size_      = 0;
            return;
        }
        if (!type_changed && capacity_ >= new_size)
        {
            size_ = new_size;
            return;
        }

        reallocate(new_size, data_type);
        data_type_ = data_type;
        size_      = new_size;
    }

    /**
     * @brief 按 TensorRT 维度调整逻辑大小。
     * @param dims TensorRT 维度对象，各维度必须为正数。
     */
    void resize(const nvinfer1::Dims &dims)
    {
        resize(irt::model::elementCount(dims));
    }

    /**
     * @brief 按 TensorRT 维度和数据类型调整逻辑大小。
     * @param dims TensorRT 维度对象，各维度必须为正数。
     * @param data_type 新元素类型。
     */
    void resize(const nvinfer1::Dims &dims, nvinfer1::DataType data_type)
    {
        resize(irt::model::elementCount(dims), data_type);
    }

    /**
     * @brief 释放当前内存并清空缓冲区。
     */
    void reset() noexcept
    {
        free_fn_(data_);
        data_     = nullptr;
        size_     = 0;
        capacity_ = 0;
    }

private:
    void reallocate(size_t new_capacity, nvinfer1::DataType data_type)
    {
        void      *new_data = nullptr;
        const auto bytes    = new_capacity * elementSize(data_type);
        if (!alloc_fn_(&new_data, bytes))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to allocate %zu bytes", bytes);
        }

        free_fn_(data_);
        data_     = new_data;
        capacity_ = new_capacity;
    }

    ///< 当前逻辑元素数量。
    size_t size_{0};

    ///< 当前容量，以元素数量表示。
    size_t capacity_{0};

    ///< 缓冲区元素类型。
    nvinfer1::DataType data_type_{nvinfer1::DataType::kFLOAT};

    ///< 底层内存指针。
    void *data_{nullptr};

    ///< 内存分配器。
    AllocFunc alloc_fn_{};

    ///< 内存释放器。
    FreeFunc free_fn_{};
};

/**
 * @brief 设备端缓冲区类型。
 *
 * 使用 CUDA `cudaMalloc/cudaFree` 管理显存。
 */
using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;

/**
 * @brief 主机端缓冲区类型。
 *
 * 使用标准 `malloc/free` 管理主机内存。
 */
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

} // namespace irt::model
