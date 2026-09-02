#pragma once

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/model/Export.h>

#include <cstddef>
#include <cstdlib>

namespace irt::model {

/** Allocates device memory for a core tensor buffer. */
class INFERRT_MODEL_API DeviceAllocator
{
public:
    bool operator()(void **ptr, size_t bytes) const noexcept;
};

/** Releases device memory allocated by DeviceAllocator. */
class INFERRT_MODEL_API DeviceFree
{
public:
    void operator()(void *ptr) const noexcept;
};

/** Allocates ordinary host memory. */
class INFERRT_MODEL_API HostAllocator
{
public:
    bool operator()(void **ptr, size_t bytes) const noexcept;
};

/** Releases ordinary host memory. */
class INFERRT_MODEL_API HostFree
{
public:
    void operator()(void *ptr) const noexcept;
};

/** Allocates page-locked host memory. */
class INFERRT_MODEL_API PinnedHostAllocator
{
public:
    bool operator()(void **ptr, size_t bytes) const noexcept;
};

/** Releases page-locked host memory. */
class INFERRT_MODEL_API PinnedHostFree
{
public:
    void operator()(void *ptr) const noexcept;
};

/** RAII buffer whose size and type are expressed using core tensor types. */
template<typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    explicit GenericBuffer(irt::TensorDataType type = irt::TensorDataType::F32) noexcept : data_type_(type) {}
    GenericBuffer(size_t elements, irt::TensorDataType type = irt::TensorDataType::F32) : data_type_(type)
    {
        resize(elements);
    }
    ~GenericBuffer() { reset(); }

    GenericBuffer(const GenericBuffer &) = delete;
    GenericBuffer &operator=(const GenericBuffer &) = delete;
    GenericBuffer(GenericBuffer &&other) noexcept
        : size_(other.size_), capacity_(other.capacity_), data_type_(other.data_type_), data_(other.data_)
    {
        other.size_ = other.capacity_ = 0;
        other.data_ = nullptr;
    }
    GenericBuffer &operator=(GenericBuffer &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_type_ = other.data_type_;
            data_ = other.data_;
            other.size_ = other.capacity_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] void *data() noexcept { return data_; }
    [[nodiscard]] const void *data() const noexcept { return data_; }
    [[nodiscard]] void *get() noexcept { return data_; }
    [[nodiscard]] const void *get() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr || size_ == 0; }
    [[nodiscard]] irt::TensorDataType dataType() const noexcept { return data_type_; }
    [[nodiscard]] size_t sizeBytes() const { return irt::checkedSizeMul(size_, irt::dataTypeSize(data_type_), "Buffer bytes"); }
    [[nodiscard]] size_t nbBytes() const { return sizeBytes(); }

    void resize(size_t elements) { resize(elements, data_type_); }
    void resize(size_t elements, irt::TensorDataType type)
    {
        if (elements == 0)
        {
            if (type != data_type_) reset();
            data_type_ = type;
            size_ = 0;
            return;
        }
        if (type == data_type_ && elements <= capacity_)
        {
            size_ = elements;
            return;
        }
        void *new_data = nullptr;
        const auto bytes = irt::checkedSizeMul(elements, irt::dataTypeSize(type), "Buffer bytes");
        if (!alloc_fn_(&new_data, bytes))
        {
            throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY, "Failed to allocate %zu bytes", bytes);
        }
        free_fn_(data_);
        data_ = new_data;
        size_ = capacity_ = elements;
        data_type_ = type;
    }
    void resize(const irt::Shape &shape) { resize(shape.elementCount()); }
    void resize(const irt::Shape &shape, irt::TensorDataType type) { resize(shape.elementCount(), type); }
    void reset() noexcept
    {
        free_fn_(data_);
        data_ = nullptr;
        size_ = capacity_ = 0;
    }

private:
    size_t          size_{0};
    size_t          capacity_{0};
    irt::TensorDataType data_type_{irt::TensorDataType::F32};
    void            *data_{nullptr};
    AllocFunc        alloc_fn_{};
    FreeFunc         free_fn_{};
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;
using PinnedHostBuffer = GenericBuffer<PinnedHostAllocator, PinnedHostFree>;

} // namespace irt::model
