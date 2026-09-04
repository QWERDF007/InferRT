#pragma once

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.hpp>

#include <cstddef>
#include <cstdint>
#include <ios>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace irt {

enum class TensorDataType
{
    U8,
    I8,
    F16,
    F32,
    I32,
    I64,
    Bool,
};

using DType = TensorDataType;

enum class TensorLayout
{
    HWC,
    NCHW,
    NHWC,
    CHW,
    Opaque,
};

using Layout = TensorLayout;

enum class MemoryKind
{
    HOST,
    DEVICE,
    PINNED_HOST,
};

[[nodiscard]] inline constexpr size_t dataTypeSize(TensorDataType type) noexcept
{
    switch (type)
    {
    case TensorDataType::U8:
    case TensorDataType::I8:
    case TensorDataType::Bool:
        return 1;
    case TensorDataType::F16:
        return 2;
    case TensorDataType::F32:
    case TensorDataType::I32:
        return 4;
    case TensorDataType::I64:
        return 8;
    }
    return 0;
}

/** Checked size arithmetic shared by tensor, allocation and backend adapters. */
[[nodiscard]] inline size_t checkedSizeAdd(size_t lhs, size_t rhs, const char *what)
{
    if (lhs > (std::numeric_limits<size_t>::max)() - rhs)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s size overflows size_t", what ? what : "Value");
    }
    return lhs + rhs;
}

[[nodiscard]] inline size_t checkedSizeMul(size_t lhs, size_t rhs, const char *what)
{
    if (rhs != 0 && lhs > (std::numeric_limits<size_t>::max)() / rhs)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s size overflows size_t", what ? what : "Value");
    }
    return lhs * rhs;
}

/** Checked product for a dimension or stride chain. */
[[nodiscard]] inline size_t checkedSizeProduct(std::initializer_list<size_t> values, const char *what)
{
    size_t result = 1;
    for (const size_t value : values)
    {
        result = checkedSizeMul(result, value, what);
    }
    return result;
}

/** Checked narrowing conversions used at ABI and file-stream boundaries. */
[[nodiscard]] inline int checkedSizeToInt(size_t value, const char *what)
{
    if (value > static_cast<size_t>((std::numeric_limits<int>::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s exceeds int range", what ? what : "Value");
    }
    return static_cast<int>(value);
}

[[nodiscard]] inline size_t checkedInt64ToSize(int64_t value, const char *what)
{
    if (value < 0 || static_cast<uint64_t>(value) > static_cast<uint64_t>((std::numeric_limits<size_t>::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s exceeds size_t range", what ? what : "Value");
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] inline int64_t checkedSizeToInt64(size_t value, const char *what)
{
    if (value > static_cast<size_t>((std::numeric_limits<int64_t>::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s exceeds int64 range", what ? what : "Value");
    }
    return static_cast<int64_t>(value);
}

[[nodiscard]] inline std::streamoff checkedSizeToStreamoff(size_t value, const char *what)
{
    using Limit = std::numeric_limits<std::streamoff>;
    if (value > static_cast<size_t>((Limit::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s exceeds stream offset range",
                             what ? what : "Value");
    }
    return static_cast<std::streamoff>(value);
}

[[nodiscard]] inline std::streamsize checkedSizeToStreamsize(size_t value, const char *what)
{
    using Limit = std::numeric_limits<std::streamsize>;
    if (value > static_cast<size_t>((Limit::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s exceeds stream size range",
                             what ? what : "Value");
    }
    return static_cast<std::streamsize>(value);
}

[[nodiscard]] inline std::string_view dataTypeToString(TensorDataType type) noexcept
{
    switch (type)
    {
    case TensorDataType::U8:
        return "uint8";
    case TensorDataType::I8:
        return "int8";
    case TensorDataType::F16:
        return "float16";
    case TensorDataType::F32:
        return "float32";
    case TensorDataType::I32:
        return "int32";
    case TensorDataType::I64:
        return "int64";
    case TensorDataType::Bool:
        return "bool";
    }
    return "unknown";
}

struct Shape
{
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> init) : dims(init) {}
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    [[nodiscard]] size_t rank() const noexcept { return dims.size(); }
    [[nodiscard]] bool empty() const noexcept { return dims.empty(); }

    [[nodiscard]] int64_t operator[](size_t index) const { return dims[index]; }
    [[nodiscard]] int64_t &operator[](size_t index) { return dims[index]; }

    [[nodiscard]] bool isDynamic() const noexcept
    {
        for (const auto d : dims)
        {
            if (d < 0) return true;
        }
        return false;
    }

    [[nodiscard]] size_t elementCount() const
    {
        if (dims.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Shape must not be empty");
        }
        size_t total = 1;
        for (const auto d : dims)
        {
            if (d <= 0)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Shape dimension must be positive for concrete buffer, got %lld", d);
            }
            const auto ud = checkedInt64ToSize(d, "Shape dimension");
            total = checkedSizeMul(total, ud, "Shape element count");
        }
        return total;
    }

    [[nodiscard]] bool operator==(const Shape &other) const noexcept
    {
        return dims == other.dims;
    }

    [[nodiscard]] bool operator!=(const Shape &other) const noexcept
    {
        return dims != other.dims;
    }
};

/**
 * @brief Validate a concrete runtime shape against a model-declared shape.
 *
 * Negative declared dimensions represent dynamic dimensions.  Every runtime
 * dimension must be positive, preserve the declared rank, and match each
 * declared static dimension.  The concrete element count is checked as part
 * of validation so an accepted shape can be used for allocation safely.
 */
inline void validateRuntimeShape(const Shape &declared_shape, const Shape &runtime_shape,
                                 std::string_view tensor_name)
{
    const std::string name(tensor_name);
    if (declared_shape.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Declared tensor shape must not be empty: %s",
                             name.c_str());
    }
    if (runtime_shape.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Runtime tensor shape must not be empty: %s",
                             name.c_str());
    }
    if (runtime_shape.rank() != declared_shape.rank())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Runtime tensor rank mismatch for %s: expected %zu, got %zu", name.c_str(),
                             declared_shape.rank(), runtime_shape.rank());
    }

    for (size_t index = 0; index < runtime_shape.rank(); ++index)
    {
        const auto declared_dimension = declared_shape[index];
        const auto runtime_dimension   = runtime_shape[index];
        if (runtime_dimension <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Runtime tensor dimension must be positive for %s dim[%zu]=%lld", name.c_str(),
                                 index, static_cast<long long>(runtime_dimension));
        }
        if (declared_dimension == 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Declared tensor dimension must not be zero for %s dim[%zu]", name.c_str(), index);
        }
        if (declared_dimension > 0 && runtime_dimension != declared_dimension)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Runtime tensor dimension mismatch for %s dim[%zu]: expected %lld, got %lld",
                                 name.c_str(), index, static_cast<long long>(declared_dimension),
                                 static_cast<long long>(runtime_dimension));
        }
    }
    (void)runtime_shape.elementCount();
}

struct TensorDesc
{
    TensorDataType data_type{TensorDataType::U8};
    TensorLayout   layout{TensorLayout::HWC};
    MemoryKind     memory_kind{MemoryKind::HOST};
    Shape          shape;

    TensorDesc() = default;

    TensorDesc(TensorDataType dt, TensorLayout lay, MemoryKind mem, int w, int h, int c)
        : data_type(dt), layout(lay), memory_kind(mem)
    {
        if (lay == TensorLayout::NCHW || lay == TensorLayout::CHW)
        {
            shape = Shape{c, h, w};
        }
        else
        {
            shape = Shape{h, w, c};
        }
    }

    TensorDesc(TensorDataType dt, TensorLayout lay, MemoryKind mem, Shape s)
        : data_type(dt), layout(lay), memory_kind(mem), shape(std::move(s))
    {
    }

    /**
     * @brief Return the width encoded by the descriptor shape.
     *
     * Geometry access is intentionally derived from `shape`; there is no
     * second mutable width/height/channels representation to drift out of
     * sync with the tensor dimensions.
     */
    [[nodiscard]] int width() const
    {
        return geometryDimension(geometryIndex(GeometryDimension::Width), "width");
    }

    [[nodiscard]] int height() const
    {
        return geometryDimension(geometryIndex(GeometryDimension::Height), "height");
    }

    [[nodiscard]] int channels() const
    {
        return geometryDimension(geometryIndex(GeometryDimension::Channels), "channels");
    }

    [[nodiscard]] bool hasGeometry() const noexcept
    {
        switch (layout)
        {
        case TensorLayout::HWC:
        case TensorLayout::CHW:
            return shape.rank() == 3;
        case TensorLayout::NCHW:
            return shape.rank() == 3 || shape.rank() == 4;
        case TensorLayout::NHWC:
            return shape.rank() == 4;
        case TensorLayout::Opaque:
            return false;
        }
        return false;
    }

    [[nodiscard]] size_t elementCount() const
    {
        if (shape.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor shape must not be empty");
        }
        return shape.elementCount();
    }

    [[nodiscard]] size_t byteSize() const
    {
        const size_t elem_size = dataTypeSize(data_type);
        if (elem_size == 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unsupported tensor data type");
        }
        const size_t elements = elementCount();
        return checkedSizeMul(elements, elem_size, "Tensor byte size");
    }

    [[nodiscard]] size_t bytesPerRequest() const
    {
        return byteSize();
    }

    void validate(const std::string &name) const
    {
        if (shape.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor '%s' shape must not be empty", name.c_str());
        }
        (void)shape.elementCount();
        if (layout != TensorLayout::Opaque && !hasGeometry())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor '%s' shape rank %zu is incompatible with layout", name.c_str(), shape.rank());
        }
        (void)byteSize();
    }

private:
    enum class GeometryDimension
    {
        Width,
        Height,
        Channels,
    };

    [[nodiscard]] size_t geometryIndex(const GeometryDimension dimension) const
    {
        const bool channel_first = layout == TensorLayout::NCHW || layout == TensorLayout::CHW;
        const bool batched = shape.rank() == 4;
        if ((layout == TensorLayout::NCHW && !batched && shape.rank() != 3)
            || (layout == TensorLayout::CHW && shape.rank() != 3)
            || (layout == TensorLayout::HWC && shape.rank() != 3)
            || (layout == TensorLayout::NHWC && !batched)
            || layout == TensorLayout::Opaque)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor layout does not expose image geometry");
        }

        if (channel_first)
        {
            const size_t offset = batched ? 1U : 0U;
            switch (dimension)
            {
            case GeometryDimension::Channels:
                return offset;
            case GeometryDimension::Height:
                return offset + 1U;
            case GeometryDimension::Width:
                return offset + 2U;
            }
        }
        else
        {
            const size_t offset = batched ? 1U : 0U;
            switch (dimension)
            {
            case GeometryDimension::Height:
                return offset;
            case GeometryDimension::Width:
                return offset + 1U;
            case GeometryDimension::Channels:
                return offset + 2U;
            }
        }
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unknown tensor geometry dimension");
    }

    [[nodiscard]] int geometryDimension(const size_t index, const char *name) const
    {
        if (index >= shape.rank() || shape[index] <= 0 || shape[index] > (std::numeric_limits<int>::max)())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor %s dimension must be in the positive int range", name);
        }
        return static_cast<int>(shape[index]);
    }
};

struct BufferView
{
    void      *data{nullptr};
    TensorDesc desc;
    size_t     bytes_per_request{0};
    int        capacity_batch{0};
    // Optional backing allocation size. Zero means the owner did not expose it.
    size_t     capacity_bytes{0};
    // Optional stable tensor name. Backend adapters use it when present to
    // validate bindings independently of container order.
    std::string tensor_name;

    /**
     * @brief Create an explicitly byte-typed view for an opaque allocation.
     *
     * This is the escape hatch for callers that only own an allocation (for
     * example a staging buffer). Typed model and operator paths should pass a
     * descriptor instead so shape and dtype are validated at the boundary.
     */
    [[nodiscard]] static BufferView fromBytes(void *pointer, size_t bytes,
                                              MemoryKind memory = MemoryKind::HOST)
    {
        BufferView view;
        view.data              = pointer;
        view.bytes_per_request = bytes;
        view.capacity_batch    = 1;
        view.capacity_bytes    = bytes;
        if (bytes > 0)
        {
            if (bytes > static_cast<size_t>((std::numeric_limits<int64_t>::max)()))
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Opaque buffer size exceeds int64 range");
            }
            view.desc = TensorDesc{TensorDataType::U8, TensorLayout::Opaque, memory,
                                   Shape{static_cast<int64_t>(bytes)}};
        }
        else
        {
            view.desc.memory_kind = memory;
        }
        return view;
    }

    [[nodiscard]] static BufferView fromBytes(void *pointer, size_t bytes, MemoryKind memory,
                                              std::string tensor_name)
    {
        auto view        = fromBytes(pointer, bytes, memory);
        view.tensor_name = std::move(tensor_name);
        return view;
    }

    [[nodiscard]] void *dataForRequest(int request_index) noexcept
    {
        if (request_index < 0 || capacity_batch <= 0 || request_index >= capacity_batch || !data)
        {
            return nullptr;
        }
        const auto req_idx = static_cast<size_t>(request_index);
        if (bytes_per_request > 0 && req_idx > (std::numeric_limits<size_t>::max)() / bytes_per_request)
        {
            return nullptr;
        }
        if (capacity_bytes != 0
            && (req_idx * bytes_per_request > capacity_bytes
                || bytes_per_request > capacity_bytes - req_idx * bytes_per_request))
        {
            return nullptr;
        }
        return static_cast<uint8_t *>(data) + req_idx * bytes_per_request;
    }

    [[nodiscard]] const void *dataForRequest(int request_index) const noexcept
    {
        if (request_index < 0 || capacity_batch <= 0 || request_index >= capacity_batch || !data)
        {
            return nullptr;
        }
        const auto req_idx = static_cast<size_t>(request_index);
        if (bytes_per_request > 0 && req_idx > (std::numeric_limits<size_t>::max)() / bytes_per_request)
        {
            return nullptr;
        }
        if (capacity_bytes != 0
            && (req_idx * bytes_per_request > capacity_bytes
                || bytes_per_request > capacity_bytes - req_idx * bytes_per_request))
        {
            return nullptr;
        }
        return static_cast<const uint8_t *>(data) + req_idx * bytes_per_request;
    }

    [[nodiscard]] size_t byteSize() const
    {
        if (capacity_batch < 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "BufferView capacity_batch must not be negative");
        }
        if (bytes_per_request != 0 && capacity_batch == 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "BufferView capacity_batch must be positive for a non-empty buffer");
        }
        if (bytes_per_request != 0)
        {
            const size_t descriptor_bytes = desc.byteSize();
            if (descriptor_bytes != bytes_per_request)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "BufferView bytes_per_request (%zu) does not match descriptor bytes (%zu)",
                                     bytes_per_request, descriptor_bytes);
            }
        }
        const size_t batch    = static_cast<size_t>(capacity_batch);
        const size_t required = checkedSizeMul(bytes_per_request, batch, "BufferView byte size");
        if (capacity_bytes != 0 && required > capacity_bytes)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "BufferView capacity (%zu) is smaller than required bytes (%zu)", capacity_bytes,
                                 required);
        }
        return required;
    }
};

using TensorView = BufferView;

} // namespace irt
