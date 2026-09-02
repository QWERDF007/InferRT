#include <inferrt/core/ModelContract.hpp>

#include <unordered_map>
#include <unordered_set>

namespace irt {

namespace {

bool matchesExecutionShape(const Shape &buffer_shape, const Shape &declared_shape)
{
    if (buffer_shape == declared_shape)
    {
        return true;
    }

    // BufferView describes one request while TensorInfo may describe the
    // complete runtime batch.  Compare the non-batch dimensions in that case.
    if (declared_shape.rank() != buffer_shape.rank() + 1 || declared_shape.empty() || declared_shape[0] <= 0)
    {
        return false;
    }
    for (size_t index = 0; index < buffer_shape.rank(); ++index)
    {
        if (declared_shape[index + 1] != buffer_shape[index])
        {
            return false;
        }
    }
    return true;
}

} // namespace

std::vector<BufferView> normalizeExecutionBuffers(const std::span<const BufferView> buffers,
                                                  const std::span<const TensorInfo> inputs,
                                                  const std::span<const TensorInfo> outputs)
{
    const size_t expected_count = checkedSizeAdd(inputs.size(), outputs.size(), "Execution binding count");
    if (buffers.size() != expected_count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Expected %zu execution buffers (%zu inputs and %zu outputs), got %zu", expected_count,
                             inputs.size(), outputs.size(), buffers.size());
    }

    std::unordered_map<std::string, size_t> expected_indices;
    expected_indices.reserve(expected_count);
    std::vector<const TensorInfo *> expected_infos;
    expected_infos.reserve(expected_count);

    const auto append_expected = [&](const std::span<const TensorInfo> infos, const TensorIOMode mode)
    {
        for (const auto &info : infos)
        {
            if (info.name.empty() || info.mode != mode)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Execution descriptor has an invalid tensor name or I/O mode");
            }
            if (!expected_indices.emplace(info.name, expected_infos.size()).second)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Duplicate execution tensor descriptor: %s", info.name.c_str());
            }
            expected_infos.push_back(&info);
        }
    };
    append_expected(inputs, TensorIOMode::Input);
    append_expected(outputs, TensorIOMode::Output);

    std::vector<BufferView> normalized(expected_count);
    std::unordered_set<std::string> seen_names;
    seen_names.reserve(buffers.size());
    for (const auto &buffer : buffers)
    {
        if (buffer.tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Execution buffer must carry a tensor name");
        }
        const auto expected_it = expected_indices.find(buffer.tensor_name);
        if (expected_it == expected_indices.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Unexpected execution tensor buffer: %s", buffer.tensor_name.c_str());
        }
        if (!seen_names.emplace(buffer.tensor_name).second)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Duplicate execution tensor buffer: %s", buffer.tensor_name.c_str());
        }

        const auto &expected = *expected_infos[expected_it->second];
        if (buffer.data == nullptr || buffer.bytes_per_request == 0 || buffer.capacity_batch <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Execution buffer is incomplete: %s", buffer.tensor_name.c_str());
        }
        if (buffer.desc.memory_kind != expected.desc.memory_kind)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Execution buffer memory kind mismatch: %s", buffer.tensor_name.c_str());
        }
        if (buffer.desc.layout != TensorLayout::Opaque)
        {
            if (buffer.desc.data_type != expected.desc.data_type)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Execution buffer data type mismatch: %s", buffer.tensor_name.c_str());
            }
            if (!matchesExecutionShape(buffer.desc.shape, expected.desc.shape))
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Execution buffer shape mismatch: %s", buffer.tensor_name.c_str());
            }
        }

        const size_t available_bytes = buffer.byteSize();
        if (!expected.desc.shape.isDynamic())
        {
            const size_t required_bytes = expected.desc.byteSize();
            if (available_bytes < required_bytes)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Execution buffer is too small: %s, required=%zu, available=%zu",
                                     buffer.tensor_name.c_str(), required_bytes, available_bytes);
            }
        }
        normalized[expected_it->second] = buffer;
    }

    if (seen_names.size() != expected_count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Execution buffer set does not cover every declared tensor");
    }
    return normalized;
}

IExecutionDescriptor::~IExecutionDescriptor() = default;
IExecutionPlan::~IExecutionPlan() = default;
IExecutableModel::~IExecutableModel() = default;
ITensorRuntimeSession::~ITensorRuntimeSession() = default;

} // namespace irt
