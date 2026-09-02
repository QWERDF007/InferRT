#include "EngineSlot.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace irt::engine::priv {

void initializeEngineSlot(Slot &slot, const EngineConfig &config,
                          const std::shared_ptr<const PipelinePlan> &pipeline,
                          const int fixed_batch_size,
                          size_t &device_arena_bytes,
                          std::unordered_map<int, size_t> &device_arena_bytes_by_device)
{
    const auto &preprocess = config.preprocessSpec();
    irt::model::setCudaDevice(slot.device_id);
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.h2d_stream, cudaStreamNonBlocking), "cudaStreamCreate(h2d)");
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.compute_stream, cudaStreamNonBlocking),
                          "cudaStreamCreate(compute)");
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.d2h_stream, cudaStreamNonBlocking), "cudaStreamCreate(d2h)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.h2d_done, cudaEventDisableTiming), "cudaEventCreate(h2d)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.compute_done, cudaEventDisableTiming),
                          "cudaEventCreate(compute)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.d2h_done, cudaEventDisableTiming), "cudaEventCreate(d2h)");

    if (!slot.runtime_plan)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ExecutionSlot has no TensorRT runtime plan");
    }
    slot.context        = slot.runtime_plan->createSession();
    slot.output_infos   = slot.runtime_plan->outputs();
    slot.capacity_batch = fixed_batch_size > 0 ? fixed_batch_size : config.max_batch_size;

    for (const auto &binding : pipeline->results())
    {
        if (std::any_of(slot.output_infos.begin(), slot.output_infos.end(), [&binding](const irt::TensorInfo &info)
                        { return info.name == binding.result_name; }))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline result name '%s' conflicts with TensorRT output tensor name",
                                 binding.result_name.c_str());
        }
    }

    const auto  runtime_inputs  = slot.runtime_plan->inputs();
    const auto &pipeline_inputs = pipeline->modelInputs();
    if (pipeline_inputs.size() != runtime_inputs.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Pipeline binds %zu model inputs but TensorRT engine has %zu", pipeline_inputs.size(),
                             runtime_inputs.size());
    }
    std::unordered_map<std::string, bool> bound_runtime_inputs;
    for (const auto &binding : pipeline_inputs)
    {
        const auto pipeline_input = pipeline->tensors().find(binding.tensor_name);
        if (pipeline_input == pipeline->tensors().end())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pipeline model input is not allocated: %s",
                                 binding.tensor_name.c_str());
        }
        const auto &input_desc = pipeline_input->second;
        if (input_desc.data_type != TensorDataType::F32 || input_desc.layout != TensorLayout::NCHW
            || input_desc.memory_kind != MemoryKind::DEVICE)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input must be a device float32 NCHW tensor: %s",
                                 binding.tensor_name.c_str());
        }
        std::string engine_name = binding.engine_tensor_name;
        if (engine_name.empty())
        {
            if (runtime_inputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Multi-input TensorRT engines require explicit PipelineBuilder::bindModelInput()");
            }
            engine_name = runtime_inputs.front().name;
        }
        const auto runtime_input = std::find_if(
            runtime_inputs.begin(), runtime_inputs.end(), [&engine_name](const irt::TensorInfo &info)
            { return info.name == engine_name; });
        if (runtime_input == runtime_inputs.end() || !bound_runtime_inputs.emplace(engine_name, true).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid or duplicate TensorRT input binding: %s",
                                 engine_name.c_str());
        }
        if (runtime_input->desc.data_type != input_desc.data_type
            || runtime_input->desc.memory_kind != input_desc.memory_kind)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input descriptor does not match TensorRT input: %s",
                                 binding.tensor_name.c_str());
        }
        if (binding.tensor_name == pipeline->modelInput()
            && (input_desc.width() != preprocess.input_width || input_desc.height() != preprocess.input_height
                || input_desc.channels() != preprocess.input_channels))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Primary pipeline model input does not match EngineConfig image input");
        }
        slot.context->setTensorShape(
            engine_name,
            irt::Shape{slot.capacity_batch, input_desc.channels(), input_desc.height(), input_desc.width()});
        slot.model_inputs.emplace_back(binding.tensor_name, std::move(engine_name));
    }

    struct ArenaEntry
    {
        std::string name;
        TensorDesc  desc;
        size_t      bytes{0};
        size_t      first_use{0};
        size_t      last_use{0};
        size_t      offset{0};
    };

    struct ArenaBlock
    {
        size_t offset{0};
        size_t bytes{0};
        size_t last_use{0};
    };

    const size_t                                               terminal_use = pipeline->nodes().size() + 2;
    std::unordered_map<std::string, std::pair<size_t, size_t>> lifetimes;
    for (const auto &[name, desc] : pipeline->tensors())
    {
        if (desc.memory_kind == MemoryKind::DEVICE)
        {
            lifetimes.emplace(name, std::make_pair(terminal_use, 0U));
        }
    }
    for (size_t node_index = 0; node_index < pipeline->nodes().size(); ++node_index)
    {
        const auto note_use = [&](const std::string &name)
        {
            const auto found = lifetimes.find(name);
            if (found == lifetimes.end())
            {
                return;
            }
            found->second.first  = std::min(found->second.first, node_index);
            found->second.second = std::max(found->second.second, node_index);
        };
        for (const auto &name : pipeline->nodes()[node_index].config.inputs)
        {
            note_use(name);
        }
        for (const auto &name : pipeline->nodes()[node_index].config.outputs)
        {
            note_use(name);
        }
    }
    for (const auto &[tensor_name, _] : slot.model_inputs)
    {
        lifetimes.at(tensor_name).second = terminal_use - 1;
    }
    for (const auto &binding : pipeline->results())
    {
        lifetimes.at(binding.tensor_name).second = terminal_use;
    }

    std::vector<ArenaEntry> entries;
    for (const auto &[name, desc] : pipeline->tensors())
    {
        if (desc.memory_kind != MemoryKind::DEVICE)
        {
            continue;
        }
        const auto lifetime = lifetimes.at(name);
        entries.push_back({name, desc,
                           irt::checkedSizeMul(static_cast<size_t>(slot.capacity_batch), desc.bytesPerRequest(),
                                               "Device tensor arena entry"),
                           lifetime.first == terminal_use ? 0U : lifetime.first,
                           std::max(lifetime.first == terminal_use ? 0U : lifetime.first, lifetime.second), 0});
    }
    std::sort(entries.begin(), entries.end(),
              [](const ArenaEntry &lhs, const ArenaEntry &rhs)
              { return lhs.first_use == rhs.first_use ? lhs.name < rhs.name : lhs.first_use < rhs.first_use; });

    size_t                  arena_bytes = 0;
    std::vector<ArenaBlock> blocks;
    for (auto &entry : entries)
    {
        auto selected = blocks.end();
        for (auto block = blocks.begin(); block != blocks.end(); ++block)
        {
            if (block->last_use < entry.first_use && block->bytes >= entry.bytes
                && (selected == blocks.end() || block->bytes < selected->bytes))
            {
                selected = block;
            }
        }
        if (selected == blocks.end())
        {
            arena_bytes  = alignDeviceOffset(arena_bytes);
            entry.offset = arena_bytes;
            blocks.push_back({entry.offset, entry.bytes, entry.last_use});
            arena_bytes = irt::checkedSizeAdd(arena_bytes, entry.bytes, "Device tensor arena size");
        }
        else
        {
            entry.offset       = selected->offset;
            selected->last_use = entry.last_use;
        }
    }

    slot.output_capacity_bytes.reserve(slot.output_infos.size());
    for (const auto &output_info : slot.output_infos)
    {
        const auto &name  = output_info.name;
        const auto   shape = slot.context->tensorShape(name);
        const size_t bytes = irt::checkedSizeMul(shape.elementCount(), sizeof(float), "TensorRT output bytes");
        if (bytes == 0 || bytes % static_cast<size_t>(slot.capacity_batch) != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "TensorRT output shape is incompatible with execution batch: %s", name.c_str());
        }
        const size_t per_request = bytes / static_cast<size_t>(slot.capacity_batch);
        arena_bytes        = alignDeviceOffset(arena_bytes);
        slot.output_capacity_bytes.push_back(bytes);
        slot.device_outputs.push_back(
            irt::BufferView{nullptr,
                            {TensorDataType::F32, TensorLayout::Opaque, MemoryKind::DEVICE,
                             Shape{static_cast<int64_t>(per_request / sizeof(float))}},
                            per_request,
                            slot.capacity_batch,
                            bytes,
                            name});
        entries.push_back({
            "model." + name,
            {TensorDataType::F32, TensorLayout::Opaque, MemoryKind::DEVICE,
             Shape{static_cast<int64_t>(per_request / sizeof(float))}},
            bytes,
            terminal_use - 1,
            terminal_use,
            arena_bytes
        });
        arena_bytes = irt::checkedSizeAdd(arena_bytes, bytes, "Device tensor arena size");
    }
    if (arena_bytes == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ExecutionSlot has no device buffers");
    }

    const size_t existing_device_bytes = device_arena_bytes_by_device[slot.device_id];
    const auto total_device_bytes = irt::checkedSizeAdd(existing_device_bytes, arena_bytes,
                                                        "Device memory pool capacity");
    if (config.device_memory_limit_bytes != 0 && total_device_bytes > config.device_memory_limit_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                             "Configured device memory pool limit (%zu bytes) is insufficient for execution arenas",
                             config.device_memory_limit_bytes);
    }
    slot.device_arena.allocate(arena_bytes, slot.device_id, slot.compute_stream);
    slot.device_arena_bytes = arena_bytes;
    device_arena_bytes = irt::checkedSizeAdd(device_arena_bytes, arena_bytes,
                                              "Device arena aggregate size");
    device_arena_bytes_by_device[slot.device_id] =
        irt::checkedSizeAdd(device_arena_bytes_by_device[slot.device_id], arena_bytes,
                            "Device arena device aggregate size");
    auto *base = static_cast<std::byte *>(slot.device_arena.data());
    for (const auto &entry : entries)
    {
        void *pointer = base + entry.offset;
        if (entry.name.starts_with("model."))
        {
            const auto output_index = static_cast<size_t>(
                std::distance(slot.output_infos.begin(),
                              std::find_if(slot.output_infos.begin(), slot.output_infos.end(), [&entry](
                                               const irt::TensorInfo &info)
                                           { return info.name == entry.name.substr(std::string("model.").size()); })));
            slot.device_outputs[output_index].data = pointer;
        }
        else
        {
            slot.device_tensors.emplace(entry.name,
                                        TensorView{pointer, entry.desc, entry.desc.bytesPerRequest(), slot.capacity_batch,
                                                   irt::checkedSizeMul(entry.desc.bytesPerRequest(),
                                                                       static_cast<size_t>(slot.capacity_batch),
                                                                       "Device tensor backing capacity"),
                                                   entry.name});
        }
    }
    slot.operators = pipeline->createOperators();
}

} // namespace irt::engine::priv
