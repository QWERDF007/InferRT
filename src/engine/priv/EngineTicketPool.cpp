#include "EngineTicketPool.hpp"

#include <algorithm>
#include <condition_variable>
#include <inferrt/core/Exception.hpp>

namespace irt::engine::priv {

TicketPool::TicketPool(const EngineConfig &config, const std::shared_ptr<const PipelinePlan> &pipeline,
                       const std::vector<size_t> &output_capacity_bytes)
{
    if (!pipeline)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "TicketPool requires a PipelinePlan");
    }
    initialize(config, pipeline, output_capacity_bytes);
}

TicketPool::~TicketPool() noexcept
{
    abort();
}

void TicketPool::initialize(const EngineConfig &config, const std::shared_ptr<const PipelinePlan> &pipeline,
                            const std::vector<size_t> &output_capacity_bytes)
{
    const size_t input_count = config.pinned_input_tickets == 0
                                 ? config.execution_slots + config.cpu_preprocess_workers
                                 : config.pinned_input_tickets;
    const size_t output_count = config.pinned_output_tickets == 0
                                  ? config.execution_slots + config.cpu_postprocess_workers
                                  : config.pinned_output_tickets;
    if (input_count == 0 || output_count == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pinned ticket pool counts must be positive");
    }

    size_t input_ticket_bytes = 0;
    for (const auto &[name, desc] : pipeline->tensors())
    {
        if (desc.memory_kind != MemoryKind::HOST)
        {
            continue;
        }
        input_ticket_bytes = irt::checkedSizeAdd(
            input_ticket_bytes,
            irt::checkedSizeMul(static_cast<size_t>(config.max_batch_size), desc.bytesPerRequest(),
                                "Pinned input ticket bytes"),
            "Pinned input ticket bytes");
    }

    size_t output_ticket_bytes = 0;
    for (const size_t bytes : output_capacity_bytes)
    {
        output_ticket_bytes = irt::checkedSizeAdd(output_ticket_bytes, bytes, "Pinned model output ticket bytes");
    }
    for (const auto &binding : pipeline->results())
    {
        const auto &desc = pipeline->tensors().at(binding.tensor_name);
        output_ticket_bytes = irt::checkedSizeAdd(
            output_ticket_bytes,
            irt::checkedSizeMul(static_cast<size_t>(config.max_batch_size), desc.bytesPerRequest(),
                                "Pinned result ticket bytes"),
            "Pinned result ticket bytes");
    }

    const auto input_pool_bytes  = irt::checkedSizeMul(input_ticket_bytes, input_count,
                                                       "Pinned input ticket pool bytes");
    const auto output_pool_bytes = irt::checkedSizeMul(output_ticket_bytes, output_count,
                                                        "Pinned output ticket pool bytes");
    capacity_bytes_ = irt::checkedSizeAdd(input_pool_bytes, output_pool_bytes, "Pinned ticket pool capacity");
    if (config.pinned_memory_limit_bytes != 0 && capacity_bytes_ > config.pinned_memory_limit_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                             "Configured pinned memory pool limit (%zu bytes) is insufficient for %zu bytes",
                             config.pinned_memory_limit_bytes, capacity_bytes_);
    }

    input_tickets_.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index)
    {
        auto ticket = std::make_unique<InputTicket>();
        for (const auto &[name, desc] : pipeline->tensors())
        {
            if (desc.memory_kind != MemoryKind::HOST)
            {
                continue;
            }
            auto [buffer, _] = ticket->buffers.try_emplace(name, irt::TensorDataType::U8);
            const size_t bytes = irt::checkedSizeMul(static_cast<size_t>(config.max_batch_size),
                                                      desc.bytesPerRequest(), "Pinned input ticket bytes");
            buffer->second.resize(bytes, irt::TensorDataType::U8);
            ticket->tensors.emplace(name, TensorView{buffer->second.data(), desc, desc.bytesPerRequest(),
                                                     config.max_batch_size, buffer->second.capacity()});
        }
        free_input_tickets_.push_back(ticket.get());
        input_tickets_.push_back(std::move(ticket));
    }

    output_tickets_.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index)
    {
        auto ticket = std::make_unique<OutputTicket>();
        ticket->model_outputs.resize(output_capacity_bytes.size());
        for (size_t output_index = 0; output_index < output_capacity_bytes.size(); ++output_index)
        {
            ticket->model_outputs[output_index].resize(output_capacity_bytes[output_index], irt::TensorDataType::U8);
        }
        for (const auto &binding : pipeline->results())
        {
            const auto &desc = pipeline->tensors().at(binding.tensor_name);
            auto [buffer, _] = ticket->result_outputs.try_emplace(binding.result_name, irt::TensorDataType::U8);
            buffer->second.resize(irt::checkedSizeMul(static_cast<size_t>(config.max_batch_size),
                                                      desc.bytesPerRequest(), "Pinned result ticket bytes"),
                                  irt::TensorDataType::U8);
        }
        free_output_tickets_.push_back(ticket.get());
        output_tickets_.push_back(std::move(ticket));
    }
}

InputTicket *TicketPool::acquireInput()
{
    std::unique_lock lock(input_mutex_);
    input_condition_.wait(lock, [this] { return aborting_.load() || !free_input_tickets_.empty(); });
    if (aborting_.load() || free_input_tickets_.empty())
    {
        return nullptr;
    }
    auto *ticket = free_input_tickets_.front();
    free_input_tickets_.pop_front();
    ++input_in_use_;
    input_high_watermark_ = std::max(input_high_watermark_, input_in_use_);
    return ticket;
}

OutputTicket *TicketPool::acquireOutput()
{
    std::unique_lock lock(output_mutex_);
    output_condition_.wait(lock, [this] { return aborting_.load() || !free_output_tickets_.empty(); });
    if (aborting_.load() || free_output_tickets_.empty())
    {
        return nullptr;
    }
    auto *ticket = free_output_tickets_.front();
    free_output_tickets_.pop_front();
    ++output_in_use_;
    output_high_watermark_ = std::max(output_high_watermark_, output_in_use_);
    return ticket;
}

void TicketPool::releaseInput(InputTicket *ticket)
{
    if (!ticket)
    {
        return;
    }
    {
        std::lock_guard lock(input_mutex_);
        free_input_tickets_.push_back(ticket);
        if (input_in_use_ > 0)
        {
            --input_in_use_;
        }
    }
    input_condition_.notify_one();
}

void TicketPool::releaseOutput(OutputTicket *ticket)
{
    if (!ticket)
    {
        return;
    }
    {
        std::lock_guard lock(output_mutex_);
        free_output_tickets_.push_back(ticket);
        if (output_in_use_ > 0)
        {
            --output_in_use_;
        }
    }
    output_condition_.notify_one();
}

void TicketPool::abort() noexcept
{
    aborting_.store(true);
    input_condition_.notify_all();
    output_condition_.notify_all();
}

TicketPool::Snapshot TicketPool::snapshot() const
{
    std::scoped_lock lock(input_mutex_, output_mutex_);
    return {input_tickets_.size(),
            output_tickets_.size(),
            input_in_use_,
            output_in_use_,
            input_high_watermark_,
            output_high_watermark_,
            capacity_bytes_,
            aborting_.load()};
}

} // namespace irt::engine::priv
