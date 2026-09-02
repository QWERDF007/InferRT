#include "EngineCompletionOrder.hpp"

#include <inferrt/core/Exception.hpp>

namespace irt::engine::priv {

CompletionOrder::CompletionOrder(EngineMetricsFault &metrics) noexcept
    : metrics_(metrics)
{
}

void CompletionOrder::registerRequest(const RequestPtr &request)
{
    if (!request)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot register an empty inference request");
    }
    if (request->source_id.empty())
    {
        request->source_id = "default";
    }

    std::lock_guard lock(mutex_);
    request->source_sequence = next_source_submit_[request->source_id]++;
    next_source_deliver_.try_emplace(request->source_id, 0);
    const auto [_, inserted] = active_requests_.emplace(request->id, request);
    if (!inserted)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Inference request id is already registered: %llu",
                             static_cast<unsigned long long>(request->id));
    }
}

RequestPtr CompletionOrder::findActive(const uint64_t request_id) const
{
    std::lock_guard lock(mutex_);
    const auto      found = active_requests_.find(request_id);
    return found == active_requests_.end() ? nullptr : found->second;
}

bool CompletionOrder::cancel(const uint64_t request_id)
{
    std::vector<PendingCompletion> ready;
    {
        std::lock_guard lock(mutex_);
        const auto      found = active_requests_.find(request_id);
        if (found == active_requests_.end())
        {
            return false;
        }

        const RequestPtr request = found->second;
        if (!request || request->finished.load() || request->cancelled.exchange(true))
        {
            return false;
        }

        request->finished.store(true);
        active_requests_.erase(found);
        metrics_.recordRequestTerminal(FailureKind::Cancelled);
        PendingCompletion completion{request, false, {}, cancelledError()};
        if (request->preserve_source_order)
        {
            pending_source_completions_[request->source_id].emplace(request->source_sequence,
                                                                      std::move(completion));
        }
        else
        {
            ready.push_back(std::move(completion));
            unordered_delivered_sequences_[request->source_id].insert(request->source_sequence);
        }
        collectOrderedCompletionsLocked(ready, request->source_id);
    }
    fulfill(std::move(ready));
    return true;
}

void CompletionOrder::completeSuccess(const RequestPtr &request, InferenceResult result)
{
    if (!request)
    {
        return;
    }

    std::vector<PendingCompletion> ready;
    {
        std::lock_guard lock(mutex_);
        if (request->finished.exchange(true))
        {
            return;
        }
        result.request_id      = request->id;
        result.source_id       = request->source_id;
        result.source_sequence = request->source_sequence;
        active_requests_.erase(request->id);
        metrics_.recordRequestSuccess();
        PendingCompletion completion{request, true, std::move(result), nullptr};
        if (request->preserve_source_order)
        {
            pending_source_completions_[request->source_id].emplace(request->source_sequence, std::move(completion));
        }
        else
        {
            ready.push_back(std::move(completion));
            unordered_delivered_sequences_[request->source_id].insert(request->source_sequence);
        }
        collectOrderedCompletionsLocked(ready, request->source_id);
    }
    fulfill(std::move(ready));
}

void CompletionOrder::completeFailure(const RequestPtr &request, const std::exception_ptr error,
                                      const FailureKind kind)
{
    if (!request)
    {
        return;
    }

    std::vector<PendingCompletion> ready;
    {
        std::lock_guard lock(mutex_);
        if (request->finished.exchange(true))
        {
            return;
        }
        active_requests_.erase(request->id);
        metrics_.recordRequestTerminal(kind);
        PendingCompletion completion{request, false, {}, error};
        if (request->preserve_source_order)
        {
            pending_source_completions_[request->source_id].emplace(request->source_sequence, std::move(completion));
        }
        else
        {
            ready.push_back(std::move(completion));
            unordered_delivered_sequences_[request->source_id].insert(request->source_sequence);
        }
        collectOrderedCompletionsLocked(ready, request->source_id);
    }
    fulfill(std::move(ready));
}

void CompletionOrder::collectOrderedCompletionsLocked(std::vector<PendingCompletion> &ready,
                                                       const std::string &source_id)
{
    auto &next_sequence = next_source_deliver_[source_id];
    auto &unordered     = unordered_delivered_sequences_[source_id];
    auto &pending       = pending_source_completions_[source_id];
    for (;;)
    {
        const auto unordered_it = unordered.find(next_sequence);
        if (unordered_it != unordered.end())
        {
            unordered.erase(unordered_it);
            ++next_sequence;
            continue;
        }
        const auto pending_it = pending.find(next_sequence);
        if (pending_it == pending.end())
        {
            break;
        }
        ready.push_back(std::move(pending_it->second));
        pending.erase(pending_it);
        ++next_sequence;
    }
}

void CompletionOrder::fulfill(std::vector<PendingCompletion> completions)
{
    for (auto &completion : completions)
    {
        if (!completion.request)
        {
            continue;
        }
        bool expected = false;
        if (!completion.request->fulfilled.compare_exchange_strong(expected, true))
        {
            continue;
        }
        try
        {
            if (completion.success)
            {
                completion.request->promise.set_value(std::move(completion.result));
            }
            else
            {
                completion.request->promise.set_exception(completion.error);
            }
        }
        catch (...)
        {
            metrics_.recordFault(FaultStage::CpuPostprocess, 0, completion.request, std::current_exception());
        }
    }
}

void CompletionOrder::shutdown()
{
    std::vector<PendingCompletion> remaining;
    {
        std::lock_guard lock(mutex_);
        for (auto &[id, request] : active_requests_)
        {
            if (request && !request->finished.exchange(true))
            {
                metrics_.recordRequestTerminal(FailureKind::Failed);
                remaining.push_back(
                    {request, false, {},
                     std::make_exception_ptr(
                         irt::Exception(irt::Status::INVALID_OPERATION, "Inference engine stopped"))});
            }
        }
        active_requests_.clear();
        for (auto &[source_id, pending] : pending_source_completions_)
        {
            for (auto &[sequence, completion] : pending)
            {
                remaining.push_back(std::move(completion));
            }
        }
        pending_source_completions_.clear();
        unordered_delivered_sequences_.clear();
    }
    fulfill(std::move(remaining));
}

size_t CompletionOrder::activeCount() const
{
    std::lock_guard lock(mutex_);
    return active_requests_.size();
}

} // namespace irt::engine::priv
