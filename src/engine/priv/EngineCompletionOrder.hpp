#pragma once

#include "EngineMetricsFault.hpp"

#include <inferrt/engine/Export.h>

#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace irt::engine::priv {

/** Owns active requests, source ordering, and exactly-once future delivery. */
class INFERRT_ENGINE_API CompletionOrder final
{
public:
    explicit CompletionOrder(EngineMetricsFault &metrics) noexcept;

    void registerRequest(const RequestPtr &request);
    [[nodiscard]] RequestPtr findActive(uint64_t request_id) const;
    [[nodiscard]] bool       cancel(uint64_t request_id);

    void completeSuccess(const RequestPtr &request, InferenceResult result);
    void completeFailure(const RequestPtr &request, std::exception_ptr error, FailureKind kind);

    void                  shutdown();
    [[nodiscard]] size_t  activeCount() const;

private:
    void collectOrderedCompletionsLocked(std::vector<PendingCompletion> &ready, const std::string &source_id);
    void fulfill(std::vector<PendingCompletion> completions);

    EngineMetricsFault &metrics_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, RequestPtr>                               active_requests_;
    std::unordered_map<std::string, uint64_t>                              next_source_submit_;
    std::unordered_map<std::string, uint64_t>                              next_source_deliver_;
    std::unordered_map<std::string, std::map<uint64_t, PendingCompletion>> pending_source_completions_;
    std::unordered_map<std::string, std::set<uint64_t>>                    unordered_delivered_sequences_;
};

} // namespace irt::engine::priv
