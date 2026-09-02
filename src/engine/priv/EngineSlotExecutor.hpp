#pragma once

#include "EngineStageQueue.hpp"
#include "EngineTestHooks.hpp"
#include "EngineTicketPool.hpp"

#include <inferrt/engine/Export.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace irt::engine::priv {

/** Owns execution slots and the asynchronous GPU dispatch/completion workers. */
class INFERRT_ENGINE_API SlotExecutor final
{
public:
    struct Snapshot
    {
        size_t slot_count{0};
        size_t idle_slot_count{0};
        size_t active_slot_count{0};
        bool   dispatcher_joinable{false};
        bool   completion_poller_joinable{false};
    };

    using FailureStateReader  = std::function<bool()>;
    using BatchFailureHandler = std::function<void(const BatchPtr &, std::exception_ptr)>;
    using FaultRecorder       = std::function<void(FaultStage, int, const RequestPtr &, std::exception_ptr, bool)>;
    using FailureTransition   = std::function<void()>;
    using BatchFinishedHandler = std::function<void(const BatchPtr &)>;

    SlotExecutor(const PipelinePlan &pipeline, StageQueue &gpu_queue, StageQueue &postprocess_queue,
                 TicketPool &ticket_pool, std::vector<std::unique_ptr<Slot>> slots, EngineTestOptions test_options,
                 FailureStateReader state_reader, BatchFailureHandler fail_batch, FaultRecorder record_fault,
                 FailureTransition transition_failed, BatchFinishedHandler finish_batch);
    ~SlotExecutor() noexcept;

    SlotExecutor(const SlotExecutor &)            = delete;
    SlotExecutor &operator=(const SlotExecutor &) = delete;

    void startDispatcher();
    void startCompletionPoller();
    void stop() noexcept;
    void notifyFailure() noexcept;

    [[nodiscard]] Snapshot snapshot() const;

private:
    void dispatcherLoop();
    void completionPollerLoop();

    [[nodiscard]] Slot *acquireIdleSlot();
    void                returnSlot(Slot &slot);
    void                synchronizeSlot(Slot &slot) const noexcept;

    const PipelinePlan &pipeline_;
    StageQueue         &gpu_queue_;
    StageQueue         &postprocess_queue_;
    TicketPool         &ticket_pool_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::deque<Slot *>                 idle_slots_;
    EngineTestOptions                  test_options_;

    FailureStateReader   state_reader_;
    BatchFailureHandler  fail_batch_;
    FaultRecorder        record_fault_;
    FailureTransition    transition_failed_;
    BatchFinishedHandler finish_batch_;

    mutable std::mutex lifecycle_mutex_;
    std::thread         dispatcher_;
    std::thread         completion_poller_;
    bool                dispatcher_started_{false};
    bool                completion_poller_started_{false};

    mutable std::mutex       slot_mutex_;
    std::condition_variable  slot_condition_;
    std::atomic_bool         stop_requested_{false};
    std::atomic_bool         failure_requested_{false};

    std::mutex              completion_mutex_;
    std::condition_variable completion_condition_;
    bool                    dispatcher_closed_{false};
    std::atomic_bool        completion_fault_injected_{false};
};

} // namespace irt::engine::priv
