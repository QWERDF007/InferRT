#pragma once

#include <inferrt/engine/Export.h>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace irt::engine::priv {

/** Resource construction points used only by lifecycle regression tests. */
enum class StartFaultPoint : uint8_t
{
    None = 0,
    RuntimePlan,
    Slot,
    TicketPool,
    PrepareWorker,
    PostprocessWorker,
    GpuDispatcher,
    CompletionPoller,
    Scheduler,
};

/** Private test-only controls; production callers never need these fields. */
struct EngineTestOptions
{
    int             start_fault_stage{0};
    StartFaultPoint start_fault_point{StartFaultPoint::None};
    size_t          start_fault_index{0};
    bool            completion_fatal{false};
    int             completion_fatal_delay_ms{0};
    bool            duplicate_completion{false};
};

INFERRT_ENGINE_API EngineTestOptions GetEngineTestOptions() noexcept;
INFERRT_ENGINE_API void SetEngineTestOptions(EngineTestOptions options) noexcept;

/** Builds the config-derived legacy plan for focused pipeline contract tests. */
INFERRT_ENGINE_API std::shared_ptr<const PipelinePlan> CreateLegacyPipelineForTest(const EngineConfig &config);

/** Restores the previous process-local test options when leaving a test scope. */
class EngineTestOptionsGuard final
{
public:
    INFERRT_ENGINE_API explicit EngineTestOptionsGuard(EngineTestOptions options) noexcept;
    INFERRT_ENGINE_API ~EngineTestOptionsGuard() noexcept;

    EngineTestOptionsGuard(const EngineTestOptionsGuard &)            = delete;
    EngineTestOptionsGuard &operator=(const EngineTestOptionsGuard &) = delete;

private:
    EngineTestOptions previous_{};
};

} // namespace irt::engine::priv
