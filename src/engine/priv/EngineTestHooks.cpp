#include "EngineTestHooks.hpp"

#include "EngineLegacyPipeline.hpp"

#include <mutex>

namespace irt::engine::priv {
namespace {

std::mutex          g_options_mutex;
EngineTestOptions   g_options{};

} // namespace

EngineTestOptions GetEngineTestOptions() noexcept
{
    std::lock_guard lock(g_options_mutex);
    return g_options;
}

void SetEngineTestOptions(const EngineTestOptions options) noexcept
{
    std::lock_guard lock(g_options_mutex);
    g_options = options;
}

std::shared_ptr<const PipelinePlan> CreateLegacyPipelineForTest(const EngineConfig &config)
{
    return makeLegacyPipeline(config);
}

EngineTestOptionsGuard::EngineTestOptionsGuard(const EngineTestOptions options) noexcept
    : previous_(GetEngineTestOptions())
{
    SetEngineTestOptions(options);
}

EngineTestOptionsGuard::~EngineTestOptionsGuard() noexcept
{
    SetEngineTestOptions(previous_);
}

} // namespace irt::engine::priv
