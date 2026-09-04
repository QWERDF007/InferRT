#pragma once

#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Export.h>
#include <inferrt/engine/Pipeline.hpp>
#include <inferrt/core/ModelContract.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace irt::engine::priv {

/** The engine consumes the same core execution-plan interface as model adapters. */
using EngineRuntimePlanFactory = std::shared_ptr<irt::IExecutionPlan> (*)(const EngineConfig &, int,
                                                                            const PipelinePlan *);

/** Create the production TensorRT runtime-plan adapter, or the test override. */
INFERRT_ENGINE_API std::shared_ptr<irt::IExecutionPlan> CreateEngineRuntimePlan(const EngineConfig &config,
                                                                                 int device_id,
                                                                                 const PipelinePlan *pipeline);

/** Replace the private runtime-plan factory; nullptr restores the production adapter. */
INFERRT_ENGINE_API void SetEngineRuntimePlanFactoryOverride(EngineRuntimePlanFactory factory) noexcept;

} // namespace irt::engine::priv
