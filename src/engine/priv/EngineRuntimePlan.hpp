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

/**
 * @brief Engine-owned seam for a loaded executable runtime plan.
 *
 * The scheduler and slot code only depend on this backend-neutral execution
 * interface.  TensorRT deserialization is one adapter; tests can provide a
 * valid session adapter without changing the production lifecycle.
 */
class INFERRT_ENGINE_API IEngineRuntimePlan : public irt::IExecutionDescriptor, public irt::IExecutionPlan
{
public:
    virtual ~IEngineRuntimePlan() = default;

    /** Positive value means a fixed batch; zero means a dynamic batch. */
    [[nodiscard]] virtual int fixedBatchSize() const noexcept = 0;
};

using EngineRuntimePlanFactory = std::shared_ptr<IEngineRuntimePlan> (*)(const EngineConfig &, int,
                                                                          const PipelinePlan *);

/** Create the production TensorRT runtime-plan adapter, or the test override. */
INFERRT_ENGINE_API std::shared_ptr<IEngineRuntimePlan> CreateEngineRuntimePlan(const EngineConfig &config,
                                                                                 int device_id,
                                                                                 const PipelinePlan *pipeline);

/** Replace the private runtime-plan factory; nullptr restores the production adapter. */
INFERRT_ENGINE_API void SetEngineRuntimePlanFactoryOverride(EngineRuntimePlanFactory factory) noexcept;

} // namespace irt::engine::priv
