#pragma once

#include "EngineTypes.hpp"
#include "EngineRuntimePlan.hpp"

#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/InferenceEngine.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace irt::engine::priv {

void executePipelineStage(const std::vector<std::unique_ptr<IOperator>> &operators,
                          const std::vector<PipelinePlan::Node>         &nodes,
                          PipelineStage stage, const BatchState &batch, int device_id,
                          cudaStream_t stream, TensorViewMap &tensors, ResultMap &results);

void dispatchGpuBatch(Slot &slot, const BatchPtr &batch, const PipelinePlan &pipeline);

void processPostprocessBatch(const std::vector<std::unique_ptr<IOperator>>                                 &operators,
                             const BatchPtr                                                                &batch,
                             const PipelinePlan                                                            &pipeline,
                             const std::function<void(const RequestPtr &, InferenceResult)>                &on_success,
                             const std::function<void(const RequestPtr &, std::exception_ptr, FailureKind)> &on_failure);

} // namespace irt::engine::priv
