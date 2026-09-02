#pragma once

#include "EngineTypes.hpp"
#include "EngineRuntimePlan.hpp"

#include <inferrt/engine/InferenceEngine.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace irt::engine::priv {

void initializeEngineSlot(Slot &slot, const EngineConfig &config,
                          const std::shared_ptr<const PipelinePlan> &pipeline,
                          int fixed_batch_size,
                          size_t &device_arena_bytes,
                          std::unordered_map<int, size_t> &device_arena_bytes_by_device);

} // namespace irt::engine::priv
