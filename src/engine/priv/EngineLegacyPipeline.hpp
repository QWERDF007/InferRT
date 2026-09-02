#pragma once

#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include <memory>

namespace irt::engine::priv {

std::shared_ptr<const PipelinePlan> makeLegacyPipeline(const EngineConfig &config);

} // namespace irt::engine::priv
