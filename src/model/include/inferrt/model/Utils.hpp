#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>

#include <map>
#include <string>

namespace irt::model {

using WeightsMap = std::map<std::string, nvinfer1::Weights>;

INFERRT_MODEL_API WeightsMap loadWeights(const std::string &file);

} // namespace irt::model