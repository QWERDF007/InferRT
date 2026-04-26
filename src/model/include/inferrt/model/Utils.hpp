#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>

#include <map>
#include <string>
#include <vector>

namespace irt::model {

using WeightsMap = std::map<std::string, nvinfer1::Weights>;

INFERRT_MODEL_API WeightsMap loadWeights(const std::string &file);

INFERRT_MODEL_API std::vector<std::string> readImagenetLabels(const std::string &label_file);

} // namespace irt::model