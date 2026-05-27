#pragma once

#include <inferrt/model/Export.h>

#include <algorithm>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 获取指定模型的常见特征层名称列表。
 *
 * 这些名称可直接用于 IModelConfig::setFeatureTensorNames()，
 * 也可作为模型支持导出的中间层参考。
 *
 * @param model_key 模型注册 key，例如 "resnet18"、"dinov2_vits14"。
 * @return 该模型的常见特征层名称；未知 key 返回空列表。
 */
INFERRT_MODEL_API std::vector<std::string> modelFeatureNames(const std::string &model_key);

} // namespace irt::model
