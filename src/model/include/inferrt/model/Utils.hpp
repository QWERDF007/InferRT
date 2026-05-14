#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>

#include <map>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 模型权重映射表。
 *
 * key 通常与 PyTorch `state_dict` 中的参数名称一致。
 */
using WeightsMap = std::map<std::string, nvinfer1::Weights>;

/**
 * @brief 从文本格式 `.wts` 文件加载权重。
 * @param file 权重文件路径。
 * @return 解析后的权重映射表。
 */
INFERRT_MODEL_API WeightsMap loadWeights(const std::string &file);

/**
 * @brief 读取 ImageNet 1000 类标签文件。
 * @param label_file 标签文件路径。
 * @return 长度为 1000 的标签列表。
 */
INFERRT_MODEL_API std::vector<std::string> readImagenetLabels(const std::string &label_file);

} // namespace irt::model
