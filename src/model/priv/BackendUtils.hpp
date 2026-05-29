#pragma once

#include "BackendRuntime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace irt::model::priv {

/**
 * @brief 计算张量元素总数。
 * @param dims TensorRT 维度的张量形状。
 * @param tensor_name 张量名称，用于错误信息。
 * @return 各维度乘积；秩为 0 时返回 1。
 * @throws irt::Exception 秩无效或存在未解析（非正）维度时抛出。
 */
size_t TensorElementCount(const nvinfer1::Dims &dims, const std::string &tensor_name);

/**
 * @brief 将 TensorRT 维度转换为 int64_t 形状向量。
 * @param dims TensorRT 维度的张量形状。
 * @param tensor_name 张量名称，用于错误信息。
 * @return 与 ``dims`` 等长的 int64 形状列表。
 * @throws irt::Exception 秩无效或存在未解析（非正）维度时抛出。
 */
std::vector<int64_t> DimsToInt64Shape(const nvinfer1::Dims &dims, const std::string &tensor_name);

/**
 * @brief 将 TensorRT 维度转换为 size_t 形状向量。
 * @param dims TensorRT 维度的张量形状。
 * @param tensor_name 张量名称，用于错误信息。
 * @return 与 ``dims`` 等长的 size_t 形状列表。
 * @throws irt::Exception 秩无效或存在未解析（非正）维度时抛出。
 */
std::vector<size_t> DimsToSizeTShape(const nvinfer1::Dims &dims, const std::string &tensor_name);

/**
 * @brief 将 int64_t 形状向量转换为 TensorRT 维度。
 * @param shape 张量形状，各维度须能安全转为 int32_t。
 * @return 对应的 ``nvinfer1::Dims``。
 * @throws irt::Exception 秩超过 ``Dims::MAX_DIMS`` 或某维度超出 int32 范围时抛出。
 */
nvinfer1::Dims Int64ShapeToDims(const std::vector<int64_t> &shape);

/**
 * @brief 判断模型配置是否仍为默认主输出设置。
 * @param config 待检查的模型配置。
 * @return 输出名为单元素 ``"output"`` 且未启用 ``featureOnly`` 时返回 true。
 */
bool IsDefaultOutputConfig(const IModelConfig &config);

} // namespace irt::model::priv
