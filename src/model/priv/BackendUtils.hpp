#pragma once

#include <inferrt/model/IModelConfig.hpp>

#include <NvInfer.h>

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

/** Convert the public backend-neutral shape at the TensorRT adapter boundary. */
inline nvinfer1::Dims ShapeToDims(const irt::Shape &shape)
{
    return Int64ShapeToDims(shape.dims);
}

/**
 * @brief 判断模型配置是否仍为默认主输出设置。
 * @param config 待检查的模型配置。
 * @return 输出名为单元素 ``"output"`` 且未启用 ``featureOnly`` 时返回 true。
 */
bool IsDefaultOutputConfig(const IModelConfig &config);

/**
 * @brief 将已解析的输入 batch 维同步到动态 batch 输出。
 *
 * ONNX Runtime / OpenVINO 的图元数据会把符号 batch 读成 ``-1``。调用方在运行时通过
 * ``setTensorShape`` 解析输入形状后，输出的第 0 维也需要同步，否则自动分配输出缓冲区时
 * 仍会看到未解析维度。
 *
 * @tparam TensorInfoMap value 类型需包含 ``shape`` 与 ``dynamic_batch`` 字段。
 * @param output_info 输出张量元数据表。
 * @param input_shape 已解析的输入张量形状。
 */
template<typename TensorInfoMap>
void PropagateResolvedBatchDimToDynamicOutputs(TensorInfoMap &output_info, const irt::Shape &input_shape)
{
    if (input_shape.empty() || input_shape[0] <= 0)
    {
        return;
    }

    for (auto &entry : output_info)
    {
        auto &info = entry.second;
        if (info.dynamic_batch && !info.shape.empty())
        {
            info.shape[0] = input_shape[0];
        }
    }
}

} // namespace irt::model::priv
