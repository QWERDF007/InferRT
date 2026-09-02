#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <inferrt/model/Export.h>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 模型权重映射表。
 *
 * key 通常与 PyTorch state_dict 中的参数名称一致，value 为 TensorRT 权重对象。
 */
using WeightsMap = std::map<std::string, nvinfer1::Weights>;

/**
 * @brief 从文本格式 .wts 文件加载权重。
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

/**
 * @brief 计算 TensorRT 维度对象中的元素总数。
 * @param dims TensorRT 维度对象，每个维度必须为正数。
 * @return 所有维度相乘后的元素总数。
 * @throws irt::Exception 当任一维度小于等于 0 时抛出。
 */
INFERRT_MODEL_API size_t elementCount(const nvinfer1::Dims &dims);

/**
 * @brief 返回 TensorRT 数据类型对应的单元素字节数。
 * @param data_type TensorRT 数据类型。
 * @return 单个元素占用的字节数。
 * @throws irt::Exception 当数据类型暂不支持时抛出。
 */
INFERRT_MODEL_API size_t elementSize(nvinfer1::DataType data_type);

/**
 * @brief 返回 TensorRT 数据类型对应的单元素字节数。
 *
 * 该函数保留为 `elementSize()` 的兼容别名，供已有代码继续使用。
 *
 * @param data_type TensorRT 数据类型。
 * @return 单个元素占用的字节数。
 */
INFERRT_MODEL_API size_t dataTypeSize(nvinfer1::DataType data_type);

/**
 * @brief 将 TensorRT 数据类型转换为稳定的可读字符串。
 * @param data_type TensorRT 数据类型。
 * @return 类型名称，例如 `float32`、`int8`；未知类型返回 `unknown`。
 */
INFERRT_MODEL_API std::string dataTypeToString(nvinfer1::DataType data_type);

/**
 * @brief 将 TensorRT 维度格式化为逗号分隔字符串。
 * @param dims TensorRT 维度对象。
 * @return 不带括号的维度字符串，例如 `1,3,224,224`。
 */
INFERRT_MODEL_API std::string dimsToCsv(const nvinfer1::Dims &dims);

/**
 * @brief 将 TensorRT 维度格式化为可读字符串。
 * @param dims TensorRT 维度对象。
 * @return 带方括号的维度字符串，例如 `[1, 3, 224, 224]`。
 */
INFERRT_MODEL_API std::string dimsToString(const nvinfer1::Dims &dims);

/**
 * @brief 检查 CUDA 调用结果，失败时转换为 InferRT 异常。
 * @param status CUDA 返回状态。
 * @param op 当前 CUDA 操作名称，用于错误消息。
 * @throws irt::Exception 当 `status != cudaSuccess` 时抛出。
 */
INFERRT_MODEL_API void checkCuda(cudaError_t status, const char *op);

/**
 * @brief 将当前线程的 CUDA device 切换到指定编号。
 * @param device_id 从 0 开始的 CUDA 设备编号。
 * @throws irt::Exception 设备编号非法或 CUDA 切换失败时抛出。
 */
INFERRT_MODEL_API void setCudaDevice(int device_id);

} // namespace irt::model
