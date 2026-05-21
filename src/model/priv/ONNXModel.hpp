#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief ONNX 通用模型实现。
 *
 * 该模型通过 TensorRT ONNX parser 从 .onnx 文件构建 engine，适用于非内置网络结构。
 */
class ONNXModel : public priv::IModelImpl
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `onnx`。
     */
    static constexpr const char *key() noexcept
    {
        return "onnx";
    }

    /**
     * @brief 构造 ONNX 模型实现。
     */
    explicit ONNXModel()
        : priv::IModelImpl()
    {
    }

    /**
     * @brief 析构 ONNX 模型实现。
     */
    ~ONNXModel() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ONNX`。
     */
    std::string name() const noexcept override
    {
        return "ONNX";
    }

    /**
     * @brief 获取 ONNX 权重文件扩展名。
     * @return `.onnx`。
     */
    std::string wtsExtension() const noexcept override
    {
        return ".onnx";
    }

    /**
     * @brief 从 ONNX 文件构建 TensorRT engine。
     * @param onnx_file ONNX 文件路径。
     */
    void build(const std::string &onnx_file) override;

    /**
     * @brief 加载已有 TensorRT engine，并同步 engine 中的 I/O metadata。
     * @param engine_file engine 文件路径。
     */
    void load(const std::string &engine_file) override;

    /**
     * @brief 优先加载已有 engine，不存在时再从 ONNX 文件构建。
     * @param onnx_file ONNX 文件路径。
     */
    void buildOrLoad(const std::string &onnx_file) override;

    /**
     * @brief 在指定 CUDA stream 上执行 ONNX 模型推理。
     * @param buffers 输入输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     */
    void infer(const std::vector<void *> &buffers, cudaStream_t stream = nullptr,
               bool non_blocking = false) override;

    /**
     * @brief ONNX 模型不支持手工构建网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
