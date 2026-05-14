#pragma once

#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>
#include <inferrt/model/Utils.hpp>

#include <memory>
#include <string>
#include <vector>

namespace irt::model::priv {

/**
 * @brief 模型内部实现基类。
 *
 * 该类集中管理模型配置、TensorRT engine 生命周期、张量命名与绑定、
 * 以及日志初始化等公共能力。
 */
class IModelImpl
{
public:
    /**
     * @brief 使用默认配置构造内部实现对象。
     */
    IModelImpl()
        : config_(std::make_unique<IModelConfig>())
    {
    }

    virtual ~IModelImpl() = default;

    /**
     * @brief 获取模型显示名称。
     * @return 模型名称。
     */
    virtual std::string name() const noexcept = 0;

    /**
     * @brief 获取权重文件扩展名。
     * @return 权重文件扩展名。
     */
    virtual std::string wtsExtension() const noexcept
    {
        return ".wts";
    }

    /**
     * @brief 获取 engine 文件扩展名。
     * @return engine 文件扩展名。
     */
    virtual std::string engineExtension() const noexcept
    {
        return ".engine";
    }

    /**
     * @brief 根据配置生成 engine 文件名后缀。
     * @param config 模型配置。
     * @return 后缀字符串，例如 `_3x224x224_1000`。
     */
    virtual std::string generateSuffix(const IModelConfig &config) const noexcept;

    /**
     * @brief 获取当前日志级别。
     * @return TensorRT 日志严重性级别。
     */
    nvinfer1::ILogger::Severity logLevel() const noexcept;

    void build(const std::string &weights_file);
    void save(const std::string &engine_file);
    void load(const std::string &engine_file);
    void buildOrLoad(const std::string &weights_file);

    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) = 0;

    virtual void infer(const std::vector<void *> &buffers) = 0;

    /**
     * @brief 设置完整模型配置。
     * @param config 配置对象。
     */
    void setModelConfig(std::unique_ptr<IModelConfig> config);
    void setNumClasses(int num_classes);
    void setInputShape(const InputShape &shape);
    void setInputShape(int channels, int height, int width);
    void setInputTensorNames(std::vector<std::string> input_tensor_names);
    void setOutputTensorNames(std::vector<std::string> output_tensor_names);

    const IModelConfig &modelConfig() const noexcept;

    int numClasses() const noexcept;

    const InputShape &inputShape() const noexcept;

    const std::vector<std::string> &inputTensorNames() const noexcept;
    const std::vector<std::string> &outputTensorNames() const noexcept;

    void setLogLevel(nvinfer1::ILogger::Severity severity);

    /**
     * @brief 向网络中添加输入张量。
     * @param network TensorRT 网络定义。
     * @param dims 输入张量维度。
     * @param data_type 输入张量数据类型。
     * @param input_index 输入张量名称索引。
     * @return 新增的输入张量。
     */
    nvinfer1::ITensor *addInputTensor(nvinfer1::INetworkDefinition *network, const nvinfer1::Dims &dims,
                                      nvinfer1::DataType data_type   = nvinfer1::DataType::kFLOAT,
                                      size_t             input_index = 0) const;

    /**
     * @brief 为输出张量命名并标记为网络输出。
     * @param network TensorRT 网络定义。
     * @param outputs 输出张量列表。
     */
    void markOutputTensors(nvinfer1::INetworkDefinition           *network,
                           const std::vector<nvinfer1::ITensor *> &outputs) const;

    /**
     * @brief 将用户缓冲区地址绑定到 TensorRT 执行上下文。
     * @param buffers 输入输出缓冲区地址列表。
     */
    void bindTensorAddresses(const std::vector<void *> &buffers);

    /**
     * @brief 获取可写 TensorRT 运行时参数。
     * @return 运行时参数引用。
     */
    TRTParams &trtParams() noexcept
    {
        return trt_params_;
    }

    /**
     * @brief 获取只读 TensorRT 运行时参数。
     * @return 运行时参数常量引用。
     */
    const TRTParams &trtParams() const noexcept
    {
        return trt_params_;
    }

    /**
     * @brief 初始化日志对象。
     */
    void initLogger();

private:
    /// 模型配置对象。
    std::unique_ptr<IModelConfig> config_;
    /// TensorRT 相关运行时对象。
    TRTParams                     trt_params_;
};

} // namespace irt::model::priv
