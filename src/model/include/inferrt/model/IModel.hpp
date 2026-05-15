#pragma once

#include "IModelConfig.hpp"
#include "IParams.hpp"
#include "Utils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace irt::model::priv {
class IModelImpl;
}

namespace irt::model {

/**
 * @brief 面向外部的统一模型包装类。
 *
 * 该类将不同模型实现封装为一致的构建、加载、配置与推理接口，
 * 调用方无需直接接触具体的内部实现类型。
 */
class INFERRT_MODEL_API IModel
{
public:
    /**
     * @brief 构造一个空模型包装对象。
     */
    IModel();

    /**
     * @brief 使用具体模型实现构造包装对象。
     * @param impl 模型内部实现对象。
     */
    explicit IModel(std::unique_ptr<priv::IModelImpl> impl);
    ~IModel();

    IModel(const IModel &)            = delete;
    IModel &operator=(const IModel &) = delete;
    IModel(IModel &&) noexcept;
    IModel &operator=(IModel &&) noexcept;

    /**
     * @brief 获取模型显示名称。
     * @return 模型名称。
     */
    virtual std::string name() const noexcept;

    /**
     * @brief 获取模型权重文件扩展名。
     * @return 权重文件扩展名。
     */
    virtual std::string wtsExtension() const noexcept;

    /**
     * @brief 获取 TensorRT engine 文件扩展名。
     * @return engine 文件扩展名。
     */
    virtual std::string engineExtension() const noexcept;

    /**
     * @brief 获取当前日志级别。
     * @return TensorRT 日志严重性级别。
     */
    virtual nvinfer1::ILogger::Severity logLevel() const noexcept;

    /**
     * @brief 从权重文件构建 engine。
     * @param weights_file 权重文件路径。
     */
    virtual void build(const std::string &weights_file);

    /**
     * @brief 保存当前 engine。
     * @param weights_file 目标文件路径。
     */
    virtual void save(const std::string &weights_file);

    /**
     * @brief 加载已有 engine。
     * @param weights_file engine 文件路径。
     */
    virtual void load(const std::string &weights_file);

    /**
     * @brief 优先加载已有 engine，不存在时再构建。
     * @param weights_file 权重文件路径。
     */
    virtual void buildOrLoad(const std::string &weights_file);

    /**
     * @brief 构建 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map);

    /**
     * @brief 执行一次推理。
     * @param buffers 输入输出缓冲区地址列表。
     */
    virtual void infer(const std::vector<void *> &buffers);

    /**
     * @brief 设置模型配置。
     * @param config 模型配置对象。
     */
    virtual void setModelConfig(std::unique_ptr<IModelConfig> config);

    /**
     * @brief 设置类别数。
     * @param num_classes 分类类别数。
     */
    virtual void setNumClasses(int num_classes);

    /**
     * @brief 设置输入尺寸。
     * @param shape 输入尺寸。
     */
    virtual void setInputShape(const InputShape &shape);

    /**
     * @brief 设置输入尺寸。
     * @param channels 输入通道数。
     * @param height 输入高度。
     * @param width 输入宽度。
     */
    virtual void setInputShape(int channels, int height, int width);

    /**
     * @brief 设置输入张量名称列表。
     * @param input_tensor_names 输入张量名称列表。
     */
    virtual void setInputTensorNames(std::vector<std::string> input_tensor_names);

    /**
     * @brief 设置输出张量名称列表。
     * @param output_tensor_names 输出张量名称列表。
     */
    virtual void setOutputTensorNames(std::vector<std::string> output_tensor_names);

    /**
     * @brief 获取当前模型配置。
     * @return 模型配置的常量引用。
     */
    virtual const IModelConfig &modelConfig() const noexcept;

    /**
     * @brief 获取类别数。
     * @return 当前类别数。
     */
    virtual int numClasses() const noexcept;

    /**
     * @brief 获取输入尺寸。
     * @return 输入尺寸。
     */
    virtual const InputShape &inputShape() const noexcept;

    /**
     * @brief 获取输入张量名称列表。
     * @return 输入张量名称列表。
     */
    virtual const std::vector<std::string> &inputTensorNames() const noexcept;

    /**
     * @brief 获取输出张量名称列表。
     * @return 输出张量名称列表。
     */
    virtual const std::vector<std::string> &outputTensorNames() const noexcept;

    /**
     * @brief 获取当前 engine 中指定类型的 I/O 张量名称。
     * @param mode TensorRT 张量 I/O 类型。
     * @return 张量名称列表。
     */
    virtual std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const;

    /**
     * @brief 获取指定张量的运行时形状。
     * @param tensor_name 张量名称。
     * @return 张量维度。
     */
    virtual nvinfer1::Dims tensorShape(const std::string &tensor_name) const;

    /**
     * @brief 获取指定张量的数据类型。
     * @param tensor_name 张量名称。
     * @return TensorRT 数据类型。
     */
    virtual nvinfer1::DataType tensorDataType(const std::string &tensor_name) const;

    /**
     * @brief 设置输入张量的运行时形状。
     * @param tensor_name 张量名称。
     * @param dims 运行时维度。
     */
    virtual void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims);

    /**
     * @brief 设置日志级别。
     * @param severity TensorRT 日志严重性级别。
     */
    virtual void setLogLevel(nvinfer1::ILogger::Severity severity);

private:
    std::unique_ptr<priv::IModelImpl> impl_;
};

} // namespace irt::model
