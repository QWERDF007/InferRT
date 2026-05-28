#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>

#include <string>
#include <vector>

namespace irt::model {

enum class ModelBackend
{
    TensorRT,
    OpenVINO,
    ONNXRuntime
};

enum class ModelDevice
{
    CPU,
    GPU
};

/**
 * @brief 模型配置基类。
 *
 * 统一描述模型的类别数、输入张量尺寸、输入张量名称、主输出张量名称，
 * 以及可选的中间特征提取配置。
 * 输入尺寸使用 NCHW 顺序的 nvinfer1::Dims4 表示，并与输入张量名称按索引一一对应。
 */
class INFERRT_MODEL_API IModelConfig
{
public:
    /**
     * @brief 使用默认 ImageNet-1K 分类配置构造。
     *
     * 默认输入张量名为 `input`，输出张量名为 `output`，输入尺寸为 1x3x224x224。
     */
    IModelConfig()          = default;
    virtual ~IModelConfig() = default;

    /**
     * @brief 设置类别数。
     * @param num_classes 目标类别数。
     */
    virtual void setNumClasses(int num_classes)
    {
        num_classes_ = num_classes;
    }

    /**
     * @brief 设置第一个输入张量尺寸。
     * @param input_shape 输入张量尺寸，按 NCHW 顺序表示。
     *
     * 该接口面向单输入模型；若当前输入尺寸列表为空，则会新增第一个输入尺寸。
     */
    virtual void setInputShape(const nvinfer1::Dims4 &input_shape)
    {
        if (input_shapes_.empty())
        {
            input_shapes_.push_back(input_shape);
            return;
        }
        input_shapes_.front() = input_shape;
    }

    /**
     * @brief 设置多个输入张量尺寸。
     * @param input_shapes 输入张量尺寸列表，顺序与输入张量名称列表一致。
     */
    virtual void setInputShapes(std::vector<nvinfer1::Dims4> input_shapes)
    {
        input_shapes_ = std::move(input_shapes);
    }

    /**
     * @brief 设置输入张量名称列表。
     * @param input_tensor_names 输入张量名称列表。
     */
    virtual void setInputTensorNames(std::vector<std::string> input_tensor_names)
    {
        input_tensor_names_ = std::move(input_tensor_names);
    }

    /**
     * @brief 设置输出张量名称列表。
     *
     * 分类时为 logits 等主输出；``featureOnly`` 时为各特征面的 TRT 张量名，
     * 数量须与 ``feature_tensor_names`` 一致。
     */
    virtual void setOutputTensorNames(std::vector<std::string> output_tensor_names)
    {
        output_tensor_names_ = std::move(output_tensor_names);
    }

    /**
     * @brief 设置需要额外导出的中间特征层 key 列表。
     * @param feature_tensor_names 特征层 key 列表，顺序即输出顺序。
     *
     * 这些 key 由具体模型实现定义，例如 `layer1`、`layer4`、`avgpool` 等。
     */
    virtual void setFeatureTensorNames(std::vector<std::string> feature_tensor_names)
    {
        feature_tensor_names_ = std::move(feature_tensor_names);
    }

    /**
     * @brief 设置是否仅构建特征提取裁剪网络。
     * @param feature_only 为 true 时，当前模型实例只构建到请求特征为止的网络。
     */
    virtual void setFeatureOnly(bool feature_only)
    {
        feature_only_ = feature_only;
    }

    virtual void setBackend(ModelBackend backend) noexcept
    {
        backend_ = backend;
    }

    virtual void setDevice(ModelDevice device) noexcept
    {
        device_ = device;
    }

    /**
     * @brief 获取类别数。
     * @return 当前类别数。
     */
    virtual int numClasses() const noexcept
    {
        return num_classes_;
    }

    /**
     * @brief 获取第一个输入张量尺寸。
     * @return 第一个输入张量尺寸，按 NCHW 顺序表示。
     *
     * 该接口面向单输入模型；多输入模型应优先使用 inputShapes()。
     */
    virtual const nvinfer1::Dims4 &inputShape() const noexcept
    {
        return input_shapes_.front();
    }

    /**
     * @brief 获取多个输入张量尺寸。
     * @return 输入张量尺寸列表，顺序与输入张量名称列表一致。
     */
    virtual const std::vector<nvinfer1::Dims4> &inputShapes() const noexcept
    {
        return input_shapes_;
    }

    /**
     * @brief 获取输入张量名称列表。
     * @return 输入张量名称列表。
     */
    virtual const std::vector<std::string> &inputTensorNames() const noexcept
    {
        return input_tensor_names_;
    }

    /**
     * @brief 获取输出张量名称列表。
     * @return 输出张量名称列表。
     */
    virtual const std::vector<std::string> &outputTensorNames() const noexcept
    {
        return output_tensor_names_;
    }

    /**
     * @brief 获取请求导出的中间特征层 key 列表。
     * @return 特征层 key 列表。
     */
    virtual const std::vector<std::string> &featureTensorNames() const noexcept
    {
        return feature_tensor_names_;
    }

    virtual bool featureOnly() const noexcept
    {
        return feature_only_;
    }

    virtual ModelBackend backend() const noexcept
    {
        return backend_;
    }

    virtual ModelDevice device() const noexcept
    {
        return device_;
    }

protected:
    /// 类别数，默认对应 ImageNet-1K。
    int num_classes_{1000};

    /// 输入张量尺寸列表，默认输入为 1x3x224x224。
    std::vector<nvinfer1::Dims4> input_shapes_{
        nvinfer1::Dims4{1, 3, 224, 224}
    };

    std::vector<std::string> input_tensor_names_{"input"};

    /// 输出张量名称列表。
    std::vector<std::string> output_tensor_names_{"output"};

    /// 仅用于建网：在 NamedTensorMap 中选取的中间层 key。
    std::vector<std::string> feature_tensor_names_{};

    bool feature_only_{false};

    ModelBackend backend_{ModelBackend::TensorRT};

    ModelDevice device_{ModelDevice::GPU};
};

} // namespace irt::model
