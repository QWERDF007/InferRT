#pragma once

#include <inferrt/model/Export.h>

#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 模型输入张量的空间尺寸描述。
 */
struct INFERRT_MODEL_API InputShape
{
    /// 输入通道数，例如 RGB 图像通常为 3。
    int channels{3};
    /// 输入高度。
    int height{224};
    /// 输入宽度。
    int width{224};
};

/**
 * @brief 模型配置基类。
 *
 * 该类统一描述模型的类别数、输入尺寸以及输入输出张量名称。
 */
class INFERRT_MODEL_API IModelConfig
{
public:
    /**
     * @brief 使用默认参数构造模型配置。
     */
    IModelConfig()          = default;
    virtual ~IModelConfig() = default;

    /**
     * @brief 使用类别数和输入尺寸构造模型配置。
     * @param num_classes 分类类别数。
     * @param input_shape 输入张量尺寸。
     */
    IModelConfig(int num_classes, const InputShape &input_shape)
        : num_classes_(num_classes)
        , input_shape_(input_shape)
    {
    }

    /**
     * @brief 使用完整参数构造模型配置。
     * @param num_classes 分类类别数。
     * @param input_shape 输入张量尺寸。
     * @param input_tensor_names 输入张量名称列表。
     * @param output_tensor_names 输出张量名称列表。
     */
    IModelConfig(int num_classes, const InputShape &input_shape, std::vector<std::string> input_tensor_names,
                 std::vector<std::string> output_tensor_names)
        : num_classes_(num_classes)
        , input_shape_(input_shape)
        , input_tensor_names_(std::move(input_tensor_names))
        , output_tensor_names_(std::move(output_tensor_names))
    {
    }

    /**
     * @brief 设置分类类别数。
     * @param num_classes 目标类别数。
     */
    virtual void setNumClasses(int num_classes)
    {
        num_classes_ = num_classes;
    }

    /**
     * @brief 设置输入尺寸。
     * @param input_shape 输入张量尺寸。
     */
    virtual void setInputShape(const InputShape &input_shape)
    {
        input_shape_ = input_shape;
    }

    /**
     * @brief 以通道、高度、宽度形式设置输入尺寸。
     * @param channels 输入通道数。
     * @param height 输入高度。
     * @param width 输入宽度。
     */
    virtual void setInputShape(int channels, int height, int width)
    {
        input_shape_ = InputShape{channels, height, width};
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
     * @param output_tensor_names 输出张量名称列表。
     */
    virtual void setOutputTensorNames(std::vector<std::string> output_tensor_names)
    {
        output_tensor_names_ = std::move(output_tensor_names);
    }

    /**
     * @brief 获取分类类别数。
     * @return 当前类别数。
     */
    virtual int numClasses() const noexcept
    {
        return num_classes_;
    }

    /**
     * @brief 获取输入尺寸。
     * @return 输入尺寸描述。
     */
    virtual const InputShape &inputShape() const noexcept
    {
        return input_shape_;
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

protected:
    /// 分类类别数，默认对应 ImageNet-1K。
    int                      num_classes_{1000};
    /// 默认输入尺寸为 3x224x224。
    InputShape               input_shape_{3, 224, 224};
    /// 默认输入张量名称。
    std::vector<std::string> input_tensor_names_{"input"};
    /// 默认输出张量名称。
    std::vector<std::string> output_tensor_names_{"output"};
};

} // namespace irt::model
