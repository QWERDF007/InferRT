#pragma once

#include <inferrt/model/Export.h>

#include <string>
#include <vector>

namespace irt::model {

struct INFERRT_MODEL_API InputShape
{
    int channels{3};
    int height{224};
    int width{224};
};

class INFERRT_MODEL_API IModelConfig
{
public:
    IModelConfig()          = default;
    virtual ~IModelConfig() = default;

    IModelConfig(int num_classes, const InputShape &input_shape)
        : num_classes_(num_classes)
        , input_shape_(input_shape)
    {
    }

    IModelConfig(int num_classes, const InputShape &input_shape, std::vector<std::string> input_tensor_names,
                 std::vector<std::string> output_tensor_names)
        : num_classes_(num_classes)
        , input_shape_(input_shape)
        , input_tensor_names_(std::move(input_tensor_names))
        , output_tensor_names_(std::move(output_tensor_names))
    {
    }

    virtual void setNumClasses(int num_classes)
    {
        num_classes_ = num_classes;
    }

    virtual void setInputShape(const InputShape &input_shape)
    {
        input_shape_ = input_shape;
    }

    virtual void setInputShape(int channels, int height, int width)
    {
        input_shape_ = InputShape{channels, height, width};
    }

    virtual void setInputTensorNames(std::vector<std::string> input_tensor_names)
    {
        input_tensor_names_ = std::move(input_tensor_names);
    }

    virtual void setOutputTensorNames(std::vector<std::string> output_tensor_names)
    {
        output_tensor_names_ = std::move(output_tensor_names);
    }

    virtual int numClasses() const noexcept
    {
        return num_classes_;
    }

    virtual const InputShape &inputShape() const noexcept
    {
        return input_shape_;
    }

    virtual const std::vector<std::string> &inputTensorNames() const noexcept
    {
        return input_tensor_names_;
    }

    virtual const std::vector<std::string> &outputTensorNames() const noexcept
    {
        return output_tensor_names_;
    }

protected:
    int                      num_classes_{1000};
    InputShape               input_shape_{3, 224, 224};
    std::vector<std::string> input_tensor_names_{"input"};
    std::vector<std::string> output_tensor_names_{"output"};
};

} // namespace irt::model
