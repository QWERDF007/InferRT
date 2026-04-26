#pragma once

#include <inferrt/model/Export.h>

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

    virtual int numClasses() const noexcept
    {
        return num_classes_;
    }

    virtual const InputShape &inputShape() const noexcept
    {
        return input_shape_;
    }

protected:
    int        num_classes_{1000};
    InputShape input_shape_{3, 224, 224};
};

} // namespace irt::model
