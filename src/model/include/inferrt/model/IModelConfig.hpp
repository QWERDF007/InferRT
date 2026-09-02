#pragma once

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/model/Export.h>
#include <inferrt/model/ModelRuntime.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace irt::model {

/** Precision requested when a backend builds a model. */
enum class ModelPrecision
{
    FP32,
    FP16
};

INFERRT_MODEL_API inline const char *modelPrecisionName(ModelPrecision precision) noexcept
{
    switch (precision)
    {
    case ModelPrecision::FP32:
        return "FP32";
    case ModelPrecision::FP16:
        return "FP16";
    }
    return "Unknown";
}

/** Backend-neutral model configuration shared by all model implementations. */
class INFERRT_MODEL_API IModelConfig
{
public:
    IModelConfig()          = default;
    virtual ~IModelConfig() = default;

    virtual void setNumClasses(int num_classes)
    {
        if (num_classes <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Number of classes must be positive");
        }
        num_classes_ = num_classes;
    }

    virtual void setInputShape(irt::Shape input_shape)
    {
        validateShape(input_shape, "Input shape");
        if (input_shapes_.empty())
        {
            input_shapes_.push_back(std::move(input_shape));
        }
        else
        {
            input_shapes_.front() = std::move(input_shape);
        }
        syncDynamicBatchToInputBatch(input_shapes_.front()[0]);
    }

    virtual void setInputShapes(std::vector<irt::Shape> input_shapes)
    {
        if (input_shapes.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input shapes list cannot be empty");
        }
        for (const auto &shape : input_shapes)
        {
            validateShape(shape, "Input shape");
        }
        input_shapes_ = std::move(input_shapes);
        syncDynamicBatchToInputBatch(input_shapes_.front()[0]);
    }

    virtual void setInputTensorNames(std::vector<std::string> names)
    {
        validateNames(names, "Input tensor");
        input_tensor_names_ = std::move(names);
    }

    virtual void setOutputTensorNames(std::vector<std::string> names)
    {
        validateNames(names, "Output tensor");
        output_tensor_names_ = std::move(names);
    }

    virtual void setFeatureTensorNames(std::vector<std::string> names)
    {
        validateNames(names, "Feature tensor", false);
        feature_tensor_names_ = std::move(names);
    }

    virtual void setFeatureOnly(bool feature_only) noexcept
    {
        feature_only_ = feature_only;
    }

    virtual void setDynamicBatch(bool dynamic_batch) noexcept
    {
        dynamic_batch_ = dynamic_batch;
        if (!dynamic_batch_)
        {
            dynamic_batch_range_explicit_ = false;
            return;
        }
        if (!dynamic_batch_range_explicit_ && !input_shapes_.empty())
        {
            syncDynamicBatchToInputBatch(input_shapes_.front()[0]);
        }
    }

    virtual void setDynamicBatchRange(int min_batch, int opt_batch, int max_batch)
    {
        if (min_batch <= 0 || opt_batch <= 0 || max_batch <= 0 || min_batch > opt_batch || opt_batch > max_batch)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid dynamic batch range: [%d, %d, %d]",
                                 min_batch, opt_batch, max_batch);
        }
        dynamic_batch_                = true;
        dynamic_batch_range_explicit_ = true;
        min_batch_size_               = min_batch;
        opt_batch_size_               = opt_batch;
        max_batch_size_               = max_batch;
    }

    virtual void setRuntime(ModelRuntime runtime) noexcept
    {
        runtime_ = std::move(runtime);
    }

    virtual void setPrecision(ModelPrecision precision) noexcept
    {
        precision_ = precision;
    }

    [[nodiscard]] virtual int numClasses() const noexcept
    {
        return num_classes_;
    }

    [[nodiscard]] virtual const irt::Shape &inputShape() const
    {
        if (input_shapes_.empty())
        {
            throw irt::Exception(Status::INVALID_OPERATION, "No input shapes configured in IModelConfig");
        }
        return input_shapes_.front();
    }

    [[nodiscard]] virtual const std::vector<irt::Shape> &inputShapes() const noexcept
    {
        return input_shapes_;
    }

    [[nodiscard]] virtual const std::vector<std::string> &inputTensorNames() const noexcept
    {
        return input_tensor_names_;
    }

    [[nodiscard]] virtual const std::vector<std::string> &outputTensorNames() const noexcept
    {
        return output_tensor_names_;
    }

    [[nodiscard]] virtual const std::vector<std::string> &featureTensorNames() const noexcept
    {
        return feature_tensor_names_;
    }

    [[nodiscard]] virtual bool featureOnly() const noexcept
    {
        return feature_only_;
    }

    [[nodiscard]] virtual bool dynamicBatch() const noexcept
    {
        return dynamic_batch_;
    }

    [[nodiscard]] virtual int minBatchSize() const noexcept
    {
        return min_batch_size_;
    }

    [[nodiscard]] virtual int optBatchSize() const noexcept
    {
        return opt_batch_size_;
    }

    [[nodiscard]] virtual int maxBatchSize() const noexcept
    {
        return max_batch_size_;
    }

    [[nodiscard]] virtual const ModelRuntime &runtime() const noexcept
    {
        return runtime_;
    }

    [[nodiscard]] virtual ModelPrecision precision() const noexcept
    {
        return precision_;
    }

protected:
    static void validateShape(const irt::Shape &shape, const char *label)
    {
        if (shape.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s cannot be empty", label);
        }
        for (const auto dimension : shape.dims)
        {
            if (dimension <= 0)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s dimensions must be positive", label);
            }
        }
    }

    static void validateNames(const std::vector<std::string> &names, const char *label, bool require_non_empty = true)
    {
        if (require_non_empty && names.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s names cannot be empty", label);
        }
        std::unordered_set<std::string> seen;
        for (const auto &name : names)
        {
            if (name.empty())
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s tensor name cannot be empty", label);
            }
            if (!seen.insert(name).second)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Duplicate %s tensor name: %s", label,
                                     name.c_str());
            }
        }
    }

    void syncDynamicBatchToInputBatch(int64_t input_batch) noexcept
    {
        if (!dynamic_batch_ || dynamic_batch_range_explicit_ || input_batch <= 0
            || input_batch > (std::numeric_limits<int>::max)())
        {
            return;
        }
        const int batch = static_cast<int>(input_batch);
        min_batch_size_ = 1;
        opt_batch_size_ = batch;
        if (max_batch_size_ < opt_batch_size_)
        {
            max_batch_size_ = opt_batch_size_;
        }
    }

    int                         num_classes_{1000};
    std::vector<irt::Shape>     input_shapes_{irt::Shape{1, 3, 224, 224}};
    std::vector<std::string>    input_tensor_names_{"input"};
    std::vector<std::string>    output_tensor_names_{"output"};
    std::vector<std::string>    feature_tensor_names_{};
    bool                        feature_only_{false};
    bool                        dynamic_batch_{false};
    bool                        dynamic_batch_range_explicit_{false};
    int                         min_batch_size_{1};
    int                         opt_batch_size_{1};
    int                         max_batch_size_{1};
    ModelRuntime                runtime_{};
    ModelPrecision              precision_{ModelPrecision::FP32};
};

} // namespace irt::model
