#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>

#include <cstdint>
#include <limits>
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
 * @brief TensorRT 构建时使用的计算精度。
 *
 * FP16 模式保持模型 I/O 为 float32，并允许 TensorRT 为内部 layer 选择
 * FP16 tactic；这样既能与现有 Python/CPU 输入接口兼容，也能单独缓存
 * FP16 与 FP32 engine。
 */
enum class ModelPrecision
{
    FP32,
    FP16
};

INFERRT_MODEL_API inline const char *modelBackendName(irt::model::ModelBackend backend)
{
    switch (backend)
    {
    case irt::model::ModelBackend::TensorRT:
        return "tensorrt";
    case irt::model::ModelBackend::OpenVINO:
        return "openvino";
    case irt::model::ModelBackend::ONNXRuntime:
        return "onnxruntime";
    }
    return "unknown";
}

INFERRT_MODEL_API inline const char *modelDeviceName(irt::model::ModelDevice device)
{
    switch (device)
    {
    case irt::model::ModelDevice::CPU:
        return "cpu";
    case irt::model::ModelDevice::GPU:
        return "gpu";
    }
    return "unknown";
}

INFERRT_MODEL_API inline const char *modelPrecisionName(irt::model::ModelPrecision precision)
{
    switch (precision)
    {
    case irt::model::ModelPrecision::FP32:
        return "fp32";
    case irt::model::ModelPrecision::FP16:
        return "fp16";
    }
    return "unknown";
}

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
            syncDynamicBatchToInputBatch(input_shape.d[0]);
            return;
        }
        input_shapes_.front() = input_shape;
        syncDynamicBatchToInputBatch(input_shape.d[0]);
    }

    /**
     * @brief 设置多个输入张量尺寸。
     * @param input_shapes 输入张量尺寸列表，顺序与输入张量名称列表一致。
     */
    virtual void setInputShapes(std::vector<nvinfer1::Dims4> input_shapes)
    {
        input_shapes_ = std::move(input_shapes);
        if (!input_shapes_.empty())
        {
            syncDynamicBatchToInputBatch(input_shapes_.front().d[0]);
        }
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

    /**
     * @brief 设置是否启用 TensorRT 动态 batch。
     * @param dynamic_batch 为 true 时，TensorRT 建网阶段将输入 batch 维声明为动态维。
     *
     * 启用后若尚未显式设置动态 batch 范围，会使用当前第一个输入尺寸的 N 维作为 opt/max，
     * 并以 1 作为 min。实际运行时 batch 必须落在该范围内。
     */
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
            syncDynamicBatchToInputBatch(input_shapes_.front().d[0]);
        }
    }

    /**
     * @brief 设置 TensorRT 动态 batch 的 profile 范围。
     * @param min_batch 最小 batch。
     * @param opt_batch TensorRT 优化 batch。
     * @param max_batch 最大 batch。
     *
     * 调用该接口会自动启用动态 batch。参数合法性在 build/load 前统一校验。
     */
    virtual void setDynamicBatchRange(int min_batch, int opt_batch, int max_batch) noexcept
    {
        dynamic_batch_                = true;
        dynamic_batch_range_explicit_ = true;
        min_batch_size_               = min_batch;
        opt_batch_size_               = opt_batch;
        max_batch_size_               = max_batch;
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
     * @brief 设置模型使用的 GPU 设备编号。
     *
     * CPU 后端会保留该配置但不会使用它；GPU 后端使用从 0 开始的 CUDA/OpenVINO 设备编号。
     * 具体设备是否存在在模型加载或构建时由对应后端校验。
     *
     * @param device_id 从 0 开始的设备编号。
     */
    virtual void setDeviceId(int device_id) noexcept
    {
        device_id_ = device_id;
    }

    virtual void setPrecision(ModelPrecision precision) noexcept
    {
        precision_ = precision;
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

    /**
     * @brief 查询是否启用 TensorRT 动态 batch。
     * @return 启用时返回 true。
     */
    virtual bool dynamicBatch() const noexcept
    {
        return dynamic_batch_;
    }

    /**
     * @brief 获取动态 batch 的最小值。
     * @return TensorRT optimization profile 的 min batch。
     */
    virtual int minBatchSize() const noexcept
    {
        return min_batch_size_;
    }

    /**
     * @brief 获取动态 batch 的优化值。
     * @return TensorRT optimization profile 的 opt batch。
     */
    virtual int optBatchSize() const noexcept
    {
        return opt_batch_size_;
    }

    /**
     * @brief 获取动态 batch 的最大值。
     * @return TensorRT optimization profile 的 max batch。
     */
    virtual int maxBatchSize() const noexcept
    {
        return max_batch_size_;
    }

    virtual ModelBackend backend() const noexcept
    {
        return backend_;
    }

    virtual ModelDevice device() const noexcept
    {
        return device_;
    }

    /**
     * @brief 获取模型使用的 GPU 设备编号。
     * @return 从 0 开始的设备编号。
     */
    virtual int deviceId() const noexcept
    {
        return device_id_;
    }

    virtual ModelPrecision precision() const noexcept
    {
        return precision_;
    }

protected:
    /**
     * @brief 在动态 batch 已启用时，用输入 N 维同步默认 profile 范围。
     * @param input_batch 当前输入尺寸中的 batch 值。
     */
    void syncDynamicBatchToInputBatch(int64_t input_batch) noexcept
    {
        if (!dynamic_batch_ || dynamic_batch_range_explicit_ || input_batch <= 0
            || input_batch > std::numeric_limits<int>::max())
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

    bool dynamic_batch_{false};

    bool dynamic_batch_range_explicit_{false};

    int min_batch_size_{1};

    int opt_batch_size_{1};

    int max_batch_size_{1};

    ModelBackend backend_{ModelBackend::TensorRT};

    ModelDevice device_{ModelDevice::GPU};

    int device_id_{0};

    ModelPrecision precision_{ModelPrecision::FP32};
};

} // namespace irt::model
