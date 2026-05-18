#pragma once

#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>
#include <inferrt/model/Utils.hpp>

#include <memory>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

namespace irt::model::priv {

/**
 * @brief 模型内部实现基类。
 *
 * 该类集中管理模型配置、TensorRT engine 生命周期、张量命名与绑定、
 * 运行时查询以及日志初始化等公共能力。具体模型只需要实现模型名称、
 * 网络构建和推理入口。
 */
class IModelImpl
{
public:
    using NamedTensorMap = std::unordered_map<std::string, nvinfer1::ITensor *>;
    enum class BuildVariant
    {
        Primary,
        Feature,
    };

    /**
     * @brief 使用默认模型配置构造内部实现对象。
     */
    IModelImpl()
        : config_(std::make_unique<IModelConfig>())
    {
    }

    /**
     * @brief 析构内部实现对象。
     */
    virtual ~IModelImpl() = default;

    /**
     * @brief 获取模型显示名称。
     * @return 模型名称。
     */
    virtual std::string name() const noexcept = 0;

    /**
     * @brief 获取权重文件扩展名。
     * @return 权重文件扩展名，默认返回 `.wts`。
     */
    virtual std::string wtsExtension() const noexcept
    {
        return ".wts";
    }

    /**
     * @brief 获取 TensorRT engine 文件扩展名。
     * @return engine 文件扩展名，默认返回 `.engine`。
     */
    virtual std::string engineExtension() const noexcept
    {
        return ".engine";
    }

    /**
     * @brief 根据模型配置生成 engine 文件名后缀。
     * @param config 模型配置。
     * @return 后缀字符串，例如 `_3x224x224_1000`。
     */
    virtual std::string generateSuffix(const IModelConfig &config) const noexcept;

    /**
     * @brief 获取当前日志级别。
     * @return TensorRT 日志严重性级别。
     */
    nvinfer1::ILogger::Severity logLevel() const noexcept;

    /**
     * @brief 从权重文件构建模型。
     * @param weights_file 权重文件路径。
     *
     * 构建时会根据当前配置生成 TensorRT engine，并保存到派生出的 engine 文件路径。
     */
    virtual void build(const std::string &weights_file);

    /**
     * @brief 保存当前 TensorRT engine。
     * @param engine_file 目标 engine 文件路径。
     */
    virtual void save(const std::string &engine_file);

    /**
     * @brief 加载已有 TensorRT engine。
     * @param engine_file engine 文件路径。
     */
    virtual void load(const std::string &engine_file);

    /**
     * @brief 优先加载已有 engine，不存在时再从权重文件构建。
     * @param weights_file 权重文件路径。
     */
    virtual void buildOrLoad(const std::string &weights_file);

    /**
     * @brief 构建 TensorRT 网络定义。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) = 0;

    /**
     * @brief 执行一次推理。
     * @param buffers 输入输出缓冲区地址列表。
     */
    virtual void infer(const std::vector<void *> &buffers) = 0;

    /**
     * @brief 执行一次特征提取前向。
     * @param buffers 输入与特征输出缓冲区地址列表。
     */
    virtual void forwardFeatures(const std::vector<void *> &buffers);

    /**
     * @brief 设置完整模型配置。
     * @param config 模型配置对象；传入空指针时恢复为默认配置。
     */
    void setModelConfig(std::unique_ptr<IModelConfig> config);

    /**
     * @brief 获取当前模型配置。
     * @return 模型配置常量引用。
     */
    const IModelConfig &modelConfig() const noexcept;

    /**
     * @brief 获取当前 engine 中指定 I/O 类型的张量名称列表。
     * @param mode TensorRT 张量 I/O 类型。
     * @return 符合指定 I/O 类型的张量名称列表。
     */
    std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const;

    /**
     * @brief 获取指定张量的运行时形状。
     * @param tensor_name 张量名称。
     * @return 张量维度。
     */
    nvinfer1::Dims tensorShape(const std::string &tensor_name) const;

    /**
     * @brief 获取指定张量的数据类型。
     * @param tensor_name 张量名称。
     * @return TensorRT 数据类型。
     */
    nvinfer1::DataType tensorDataType(const std::string &tensor_name) const;

    /**
     * @brief 设置输入张量的运行时形状。
     * @param tensor_name 张量名称。
     * @param dims 运行时维度。
     */
    void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims);

    /**
     * @brief 设置日志级别。
     * @param severity TensorRT 日志严重性级别。
     */
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
     * @brief 按配置向网络中添加输入张量。
     * @param network TensorRT 网络定义。
     * @param data_type 输入张量数据类型。
     * @param input_index 输入张量索引，用于同时选择输入名称和输入尺寸。
     * @return 新增的输入张量。
     *
     * 输入名称来自 inputTensorNames()[input_index]，输入尺寸来自 inputShapes()[input_index]。
     */
    nvinfer1::ITensor *addInputTensor(nvinfer1::INetworkDefinition *network,
                                      nvinfer1::DataType            data_type   = nvinfer1::DataType::kFLOAT,
                                      size_t                        input_index = 0) const;

    /**
     * @brief 为输出张量命名并标记为网络输出。
     * @param network TensorRT 网络定义。
     * @param outputs 输出张量列表。
     */
    void markOutputTensors(nvinfer1::INetworkDefinition           *network,
                           const std::vector<nvinfer1::ITensor *> &outputs) const;

    /**
     * @brief 按配置选中的中间特征输出命名并标记为网络输出。
     * @param network TensorRT 网络定义。
     * @param named_tensors 可用于特征提取的命名张量表。
     */
    void markFeatureOutputTensors(nvinfer1::INetworkDefinition *network, const NamedTensorMap &named_tensors) const;

    /**
     * @brief 当所有请求的特征张量都已可用时，立即标记并返回 true。
     * @param network TensorRT 网络定义。
     * @param named_tensors 当前已可用的命名张量表。
     * @return 若已完成特征输出标记则返回 true，否则返回 false。
     */
    bool tryMarkFeatureOutputTensors(nvinfer1::INetworkDefinition *network, const NamedTensorMap &named_tensors) const;

    /**
     * @brief 将用户缓冲区地址绑定到 TensorRT 执行上下文。
     * @param buffers 输入输出缓冲区地址列表。
     */
    void bindTensorAddresses(const std::vector<void *> &buffers);

    /**
     * @brief 将用户缓冲区地址绑定到特征提取执行上下文。
     * @param buffers 输入与特征输出缓冲区地址列表。
     */
    void bindFeatureTensorAddresses(const std::vector<void *> &buffers);

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

    TRTParams &featureTrtParams() noexcept
    {
        return feature_trt_params_;
    }

    const TRTParams &featureTrtParams() const noexcept
    {
        return feature_trt_params_;
    }

    bool isBuildingFeatureEngine() const noexcept
    {
        return build_variant_ == BuildVariant::Feature;
    }

    bool isFeatureOnlyConfig() const noexcept
    {
        return modelConfig().featureOnly();
    }

    TRTParams &featureExecutionParams() noexcept
    {
        return isFeatureOnlyConfig() ? trt_params_ : feature_trt_params_;
    }

    const TRTParams &featureExecutionParams() const noexcept
    {
        return isFeatureOnlyConfig() ? trt_params_ : feature_trt_params_;
    }

    /**
     * @brief 初始化日志对象。
     */
    void initLogger();

    /**
     * @brief 获取主推理输出张量名称列表。
     */
    std::vector<std::string> primaryOutputTensorNames() const;

    /**
     * @brief 获取特征提取输出张量名称列表。
     */
    std::vector<std::string> featureOutputTensorNames() const;

    /**
     * @brief 从可用命名张量表中解析用户请求的特征输出张量。
     * @param named_tensors 可用命名张量表。
     * @return 与配置顺序一致的特征张量列表。
     */
    std::vector<nvinfer1::ITensor *> resolveFeatureTensors(const NamedTensorMap &named_tensors) const;

    /**
     * @brief 创建并缓存一套 TensorRT 运行时对象。
     * @param weights_file 权重文件路径，仅用于日志。
     * @param weights_map 权重映射表。
     * @param build_fn 网络构建回调。
     * @param params 目标运行时对象集合。
     */
    void buildRuntimeFromWeights(const std::string &weights_file, const WeightsMap &weights_map,
                                 const std::function<void(nvinfer1::INetworkDefinition *)> &build_fn,
                                 TRTParams &params);

    /**
     * @brief 从序列化 engine 文件加载一套运行时对象。
     * @param engine_file engine 文件路径。
     * @param params 目标运行时对象集合。
     */
    void loadRuntimeFromFile(const std::string &engine_file, TRTParams &params);

    /**
     * @brief 将当前 engine 序列化保存到文件。
     * @param engine_file 目标文件路径。
     * @param params 目标运行时对象集合。
     */
    void saveRuntimeToFile(const std::string &engine_file, const TRTParams &params);

    /**
     * @brief 创建特征提取 sidecar engine 文件路径。
     * @param base_engine_file 主 engine 文件路径。
     * @return 特征 engine 文件路径。
     */
    std::string featureEngineFileName(const std::string &base_engine_file) const;

    void ensurePrimaryInferenceReady() const;
    void ensureFeatureExtractionReady() const;

private:
    /// 模型配置对象。
    std::unique_ptr<IModelConfig> config_;
    /// TensorRT 相关运行时对象。
    TRTParams trt_params_;
    /// 特征提取相关运行时对象。
    TRTParams feature_trt_params_;
    /// 当前 buildNetwork 正在构建的 engine 类型。
    BuildVariant build_variant_{BuildVariant::Primary};
};

} // namespace irt::model::priv
