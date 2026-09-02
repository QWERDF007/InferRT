#pragma once

#include <inferrt/model/BackendRuntime.hpp>
#include "TensorRTBackend.hpp"
#include "TRTParams.hpp"
#include "TRTUtils.hpp"

#include <inferrt/model/IModelConfig.hpp>

#include <functional>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
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
    using BackendRuntimeFactory = std::unique_ptr<irt::model::IBackendRuntime> (*)(ModelRuntime::Backend);

    /// 构建期网络名称到 TensorRT 张量指针的映射表。
    using NamedTensorMap = std::unordered_map<std::string, nvinfer1::ITensor *>;

    /**
     * @brief 标识 buildNetwork 当前正在构建的网络类型。
     */
    enum class BuildVariant
    {
        Primary, ///< 主分类（或完整）推理网络。
        Feature, ///< 仅导出配置中选定中间特征的裁剪网络。
    };

    /**
     * @brief 使用默认模型配置构造内部实现对象。
     */
    explicit IModelImpl(BackendRuntimeFactory backend_factory = &irt::model::CreateBackendRuntime)
        : config_(std::make_unique<IModelConfig>())
        , backend_factory_(backend_factory != nullptr ? backend_factory : &irt::model::CreateBackendRuntime)
    {
        // Route construction through the same checked factory path used by
        // transactional reloads.  A backend factory that returns nullptr is
        // a construction error, never a partially usable model state.
        backend_runtime_ = createBackendRuntime(config_->runtime().backend());
    }

    /**
     * @brief 析构内部实现对象。
     */
    virtual ~IModelImpl();

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

    /** @brief 获取当前 backend-neutral 日志级别。 */
    LogLevel logLevel() const noexcept;

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
     * @brief 查询当前手写 TensorRT 网络是否支持动态 batch。
     * @return 支持时返回 true；默认实现保持静态 batch 约束。
     */
    virtual bool supportsDynamicBatch() const noexcept
    {
        return false;
    }

    /**
     * @brief 获取 TensorRT engine 缓存契约版本。
     * @return 用于 engine manifest 的版本字符串；派生类图结构变化时可覆盖以触发重建。
     */
    virtual std::string engineCacheVersion() const noexcept
    {
        return "1";
    }

    /**
     * @brief 在指定 CUDA stream 上执行一次推理。
     * @param buffers 输入输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     * @param non_blocking 为 true 时仅提交执行，不在函数内等待 stream 完成。
     */
    virtual void infer(std::span<const irt::BufferView> buffers, std::uintptr_t stream = 0,
                       bool non_blocking = false);

    /**
     * @brief 在指定 CUDA stream 上执行一次特征提取前向。
     * @param buffers 输入与特征输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     * @param non_blocking 为 true 时仅提交执行，不在函数内等待 stream 完成。
     */
    virtual void forwardFeatures(std::span<const irt::BufferView> buffers, std::uintptr_t stream = 0,
                                 bool non_blocking = false);

    /**
     * @brief 设置完整模型配置。
     * @param config 模型配置对象；传入空指针时恢复为默认配置。
     */
    void setModelConfig(std::unique_ptr<IModelConfig> config);

    void replaceModelConfigWithoutReset(std::unique_ptr<IModelConfig> config);

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
    std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const;

    /** @brief Return backend-owned I/O descriptors through the core contract. */
    std::vector<irt::TensorInfo> inputs() const;
    std::vector<irt::TensorInfo> outputs() const;

    /**
     * @brief 获取指定张量的运行时形状。
     * @param tensor_name 张量名称。
     * @return 张量维度。
     */
    irt::Shape tensorShape(const std::string &tensor_name) const;

    /**
     * @brief 获取指定张量的数据类型。
     * @param tensor_name 张量名称。
     * @return TensorRT 数据类型。
     */
    irt::TensorDataType tensorDataType(const std::string &tensor_name) const;

    /** @brief 获取当前后端 I/O buffer 要求的地址空间。 */
    irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const;

    /**
     * @brief 设置输入张量的运行时形状。
     * @param tensor_name 张量名称。
     * @param dims 运行时维度。
     */
    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape);

    /**
     * @brief 设置模型默认使用的外部 CUDA stream。
     * @param stream 调用方提供的 CUDA stream。
     */
    void setStream(std::uintptr_t stream);

    /**
     * @brief 清除模型默认外部 CUDA stream，恢复为内部自建 stream。
     */
    void clearStream();

    /**
     * @brief 设置日志级别。
     * @param severity TensorRT 日志严重性级别。
     */
    void setLogLevel(LogLevel level);

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
     * @brief 获取可写 TensorRT 运行时参数。
     * @return 运行时参数引用。
     */
    TRTParams &trtParams();

    /**
     * @brief 获取只读 TensorRT 运行时参数。
     * @return 运行时参数常量引用。
     */
    const TRTParams &trtParams() const;

    /**
     * @brief 判断 buildNetwork 是否正在为特征裁剪网络构建。
     * @return 为 true 时表示当前应只标记配置中的特征输出。
     */
    bool isBuildingFeatureEngine() const noexcept
    {
        return build_variant_ == BuildVariant::Feature;
    }

    /**
     * @brief 判断模型配置是否处于仅特征提取模式。
     * @return 为 true 时当前 engine 构建为 feature-only 网络。
     */
    bool isFeatureOnlyConfig() const noexcept
    {
        return modelConfig().featureOnly();
    }

    /**
     * @brief 初始化日志对象。
     */
    void initLogger();

    /**
     * @brief 从可用命名张量表中解析用户请求的特征输出张量。
     * @param named_tensors 可用命名张量表。
     * @return 与配置顺序一致的特征张量列表。
     */
    std::vector<nvinfer1::ITensor *> resolveFeatureTensors(const NamedTensorMap &named_tensors) const;

    /**
     * @brief 创建并缓存当前 TensorRT 运行时对象。
     * @param weights_file 权重文件路径，仅用于日志。
     * @param build_fn 网络构建回调。
     */
    void buildRuntimeFromWeights(const std::string                                         &weights_file,
                                 const std::function<void(nvinfer1::INetworkDefinition *)> &build_fn);

    /**
     * @brief 从序列化 engine 文件加载当前运行时对象。
     * @param engine_file engine 文件路径。
     */
    void loadRuntimeFromFile(const std::string &engine_file);

    /**
     * @brief 将当前 engine 序列化保存到文件。
     * @param engine_file 目标文件路径。
     */
    void saveRuntimeToFile(const std::string &engine_file) const;

    bool usesTensorRTBackend() const noexcept;

    /**
     * @brief 解析本次 enqueue 应使用的 CUDA stream。
     * @param stream_override 单次调用覆盖；非空时优先级最高。
     * @return 生效的 stream；runtime 未就绪且尚无内部 stream 时返回 nullptr。
     *
     * 优先级：stream_override > external_stream > 惰性创建的内部 stream。
     */
    std::uintptr_t resolveExecutionStream(std::uintptr_t stream_override = 0);

protected:
    /**
     * @brief 允许派生模型规整外部传入的配置。
     * @param config 已经设置到模型上的配置对象。
     *
     * 该 hook 用于补齐模型族的专属默认值，例如不同输入分辨率的 ViT 变体。
     * 默认实现不修改配置，普通模型无需关心。
     */
    virtual void normalizeModelConfig(IModelConfig &config) const
    {
        (void)config;
    }

private:
    void execute(std::span<const irt::BufferView> buffers, std::uintptr_t stream, bool non_blocking);

    [[nodiscard]] std::unique_ptr<irt::model::IBackendRuntime> createBackendRuntime(ModelRuntime::Backend backend) const;

    void buildBackendRuntimeFromFile(const std::string &model_file);

    void syncModelConfigFromBackendRuntime(IModelConfig &config, const irt::model::IBackendRuntime &backend);

    TensorRTBackend &tensorRTBackend();

    const TensorRTBackend &tensorRTBackend() const;

    /// 模型配置对象。
    std::unique_ptr<IModelConfig> config_;

    /// 后端创建策略；生产模型使用 CreateBackendRuntime，测试可注入最小后端实现。
    BackendRuntimeFactory backend_factory_{&irt::model::CreateBackendRuntime};

    /// 当前 buildNetwork 正在构建的 engine 类型。
    BuildVariant build_variant_{BuildVariant::Primary};

    /// 当前后端运行时；由 modelConfig().runtime().backend() 决定具体派生实现。
    std::unique_ptr<irt::model::IBackendRuntime> backend_runtime_;
};

} // namespace irt::model::priv
