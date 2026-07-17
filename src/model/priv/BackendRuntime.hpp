#pragma once

#include <NvInfer.h>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifndef INFERRT_BUILD_ONNX
#define INFERRT_BUILD_ONNX 0
#endif

#ifndef INFERRT_BUILD_OPENVINO
#define INFERRT_BUILD_OPENVINO 0
#endif

namespace irt::model::priv {

class TensorRTBackend;

/**
 * @brief 模型推理后端运行时抽象接口。
 *
 * 统一 TensorRT、ONNX Runtime、OpenVINO 等后端的加载、元数据查询与推理入口。
 * I/O 张量元数据使用 TensorRT 的 ``Dims`` / ``DataType`` 表示，便于与 ``IModelImpl`` 共用。
 */
class IBackendRuntime
{
public:
    virtual ~IBackendRuntime() = default;

    /**
     * @brief 返回当前后端类型。
     * @return 对应的 ``ModelRuntime::Backend`` 枚举值。
     */
    virtual ModelRuntime::Backend backend() const noexcept = 0;

    /**
     * @brief 从磁盘加载已序列化的模型或图文件。
     * @param model_file 模型文件路径（TensorRT 为 engine，图后端为 ONNX / OpenVINO IR）。
     * @param config 模型配置，用于选择设备、输出张量名等。
     * @param model_name 模型显示名称，供日志或会话标识使用。
     */
    virtual void load(const std::string &model_file, const IModelConfig &config, const std::string &model_name) = 0;

    /**
     * @brief 将当前运行时状态保存到磁盘。
     * @param engine_file 输出文件路径。
     * @throws irt::Exception 非 TensorRT 后端默认抛出 ``ERROR_NOT_IMPLEMENTED``。
     */
    virtual void save(const std::string &engine_file) const;

    /**
     * @brief 枚举输入或输出张量名称。
     * @param mode ``kINPUT`` 或 ``kOUTPUT``。
     * @return 与引擎/图定义顺序一致的张量名列表。
     */
    virtual std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const = 0;

    /**
     * @brief 查询张量形状。
     * @param tensor_name 张量名称。
     * @return 当前解析后的 ``nvinfer1::Dims``。
     */
    virtual nvinfer1::Dims tensorShape(const std::string &tensor_name) const = 0;

    /**
     * @brief 查询张量数据类型。
     * @param tensor_name 张量名称。
     * @return 映射到 TensorRT 枚举的数据类型。
     */
    virtual nvinfer1::DataType tensorDataType(const std::string &tensor_name) const = 0;

    /**
     * @brief 设置输入张量形状（动态维场景）。
     * @param tensor_name 输入张量名称。
     * @param dims 新的张量维度。
     */
    virtual void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) = 0;

    /**
     * @brief 执行一次推理。
     * @param buffers 按 ``输入张量 + 输出张量`` 顺序排列的设备或主机缓冲区指针。
     *                TensorRT 使用 GPU 指针；ONNX Runtime / OpenVINO 使用主机指针。
     */
    virtual void infer(const std::vector<void *> &buffers) = 0;

    /**
     * @brief 设置外部 CUDA 执行流。
     * @param stream 非空的 CUDA stream 句柄。
     * @throws irt::Exception ``stream`` 为空时抛出。
     */
    virtual void setStream(cudaStream_t stream);

    /**
     * @brief 清除外部 CUDA 执行流，恢复默认解析逻辑。
     */
    virtual void clearStream();

    /**
     * @brief 解析本次推理应使用的 CUDA stream。
     * @param stream_override 调用方临时指定的 stream；为 nullptr 时使用已注册的外部流。
     * @return 最终选用的 stream，图后端可能返回 nullptr。
     */
    virtual cudaStream_t resolveExecutionStream(cudaStream_t stream_override = nullptr);

    /**
     * @brief 获取 TensorRT 日志级别。
     * @return 当前日志严重级别。
     */
    virtual nvinfer1::ILogger::Severity logLevel() const noexcept;

    /**
     * @brief 设置 TensorRT 日志级别。
     * @param severity 新的日志严重级别。
     */
    virtual void setLogLevel(nvinfer1::ILogger::Severity severity);

    /**
     * @brief 向下转型为 TensorRT 后端。
     * @return TensorRT 实例指针；非 TensorRT 后端返回 nullptr。
     */
    virtual TensorRTBackend *asTensorRT() noexcept;

    /**
     * @brief 向下转型为 TensorRT 后端（const 版本）。
     * @return TensorRT 实例指针；非 TensorRT 后端返回 nullptr。
     */
    virtual const TensorRTBackend *asTensorRT() const noexcept;

protected:
    /// TensorRT 构建与运行时的日志级别。
    nvinfer1::ILogger::Severity log_level_{nvinfer1::ILogger::Severity::kWARNING};
    /// 由 ``setStream`` 注册的外部 CUDA 执行流。
    cudaStream_t                external_stream_{nullptr};
};

/**
 * @brief TensorRT 推理后端实现。
 *
 * 负责从网络定义构建 engine、反序列化已有 engine，并在 GPU 上执行推理。
 */
class TensorRTBackend final : public IBackendRuntime
{
public:
    /// 网络构建回调：向 ``INetworkDefinition`` 填入层与张量。
    using NetworkBuildFn = std::function<void(nvinfer1::INetworkDefinition *)>;

    ModelRuntime::Backend backend() const noexcept override;

    void load(const std::string &engine_file, const IModelConfig &config, const std::string &model_name) override;

    void save(const std::string &engine_file) const override;

    std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const override;

    nvinfer1::Dims tensorShape(const std::string &tensor_name) const override;

    nvinfer1::DataType tensorDataType(const std::string &tensor_name) const override;

    void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) override;

    void infer(const std::vector<void *> &buffers) override;

    void setStream(cudaStream_t stream) override;

    void clearStream() override;

    cudaStream_t resolveExecutionStream(cudaStream_t stream_override = nullptr) override;

    nvinfer1::ILogger::Severity logLevel() const noexcept override;

    void setLogLevel(nvinfer1::ILogger::Severity severity) override;

    TensorRTBackend *asTensorRT() noexcept override;

    const TensorRTBackend *asTensorRT() const noexcept override;

    /**
     * @brief 获取可变的 TensorRT 运行时参数集合。
     * @return 持有 engine、context、logger 等的 ``TRTParams`` 引用。
     */
    TRTParams &params() noexcept;

    /**
     * @brief 获取只读的 TensorRT 运行时参数集合。
     * @return ``TRTParams`` 常量引用。
     */
    const TRTParams &params() const noexcept;

    /**
     * @brief 按模型名初始化 TensorRT 日志器。
     * @param model_name 写入日志前缀的模型名称。
     */
    void initLogger(const std::string &model_name);

    /**
     * @brief 通过回调构建网络并生成 engine。
     * @param source_file 权重或源文件路径，仅用于日志。
     * @param model_name 模型显示名称。
     * @param config 模型配置，用于生成动态 batch profile 和默认运行时形状。
     * @param build_fn 向 ``INetworkDefinition`` 填充层的构建函数。
     */
    void buildFromNetwork(const std::string &source_file, const std::string &model_name, const IModelConfig &config,
                          NetworkBuildFn build_fn);

    /**
     * @brief 在指定 CUDA stream 上执行推理。
     * @param buffers 按 I/O 顺序排列的 GPU 缓冲区指针。
     * @param stream_override 本次调用临时使用的 stream；为 nullptr 时走默认解析。
     * @param non_blocking 为 true 时使用异步 H2D/D2H 拷贝。
     */
    void execute(const std::vector<void *> &buffers, cudaStream_t stream_override, bool non_blocking);

    /**
     * @brief 标记当前 engine 是否仅用于特征提取网络。
     * @param feature_only 为 true 时表示输出为中间特征而非分类 logits。
     */
    void setFeatureOnly(bool feature_only) noexcept;

private:
    /// 将 ``buffers`` 中的地址绑定到 execution context 的 I/O 张量。
    void bindTensorAddresses(const std::vector<void *> &buffers);

    TRTParams params_;
    int       device_id_{0}; ///< 当前 TensorRT engine 使用的 CUDA 设备编号。
};

/**
 * @brief 按后端类型创建对应的运行时实例。
 * @param backend 目标 ``ModelRuntime::Backend``。
 * @return 新创建的后端运行时；不支持的后端抛出异常。
 */
std::unique_ptr<IBackendRuntime> CreateBackendRuntime(ModelRuntime::Backend backend);

/**
 * @brief 创建 ONNX Runtime 后端实例。
 * @return ONNX Runtime 运行时；未启用编译选项时抛出异常。
 */
#if INFERRT_BUILD_ONNX
std::unique_ptr<IBackendRuntime> CreateONNXRuntimeBackend();
#endif

/**
 * @brief 创建 OpenVINO 后端实例。
 * @return OpenVINO 运行时；未启用编译选项时抛出异常。
 */
#if INFERRT_BUILD_OPENVINO
std::unique_ptr<IBackendRuntime> CreateOpenVINOBackend();
#endif

} // namespace irt::model::priv
