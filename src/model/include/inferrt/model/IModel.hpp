#pragma once

#include "IModelConfig.hpp"
#include <inferrt/core/ModelContract.hpp>
#include <inferrt/model/Logging.hpp>
#include <inferrt/model/ModelRuntime.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace irt::model::priv {
class IModelImpl;
}

namespace irt::model {

/**
 * @brief 面向外部的统一模型包装类。
 *
 * 该类将不同模型实现封装为一致的构建、加载、配置、运行时查询与推理接口，
 * 调用方无需直接接触具体的内部实现类型。
 */
class INFERRT_MODEL_API IModel : public irt::IExecutableModel
{
public:
    /**
     * @brief 构造一个空模型包装对象（处于 invalid 状态）。
     */
    IModel();

    /**
     * @brief 将库内具体模型实现包装为公共模型对象。
     * @tparam Impl 继承自模型内部实现基类的具体类型。
     * @param impl 已创建的具体模型实现。
     * @return 持有该实现的公共模型对象。
     */
    template<typename Impl>
    static std::unique_ptr<IModel> fromImplementation(std::unique_ptr<Impl> impl)
    {
        return std::unique_ptr<IModel>(new IModel(std::move(impl)));
    }

    /**
     * @brief 析构模型包装对象。
     */
    ~IModel();

    /** @brief 禁止拷贝构造。 */
    IModel(const IModel &) = delete;
    /** @brief 禁止拷贝赋值。 */
    IModel &operator=(const IModel &) = delete;
    /**
     * @brief 移动构造模型包装对象。
     * @param other 被移动的模型包装对象。
     */
    IModel(IModel &&) noexcept;
    /**
     * @brief 移动赋值模型包装对象。
     * @param other 被移动的模型包装对象。
     * @return 当前对象引用。
     */
    IModel &operator=(IModel &&) noexcept;

    /**
     * @brief 判断当前模型包装对象是否包含有效实现。
     */
    [[nodiscard]] bool isValid() const noexcept;

    /**
     * @brief 显式布尔转换，用于判断当前模型是否有效。
     */
    explicit operator bool() const noexcept
    {
        return isValid();
    }

    /**
     * @brief 获取模型显示名称。
     * @return 模型名称。
     */
    virtual std::string name() const;

    /**
     * @brief 获取模型权重文件扩展名。
     * @return 权重文件扩展名。
     */
    virtual std::string wtsExtension() const;

    /**
     * @brief 获取 TensorRT engine 文件扩展名。
     * @return engine 文件扩展名。
     */
    virtual std::string engineExtension() const;

    /** @brief 获取当前日志级别。 */
    virtual LogLevel logLevel() const;

    /**
     * @brief 从权重文件构建 TensorRT engine。
     * @param weights_file 权重文件路径。
     */
    virtual void build(const std::string &weights_file);

    /**
     * @brief 保存当前 TensorRT engine。
     * @param weights_file 目标文件路径。
     */
    virtual void save(const std::string &weights_file);

    /**
     * @brief 加载已有 TensorRT engine。
     * @param weights_file engine 文件路径。
     */
    virtual void load(const std::string &weights_file);

    /**
     * @brief 优先加载已有 engine，不存在时再从权重文件构建。
     * @param weights_file 权重文件路径。
     */
    virtual void buildOrLoad(const std::string &weights_file);

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
     * @brief 设置模型配置。
     * @param config 模型配置对象；为空时使用默认配置。
     */
    virtual void setModelConfig(std::unique_ptr<IModelConfig> config);

    /**
     * @brief 获取当前模型配置。
     * @return 模型配置常量引用。
     */
    virtual const IModelConfig &modelConfig() const;

    /**
     * @brief 获取当前模型运行目标。
     * @return 同时包含后端和设备信息的运行目标。
     */
    virtual const ModelRuntime &runtime() const;

    /** @brief 获取当前运行时中指定类型的 I/O 张量名称。 */
    virtual std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const;

    /**
     * @brief 获取指定张量的运行时形状。
     * @param tensor_name 张量名称。
     * @return 张量维度。
     */
    virtual irt::Shape tensorShape(const std::string &tensor_name) const;

    /**
     * @brief 获取指定张量的数据类型。
     * @param tensor_name 张量名称。
     * @return TensorRT 数据类型。
     */
    virtual irt::TensorDataType tensorDataType(const std::string &tensor_name) const;

    /**
     * @brief 设置输入张量的运行时形状。
     * @param tensor_name 张量名称。
     * @param dims 运行时维度。
     */
    virtual void setTensorShape(const std::string &tensor_name, irt::Shape shape);

    /**
     * @brief 设置模型默认使用的外部 CUDA stream。
     * @param stream 调用方提供的 CUDA stream；为空时行为未定义，请改用 clearStream。
     *
     * 该设置会同时作用于主推理与特征提取执行路径。
     */
    virtual void setStream(std::uintptr_t stream);

    /**
     * @brief 清除模型默认外部 CUDA stream，恢复为内部自建 stream。
     *
     * 该设置会同时作用于主推理与特征提取执行路径。
     */
    virtual void clearStream();

    /**
     * @brief 解析本次执行应使用的 CUDA stream。
     * @param stream_override 单次调用覆盖；非空时优先级最高。
     * @return 生效的 stream；runtime 未就绪时返回 nullptr。
     */
    std::uintptr_t resolveExecutionStream(std::uintptr_t stream_override = 0) const;

    /** @brief 设置日志级别。 */
    virtual void setLogLevel(LogLevel level);

    /** Backend-neutral I/O descriptors for new consumers. */
    std::vector<irt::TensorInfo> inputs() const override;
    std::vector<irt::TensorInfo> outputs() const override;
    void setInputShape(const std::string &name, irt::Shape shape) override;
    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options = {}) override;

private:
    template<typename Impl>
    explicit IModel(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl))
    {
    }

    [[nodiscard]] std::vector<irt::BufferView> normalizeExecutionBuffers(
        std::span<const irt::BufferView> buffers) const;

    /// 模型内部实现对象。
    std::unique_ptr<priv::IModelImpl> impl_;
};

} // namespace irt::model
