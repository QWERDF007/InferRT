/**
 * @file Pipeline.hpp
 * @brief 代码定义的推理数据流水线。
 *
 * Pipeline 不从 YAML 或 JSON 解析。调用方通过 PipelineBuilder 定义有类型 DAG，
 * 在 Engine 启动前编译为不可变 PipelinePlan。需要改变算子链时，应构建新的
 * PipelinePlan 并用它创建新的 Engine；运行中的 Plan 不支持增删节点。
 */

#pragma once

#include <cuda_runtime_api.h>
#include <inferrt/engine/Export.h>
#include <opencv2/core.hpp>

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace irt::engine {

enum class ExecutionKind
{
    CPU,
    CUDA,
};

/** @brief 热路径中节点的固定执行阶段。 */
enum class PipelineStage
{
    CPU_PREPROCESS,
    H2D,
    CUDA_PREPROCESS,
    CUDA_POSTPROCESS,
    CPU_POSTPROCESS,
};

enum class TensorDataType
{
    U8,
    F32,
    I32,
    I64,
};

enum class TensorLayout
{
    HWC,
    NCHW,
};

enum class MemoryKind
{
    HOST,
    DEVICE,
};

/**
 * @brief 单张图像或单个请求对应的固定张量描述。
 *
 * batch 维由 ExecutionSlot 的 max batch 统一管理，故不在本描述中重复存储。
 */
struct INFERRT_ENGINE_API TensorDesc
{
    TensorDataType data_type{TensorDataType::U8};
    TensorLayout   layout{TensorLayout::HWC};
    MemoryKind     memory_kind{MemoryKind::HOST};
    int            width{0};
    int            height{0};
    int            channels{0};

    [[nodiscard]] size_t bytesPerRequest() const;
    void                 validate(const std::string &name) const;
};

/** @brief 一个逻辑张量在某个 ExecutionSlot 中的实际视图。 */
struct INFERRT_ENGINE_API TensorView
{
    void      *data{nullptr};
    TensorDesc desc;
    size_t     bytes_per_request{0};

    [[nodiscard]] void       *dataForRequest(int request_index);
    [[nodiscard]] const void *dataForRequest(int request_index) const;
};

using TensorViewMap  = std::unordered_map<std::string, TensorView>;
using ResultMap      = std::unordered_map<std::string, std::vector<float>>;
using TensorInputMap = std::unordered_map<std::string, std::vector<float>>;

/** @brief 节点声明的执行位置和输入/输出数目约束。 */
struct INFERRT_ENGINE_API OperatorContract
{
    ExecutionKind kind{ExecutionKind::CPU};
    PipelineStage stage{PipelineStage::CPU_PREPROCESS};
    size_t        input_count{0};
    size_t        output_count{0};
    bool          in_place{false};
    bool          thread_safe{false};
    size_t        scratch_bytes{0};
};

/**
 * @brief 由 Builder 传给算子创建器的代码配置。
 *
 * `parameters` 承载调用方传入的强类型 C++ 参数；Registry 和 Pipeline 从不
 * 解析序列化文本。内置算子的参数类型定义在 BuiltinOperators.hpp。
 */
struct INFERRT_ENGINE_API NodeConfig
{
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::any                 parameters;
};

/** @brief 单次节点执行时提供的资源和数据。 */
struct INFERRT_ENGINE_API OperatorContext
{
    int                   device_id{0};
    int                   actual_batch{0};
    int                   request_index{0};
    cudaStream_t          stream{nullptr};
    const cv::Mat        *image{nullptr};
    TensorViewMap        &tensors;
    ResultMap            &results;
    const TensorInputMap *inputs{nullptr};
};

class INFERRT_ENGINE_API IOperator
{
public:
    virtual ~IOperator() = default;

    [[nodiscard]] virtual OperatorContract contract() const = 0;

    virtual void prepare() {}

    virtual void execute(OperatorContext &context) = 0;
};

using OperatorCreator = std::function<std::unique_ptr<IOperator>(const NodeConfig &)>;

/**
 * @brief 显式注册的算子工厂集合。
 *
 * Engine 启动前可注册自定义算子；PipelineBuilder 编译 Plan 时冻结 Registry。
 * 冻结后仅允许 create，不允许覆写或追加类型。
 */
class INFERRT_ENGINE_API OperatorRegistry final
{
public:
    bool registerCreator(std::string type, OperatorCreator creator);
    void freeze();

    [[nodiscard]] std::unique_ptr<IOperator> create(std::string_view type, const NodeConfig &config) const;
    [[nodiscard]] std::vector<std::string>   types() const;
    [[nodiscard]] bool                       frozen() const noexcept;

private:
    std::unordered_map<std::string, OperatorCreator> creators_;
    bool                                             frozen_{false};
};

class PipelinePlan;

/** @brief 在 C++ 代码中定义有向无环流水线。 */
class INFERRT_ENGINE_API PipelineBuilder final
{
public:
    PipelineBuilder &addTensor(std::string name, TensorDesc desc);
    PipelineBuilder &addNode(std::string type, NodeConfig config);
    /**
     * @brief 将 CUDA 后处理产生的 device tensor 绑定为 D2H 结果。
     *
     * float32 tensor 会自动写入 InferenceResult::outputs；所有类型同时以
     * `result.<result_name>` 的 host TensorView 暴露给 CPU_POSTPROCESS 算子。
     */
    PipelineBuilder &addResult(std::string tensor_name, std::string result_name = {});
    PipelineBuilder &setModelInput(std::string tensor_name);
    PipelineBuilder &bindModelInput(std::string tensor_name, std::string engine_tensor_name);

    [[nodiscard]] std::shared_ptr<const PipelinePlan> build(std::shared_ptr<OperatorRegistry> registry) const;

private:
    struct Node
    {
        std::string type;
        NodeConfig  config;
    };

    std::unordered_map<std::string, TensorDesc>      tensors_;
    std::vector<Node>                                nodes_;
    std::vector<std::pair<std::string, std::string>> results_;
    std::string                                      model_input_;
    std::vector<std::pair<std::string, std::string>> model_inputs_;
};

/**
 * @brief 编译后的不可变 DAG。
 *
 * Plan 只保存拓扑排序后的节点定义；每个 ExecutionSlot 通过 createOperators()
 * 创建自己的算子实例，避免共享含 CUDA 状态的对象。
 */
class INFERRT_ENGINE_API PipelinePlan final
{
public:
    struct ResultBinding
    {
        std::string tensor_name;
        std::string result_name;
    };

    struct ModelInputBinding
    {
        std::string tensor_name;
        std::string engine_tensor_name;
    };

    struct Node
    {
        std::string      type;
        NodeConfig       config;
        OperatorContract contract;
    };

    [[nodiscard]] const std::unordered_map<std::string, TensorDesc> &tensors() const noexcept;
    [[nodiscard]] const std::vector<Node>                           &nodes() const noexcept;
    [[nodiscard]] const std::vector<ResultBinding>                  &results() const noexcept;
    [[nodiscard]] const std::string                                 &modelInput() const noexcept;
    [[nodiscard]] const std::vector<ModelInputBinding>              &modelInputs() const noexcept;
    [[nodiscard]] std::vector<std::unique_ptr<IOperator>>            createOperators() const;

private:
    friend class PipelineBuilder;

    PipelinePlan(std::shared_ptr<const OperatorRegistry> registry, std::unordered_map<std::string, TensorDesc> tensors,
                 std::vector<Node> nodes, std::vector<ResultBinding> results, std::string model_input,
                 std::vector<ModelInputBinding> model_inputs);

    std::shared_ptr<const OperatorRegistry>     registry_;
    std::unordered_map<std::string, TensorDesc> tensors_;
    std::vector<Node>                           nodes_;
    std::vector<ResultBinding>                  results_;
    std::string                                 model_input_;
    std::vector<ModelInputBinding>              model_inputs_;
};

} // namespace irt::engine
