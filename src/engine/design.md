# InferRT Engine 实现索引

`engine` 是 InferRT 的部署编排模块，负责请求接入、流水线调度、批处理、执行槽资源和结果交付。公共接口、实施目标与验收条件分别以源码和 [`sol_plan.md`](../../sol_plan.md) 为准；本文只维护入口索引，不复制实现细节。

## 公共接口

- 引擎配置：[`include/inferrt/engine/EngineConfig.hpp`](include/inferrt/engine/EngineConfig.hpp)
- 推理引擎：[`include/inferrt/engine/InferenceEngine.hpp`](include/inferrt/engine/InferenceEngine.hpp)
- Pipeline 构建与节点接口：[`include/inferrt/engine/Pipeline.hpp`](include/inferrt/engine/Pipeline.hpp)
- 内置算子声明：[`include/inferrt/engine/BuiltinOperators.hpp`](include/inferrt/engine/BuiltinOperators.hpp)

## 实现入口

- 生命周期编排：[`InferenceEngine.cpp`](InferenceEngine.cpp)
- CPU 阶段 worker 与批次完成：[`priv/EngineStageWorkers.hpp`](priv/EngineStageWorkers.hpp)、[`priv/EngineStageWorkers.cpp`](priv/EngineStageWorkers.cpp)
- GPU 请求执行与阶段衔接：[`priv/EngineExecution.cpp`](priv/EngineExecution.cpp)
- 兼容 key、优先级和批调度：[`priv/EngineScheduler.hpp`](priv/EngineScheduler.hpp)、[`priv/EngineScheduler.cpp`](priv/EngineScheduler.cpp)
- 有界阶段队列：[`priv/EngineStageQueue.hpp`](priv/EngineStageQueue.hpp)、[`priv/EngineStageQueue.cpp`](priv/EngineStageQueue.cpp)
- 请求 ticket：[`priv/EngineTicketPool.hpp`](priv/EngineTicketPool.hpp)、[`priv/EngineTicketPool.cpp`](priv/EngineTicketPool.cpp)
- 聚合执行槽：[`priv/EngineSlot.hpp`](priv/EngineSlot.hpp)、[`priv/EngineSlot.cpp`](priv/EngineSlot.cpp)
- GPU 执行：[`priv/EngineSlotExecutor.hpp`](priv/EngineSlotExecutor.hpp)、[`priv/EngineSlotExecutor.cpp`](priv/EngineSlotExecutor.cpp)
- 完成顺序与结果交付：[`priv/EngineCompletionOrder.hpp`](priv/EngineCompletionOrder.hpp)、[`priv/EngineCompletionOrder.cpp`](priv/EngineCompletionOrder.cpp)
- 指标和故障记录：[`priv/EngineMetricsFault.hpp`](priv/EngineMetricsFault.hpp)、[`priv/EngineMetricsFault.cpp`](priv/EngineMetricsFault.cpp)
- TensorRT runtime plan：[`priv/EngineRuntimePlan.hpp`](priv/EngineRuntimePlan.hpp)、[`priv/TensorRTRuntimePlan.hpp`](priv/TensorRTRuntimePlan.hpp)、[`priv/TensorRTRuntimePlan.cpp`](priv/TensorRTRuntimePlan.cpp)
- 测试用默认流水线构造：[`priv/EngineLegacyPipeline.cpp`](priv/EngineLegacyPipeline.cpp)、[`priv/EngineTestHooks.cpp`](priv/EngineTestHooks.cpp)

## 依赖关系

`engine` 使用 `core` 的张量和执行契约，使用 `model` 的后端 runtime，使用 `cvcuda` 和 `ops` 的前后处理实现。模型后端接口见 [`src/model/include/inferrt/model/BackendRuntime.hpp`](../model/include/inferrt/model/BackendRuntime.hpp)，核心契约见 [`src/core/include/inferrt/core/ModelContract.hpp`](../core/include/inferrt/core/ModelContract.hpp) 与 [`src/core/include/inferrt/core/Tensor.hpp`](../core/include/inferrt/core/Tensor.hpp)。

## 验证入口

- Engine 单元和集成测试：[`../../tests/engine`](../../tests/engine)
- Engine 样例：[`../../samples/engine`](../../samples/engine)
- Engine benchmark：[`../../benchmark/engine`](../../benchmark/engine)
- 当前 Windows Release 验收报告：[`../../sol_review_18.md`](../../sol_review_18.md)

修改 engine 时，先更新对应公共接口或私有实现，再在同一 seam 增加行为测试；不在本文维护第二套生命周期、线程模型或资源不变量。
