# WORKLOG

> 本文件是项目唯一的任务账本。真实日志按最新在前追加在固定示例条目之后，并固定位于其他真实日志之上；`⏳ 待你裁决` 始终固定在顶部。

## ⏳ 待你裁决

<!-- 没有待裁决事项时保持本节为空。 -->

---

## 日志

<!--
建议格式：

### YYYY-MM-DD — 简短任务名

**目标**
- ...

**当前状态**
- 已完成：...
- 未完成：...

**验证证据**
- `command ...` → 关键结果
- 未验证项请明确写“未验证”

**下一步**
- ...
-->




## [示例] 修复订单导出超时

**总目标**：后台订单导出在 1 万行数据量下 30 秒内完成，不再 504。

**状态**：✅ 完成

**干到哪了**：
- [x] 定位根因：导出走了逐行 N+1 查询 —— 证据：慢日志中同款 SELECT 出现 10,412 次
- [x] 改为批量查询 + 流式写出 —— 证据：`export_test.go` 新增用例通过；本地 1 万行实测 4.2s
- [x] 隔离实例真实触发目标路径 —— 证据：staging 实测导出 12,000 行 5.1s，HTTP 200
- [x] 开关两态验证：`export_v2=off` 时回退旧路径正常

**边界**：不动导出的字段结构；不顺手重构 handler。

### 2026-09-04 — 重建 InferRT 0.0.3 并运行 full 测试

**目标**
- 清理旧生成物，将项目版本更新为 `0.0.3`，重新配置、构建、软链接依赖并安装，完成 Windows full 测试。

**当前状态**
- 已完成：删除 `artifacts/`、`build/` 和 `InferRT-0.0.2/`，将根 `CMakeLists.txt` 版本改为 `0.0.3`。
- 已完成：重新配置并构建 `full-release`；构建缓存安装前缀为 `F:/Projects/InferRT/InferRT-0.0.3`，ONNX/OpenVINO 均为关闭状态。
- 已完成：以真实 UAC 管理员权限运行 `link_dependencies.py --mode symlink`，`build/bin` 中 35 个第三方运行库均为符号链接，未保留大体积复制文件。
- 已完成：安装树 `InferRT-0.0.3/` 已生成，CMake package 版本为 `0.0.3`。
- 已完成：full CTest 在显式加入 `build/bin` 运行库路径后全部通过。
- 已完成：同步清理 docs 中已删除临时报告目录的失效索引，当前本地验证统一指向本账本最新条目。

**验证证据**
- `cmake --preset full ...`（提权）→ 配置成功，版本 `0.0.3`，安装前缀 `F:/Projects/InferRT/InferRT-0.0.3`。
- `cmake --build --preset full-release --parallel` → 退出码 0。
- `Start-Process -Verb RunAs ... link_dependencies.py --build-dir build/presets/full --config Release --mode symlink` → 退出码 0；35 个目标均为 `SymbolicLink`。
- `cmake --install build/presets/full --config Release` → 安装成功。
- `ctest --test-dir build/presets/full -C Release --output-on-failure --no-tests=error`（提权，PATH 加入 `build/bin`）→ `9/9 passed`，包含 C++、Python API 与 relocation。
- `artifacts/`、`InferRT-0.0.2/` → 已确认不存在。

**下一步**
- 当前请求已完成，无需继续生成测试报告或其他临时产物。

---

### 2026-09-04 — 收口当前 Release 验收证据与生成物

**目标**
- 按 `sol_plan.md` 收口当前 Windows Release 验收证据，更新唯一验收报告，并清理不再使用的临时生成物。

**当前状态**
- 已完成：提权 py312 严格部署当前隔离安装包，处理 44 个运行库文件；当前质量门禁 `finding_count=0`。
- 已完成：当前 CUDA NMS benchmark 三种规模均为 `steady_allocations=0`，并确认 CUDA 内存检查为 `ERROR SUMMARY: 0 errors`。
- 已完成：最新完整 CTest 为 `9/9 passed`；最新 `inferrt_test_engine` 为 `106` 项，其中 `105 passed、1 skipped、0 failed`。
- 已完成：更新 `sol_review_18.md` 与架构索引中的当前证据引用；保留 ONNX/OpenVINO 为可选能力。
- 已完成：删除旧 relocation 安装树、旧 consumer/benchmark/sanitizer/session 目录及 pytest 缓存；保留当前、基线和可选后端证据。
- 尚未完成：Linux core-only、Linux CUDA、Linux sanitizer 的真实 runner 证据仍需 CI 产生，当前 Windows 工作区无法替代；本机 WSL 无可用发行版，Docker/Podman/`gh` 不可用，远端分支不包含当前未提交改动。

**验证证据**
- `ctest --test-dir build\\presets\\full -C Release --output-on-failure --no-tests=error --output-junit ...\\ctest-current.xml`（提权）→ `9/9 passed`。
- `inferrt_test_engine --gtest_output=xml:...\\engine-gtest-final-current.xml` → `105 passed、1 skipped、0 failed`。
- `D:\\Software\\anaconda3\\envs\\py312\\python.exe tools\\package_runtime_dlls.py --strict`（提权）→ `Files processed: 44`。
- `D:\\Software\\anaconda3\\envs\\py312\\python.exe tools\\quality_gates.py ...`（提权）→ `finding_count=0`。
- `inferrt_benchmark_cuda_ops.exe --benchmark_filter=^BM_NMS` → 64/160/512 boxes 均 `steady_allocations=0`。

**下一步**
- 等待或触发 CI Linux runner，归档 core-only、CUDA 和 sanitizer 报告后，再闭合跨平台验收结论。

---

### 2026-09-04 — 收口 Engine 阶段职责与 features 安装依赖

**目标**

- 按 `sol_plan.md` 的结构收口 Engine CPU 阶段职责，并修复安装包导出 `features` 时 Faiss 目标无法自动闭合的问题。

**当前状态**

- 已完成：新增 `EngineStageWorkers`，集中拥有 prepare/postprocess worker、阶段执行、批次失败/完成和阶段指标逻辑；`InferenceEngine` 保留生命周期与故障编排。
- 已完成：保留 scheduler → prepare → GPU executor → postprocess 的关闭顺序，并处理 ticket pool 阻塞时的失败回滚。
- 已完成：安装包按组件导出 targets；可导出 `features` 时始终安装 `ConfigFaiss.cmake`，消费者通过 `CMAKE_PREFIX_PATH` 自动解析 Faiss 原生 target 或根目录布局。
- 已完成：同步更新 Engine 设计及项目文档索引。
- 尚未完成：本轮未重新执行构建、测试、安装或其他验证；ONNX/OpenVINO 仍为可选项，不作为当前交付缺口。

**验证证据**

- 未验证：本轮按要求不运行验证命令，仅完成源码和文档静态收口。

**下一步**

- 若恢复验收，使用当前源码重新执行受影响的 Engine focused 与安装迁移测试；在此之前不将本轮修改标记为已验证。

---

### 2026-09-04 — 收口统一测试报告门禁

**目标**

- 将 CTest、Python 和工具测试的验收统计统一为机器可读报告门禁，并同步当前 Windows Release 验收记录。

**当前状态**

- 已完成：新增 [`tools/validate_test_report.py`](tools/validate_test_report.py) 及其测试，统一校验执行数量、失败数量、必需用例和允许的可选 skip。
- 已完成：CI 接入核心、CUDA、安装迁移和 Python 报告的 fail-fast 检查；可选后端、可选权重及 Windows 符号链接权限不足按显式规则放行。
- 已完成：更新项目文档索引和 [`sol_review_18.md`](sol_review_18.md)，当前工具测试统计为 56 total、55 passed、1 skipped。
- 尚未完成：Linux core-only、Linux CUDA、sanitizer 和跨平台 CI 的真实 runner 证据仍待补齐。

**验证证据**

- `tests/tools`：55 passed、1 skipped，退出码 0；唯一 skip 为 Windows 符号链接权限不足。
- full CTest：9/9 通过；core-only CTest：4/4 通过；CUDA CTest：5/5 通过。
- Python：169 total、151 passed、18 optional skipped、required_skipped=0。
- 本轮未重新执行验证命令；以上为已有机器可读报告和前序验证结果。

**下一步**

- 在具备对应 runner 后补齐 Linux、CUDA、sanitizer 和跨平台证据，再按 [`sol_plan.md`](sol_plan.md) 更新最终验收结论。

---


### 2026-09-04 — 修正隔离 session 的执行描述来源

**目标**

- 让并发或动态 batch session 的执行缓冲校验使用 session 当前 shape/dtype，避免复用共享 backend 的 mutable I/O 状态。

**当前状态**

- 已完成：`IBackendRuntime::executeSession()` 保留 backend 的 I/O 名称和内存类型来源，改为从具体 `ITensorRuntimeSession` 读取 tensor shape 与 data type 后再调用统一规范化校验。
- 已完成：新增 model focused 回归用例，覆盖共享 backend 固定 batch 与独立 session batch 不同、乱序 buffer 绑定的执行路径。
- 尚未完成：本轮按要求未执行构建、测试、安装或 Python 验证。

**验证证据**

- 未验证：本轮明确不运行验证命令。

**下一步**

- 后续若恢复验收，先构建并运行受影响的 model focused 测试，再据结果更新正式验收记录。

---


### 2026-09-04 — 收敛 TensorRT runtime 工具边界

**目标**

- 消除 TensorRT engine 读取、反序列化和 execution context 创建在不同模型路径中的重复实现，完成 Engine runtime 适配边界的静态收口。

**当前状态**

- 已完成：在 [`src/model/priv/TRTUtils.hpp`](src/model/priv/TRTUtils.hpp) / [`src/model/priv/TRTUtils.cpp`](src/model/priv/TRTUtils.cpp) 集中提供二进制文件读取、engine 反序列化和独立 context 创建。
- 已完成：`TensorRTBackend` 的 engine load/build/session 路径和 ONNX 的 TensorRT 构建路径复用统一工具；Backend 保留 binding、enqueue 和 backend-specific shape 逻辑。
- 已完成：同步更新项目总览、架构图谱和验收报告索引；阶段 5 仍保留 SAM 子模块职责审查、跨平台 runner、sanitizer 和发布 ABI 清单等未闭合项。
- 尚未完成：本轮按要求未执行构建、测试、安装或其他验证。

**验证证据**

- 未验证：本轮明确不运行验证命令，仅完成源码与文档静态修改。

**下一步**

- 后续若恢复验收，先编译受影响的 model、engine 和 ONNX 目标，再运行对应 focused 测试并据结果更新正式报告。

---


### 2026-09-04 — 统一依赖缓存来源标记

**目标**

- 修复依赖别名与缓存变量名不一致导致的 provenance 标记失配，确保已有构建树中新的 `-D` 依赖路径仍优先于项目默认值。

**当前状态**

- 已完成：`inferrt_dependency_resolve_path()` 按实际缓存变量名生成统一来源标记，覆盖 TensorRT、Faiss、OpenCV 等依赖路径。
- 已完成：新增连续两次 `-D` 修改依赖路径的 CMake 回归用例。
- 尚未完成：本轮按要求未执行构建、测试、安装或其他验证。

**验证证据**

- 未验证：本轮明确不运行验证命令。

**下一步**

- 后续验收时运行新增依赖来源回归用例，并据结果更新验收记录。

---


### 2026-09-04 — 统一后端无关执行入口

**目标**

- 将模型 backend 的 buffer 校验、tensor name 规范化和实际后端执行收敛到单一执行边界。

**当前状态**

- 已完成：`IBackendRuntime::execute()` 统一调用 `normalizeExecutionBuffers()`，TensorRT、ONNX Runtime、OpenVINO backend 通过 `executeNormalized()` 接收规范化缓冲区。
- 已完成：`IModel`/`IModelImpl` 移除重复的外层 buffer 规范化；Engine runtime plan、model 和 backend 统一使用 core `IExecutionPlan`/`IExecutableModel` 契约；补充按 tensor name 重排输入输出的回归用例。
- 已完成：静态确认 ONNX/OpenVINO 会话层和 Engine fake session 仍实现原 `execute()` 接口。
- 尚未完成：本轮按要求未执行构建、测试、安装或其他验证；`sol_plan.md` 中的跨平台 runner、sanitizer、性能基线和长期 API/ABI 收口项仍未改变。

**验证证据**

- 未验证：本轮明确不运行构建、测试、安装或其他验证命令；仅完成源码静态检查。

**下一步**

- 后续验收时先构建受影响的 model、engine 和 core 测试目标，再据结果更新正式验收记录。

---


### 2026-09-04 — 修正 CompletionOrder 注册与关闭顺序

**目标**

- 消除重复请求 ID 消耗来源序号，以及 shutdown 在同一来源下乱序交付完成结果的问题。

**当前状态**

- 已完成：`registerRequest()` 在分配来源序号前拒绝活动请求 ID 重复注册；`shutdown()` 将活动请求失败结果与 pending completion 按来源和序号整理后统一交付。
- 已完成：补充重复 ID 序号连续性、shutdown 同时清理活动与 pending completion 的 focused 测试。
- 尚未完成：本轮按要求未执行构建、测试或安装验证；`sol_plan.md` 中的跨平台 runner、sanitizer 和长期 API/ABI 收口项仍未改变。

**验证证据**

- 未验证：本轮明确不运行构建、测试、安装或其他验证命令。

**下一步**

- 后续验收时先构建并运行 `inferrt_test_engine`，确认新增 CompletionOrder 用例通过后再更新正式验收报告。

---


### 2026-09-04 — 收口 Pipeline 与异步失败完成契约

**目标**

- 补齐 Pipeline 构建边界和异步失败结果的明确错误契约，避免非法输入或空异常指针形成未定义行为。

**当前状态**

- 已完成：`PipelineBuilder::setModelInput()` 拒绝空名称；Pipeline DAG 对同一节点的自引用不再错误增加边；空 `std::exception_ptr` 按失败类型转换为明确的 `irt::Exception`；completion 回归测试显式包含所需标准头。
- 已完成：相关测试覆盖入口拒绝和空失败异常仍能完成 future；既有构建产物、安装快照和用户未跟踪文件均保留。
- 尚未完成：本轮按要求未执行构建、测试、安装或 Python 验证；跨平台 runner、sanitizer 和 `sol_plan.md` 中未闭合的长期结构项仍以现有报告为准。

**验证证据**

- 未验证：本轮明确不运行构建、测试、安装或 Python。

**下一步**

- 后续如继续验收，先使用当前源码重新执行受影响目标，再更新验收报告；在此之前不得将本轮改动标记为已验证。

---


### 2026-09-04 — 完善动态维度执行缓冲契约

**目标**

- 使执行缓冲校验正确接受模型声明中的动态维度，同时继续拒绝静态维度、类型、内存类型和容量错误。

**当前状态**

- 已完成：`normalizeExecutionBuffers()` 按 wildcard 匹配动态声明维度，并保留 concrete shape 的正数校验。
- 已完成：补充动态 batch、动态空间维度和输出缓冲的核心契约测试。
- 尚未完成：TensorRT、ONNX Runtime、OpenVINO 适配层的跨后端统一证据仍以 `sol_plan.md` 验收矩阵为准；可选后端不计入当前正式包缺口。

**验证证据**

- `cmake --build build/presets/full --config Release --target inferrt_test_core --parallel 4` -> 构建通过。
- `ModelContractTest.AcceptsConcreteBuffersForDeclaredDynamicDimensions` -> 1 test passed。

**下一步**

- 继续按 `sol_plan.md` 核对后端无关执行契约和跨平台证据，未产生真实 runner 结果的项目保持未验证。

---

### 2026-09-04 — Windows Release 验收证据收口

**目标**

- 以当前源码和 `sol_plan.md` 为准收口 Windows Release 验收证据，并区分沙箱权限问题与源码失败。

**当前状态**

- 已完成：提权 py312 下 full CTest、工具测试和质量门禁复核；`0xc0000022` 确认为 Python 沙箱权限问题。
- 已完成：保留正式安装包和验收证据，清理 Python 模型测试中间产物、relocation 临时目录及重复证据文件。
- 尚未完成：Linux core-only/CUDA/sanitizer runner 和跨平台 full CI 尚未在本机产生真实证据；阶段 2/5 的完整后端无关 API 收敛、SAM/shape matcher 结构收敛及最终 ABI 消费者清单仍按 [`sol_review_18.md`](sol_review_18.md) 标记为未闭合。

**验证证据**

- `ctest --test-dir build\presets\full -C Release --output-on-failure --no-tests=error` -> 9/9 passed，0 failed。
- `D:\Software\anaconda3\envs\py312\python.exe -m pytest tests\tools -v --tb=short` -> 44 passed，1 个 Windows 符号链接权限 skip，退出码 0。
- `D:\Software\anaconda3\envs\py312\python.exe tools\quality_gates.py --json ...` -> `finding_count=0`。
- 当前正式 Python 门禁证据 -> 169 tests，151 passed，0 failed，18 个可选后端/可选权重 skip，`required_skipped=0`。

**下一步**

- 以 [`sol_review_18.md`](sol_review_18.md) 和 [`sol_plan.md`](sol_plan.md) 的未闭合项为准，不把本机未执行的跨平台 runner 结果标记为通过。

---
