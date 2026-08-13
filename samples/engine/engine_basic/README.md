# Engine 示例

`inferrt_sample_engine` 使用同一张图和同一份 TensorRT engine 对比普通 `IModel` 与 `InferenceEngine`。
不提供或读取 YAML：模型配置和 Pipeline DAG 都在 [EngineExample.hpp](EngineExample.hpp) 中通过 C++ API 定义。

内置示例为：

| `--example` | 输入和 batch profile | CUDA 预处理 |
|---|---|---|
| `resnet18` | 224 × 224，`1/4/8` | `resize → BGR2RGB → Normalize` |
| `yolov8n` | 640 × 640，`1/4/8` | `letterbox` |
| `dinov2_vits14` | 518 × 518，固定 batch 1 | `resize → BGR2RGB → Normalize` |

```bash
./build/bin/inferrt_sample_engine \
  --example yolov8n \
  --engine /mnt/d/Models/yolov8/engine_compare/yolov8n_dynamic.engine \
  --image assets/pics/dog.jpg \
  --preprocess cuda --mode compare
```

模式：

- `model`：普通 `IModel` 同步 H2D → infer → D2H。
- `engine`：Engine 的 Pinned Memory、动态组批和异步执行槽路径。
- `compare`：执行两条路径并输出最大绝对误差。

## 代码定义 Pipeline

算子 DAG 不从 YAML/JSON 读取。调用方先显式注册算子，再在 C++ 中声明张量、节点依赖和模型输入，最后将冻结后的
`PipelinePlan` 传给 `InferenceEngine::create(config, plan)`。`EngineExample.hpp` 中的 `makeConfig()` 和
`makePipeline()` 是 sample/benchmark 共用的可运行 C++ 定义。

```cpp
auto registry = std::make_shared<irt::engine::OperatorRegistry>();
irt::engine::registerBuiltinOperators(*registry);
using namespace irt::engine;

PipelineBuilder builder;
builder
    .addTensor("host_bgr", {TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, 768, 576, 3})
    .addTensor("device_bgr", {TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE, 768, 576, 3})
    .addTensor("resized_bgr", {TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE, 518, 518, 3})
    .addTensor("rgb", {TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE, 518, 518, 3})
    .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 518, 518, 3})
    .addNode("cpu.copy_image", {{}, {"host_bgr"}, {}})
    .addNode("cuda.upload", {{"host_bgr"}, {"device_bgr"}, {}})
    .addNode("cvcuda.resize", {{"device_bgr"}, {"resized_bgr"}, ResizeOptions{}})
    .addNode("cvcuda.cvt_color", {{"resized_bgr"}, {"rgb"}, CvtColorOptions{}})
    .addNode("cvcuda.normalize", {{"rgb"}, {"model_input"}, NormalizeOptions{mean, stddev}})
    .setModelInput("model_input");

const auto plan = builder.build(registry);
auto engine = irt::engine::InferenceEngine::create(config, plan);
```

`build()` 会验证张量、输入输出数量和 DAG 环路，做拓扑排序后冻结 Registry。Plan 是不可变对象，每个 ExecutionSlot
按 Plan 创建自己的算子实例。需要增删算子时，应重新定义并编译 DAG，然后用新 Plan 创建新 Engine；不会在运行中的
Engine 内修改节点。

## 外部后处理算子

Engine 不内置任何模型相关的 Decode/NMS。应用在构建 Plan 前注册自己的 CPU/CUDA 算子即可；CUDA 后处理可读取
`model.<TensorRT 输出名>` 的 device tensor。若它产生新的 tensor，以 `addResult()` 声明后，Engine 会在 CUDA
后处理结束时自动 D2H；float32 结果写入 `InferenceResult::outputs`，其他类型通过 CPU 后处理中的
`result.<名称>` host tensor 访问。

```cpp
registry->registerCreator("detector.nms.cuda", makeDetectorNmsCreator());

builder
    .addTensor("keep", {TensorDataType::I64, TensorLayout::HWC, MemoryKind::DEVICE, max_boxes, 1, 1})
    .addTensor("keep_count", {TensorDataType::I32, TensorLayout::HWC, MemoryKind::DEVICE, 1, 1, 1})
    .addNode("detector.nms.cuda", {{"model.boxes", "model.scores"}, {"keep", "keep_count"}, options})
    .addResult("keep", "keep_indices")
    .addResult("keep_count", "keep_count");
```

`detector.nms.cuda` 是调用方实现的 `IOperator`；它可复用现有 [cvcuda::NMS](../../src/cvcuda/include/inferrt/cvcuda/OpNMS.hpp)，
并在 `CUDA_POSTPROCESS` 阶段仅向 `context.stream` enqueue。随后注册一个 `CPU_POSTPROCESS` 算子读取
`result.keep_indices` 和 `result.keep_count`，转换成项目自己的检测结果格式。

## 队列、取消与指标

`EngineConfig::queue_policy` 支持 `Reject`、`Block`、`DropOldest`、`DropNewest`。需要请求级 deadline、优先级、
兼容组或取消时，
使用带 `RequestOptions` 的重载：

```cpp
auto handle = engine->submit(frame, {.deadline = std::chrono::milliseconds(40),
                                     .priority = 10,
                                     .compatibility_key = "camera-a"});
if (!engine->cancel(handle.id())) {
    // 已完成，或已进入最终完成路径。
}
auto result = std::move(handle).takeFuture().get();
auto metrics = engine->metrics();
```

同一 `compatibility_key` 和 priority 的请求才会被组入同一 batch；同优先级的组按最早请求轮转。取消已提交至 GPU
的 batch 不会中断 kernel；完成后该请求的 future 以取消错误结束，资源仍按正常路径回收。固定 batch TensorRT
engine 可通过 `StaticBatchPolicy::Pad` 以零填充并只返回有效请求，或显式选择 `Reject`。

`--engine` 接收 TensorRT engine 的绝对或当前工作目录相对路径。DINO 特征示例需要 feature-only engine；它在
C++ `EngineConfig` 中声明 `x_norm_clstoken` 作为唯一 feature/output。可使用已有 feature extraction sample 生成，
然后用 Engine sample 做数值对比：

```bash
build/bin/inferrt_sample_engine \
  --example dinov2_vits14 \
  --engine /mnt/d/Models/dinov2/dinov2_vits14_feature.engine \
  --image assets/pics/dog.jpg \
  --preprocess cuda --feature x_norm_clstoken --mode compare

build/bin/inferrt_benchmark_engine \
  --example dinov2_vits14 \
  --engine /mnt/d/Models/dinov2/dinov2_vits14_feature.engine \
  --image assets/pics/dog.jpg \
  --preprocess both --inflight 8 --warmup 10 --repeat 100
```

需要 dense patch token 时，在 `makeConfig()` 中把 DINO 的两个 C++ tensor-name 列表改为 `x_norm_patchtokens`，
并使用对应的 feature-only engine。需要新的模型或算子链时，同样直接修改或新增 C++ `makeConfig()`/`makePipeline()`
分支，再编译出新的 Plan；不会在运行的 Engine 中修改节点。

`batch.min`、`batch.opt`、`batch.max` 对应 TensorRT dynamic batch profile。原生 YOLO 可用 detection sample 构建
`1/4/8` profile：

```bash
build/bin/inferrt_sample_detection \
  --model yolov8n \
  --weights-file /mnt/d/Models/yolov8/engine_compare/yolov8n_dynamic.wts \
  --batch-min 1 --batch-opt 4 --batch-max 8
```

`buildOrLoad` 会把 `.wts` 同目录、同文件名的 `.engine` 当作缓存；为保留静态 engine，请使用不同的
`.wts` 文件名，例如上例的 `yolov8n_dynamic.wts`。

CPU 路径使用 OpenCV 做 BGR→RGB、缩放、NCHW 与归一化；CUDA 路径在代码中按模型语义定义。YOLO 调用既有
`cvcuda.letterbox`，完成 BGR→RGB、等比例缩放、padding、`uint8`→`float`、HWC→NCHW，适用于 `[0, 1]` 的
identity normalization。DINOv2 则严格使用独立算子 `resize(uint8 BGR) → cvtColor(BGR2RGB) →
normalize(uint8 HWC RGB → float NCHW)`，使用 ImageNet mean/std，不使用 letterbox。

CUDA Plan 在图片解码后以该图片尺寸声明输入 tensor；同一 Engine 生命周期内提交的图片必须保持该尺寸、连续且为
`CV_8UC3`。CPU Plan 可处理任意 OpenCV 支持的输入尺寸，作为数值和性能基线。

使用 benchmark 进行普通调用、Engine CPU 和 Engine CUDA 的端到端比较：

```bash
build/bin/inferrt_benchmark_engine \
  --example yolov8n \
  --engine /mnt/d/Models/yolov8/engine_compare/yolov8n_dynamic.engine \
  --image assets/pics/dog.jpg \
  --preprocess both --inflight 8 --warmup 20 --repeat 200
```

图片解码、模型/engine 加载、启动期分配和 warmup 不在计时范围内；计时包含预处理、数据传输、推理、D2H、结果
物化，以及 Engine 的入队和排队。同步路径报告平均端到端延迟；异步路径报告提交到 future 完成的 p50/p95/p99
延迟及墙钟吞吐量，并输出队列高水位、完成/失败/取消/丢弃计数。已在 `benchmark/engine/README.md` 中记录 YOLOv8n
和 DINOv2 ViT-S/14 的真实对比结果。
