# Engine benchmark

`inferrt_benchmark_engine` 使用 [EngineExample.hpp](../../samples/engine/EngineExample.hpp) 中同一份 C++ 配置、
代码定义 DAG、图片和 TensorRT engine 测量三条完整数据路径：

1. `IModel direct`：CPU 预处理、同步 H2D、推理、同步 D2H。
2. `Engine sync`：通过 `InferenceEngine::infer()` 走 Engine 路径。
3. `Engine async`：通过 `submit()` 保持多个 in-flight 请求，测量动态 batch 和多执行槽流水线吞吐。

```bash
./build/bin/inferrt_benchmark_engine \
  --example resnet18 \
  --engine /path/to/resnet18.engine \
  --image assets/pics/dog.jpg \
  --warmup 20 --repeat 200 --inflight 16
```

报告为端到端平均延迟和吞吐，不把模型加载或磁盘读取计入测量。异步路径额外输出各阶段队列、Pinned ticket pool
高水位、DeviceArena 大小、完成/失败/取消/丢弃计数，以及 queue/CPU prepare/GPU/CPU post 的平均阶段延迟。
使用 Nsight Systems 时，应观察 Engine async 的跨 batch copy/compute 重叠。

## 实测结果

测试环境：RTX 4090、TensorRT 10.16、CUDA 12.8；输入为 `assets/pics/dog.jpg`（768 × 576），
`warmup=10`、`repeat=100`、`inflight=8`。图片解码、模型/engine 加载、启动分配和 warmup 不计入时间；
计时包含预处理、H2D、TensorRT、D2H、结果物化以及 Engine 的入队和排队。

表中的 `IModel CPU` 指 CPU/OpenCV 预处理和同步 TensorRT 调用；模型推理本身仍在 GPU。`Engine CPU` 与
`Engine CUDA` 都使用异步 Engine 路径，后者分别按模型语义使用 CUDA 预处理。吞吐倍率以 `IModel CPU` 的墙钟吞吐为基准。

### YOLOv8n（dynamic batch 1/4/8）

| 路径 | 端到端指标 | 墙钟吞吐 | 相对 IModel CPU |
|---|---:|---:|---:|
| IModel CPU | 平均 3.295 ms | 303.53 images/s | 1.00× |
| Engine CPU | p50 9.053 ms | 833.74 images/s | 2.75× |
| Engine CUDA | p50 7.234 ms | 1078.74 images/s | 3.55× |

YOLO 的 CUDA 预处理为现有 `letterbox` 算子，完成 BGR→RGB、等比例缩放、padding、HWC→NCHW 与 `/255`；
该模型使用 identity normalization。

### DINOv2 ViT-S/14（feature-only，`x_norm_clstoken`）

| 路径 | 端到端指标 | 墙钟吞吐 | 相对 IModel CPU |
|---|---:|---:|---:|
| IModel CPU | 平均 6.775 ms | 147.60 images/s | 1.00× |
| Engine CPU | p50 37.498 ms | 211.87 images/s | 1.44× |
| Engine CUDA | p50 37.308 ms | 210.81 images/s | 1.43× |

DINO 的 CUDA 预处理严格为 `resize(uint8 BGR) → BGR2RGB → Normalize(uint8 HWC RGB → float NCHW)`，使用
ImageNet mean/std，不使用 letterbox。GPU resize 与 OpenCV resize 的插值量化不同，因此两条数值路径的 CLS
feature 最大绝对差异为 `0.00897`；两者均生成 384 维 `x_norm_clstoken`。

同步参考数据：YOLO Engine CPU/CUDA 分别为 5.130/4.400 ms，DINO Engine CPU/CUDA 分别为 5.507/4.992 ms。
异步 p50 是请求从提交到 future 完成的端到端延迟，不能与单请求同步平均延迟直接比较；吞吐倍率以各自的墙钟吞吐计算。
