# RF-DETR 高吞吐推理 Sample

`inferrt_sample_engine_rfdetr` 基于 `src/engine` 的 `InferenceEngine`（异步请求、动态组批、Pinned Memory、
多 ExecutionSlot 流水线），对 RF-DETR 动态 batch TensorRT engine 做高吞吐压测，并解码 `dets`/`labels`
输出打印/绘制检测框。同时包含普通 `IModel` 同步路径作为基线与数值参考。

## 基准模型与输入

- **模型**：RF-DETR `rfdetr-nano`（DLtools 训练，4 个前景类 + 背景，class_embed 5 维），checkpoint
  `checkpoint_best_total.pth`。
- **输入**：1×3×**1024×1024**（RGB，ImageNet 归一化 mean/std，无 letterbox），动态 batch profile 1/8/8。
- **测试图**：`F:/data/RF-DETR_TrainDatasets/chengrun_ccd34_damian/labelme/train/1.jpg`（1056×760，灰度转 BGR）。
- **GPU**：NVIDIA GeForce RTX 4060 8GB，驱动 596.36，TensorRT fp32/fp16。

两条推理链路（均转 TensorRT engine）：

1. `pth -> wts -> TensorRT`：`rfdetr_gen_wts.py` 导出（DinoV2 shape 置 1024 后 export，pos-embed 插值到 4097×384），
   原生逐层构建器建图。
2. `pth -> onnx -> TensorRT`：DLtools `defect_detect_v2.build_model` 导出 ONNX（dets 1×300×4、labels 1×300×5），
   经 `ONNXModel`（NvOnnxParser）建图。

## 构建 engine

先用 detection sample 生成动态 batch engine：

```bash
build/bin/inferrt_sample_detection.exe ^
  -m rfdetr_nano -w rfdetr_nano_1024.wts ^
  -i assets/pics/dog.jpg -l assets/coco80.names ^
  --runtime tensorrt:0 --classes 4 --input-size 1024 ^
  --batch-min 1 --batch-opt 8 --batch-max 8 --batch-size 8 [--precision fp16]
```

`buildOrLoad` 会在 wts 同目录生成 `<wts 名>.engine`，把该文件传给本 sample。

## 运行

```bash
build/bin/inferrt_sample_engine_rfdetr.exe ^
  --engine F:/tmp/rfdetr_nano_dyn_fp16.engine ^
  --image assets/pics/dog.jpg ^
  --labels F:/tmp/classes.txt ^
  --output F:/tmp/engine_fp16_result.jpg ^
  --warmup 10 --repeat 300 --inflight 32 --preprocess cuda
```

参数：

- `--engine`：动态 batch TensorRT engine（profile 必须与 `--batch-min/opt/max` 一致）。
- `--input-size`：方形输入尺寸，需与 engine 一致（默认 1024）。
- `--batch-min/opt/max`：engine 动态 batch profile（默认 1/8/8）。
- `--max-wait-us`：scheduler 组批等待窗口（默认 2000）。
- `--slots`：ExecutionSlot 数量（默认 1；fp32 1024 时单 context 已接近 8GB 显存）。
- `--inflight`：异步在途请求上限（默认 8）。
- `--preprocess`：cpu | cuda | both。
- `--conf-threshold` / `--nms-threshold` / `--max-detections`：RF-DETR 解码阈值（默认 0.35 / 0.50 / 100）。

输出：

- `IModel direct`：同步单图基线（延迟 + 吞吐 + 检测框）。
- `Engine <CPU|CUDA> sync`：Engine 同步提交延迟。
- `Engine <CPU|CUDA> async`：`--inflight` 并发窗口下的 p50/p95/p99 提交到完成延迟、墙钟吞吐、
  队列/阶段高水位与完成/失败/取消/丢弃计数、各阶段平均耗时。
- Engine 结果的 `max_abs_output_diff` 与 direct 路径对比（CPU 预处理应逐位一致，CUDA resize 有微小差异）。

## 性能测试汇总（RTX 4060 8GB，rfdetr-nano @ 1024×1024，4 类）

口径说明：detection sample 的计时为**单批总耗时**（含同步 CPU 预处理 + H2D + GPU + D2H + 后处理，
预处理在计时循环外）；engine sample 为**端到端**（预处理/H2D/D2H 与 GPU 计算在流水线中重叠，async 含排队）。

### 表 1：detection sample（串行路径，warmup=20, repeat=100，每批端到端 ms）

**FP32**

| batch | wts GPU | wts E2E | onnx GPU | onnx E2E |
|---|---|---|---|---|
| 1 | 56.62 | 92.4 | 56.45 | 74.0 |
| 2 | 114.66 | 139.7 | 113.39 | 134.6 |
| 4 | 229.91 | 259.1 | 228.70 | 261.3 |
| 8 | 476.99 | 573.6 | 458.43 | 514.8 |

**FP16**

| batch | wts GPU | wts E2E | onnx GPU | onnx E2E |
|---|---|---|---|---|
| 1 | 10.22 | 27.5 | 10.80 | 33.2 |
| 2 | 19.65 | 41.3 | 21.15 | 45.9 |
| 4 | 40.48 | 70.4 | 43.63 | 75.9 |
| 8 | 82.13 | 123.1 | 82.47 | 136.9 |

### 表 2：engine sample（动态 1/8/8 engine，端到端）

| 路径 | fp32 | fp16 |
|---|---|---|
| IModel direct 单图 | 67.9 ms，14.7 img/s | 21.2 ms，47.2 img/s |
| Engine CUDA sync | 73.7 ms，13.6 img/s | 30.0 ms，33.3 img/s |
| Engine CUDA async (inflight 8) | p50=462 ms，17.1 img/s | p50=106 ms，72.3 img/s |
| Engine CUDA async (inflight 16) | — | p50=195 ms，82.1 img/s |
| Engine CUDA async (inflight 32) | — | p50=360 ms，88.3 img/s |

### 表 3：统一换算到每图端到端（吞吐对比）

| 配置 | 每图耗时 | 吞吐 |
|---|---|---|
| detection b1 fp32 (wts) | 92.4 ms | 10.8 img/s |
| detection b8 fp32 (wts) | 71.7 ms | 14.0 img/s |
| engine async fp32 (inflight 8) | 58.3 ms | 17.1 img/s |
| detection b1 fp16 (wts) | 27.5 ms | 36.4 img/s |
| detection b8 fp16 (wts) | 15.4 ms | 65.0 img/s |
| engine async fp16 (inflight 8/32) | 13.8 / 11.3 ms | 72.3 / 88.3 img/s |

### 结论

- **fp16 相对 fp32**：GPU 快约 5.6×（fp32 无 tensor core 加速）；但 CPU 预处理不随精度变化，
  batch 1 时预处理占端到端 50%+（fp16），成为瓶颈。
- **detection（串行）vs engine（异步流水线）**：同口径下 engine 吞吐高约 20-35%
  （fp16 b8：65 → 88 img/s），收益来自预处理/H2D/D2H 与 GPU 计算重叠，以及攒批摊薄固定开销。
- **engine 并发扩展**：fp16 inflight 8→16→32，吞吐 72.3→82.1→88.3 img/s
  （batch-8 纯 GPU 上限约 97.6 img/s，差在 2ms 攒批窗口与不满批）；延迟 p50 106→195→360 ms，
  随 inflight 近似线性增长（排队深度）。这是批处理服务的标准吞吐/延迟权衡。
- **数值一致性**：CPU 预处理路径 engine 与 IModel 输出逐位一致（diff=0.000）；CUDA resize 与 OpenCV
  双线性实现有微小差异（输出 diff≈3.8，框坐标差 <1px）；fp16 相对 fp32 置信度差 <0.03。
- **显存**：fp32 1024 动态 batch-8 context 峰值约 7GB（8GB 卡上只能 1 slot）；fp16 context 更小。

## 显存占用计算（fp32 @1024，数据来自 TensorRT 构建日志与运行时日志）

单位换算：fp32 每个元素 4 字节（`sizeof(float)`），fp16 为 2 字节；1 MiB = 1024² B。

### batch 1（profile 1/1/1，实测 context +859 MiB，总显存 969 MiB）

| 项目 | 计算 | 大小 |
|---|---|---|
| 权重（Total Weights Memory） | 构建日志 `115,511,296 B` | 110.1 MiB |
| 激活池（Total Activation Memory） | 构建日志 `901,301,248 B` = 859.5 MiB | **859.5 MiB** |
| 输入张量 | 1×3×1024×1024×4B = 12,582,912 B | 12.0 MiB |
| 输出张量 | dets 1×300×4×4B + labels 1×300×5×4B = 10,800 B | ~0.01 MiB |
| **合计** | | **≈ 981.6 MiB ≈ 0.96 GB** |

日志吻合：context 分配 `GPU +859 MiB` = 激活池（`901,301,248 ÷ 1024² = 859.5` 精确相等）；建完 context 后
`now: GPU 969 MiB` = 859.5（激活）+ 110.1（权重）= 969.6 ✓。

### 激活池 860 MiB 的构成（单图 4096 token = 64×64，12 层 transformer 共享 pool）

| 单层项目（每图 fp32） | 计算（×4B = 4 字节/元素） | 大小 |
|---|---|---|
| Q/K/V 投影 | 3 × 4096×384×4B | 18.9 MB |
| 注意力分数（4 窗口×1024²） | 4 × 1024×1024×4B | 16.8 MB |
| 注意力输出 + 残差 | ~2 × 4096×384×4B | 12.6 MB |
| FFN（384→2048，双向） | 2 × 4096×2048×4B | **67.1 MB** |
| 单层合计 | | ≈ 115 MB |

12 层不是各存一份（否则 ~1.4GB），而是 pool 复用取峰值 ~860 MB（含 encoder 输入/位置编码、P4 级
multi-scale projector、decoder deformable attention 等叠加）。`Max Scratch Memory 823.5 MiB` 是构建期临时
工作区，运行时并入/按需分配，不常驻。

### batch 8（profile 1/8/8，实测 context +6876 MiB）

激活池随 batch 近似线性放大：6876 ≈ 8 × 859.5。context 内存按 profile 上限（opt=max=8）一次性分配，
**与运行 batch 无关**——即使实际跑 batch 1，context 仍占 ~6.9GB，这就是 8GB 卡上只能 1 个 ExecutionSlot、
且 DirectRunner 与 Engine 不能共存的原因（实测共存时 OOM：`Requested amount of GPU memory 7210309632 bytes`）。

fp16 时激活与权重各约减半（激活 ~430 MiB、权重 ~57 MiB），单 context ≈ 3.5GB，两个 slot 理论可容纳，
但未实测。
