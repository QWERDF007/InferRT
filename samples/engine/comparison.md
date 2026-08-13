# yolov8n vs RF-DETR 性能与显存对比（1024×1024）

## 测试环境与口径

- **GPU**：NVIDIA GeForce RTX 4060 8GB，驱动 596.36，TensorRT（fp32/fp16）。
- **测试图**：`F:/data/RF-DETR_TrainDatasets/chengrun_ccd34_damian/labelme/train/1.jpg`（1056×760，灰度转 BGR）。
- **输入**：均 1×3×1024×1024。yolov8n 用 letterbox（identity 归一化）；RF-DETR 用直接 resize + ImageNet 归一化。
- **两条推理链路**：
  1. detection sample（`samples/model/detection`）：wts 原生构建，串行 H2D→infer→D2H，计时循环不含 CPU 预处理；
  2. engine sample（`samples/engine/engine_basic` + `benchmark/engine`）：`InferenceEngine` 异步流水线，端到端含排队。
- 动态 batch profile 均 1/8/8（`--batch-min 1 --batch-opt 8 --batch-max 8`）。

## 模型

| | yolov8n | rfdetr-nano |
|---|---|---|
| 权重 | `F:\Projects\dltool\obj\models\111\train\logs\weights\best.pt` | `checkpoint_best_total.pth`（DLtools 训练） |
| 类别 | 2（names: 0, 1） | 4 前景 + 背景（class_embed 5 维） |
| 架构 | 轻量 CNN（C2f + SPPF + DFL head） | DINOv2 12 层 transformer + two-stage decoder（300 queries） |
| 输入尺寸 | 1024（train_args.imgsz=1024） | 1024（DLtools resolution=1024） |

## 一、显存占用（TensorRT 构建/运行时日志）

单位：MiB = 1024² B；fp32 每元素 4B，fp16 2B。

| 项目 | yolov8n fp32 | yolov8n fp16 | rfdetr fp32 | rfdetr fp16 |
|---|---|---|---|---|
| 激活池（Total Activation Memory） | 336 MiB（352,321,536 B） | 172 MiB（180,355,072 B） | 859.5 MiB（901,301,248 B） | ~430 MiB（约减半） |
| 权重（Total Weights Memory） | 15.6 MiB | 5.8 MiB | 110.1 MiB | ~57 MiB |
| engine 文件 | 18.3 MB | 8.2 MB | 113.6 MB | 59.8 MB |
| **IExecutionContext 分配（b8 profile）** | **+336 MiB** | **+172 MiB** | **+6876 MiB** | ~3.4 GB |
| 建 context 后 GPU 总量 | 352 MiB | 177 MiB | 6986 MiB | ~3.5 GB |
| 输入张量（1×3×1024² fp32） | 12 MiB | 6 MiB | 12 MiB | 6 MiB |

关键差异：**b8 context 显存相差约 19.5 倍**（352 vs 6876 MiB）。RF-DETR 的 transformer 激活随
batch 近似线性放大（859.5 × 8），且按 profile 上限一次性分配——8GB 卡只能 1 个 ExecutionSlot；
yolov8n 的 CNN 激活池小一个量级，可多 slot/更大 batch。

## 二、性能：detection sample（串行，warmup=10, repeat=100，每批 ms）

**yolov8n**（GPU 纯推理 / 端到端含 CPU 预处理）

| batch | fp32 | fp16 |
|---|---|---|
| 1 | 3.46 / 18.5 | 1.96 / 17.3 |
| 2 | 6.83 / 27.3 | 3.00 / 26.2 |
| 4 | 14.65 / 43.3 | 5.80 / 35.2 |
| 8 | 30.65 / 75.9 | 11.96 / 64.8 |

**rfdetr-nano**（wts 原生，warmup=20, repeat=100，每批 ms）

| batch | fp32 | fp16 |
|---|---|---|
| 1 | 56.62 / 92.4 | 10.22 / 27.5 |
| 2 | 114.66 / 139.7 | 19.65 / 41.3 |
| 4 | 229.91 / 259.1 | 40.48 / 70.4 |
| 8 | 476.99 / 573.6 | 82.13 / 123.1 |

## 三、性能：engine sample（InferenceEngine 异步流水线，端到端）

| 路径 | yolov8n fp32 | yolov8n fp16 | rfdetr fp32 | rfdetr fp16 |
|---|---|---|---|---|
| IModel direct | 18.0 ms，55.7 img/s | 16.2 ms，61.7 img/s | 67.9 ms，14.7 img/s | 21.2 ms，47.2 img/s |
| Engine CUDA sync | 30.0 ms，33.4 img/s | 29.9 ms，33.4 img/s | 73.7 ms，13.6 img/s | 30.0 ms，33.3 img/s |
| async (inflight 8) | p50=32 ms，220.1 img/s | p50=16 ms，270.4 img/s | p50=462 ms，17.1 img/s | p50=106 ms，72.3 img/s |
| async (inflight 16) | — | p50=32 ms，308.3 img/s | — | p50=195 ms，82.1 img/s |
| async (inflight 32) | — | p50=58 ms，415.6 img/s | — | p50=360 ms，88.3 img/s |

纯 GPU 上限参考：yolov8n fp16 b8 ≈ 8/0.01196 ≈ 669 img/s；rfdetr fp16 b8 ≈ 8/0.0821 ≈ 97.6 img/s。

## 四、对比结论

| 指标 | yolov8n / rfdetr 比值 |
|---|---|
| GPU fp32 batch1 | **16.0×**（3.46 vs 56.62 ms） |
| GPU fp16 batch1 | **5.2×**（1.96 vs 10.22 ms） |
| fp32 async (inflight 8) 吞吐 | **12.9×**（220 vs 17.1 img/s） |
| fp16 async (inflight 32) 吞吐 | **4.7×**（415.6 vs 88.3 img/s） |
| fp32 激活池 | 2.6×（336 vs 859.5 MiB） |
| **fp32 b8 context 显存** | **19.5×**（352 vs 6876 MiB） |

1. **计算量**：yolov8n 是轻量 CNN，GPU 推理比 RF-DETR（DINOv2 12 层注意力 + 300 query 解码）快一个量级；
   fp32 差距（16×）大于 fp16（5.2×），因为 RF-DETR 的注意力/FFN 算子在 fp32 下吃不到 tensor core。
2. **显存**：差距最大。RF-DETR 的 transformer 激活随 batch 线性放大（b8 = 8×859.5 MiB）且按 profile 上限
   一次性分配；yolov8n 卷积激活池仅 336 MiB。8GB 卡上 rfdetr fp32 只能 1 slot，yolov8n fp16 可多 slot/大 batch。
3. **流水线收益**：两者 async 吞吐都接近各自 GPU 上限（yolov8n 415/669 ≈ 62%，rfdetr 88.3/97.6 ≈ 90%），
   剩余差距来自 2ms 攒批窗口与不满批；高 inflight 时延迟 p50 随排队深度线性增长。
4. **CPU 预处理**：1024² 的 CPU 预处理（~13-35ms）在 fp16 小模型场景成为明显瓶颈（yolov8n fp16 b1 端到端
   17.3ms 中 GPU 仅 1.96ms）；CUDA 预处理路径可消除该瓶颈。

## 检测结果（同图，可复现）

- yolov8n：2 个目标，class 1，conf 0.397 / 0.302，框大致覆盖图像中部与右侧区域（阈值 0.25）。
- rfdetr：2 个目标，class 1，conf 0.79 / 0.61（fp32），框与 yolov8n 第一个目标区域一致。
- 两模型类别体系不同（yolov8n 2 类 vs rfdetr 4 类映射），检测数量与置信度不可直接比较，仅作功能验证。
