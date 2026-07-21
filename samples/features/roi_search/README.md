# LabelMe ROI 搜索示例

`inferrt_sample_roi_search` 从 LabelMe 的 `images` 和 `annotations` 目录读取多边形标注，将每个标注的外接矩形作为一个 ROI，使用空间特征图、ROIAlign 和 Faiss 建立 ROI 检索库。

## 构建

```powershell
cmake --build build --config Release --target inferrt_sample_roi_search --parallel 8
```

## 真实数据运行

`F:\data\tmp\ori-21` 的目录结构为：

```text
ori-21/
  images/*.jpg
  annotations/*.json
```

LabelMe 的每个 `shape` 会生成一个 ROI，`imagePath` 用于关联图像；如果缺失则按标注文件名尝试常见图像扩展名。

```powershell
build\bin\inferrt_sample_roi_search.exe `
  --weights-file F:\models\dinov3\dinov3_vits16.wts `
  --images-dir F:\data\tmp\ori-21\images `
  --annotations-dir F:\data\tmp\ori-21\annotations `
  --model dinov3_vits16 `
  --feature x_norm_patchtokens `
  --runtime tensorrt:0 `
  --precision fp16 `
  --model-batch-size 8 `
  --index build\roi_search\ori-21_dinov3.faiss `
  --rebuild-index `
  --query-index 0 `
  --topk 5 `
  --warmup 0 `
  --repeat 1
```

输出会分别报告：LabelMe 解析、模型加载、ROI 特征提取、Faiss 建库、查询，以及各建库阶段占总耗时的比例。建库时重点观察 `extracting_features`；查询计时包含查询图像读取、预处理、模型前向、ROIAlign、展平归一化和 Faiss 搜索。

实现会先按图像路径分组：同一张图的多个 ROI 只执行一次模型前向，并在同一张特征图上批量 ROIAlign；不同图像按模型允许的 batch 打包。`--model-batch-size 8` 与上述动态 batch engine 配合，可避免逐 ROI 单独前向。

## 真实数据瓶颈与优化验证

数据集统计：目录中 1,854 张图像、1,854 个 JSON，解析得到 3,315 个有效 ROI，实际参与 ROI 特征提取的唯一图像为 1,253 张（其余标注没有有效 polygon）。下面均为
TensorRT FP16、DINOv3 ViT-S/16、`x_norm_patchtokens`、ROIAlign `7x7`、Faiss RAM IVF-PQ、
`--warmup 0 --repeat 1` 的单次建库结果。两次使用完全相同的 ROI 顺序和索引配置。

说明：原始“逐 ROI 前向 + 串行预处理”在完整数据上 60 秒仍未完成；因此表格使用已经完成的“按图像分组复用 + 串行预处理”作为可重复的优化前基线，避免把超时外推值当成正式结果。

| 阶段 | 优化前：按图像分组复用 + 串行预处理 | 优化后：按图像分组复用 + 并行预处理 | 变化 |
| --- | ---: | ---: | ---: |
| LabelMe 解析 | 224.273 ms | 233.896 ms | +4.29%（解析抖动） |
| 模型加载 | 246.874 ms | 243.077 ms | -1.54%（加载抖动） |
| ROI 特征提取（解码/预处理/前向/ROIAlign） | 153,237.765 ms | 33,139.209 ms | **4.62x，减少 78.37%** |
| Faiss 建库 | 7,592.189 ms | 7,524.269 ms | -0.89%（建库抖动） |
| 建库总耗时 | 161,084.665 ms | 40,914.324 ms | **3.94x，减少 74.60%** |
| 单次查询（无预热） | 35.463 ms | 70.726 ms | +99.44%（GPU 冷启动抖动） |

优化后的阶段占比为：特征提取 80.997%，Faiss 建库 18.390%，模型加载 0.594%。单次查询未预热，包含查询图像解码、预处理和 TensorRT 首次执行，不能用于衡量建库优化；实际服务应使用预热后的查询统计。因此剩余主要瓶颈仍是图像解码、ImageNet 预处理和模型前向；Faiss 建库不是主要瓶颈。

并行预处理位于 `ImageFeatureExtractor` 共用路径，因此 `ImageSearch` 和 `ImageCluster` 也会获得相同的解码/预处理加速。本次并行化只改变 batch 内任务调度，不改变每张图的解码、颜色转换、缩放、归一化、NCHW 排列或 ROI 顺序。优化前后的 3,315-ROI Faiss 文件 SHA-256 均为
`02484031046D849FAF5AE995E4F0EFC994BD8A365A52B43B2C49D64E28E0D83A`，Top-K 的 ROI ID、排序和分数一致，属于严格等价优化。

优化后完整数据复现实验（优化前数据来自同一命令、同一配置的修改前构建产物，耗时记录见上表）：

```powershell
build\bin\inferrt_sample_roi_search.exe `
  --weights-file F:\models\dinov3\dinov3_vits16.wts `
  --images-dir F:\data\tmp\ori-21\images `
  --annotations-dir F:\data\tmp\ori-21\annotations `
  --model dinov3_vits16 --feature x_norm_patchtokens `
  --runtime tensorrt:0 --precision fp16 --model-batch-size 8 `
  --index build\roi_search\optimized_full_final.faiss --rebuild-index `
  --query-index 0 --topk 5 --warmup 0 --repeat 1
```

## 参数

- `--weights-file`：模型权重文件；TensorRT 会根据 `.wts` 自动查找同名 engine（例如 `dinov3_vits16.wts` 对应 `dinov3_vits16.engine`）。
- `--images-dir`、`--annotations-dir`：LabelMe 图像和 JSON 目录。
- `--model`、`--feature`、`--runtime`、`--precision`：特征模型配置；ROI 必须选择空间特征图或 DINO patch token。已有 FP16 engine 时使用 `--precision fp16`。
- `--model-batch-size`：特征提取 batch，不能超过模型支持的动态 batch。
- `--pooled-height`、`--pooled-width`、`--sampling-ratio`：ROIAlign 参数。
- `--max-items`：只取前 N 个 ROI 进行快速验证，`0` 表示全部。
- `--query-index`、`--topk`：选择一个图库 ROI 作为查询并返回 Top-K。
- `--rebuild-index`：强制重新提取特征和建库；不指定时若索引 manifest 匹配会直接加载。
- `--warmup`、`--repeat`：查询阶段预热和重复次数。

该示例的计时用于定位瓶颈；优化前后应保持相同数据、模型、ROIAlign 参数和 batch 设置，并使用 `--rebuild-index` 对比建库阶段。
