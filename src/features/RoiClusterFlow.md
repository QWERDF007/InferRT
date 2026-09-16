# RoiCluster ROI 聚类完整流程

本文档说明 `src/features` 中 ROI 聚类模块的端到端架构与执行流程。核心入口是 `irt::features::RoiCluster`，它复用图像特征图提取与 `RoIAlign` 能力，并在特征向量上执行密度聚类算法（HDBSCAN），实现免训练、无监督的 ROI 局部模式发现与异常筛选。

## 1. 主要文件与代码落点

- `include/inferrt/features/RoiCluster.hpp`：公共 API、数据结构、配置及 `RoiCluster` 类声明。
- `RoiCluster.cpp`：公共 API 的 PIMPL 转发与生命周期管理。
- `priv/RoiClusterImpl.hpp/.cpp`：ROI 图像分组前向、特征提取调度与 HDBSCAN 聚类执行。
- `priv/RoiFeatureExtractor.hpp/.cpp`：ROI 聚类与 ROI 检索共享的模型特征图抽取、RoIAlign、PCA 降维与向量归一化。
- `src/ops/HDBSCAN.cpp`：纯 C++ 高性能 HDBSCAN 聚类算子（最小生成树、互可达图、树状图收缩与稳定聚类提取）。
- `tests/features/TestRoiCluster.cpp`：ROI 聚类单元测试与数值验证。

## 2. 配置入口

`RoiClusterConfig` 继承 `RoiFeatureConfig` 并聚合 `irt::ops::HDBSCANConfig`：

- **特征抽取配置（继承自 RoiFeatureConfig）**：
  - `model_name`：特征提取骨干网络，默认 `resnet18`。
  - `feature_name`：用于 RoIAlign 的空间特征图张量名称，默认 `layer4`。
  - `model_runtime`：运行目标（如 `tensorrt:0`、`onnxruntime:cpu`、`openvino:cpu`）。
  - `model_precision`：模型精度（`FP32` 或 `FP16`）。
  - `preprocess_backend`：预处理后端（`CPU` 或 `GPU`）。
  - `norm`：向量归一化方式（默认 L2）。
  - `pooled_height` / `pooled_width`：`RoIAlign` 输出网格大小，默认 `7 x 7`。
  - `sampling_ratio`：`RoIAlign` 每个 bin 采样率（`-1` 自适应）。
  - `aligned`：是否启用 aligned 坐标规则。
  - `use_pca` / `pca_dim`：是否启用每图本地 PCA 降维。
- **HDBSCAN 聚类配置（hdbscan）**：
  - `min_cluster_size`：成簇的最小样本数。
  - `min_samples`：核心点邻域样本数阈值。
  - `cluster_selection_epsilon`：距离合并阈值。
  - `metric`：距离度量方式（Euclidean / Cosine 等）。

## 3. 输入数据模型

聚类条目由 `std::vector<RoiClusterItem>` 输入：

```cpp
struct RoiClusterItem
{
    int64_t               roi_id;
    std::filesystem::path image_path;
    RoiClusterBox         roi; // 浮点半开区间 [x1, y1, x2, y2)
};
```

要求：
- `roi_id` 唯一且顺序稳定；
- `image_path` 真实有效且为支持的图片格式；
- 坐标基于原始图像像素，无需预先缩放。

## 4. 端到端执行流程

```
[输入 std::vector<RoiClusterItem>]
       │
       ▼
[按 image_path 分组与批处理]
 └── 同一张图仅执行一次模型前向提取特征图
 └── 同图所有 ROI 在共享特征图上批量执行 RoIAlign
       │
       ▼
[特征展平与归一化]
 └── 展平为 pooled_height * pooled_width * channels
 └── 可选 PCA 降维与 L2 归一化，保持输入顺序一致
       │
       ▼
[HDBSCAN 聚类计算 (irt::ops::HDBSCAN)]
 └── 构建互可达图与最小生成树 (MST)
 └── 层次树凝聚与稠密簇提取
       │
       ▼
[封装输出 RoiClusterResult]
 └── assignments: 每条 ROI 的 cluster_id 与置信度 probability
 └── 统计非噪声簇数 cluster_count 与噪声数 noise_count
```

## 5. 进度阶段

聚类进度通过 `RoiClusterProgressCallback` 汇报，包含三个主要阶段：
1. `LoadingModel`：加载特征提取模型与创建运行时；
2. `ExtractingFeatures`：图像分批前向推理与批量 RoIAlign 抽取特征；
3. `Clustering`：执行高维向量空间 HDBSCAN 计算。
