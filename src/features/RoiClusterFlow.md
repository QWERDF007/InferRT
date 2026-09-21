# RoiCluster ROI 聚类完整流程

本文档说明 `src/features` 中 ROI 聚类模块的端到端架构与执行流程。核心入口是 [`irt::features::RoiCluster`](include/inferrt/features/RoiCluster.hpp)，采用 DINO 原图裁剪与掩膜加权平均特征（`CropMaskedMean`），执行自包含的特征提取与 HDBSCAN 密度聚类。

## 1. 主要文件与代码索引

- [`include/inferrt/features/RoiCluster.hpp`](include/inferrt/features/RoiCluster.hpp)：公共 API、数据结构、进度回调、聚类结果及 `RoiCluster` 类声明。
- [`RoiCluster.cpp`](RoiCluster.cpp)：公共 API 的 PIMPL 转发。
- [`priv/RoiClusterImpl.hpp`](priv/RoiClusterImpl.hpp) / [`priv/RoiClusterImpl.cpp`](priv/RoiClusterImpl.cpp)：ROI 条目特征提取与 HDBSCAN 聚类实现。
- [`priv/RoiFeatureExtractor.hpp`](priv/RoiFeatureExtractor.hpp)：语义特征提取器。
- [`src/ops/HDBSCAN.cpp`](../ops/HDBSCAN.cpp)：纯 C++ 高性能 HDBSCAN 算子（默认采用内存无完整距离矩阵的 Brute + Euclidean 路径）。

## 2. 配置与参数

`RoiClusterConfig` 继承自 `RoiFeatureConfig` 并聚合 `irt::ops::HDBSCANConfig`：

- **特征配置（继承自 RoiFeatureConfig）**：
  - `mode`：特征提取模式，默认 `RoiFeatureMode::CropMaskedMean`。
  - `model_name`：默认 `dinov3_vits16`。
  - `feature_name`：默认 `x_norm_patchtokens`。
  - `patch_size`：默认 `16`。
  - `crop_margin` / `background_keep` / `spatial_weight` / `max_detail_views`：统一的描述子空间参数。
- **HDBSCAN 聚类配置（hdbscan）**：
  - `min_cluster_size`：形成簇的最小样本数。
  - `min_samples`：核心点邻域样本数阈值。
  - `cluster_selection_epsilon`：距离合并阈值。
  - `metric`：距离度量方式（默认 `ClusteringMetric::Euclidean`）。
  - `algorithm`：聚类算法（默认 `ClusteringAlgorithm::Brute`，不显式保留完整 N×N 距离矩阵）。

## 3. 调用模式与流程

`RoiCluster` 为独立的自包含模块，对外仅提供 `cluster(weights_file, items, progress_callback)`：

```cpp
RoiClusterConfig config;
config.hdbscan.min_cluster_size = 5;
config.hdbscan.min_samples = 3;

RoiCluster cluster(config);
auto result = cluster.cluster(weights_path, items, progress_callback);

// 遍历聚类结果：
for (const auto &assignment : result.assignments)
{
    // assignment.roi_id
    // assignment.cluster_id (-1 表示噪声)
    // assignment.probability
}
```

内部自动完成跨图像合批、单图按需解码、DINO 特征提取与流式 HDBSCAN 密度聚类全流程。
