# RoiCluster ROI 聚类完整流程

本文档说明 `src/features` 中 ROI 聚类模块的端到端架构与执行流程。核心入口是 [`irt::features::RoiCluster`](include/inferrt/features/RoiCluster.hpp)，采用与 ROI 检索统一的 DINO 原图裁剪与掩膜加权平均特征（`CropMaskedMean`），支持通过 [`RoiFeatureMatrixView`](include/inferrt/features/RoiFeature.hpp) 借用现有搜索库向量，执行免重复特征提取的 HDBSCAN 密度聚类。

## 1. 主要文件与代码索引

- [`include/inferrt/features/RoiCluster.hpp`](include/inferrt/features/RoiCluster.hpp)：公共 API、数据结构、进度回调、聚类结果及 `RoiCluster` 类声明。
- [`RoiCluster.cpp`](RoiCluster.cpp)：公共 API 的 PIMPL 转发。
- [`priv/RoiClusterImpl.hpp`](priv/RoiClusterImpl.hpp) / [`priv/RoiClusterImpl.cpp`](priv/RoiClusterImpl.cpp)：独立条目聚类与矩阵视图（`RoiFeatureMatrixView`）零拷贝聚类实现。
- [`priv/RoiFeatureExtractor.hpp`](priv/RoiFeatureExtractor.hpp)：与 ROI 搜索共用的语义特征提取器。
- [`src/ops/HDBSCAN.cpp`](../ops/HDBSCAN.cpp)：纯 C++ 高性能 HDBSCAN 算子（默认采用内存无完整距离矩阵的 Brute + Euclidean 路径）。

## 2. 配置与参数

`RoiClusterConfig` 继承自 `RoiFeatureConfig` 并聚合 `irt::ops::HDBSCANConfig`：

- **特征配置（继承自 RoiFeatureConfig）**：
  - `mode`：特征提取模式，默认 `RoiFeatureMode::CropMaskedMean`。
  - `model_name`：默认 `dinov3_vits16`。
  - `feature_name`：默认 `x_norm_patchtokens`。
  - `patch_size`：默认 `16`。
  - `crop_margin` / `background_keep` / `spatial_weight` / `max_detail_views`：与搜索保持统一的描述子空间。
- **HDBSCAN 聚类配置（hdbscan）**：
  - `min_cluster_size`：形成簇的最小样本数。
  - `min_samples`：核心点邻域样本数阈值。
  - `cluster_selection_epsilon`：距离合并阈值。
  - `metric`：距离度量方式（默认 `ClusteringMetric::Euclidean`）。
  - `algorithm`：聚类算法（默认 `ClusteringAlgorithm::Brute`，不显式保留完整 N×N 距离矩阵）。

## 3. 共享库调用模式

为了避免重复推理同一个特征库，推荐在持有 `RoiSearch` 实例的前提下，通过共享矩阵视图执行聚类：

```cpp
// 1. 搜索库准备好后：
RoiSearch library(search_config);
library.buildOrLoad(weights, items, index_path);

// 2. 聚类直接借用搜索库的特征向量内存（零拷贝，不触发模型前向与图像解码）：
RoiCluster cluster(cluster_config);
auto result = cluster.cluster(library.featureView(), progress_callback);

// 3. 调参再次聚类同样无需重提特征：
cluster_config.hdbscan.min_samples = 2;
RoiCluster recluster(cluster_config);
auto new_result = recluster.cluster(library.featureView());
```

## 4. 独立调用与异常边界

- 如果用于独立导出或脱机任务，可调用 `cluster(weights_file, items, progress)`，此时提取器统一完成图像解码、裁剪汇聚并执行聚类。
- 借用视图 `cluster(view)` 要求 `view.data` 有效且在聚类执行期间其宿主（`RoiSearch`）不被并发销毁或重建。
