# RoiSearch ROI 搜索匹配完整流程

本文档说明 `src/features` 中 ROI 搜索匹配模块的端到端流程。核心入口是
[`irt::features::RoiSearch`](include/inferrt/features/RoiSearch.hpp)，默认采用 DINO 图像原图裁剪与掩膜加权平均汇聚（`CropMaskedMean`），生成统一 D 维语义特征向量，并通过精确内积索引（默认 GPU FP16 `GpuIndexFlatIP`，支持通过配置自由选择 CPU/GPU 与 FP16/FP32）完成 Top-K 检索。同时支持通过 [`featureView()`](include/inferrt/features/RoiSearch.hpp) 借用特征向量供聚类使用，无需重复提取。

## 1. 主要文件与职责索引

- [`include/inferrt/features/RoiFeature.hpp`](include/inferrt/features/RoiFeature.hpp)：ROI 特征提取共用数据结构（`RoiFeatureItem` 支持矩形与多边形 `polygon`）、配置（`RoiFeatureConfig`）、矩阵视图（`RoiFeatureMatrixView`）与工作量统计（`RoiFeatureWorkStats`）。
- [`include/inferrt/features/RoiSearch.hpp`](include/inferrt/features/RoiSearch.hpp)：公共搜索 API、配置结构（`RoiSearchConfig`，默认 `exact_search = true`）与 `RoiSearch` 类声明。
- [`RoiSearch.cpp`](RoiSearch.cpp)：公共 API 的 PIMPL 转发与默认 ROI 索引路径生成。
- [`priv/RoiSearchImpl.hpp`](priv/RoiSearchImpl.hpp) / [`priv/RoiSearchImpl.cpp`](priv/RoiSearchImpl.cpp)：ROI 条目校验、流式索引构建、矩阵借用与多边形/库内/调参查询实现。
- [`priv/RoiEmbeddingCore.hpp`](priv/RoiEmbeddingCore.hpp)：几何变换、光栅化掩膜、patch 覆盖权重计算、L2 汇聚与多视图融合纯算法核。
- [`priv/RoiFeatureExtractor.hpp`](priv/RoiFeatureExtractor.hpp) / [`priv/RoiFeatureExtractor.cpp`](priv/RoiFeatureExtractor.cpp)：跨图像合批、单图按需解码、模型推导与流式提取（`extractTo`）。
- [`priv/LegacyRoiFeatureExtractor.hpp`](priv/LegacyRoiFeatureExtractor.hpp) / [`priv/LegacyRoiFeatureExtractor.cpp`](priv/LegacyRoiFeatureExtractor.cpp)：用于历史对照的旧版本 RoIAlign + 局部 PCA 提取路径。
- [`include/inferrt/features/RoiFeatureEncoder.hpp`](include/inferrt/features/RoiFeatureEncoder.hpp)：独立特征提取器公共包装。

## 2. 配置与核心参数

`RoiSearchConfig` 继承自 `RoiFeatureConfig` 与 `ImageSearchConfig`：

- `mode`：特征提取模式，默认 `RoiFeatureMode::CropMaskedMean`；可选 `RoiFeatureMode::LegacyRoiAlign`（仅历史对照）。
- `model_name`：特征模型名称，默认 `dinov3_vits16`。
- `feature_name`：特征张量名称，默认 `x_norm_patchtokens`。
- `patch_size`：DINO patch 大小（DINOv3 为 16，DINOv2 为 14）。
- `crop_margin`：ROI 裁剪留边比例，默认 `0.05`。
- `background_keep`：背景抑制系数，默认 `1.0`（保留自然上下文）。
- `spatial_weight`：2×2 空间网格权重，默认 `0.0`（输出 D 维；大于 0 时输出 5D）。
- `max_detail_views`：细长 ROI 附加分区视图上限，默认 `0`（可选 2 或 3）。
- `exact_search`：搜索精确度，默认 `true`（使用精确内积检索，无近似量化损失；遵循 `faiss_backend` 与 `model_precision` 配置，默认 GPU FP16）。

## 3. 输入数据模型

```cpp
struct RoiFeaturePoint { float x{0}, y{0}; };

struct RoiFeatureItem
{
    int64_t                      roi_id{0};  ///< 调用方提供的唯一 ID
    std::filesystem::path        image_path; ///< 图像路径
    RoiFeatureBox                roi;        ///< 矩形 [x1, y1, x2, y2)
    std::vector<RoiFeaturePoint> polygon;    ///< 非空时 polygon 为权威几何，roi 为推导外接框
};
```

## 4. 特征构建与检索核心机制

1. **流式建库与跨图合批**：
   - 提取任务按 `image_path` 组织，同一张图像只解码一次。
   - 每个 ROI 根据几何规则生成裁剪图与掩膜输入，跨图打包至 `model_batch_size` 统一前向。
   - `extractTo` 直接流式写入预分配的 Flat 向量内存中，消除中间临时文件和双份内存占用。
2. **掩膜与 Patch 汇聚**：
   - 矩形使用精确水平与垂直相交覆盖率；多边形使用扫描线多采样点精确积分覆盖率。
   - patch 权重为掩膜在每个 patch 网格内的平均覆盖面积。
   - foreground token 归一化后按权重加权平均，再进行整向量 L2 归一化。
3. **共享库与免重复推理**：
   - `featureView()`：返回底层 `IndexFlatIP` 的只读 `RoiFeatureMatrixView`，供 `RoiCluster` 直接借用聚类，无需重复推导。
   - `searchByRoiId(roi_id, k)`：直接查找库内已有向量检索，无需图像解码与模型推理。
   - `repeatSearch(k)`：复用上一条查询向量（用于调整 Top-K），无需重新推理。
