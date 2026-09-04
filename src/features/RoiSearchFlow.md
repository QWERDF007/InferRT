# RoiSearch ROI 搜索匹配完整流程

本文档说明 `src/features` 中 ROI 搜索匹配模块的端到端流程。核心入口是
`irt::features::RoiSearch`，它复用图像搜索模块中的模型特征抽取、特征归一化和 Faiss
索引能力，并在空间特征图上增加 ROI 坐标映射与 `RoIAlign`，最终完成 ROI 级特征库构建和 Top-K 匹配。

## 1. 主要文件

- `include/inferrt/features/RoiSearch.hpp`：公共 API、ROI 数据结构、配置结构和 `RoiSearch` 类声明。
- `RoiSearch.cpp`：公共 API 的 PIMPL 转发与默认 ROI 索引路径生成。
- `priv/RoiSearchImpl.hpp`：`RoiSearch::Impl` 私有实现声明。
- `priv/RoiSearchImpl.cpp`：ROI 条目校验、索引构建/加载和查询实现。
- `priv/RoiFeatureExtractor.hpp`：ROI 搜索与 ROI 聚类共用的模型特征图、ROIAlign、PCA 和归一化实现。
- `priv/ImageFeatureExtractor.hpp/.cpp`：图像级与 ROI 级检索共用的模型特征图抽取器。
- `priv/FeatureSearchCommon.hpp/.cpp`：图像级与 ROI 级检索共用的配置校验、归一化和 Faiss 索引工具。
- `priv/ImageSearchFaissIndex.hpp`：Faiss RAM IVF-PQ 与 CPU 磁盘 IVF+Flat 索引构建工具。
- `src/ops/RoIAlign.cpp`：ROIAlign 算子实现，用于把任意 ROI 特征映射到统一空间大小。

## 2. 配置入口

`RoiSearchConfig` 继承 `ImageSearchConfig`，因此会复用图像搜索已有配置：

- `model_name`：用于抽取特征图的模型名称，默认 `resnet18`。
- `feature_name`：用于 ROIAlign 的空间特征图张量名称，默认 `layer4`。
- `model_runtime`：特征抽取模型运行目标，统一包含后端和设备；GPU 使用 `tensorrt:0`、`onnxruntime:0` 或
  `openvino:1`，CPU 使用 `onnxruntime:cpu` 或 `openvino:cpu`。也支持裸设备简写 `cpu`、`gpu:0`、`cuda:0`。
  GPU Faiss 使用该目标的设备编号。
- `model_precision`：底层模型构建/加载精度，支持 `FP32` 和 `FP16`。
- `preprocess_backend`：支持 CPU 或 GPU 预处理。GPU 路径使用 CVCUDA 在设备端执行共享 `PreprocessSpec`，并在图后端需要时回传主机输入。
- `norm`：ROI 展平特征的归一化方式，默认 L2。
- `faiss_backend`：Faiss 搜索后端，支持 CPU 或 GPU。
- `index_storage`：CPU Faiss 可选择 RAM 或 Disk；GPU Faiss 会强制使用 RAM。
- `model_batch_size`：模型前向批量；ROI 特征提取和 Faiss 添加/落盘都会使用该批量推进。

ROI 检索额外增加：

- `pooled_height` / `pooled_width`：`RoIAlign` 输出空间大小，默认 `7 x 7`。
- `sampling_ratio`：`RoIAlign` 每个 bin 的采样率，`-1` 表示自适应。
- `aligned`：是否启用 aligned ROIAlign 坐标规则。
- `use_pca`：是否对每张图自己的特征图通道维训练本地 OpenCV PCA 并降维，默认关闭。
- `pca_dim`：PCA 输出通道数；`use_pca=true` 时必须大于 0，且不能超过原始特征图通道数和当前特征图空间位置数量。

## 3. 输入数据模型

ROI 特征库由 `std::vector<RoiSearchItem>` 描述：

```cpp
struct RoiSearchItem
{
    int64_t roi_id;
    std::filesystem::path image_path;
    RoiSearchBox roi;
};
```

`RoiSearchBox` 使用原始图像像素坐标：

```cpp
struct RoiSearchBox
{
    float x1;
    float y1;
    float x2;
    float y2;
};
```

要求：

- `image_path` 必须存在，且必须是 `ImageSearch::isImageFile()` 支持的图片格式。
- `roi_id` 由调用方提供，调用方负责保证顺序稳定且唯一。
- ROI 坐标必须为有限数。
- `x2 > x1` 且 `y2 > y1`。
- ROI 坐标以原图为基准，调用方不需要预先缩放到模型输入尺寸或特征图尺寸。

## 4. 构建或加载索引

常用入口：

```cpp
RoiSearch searcher(config);
searcher.buildOrLoad(weights_file, gallery_items, index_file, rebuild_index, progress_callback);
```

`index_file` 为空时会在当前工作目录生成时间戳 `.faiss` 文件。`buildOrLoad` 的决策流程：

1. 解析最终 `.faiss` 路径。
2. 校验并规范化 `gallery_items`，图像路径统一转为绝对路径。
3. 如果 `rebuild_index == false`，并且 `.faiss`、`.manifest.yaml` 均存在：
   - 校验 manifest 是否匹配当前模型、特征、后端、归一化、索引类型和 ROIAlign 配置。
   - 加载 manifest 中的 ROI ID 序列，确认与本次输入一致。
   - 匹配成功则直接加载 Faiss 索引。
4. 否则创建模型和 ROI 特征抽取器，重新构建索引。

## 5. ROI 特征提取流程

单个 ROI 特征的生成流程如下：

1. `ImageFeatureExtractor` 读取原图，记录原始宽高。
2. 在 batch 内执行共享 `PreprocessSpec` 预处理，把原图缩放到模型输入尺寸，并转为 NCHW float；CPU 路径并行执行 OpenCV 预处理，GPU 路径使用 CVCUDA，写入位置仍按输入顺序固定。
3. 调用 `IModel::forwardFeatures()` 输出指定 `feature_name` 的特征张量。
4. 校验输出特征张量：
   - 标准 CNN/空间特征必须是 `B x C x H x W`。
   - DINOv2/DINOv3 可使用 `x_norm_patchtokens`，其形状为 `B x tokens x dim`；ROI 模块会根据模型输入宽高比例推断 `patch_h x patch_w`，并重排为 `B x dim x patch_h x patch_w`。
5. 将原图 ROI 映射到特征图坐标：

```text
scale_x = feature_width  / original_image_width
scale_y = feature_height / original_image_height

mapped_x1 = roi.x1 * scale_x
mapped_y1 = roi.y1 * scale_y
mapped_x2 = roi.x2 * scale_x
mapped_y2 = roi.y2 * scale_y
```

6. 如果 `use_pca=true`，对当前图的特征图空间位置通道向量训练本地 OpenCV PCA，并在抽取 ROI 前把特征图从 `C x H x W` 投影为 `pca_dim x H x W`。
7. 构造 ROIAlign 输入 `[batch_index, mapped_x1, mapped_y1, mapped_x2, mapped_y2]`。
8. 调用 `irt::ops::RoIAlign`，输出形状为 `C x pooled_height x pooled_width`；启用 PCA 时输出形状为 `pca_dim x pooled_height x pooled_width`。
9. 将 ROIAlign 输出展平为一维向量。
10. 按 `config.norm` 执行 L1、L2 或 None 归一化。

最终 ROI 向量维度为：

```text
feature_dim = feature_channels * pooled_height * pooled_width
```

启用 PCA 时，Faiss 中保存和搜索的最终维度为：

```text
search_dim = pca_dim * pooled_height * pooled_width
```

## 6. 特征库构建流程

重建索引时，`RoiSearch::Impl::buildWithItems()` 执行以下步骤：

1. `LoadingModel`：创建 `RoiFeatureExtractor`，内部持有 `ImageFeatureExtractor`。
2. 根据输出特征图通道数和 ROIAlign 输出尺寸确定原始 `feature_dim`。
3. 按规范化后的图像路径分组，并按模型最大 batch 打包不同图像；同一图像的所有 ROI 只进入一次模型前向。
4. 对每个图像 batch 执行图像解码和共享 `PreprocessSpec` 预处理；CPU 路径并行执行 OpenCV 预处理，GPU 路径使用 CVCUDA。随后调用一次模型前向，在共享特征图上批量 ROIAlign，按原始 ROI 顺序回写临时特征文件。
5. 如果 `use_pca=true`，对每张图的特征图训练本地 PCA，投影为 `pca_dim x H x W` 后再执行该图的 ROIAlign、展平和归一化。
6. 调用 `buildConfiguredFaissIndex()`，通过临时特征存储的连续区间/索引批量回调训练和添加 Faiss 索引。
7. 写入 `<index>.manifest.yaml`，包含 ROI ID 映射和配置。
8. 保存或加载完成后的 Faiss 索引进入可查询状态。

ROI 批量接口复用相同的特征抽取和 ROIAlign 逻辑；构建时会先按图像路径分组，同一张图的多个 ROI
只执行一次模型前向，再在同一份特征图上批量 ROIAlign，并按原始 ROI 顺序回写特征。

## 7. Faiss 索引模式

ROI 检索复用图像搜索的 Faiss 构建工具，支持两条路径。

### RAM IVF-PQ

默认使用内存驻留的 `IndexIVFPQ`：

1. 抽样 ROI 特征作为训练数据。
2. 按特征维度和样本量选择 IVF/PQ 参数。
3. 训练 IVF-PQ。
4. 释放训练样本缓存；按 `model_batch_size` 从构建阶段的临时特征存储读取并立即添加到索引。
5. 写入 `.faiss`。
6. 如果配置为 GPU Faiss，将 CPU 索引迁移到 GPU。

### CPU 磁盘 IVF+Flat

当 `faiss_backend == CPU` 且 `index_storage == Disk` 时使用磁盘倒排列表：

1. 抽样 ROI 特征生成 IVF 聚类中心。
2. 写入 `.faiss` 索引骨架。
3. 两遍按 `model_batch_size` 扫描 ROI 特征：
   - 第一遍统计每个倒排列表大小。
   - 第二遍将 ID 和向量写入 `.ivfdata`。
4. 重新加载 `.faiss`，并挂接磁盘上的 `.ivfdata`。

该模式适合 ROI 数量很大、RAM 不适合常驻完整索引的场景。

## 8. 旁路文件

给定索引路径 `<index>.faiss`，ROI 检索会生成：

- `<index>.manifest.yaml`：记录 ROI ID 映射、模型、特征、后端、归一化、Faiss 配置和 ROIAlign 配置。
- `<index>.faiss.ivfdata`：仅 CPU 磁盘 IVF 模式使用，保存倒排列表数据。

manifest 中的 ROI ID 顺序与 Faiss 向量 ID 一一对应，因此查询结果可以从 Faiss ID 还原到调用方提供的 ROI ID。

## 9. 查询流程

查询入口：

```cpp
auto results = searcher.search(query_image, query_roi, top_k);
```

查询流程：

1. 校验索引已就绪，且 `top_k > 0`。
2. 校验查询图像路径和查询 ROI。
3. 如果索引是从磁盘加载的，首次查询前懒加载 `RoiFeatureExtractor`。
4. 对查询图像执行模型特征图抽取。
5. 将查询 ROI 从原图坐标映射到特征图坐标。
6. 如果启用 PCA，对当前查询图的特征图训练本地 PCA 并做通道投影。
7. 对当前查询特征图执行 ROIAlign 和展平；启用 PCA 时这里使用的是投影后的特征图。
8. 按 `config.norm` 执行归一化。
9. 调用 Faiss：

```cpp
index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());
```

10. 使用 manifest 中的 ID 映射将 Faiss ID 还原为 `RoiSearchResult`。
11. 返回按相似度从高到低排序的结果列表。

`RoiSearchResult` 包含：

- `score`：相似度分数。
- `roi_id`：命中的图库 ROI ID。

## 10. 进度回调

ROI 搜索复用 `ImageSearchBuildProgress` 和 `ImageSearchBuildStage`：

- `LoadingModel`：加载模型与创建 ROI 特征抽取器。
- `ExtractingFeatures`：按图像 batch 提取特征图、执行批量 ROIAlign，并写入构建期间的临时特征存储。
- `BuildingIndex`：按 batch 从临时特征存储构建 Faiss 特征库；CPU 磁盘 IVF 的两次扫描也统一归入此阶段。
- `LoadingIndex`：已有索引被复用时，加载索引或迁移到 GPU。

进度回调只报告上述主要阶段，不再报告 ROI 校验、Faiss 内部训练/分配、写文件和保存元数据等实现细节。阶段完成通过最后一次进度事件的 `processed_count == total_count` 表示。

`processed_count` 与 `total_count` 表示当前阶段已处理和总计的 ROI 条目数或向量数。

## 11. 注意事项

- ROI 坐标始终使用原图坐标，不使用模型输入尺寸坐标。
- `feature_name` 应选择保留空间信息的特征：标准空间特征图使用 NCHW；DINOv2/DINOv3 可使用 `x_norm_patchtokens`。
- 如果输出是 CLS token、register/storage token 或纯向量，ROI 检索会拒绝构建。
- 已加载索引的 Faiss 维度必须等于当前配置下的 ROIAlign 输出维度，否则查询前会报错并要求重建索引。
- ROIAlign 输出越大，单条 ROI 向量维度越高，索引训练和搜索成本也越高。
- PCA 在每张图自己的特征图通道维上训练并投影；ROIAlign 只作用于降维后的空间特征图，因此修改 `use_pca` 或 `pca_dim` 后需要重建索引。
- 同一张图的多个 ROI 会共享一次图像特征图前向；不同图像按模型最大 batch 分组处理。
- GPU Faiss 只支持 RAM 索引路径，`index_storage=Disk` 对 GPU Faiss 会被归一化为 RAM。
