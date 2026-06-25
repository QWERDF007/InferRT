# RoiSearch ROI 搜索匹配完整流程

本文档说明 `src/features` 中 ROI 搜索匹配模块的端到端流程。核心入口是
`irt::features::RoiSearch`，它复用图像搜索模块中的模型特征抽取、特征归一化和 Faiss
索引能力，并在空间特征图上增加 ROI 坐标映射与 `RoIAlign`，最终完成 ROI 级特征库构建和 Top-K 匹配。

## 1. 主要文件

- `include/inferrt/features/RoiSearch.hpp`：公共 API、ROI 数据结构、配置结构和 `RoiSearch` 类声明。
- `RoiSearch.cpp`：公共 API 的 PIMPL 转发与默认 ROI 索引路径生成。
- `priv/RoiSearchImpl.hpp`：`RoiSearch::Impl` 私有实现声明。
- `priv/RoiSearchImpl.cpp`：ROI 条目校验、ROI 特征抽取、索引构建/加载和查询实现。
- `priv/ImageFeatureExtractor.hpp/.cpp`：图像级与 ROI 级检索共用的模型特征图抽取器。
- `priv/FeatureSearchCommon.hpp/.cpp`：图像级与 ROI 级检索共用的配置校验、归一化和 Faiss 索引工具。
- `priv/ImageSearchFaissIndex.hpp`：Faiss RAM IVF-PQ 与 CPU 磁盘 IVF+Flat 索引构建工具。
- `src/ops/RoIAlign.cpp`：ROIAlign 算子实现，用于把任意 ROI 特征映射到统一空间大小。

## 2. 配置入口

`RoiSearchConfig` 继承 `ImageSearchConfig`，因此会复用图像搜索已有配置：

- `model_name`：用于抽取特征图的模型名称，默认 `resnet18`。
- `feature_name`：用于 ROIAlign 的空间特征图张量名称，默认 `layer4`。
- `model_backend` / `model_device`：特征抽取模型后端和设备。
- `preprocess_backend`：当前实现支持 CPU 预处理。
- `norm`：ROI 展平特征的归一化方式，默认 L2。
- `faiss_backend`：Faiss 搜索后端，支持 CPU 或 GPU。
- `index_storage`：CPU Faiss 可选择 RAM 或 Disk；GPU Faiss 会强制使用 RAM。
- `model_batch_size`：模型前向批量；ROI 特征提取和 Faiss 添加/落盘都会使用该批量推进。

ROI 检索额外增加：

- `pooled_height` / `pooled_width`：`RoIAlign` 输出空间大小，默认 `7 x 7`。
- `sampling_ratio`：`RoIAlign` 每个 bin 的采样率，`-1` 表示自适应。
- `aligned`：是否启用 aligned ROIAlign 坐标规则。
- `use_pca`：是否先对特征图通道维使用 OpenCV PCA 降维，默认关闭；该配置适用于所有可输出空间特征图的网络，不限于 DINO 系列。
- `pca_dim`：PCA 输出通道数；`use_pca=true` 时必须大于 0，且不能超过原始特征图通道数和 PCA 训练样本数量。

## 3. 输入数据模型

ROI 特征库由 `std::vector<RoiSearchItem>` 描述：

```cpp
struct RoiSearchItem
{
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
- ROI 坐标必须为有限数。
- `x2 > x1` 且 `y2 > y1`。
- ROI 坐标以原图为基准，调用方不需要预先缩放到模型输入尺寸或特征图尺寸。

## 4. 构建或加载索引

常用入口：

```cpp
RoiSearch searcher(config);
searcher.buildOrLoad(weights_file, gallery_items, index_file, rebuild_index, progress_callback);
```

`index_file` 必须显式指定。`buildOrLoad` 的决策流程：

1. 校验 `index_file` 非空。
2. 校验并规范化 `gallery_items`，图像路径统一转为绝对路径。
3. 如果 `rebuild_index == false`，并且 `.faiss`、`.rois.txt`、`.meta.txt` 均存在：
   - 校验元数据是否匹配当前模型、特征、后端、归一化、索引类型和 ROIAlign 配置。
   - 加载 `.rois.txt`，确认 ROI 条目列表与本次输入一致。
   - 匹配成功则直接加载 Faiss 索引。
4. 否则创建模型和 ROI 特征抽取器，重新构建索引。

## 5. ROI 特征提取流程

单个 ROI 特征的生成流程如下：

1. `ImageFeatureExtractor` 读取原图，记录原始宽高。
2. 使用 ImageNet 预处理把原图缩放到模型输入尺寸，并转为 NCHW float。
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

6. 如果 `use_pca=true`，构建阶段先用特征图每个空间位置的通道向量训练 OpenCV PCA，并在抽取 ROI 前把特征图从 `C x H x W` 投影为 `pca_dim x H x W`。
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
3. 如果 `use_pca=true`：
   - 先提取图库图像的空间特征图。
   - 将特征图整理为 `N*H*W x C` 的通道向量矩阵，用 `cv::PCA` 训练 PCA 参数。
   - 后续 Faiss 构建回调先把整张特征图投影为 `pca_dim x H x W`，再执行 ROIAlign、展平和归一化。
   - 最终写入 Faiss 的单条 ROI 向量维度为 `pca_dim * pooled_height * pooled_width`。
4. 如果 `use_pca=false`，直接通过 ROI 特征抽取回调现算特征。
5. 调用 `buildConfiguredFaissIndex()` 构建 Faiss 索引。
6. Faiss 构建过程中通过回调按 ROI 条目下标提取特征：
   - 单条回调：`extract(gallery_items[index])`
   - 连续批量回调：`extractBatch(gallery_items, begin, count)`
   - 任意下标批量回调：`extractBatch(gallery_items, indices)`
7. 写入 ROI 映射文件 `<index>.rois.txt`。
8. 如果启用 PCA，写入 PCA 参数文件 `<index>.pca.yml`。
9. 写入元数据文件 `<index>.meta.txt`。
10. 保存或加载完成后的 Faiss 索引进入可查询状态。

当前 ROI 批量接口会复用相同的特征抽取和 ROIAlign 逻辑。后续如果同一张图有多个 ROI，可以在 `RoiFeatureExtractor`
中进一步合并同图 ROI，减少重复模型前向。

## 7. Faiss 索引模式

ROI 检索复用图像搜索的 Faiss 构建工具，支持两条路径。

### RAM IVF-PQ

默认使用内存驻留的 `IndexIVFPQ`：

1. 抽样 ROI 特征作为训练数据。
2. 按特征维度和样本量选择 IVF/PQ 参数。
3. 训练 IVF-PQ。
4. 释放训练样本缓存；按 `model_batch_size` 分批重新提取 ROI 特征，并立即添加到索引。
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

- `<index>.faiss.rois.txt`：每行一个 ROI 条目，包含图像路径和原图 ROI 坐标。
- `<index>.faiss.meta.txt`：记录模型、特征、后端、归一化、Faiss 配置和 ROIAlign 配置。
- `<index>.faiss.pca.yml`：仅 `use_pca=true` 时生成，保存 OpenCV PCA 的均值、特征向量、特征值和输入/输出维度。
- `<index>.faiss.ivfdata`：仅 CPU 磁盘 IVF 模式使用，保存倒排列表数据。

`.rois.txt` 的行号与 Faiss 向量 ID 一一对应，因此查询结果可以从 Faiss ID 还原到图像路径和 ROI 框。

## 9. 查询流程

查询入口：

```cpp
auto results = searcher.search(query_image, query_roi, top_k);
```

查询流程：

1. 校验索引已就绪，且 `top_k > 0`。
2. 校验查询图像路径和查询 ROI。
3. 如果索引是从磁盘加载的，首次查询前懒加载 `RoiFeatureExtractor`；启用 PCA 时同步加载 `<index>.pca.yml`。
4. 对查询图像执行模型特征图抽取。
5. 将查询 ROI 从原图坐标映射到特征图坐标。
6. 如果启用 PCA，使用构建阶段保存的 PCA 参数先对整张查询特征图做通道投影。
7. 对当前查询特征图执行 ROIAlign 和展平；启用 PCA 时这里使用的是投影后的特征图。
8. 按 `config.norm` 执行归一化。
9. 调用 Faiss：

```cpp
index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());
```

10. 使用 `.rois.txt` 中的映射将 Faiss ID 还原为 `RoiSearchResult`。
11. 返回按相似度从高到低排序的结果列表。

`RoiSearchResult` 包含：

- `score`：相似度分数。
- `image_path`：命中 ROI 所属图像。
- `roi`：命中 ROI 的原图坐标。
- `item_index`：命中条目在特征库中的下标。

## 10. 进度回调

ROI 搜索复用 `ImageSearchBuildProgress` 和 `ImageSearchBuildStage`：

- `Started`：流程开始。
- `CollectingImages`：校验并规范化 ROI 条目。
- `LoadingModel`：加载模型与创建 ROI 特征抽取器。
- `TrainingFeatures`：提取 PCA 训练用的特征图通道向量，或抽样提取 Faiss 训练特征。
- `TrainingIndex`：训练 Faiss 索引结构。
- `AssigningVectors`：CPU 磁盘 IVF 模式统计倒排列表。
- `AddingVectors`：提取 ROI 特征并添加到索引或写入磁盘倒排列表。
- `WritingIndex`：写入 `.faiss`。
- `LoadingIndex`：加载索引或迁移到 GPU。
- `SavingMetadata`：写入 ROI 映射和元数据。
- `Finished`：构建或加载完成。

`processed_count` 与 `total_count` 表示当前阶段已处理和总计的 ROI 条目数或向量数。

## 11. 注意事项

- ROI 坐标始终使用原图坐标，不使用模型输入尺寸坐标。
- `feature_name` 应选择保留空间信息的特征：标准空间特征图使用 NCHW；DINOv2/DINOv3 可使用 `x_norm_patchtokens`。
- 如果输出是 CLS token、register/storage token 或纯向量，ROI 检索会拒绝构建。
- ROIAlign 输出越大，单条 ROI 向量维度越高，索引训练和搜索成本也越高。
- PCA 先在特征图通道维上训练，再对整张特征图投影；ROIAlign 只作用于降维后的空间特征图，因此修改 `use_pca` 或 `pca_dim` 后需要重建索引。
- 同一张图多个 ROI 当前会逐条抽取特征图，后续可按图像分组优化。
- GPU Faiss 只支持 RAM 索引路径，`index_storage=Disk` 对 GPU Faiss 会被归一化为 RAM。
