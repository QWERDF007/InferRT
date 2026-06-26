# ImageSearch 图像搜索完整流程

本文档说明 `src/features` 中图像搜索模块的端到端流程。核心入口是
`irt::features::ImageSearch`，它把 InferRT 特征提取模型、OpenCV 预处理和 Faiss
向量索引串成一个图库检索流程。

## 1. 主要文件

- `include/inferrt/features/ImageSearch.hpp`：公共 API、配置、结果结构和构建进度结构。
- `ImageSearch.cpp`：公共 API 的薄封装、图库扫描、默认索引路径生成。
- `priv/ImageSearchImpl.hpp`：`ImageSearch::Impl` 的 PIMPL 声明。
- `priv/ImageSearchImpl.cpp`：模型加载、图片预处理、特征提取、索引构建/加载和查询实现。
- `priv/ImageSearchFaissIndex.hpp`：Faiss RAM IVF-PQ 和 CPU 磁盘 IVF+Flat 索引构建工具。

## 2. 配置入口

调用方通过 `ImageSearchConfig` 控制检索行为：

- `model_name`：用于提取特征的内置模型名，如 `resnet18`、`dinov2_vits14`。
- `feature_name`：用于检索的中间特征名，如 `layer4`、`x_norm_clstoken`。
- `model_backend` / `model_device`：特征提取后端和设备。
- `preprocess_backend`：当前实现支持 CPU 预处理。
- `norm`：特征归一化方式，默认 L2。
- `faiss_backend`：Faiss 搜索后端，支持 CPU 或 GPU。
- `index_storage`：CPU Faiss 搜索时索引常驻 RAM 或使用磁盘倒排列表。
- `model_batch_size`：特征提取模型前向批量；训练特征采样、图库向量提取以及 Faiss 添加/落盘都会使用该批量推进。

## 3. 构建或加载索引

常用入口是：

```cpp
ImageSearch searcher(config);
searcher.buildOrLoad(weights_file, gallery_dir, index_file, rebuild_index, progress_callback);
```

`buildOrLoad` 的决策如下：

1. 根据可选 `index_file` 解析最终 `.faiss` 路径；未指定时在 `gallery_dir` 下生成时间戳文件名。
2. 如果 `rebuild_index == false`，且 `.faiss`、`.manifest.yaml` 都存在并匹配当前配置，则直接加载索引。
3. 否则进入完整重建流程。

元数据匹配会校验模型名、特征名、图库目录、模型后端、设备、归一化方式、Faiss 后端、索引存储类型和索引类型。这样可以避免使用旧配置生成的索引。

## 4. 重建索引流程

重建索引的主要阶段对应 `ImageSearchBuildStage`：

1. `Started`：流程开始。
2. `CollectingImages`：递归扫描图库目录，收集支持的图片格式并排序。
3. `LoadingModel`：创建 `ImageSearchFeatureExtractor`，构建或加载模型。
4. `TrainingFeatures`：抽样图库图片并提取训练特征。
5. `TrainingIndex`：训练 Faiss IVF/PQ 索引结构。
6. `AssigningVectors`：CPU 磁盘 IVF 模式下统计每个倒排列表需要容纳的向量数量。
7. `AddingVectors`：提取图库特征并添加到 Faiss 索引或写入磁盘倒排列表。
8. `WritingIndex`：写入 `.faiss` 文件。
9. `LoadingIndex`：按配置加载索引，RAM 索引可迁移到 GPU。
10. `SavingMetadata`：写入 manifest。
11. `Finished`：构建完成。

## 5. 特征提取

`ImageSearchFeatureExtractor` 负责把图片转换成检索向量：

1. 构造 `IModelConfig`，开启 `featureOnly`，把 `feature_name` 设置为输出张量。
2. TensorRT 后端会用 `model_batch_size` 设置动态 batch profile。
3. OpenCV 读取图片，使用 ImageNet 预处理转换为 NCHW float 输入。
4. 调用 `forwardFeatures` 得到指定中间层输出。
5. 对每条特征按配置做 L1/L2/None 归一化。

连续图片区间通过 `extractBatch(begin, count)` 批量提取；训练采样如果产生非连续下标，会通过 indexed batch 重排为临时批次，避免退化为逐图推理。

## 6. Faiss 索引模式

模块根据配置选择两种索引构建路径。

### RAM IVF-PQ

默认路径是 RAM 常驻的 `IndexIVFPQ`：

1. 按图库规模和内存预算选择 `nlist`。
2. 批量抽样训练特征。
3. 选择 PQ 子量化器数量和 code 位宽。
4. 训练 IVF-PQ。
5. 释放训练样本缓存；按 `model_batch_size` 分批重新提取图库向量，并立即添加到索引。
6. 写入 `.faiss`，并在需要时迁移到 GPU Faiss。

### CPU 磁盘 IVF+Flat

当 `faiss_backend == CPU` 且 `index_storage == Disk` 时使用磁盘倒排列表：

1. 创建 `IndexIVFFlat` 骨架。
2. 抽样训练特征并生成 IVF 聚类中心。
3. 写入 `.faiss` 骨架。
4. 第一遍按 `model_batch_size` 扫描图库特征，统计每个倒排列表大小。
5. 初始化 `.ivfdata` 侧车文件，写入文件头和列表元数据。
6. 第二遍按 `model_batch_size` 扫描图库特征，把 ID 和 float code 分组写入 `.ivfdata`。
7. 重新加载 `.faiss`，并用 `InferRtOnDiskInvertedLists` 挂接 `.ivfdata`。

## 7. 旁路文件

给定索引路径 `<index>.faiss`，模块还会生成：

- `<index>.manifest.yaml`：记录模型、特征、图库路径映射、后端、归一化、批量和索引类型。
- `<index>.faiss.ivfdata`：仅 CPU 磁盘 IVF 模式使用，保存倒排列表中的 ID 和向量编码。

## 8. 查询流程

调用：

```cpp
auto results = searcher.search(query_image, top_k);
```

查询时流程如下：

1. 校验索引已就绪且 `top_k > 0`。
2. 如果索引是从磁盘直接加载的，首次查询前懒加载 `ImageSearchFeatureExtractor`。
3. 对查询图片执行同样的预处理、特征提取和归一化。
4. 调用 Faiss `search(1, query_feature, top_k, distances, indices)`。
5. 用 manifest 中的路径映射把 Faiss ID 转成图片路径。
6. 返回按相似度从高到低排列的 `ImageSearchResult`。

## 9. 批量和进度

- `model_batch_size` 同时控制模型前向批量和 Faiss 添加/落盘批量。
- 进度回调的 `processed_count/total_count` 表示当前阶段已经处理的工作单元数量，通常是图片或向量数量，不是 batch 数。
- `batch_begin` 和 `batch_count` 表示当前完成批次覆盖的图库下标区间或采样数量。
- 构建索引时不会一次性提取完整图库特征；每批特征提取完成后会立即进入 Faiss 添加或磁盘倒排列表写入流程。
- RAM IVF-PQ 添加阶段不复用训练样本缓存，训练完成后会释放训练特征，再按图库顺序现提取现添加。

## 10. 常见注意事项

- 若未指定 `--rebuild-index` 且已有索引元数据匹配，流程会直接加载索引，不会重新提取图库特征。
- DINO 图像级检索通常使用 `x_norm_clstoken` 作为紧凑全局向量。
- ONNX Runtime/OpenVINO 后端要使用大于 1 的 `model_batch_size`，导出的图必须支持动态 batch。
- GPU Faiss 当前强制使用 RAM 索引路径，`index_storage=Disk` 只对 CPU Faiss 生效。
