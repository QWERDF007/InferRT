# 特征示例

本目录集中放置基于 `inferrt_features` 或模型特征输出的示例：

- `shape_template_matching/`：形状模板训练、保存、加载与匹配。
- `feature_extract/`：导出 InferRT 特征张量，用于与 PyTorch 结果对比。
- `image_search/`：基于 Faiss 的图像特征检索。
- `roi_search/`：读取 LabelMe 标注、批量提取 ROI 特征并建立 Faiss 检索库。
- `dino_pca_visualize/`：将 DINO patch token 通过 PCA 可视化。

构建示例：

```bash
cmake --build build --config Release --target inferrt_sample_shape_template_matching
cmake --build build --config Release --target inferrt_sample_feature_extract
cmake --build build --config Release --target inferrt_sample_image_search
cmake --build build --config Release --target inferrt_sample_roi_search
cmake --build build --config Release --target inferrt_sample_dino_pca_visualize
```

各示例的参数和运行方式见对应目录下的 README。特征模块依赖 Faiss；缺少该依赖时，整个特征示例目录会被 CMake 跳过。
