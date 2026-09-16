# DINO 区域检索 v4 CLI

本目录使用 InferRT 示例目标，支持 `build`、`search` 和 `search-batch`。配置定义与公共接口见 [DinoRegionSearch.hpp](../../../src/features/include/inferrt/features/DinoRegionSearch.hpp)，完整系统流程见 [DinoRegionSearchFlow.md](../../../src/features/DinoRegionSearchFlow.md)。

```bash
cmake --build build --config Release --target inferrt_sample_dino_region_search

build/bin/inferrt_sample_dino_region_search build --gallery gallery --index build/dino_index \
  --profile samples/features/dino_region_search/profiles/dinov3_vits16.yaml

build/bin/inferrt_sample_dino_region_search search --index build/dino_index \
  --profile samples/features/dino_region_search/profiles/dinov3_vits16.yaml \
  --request examples/query.yaml --output result.yaml

build/bin/inferrt_sample_dino_region_search search-batch --index build/dino_index \
  --profile samples/features/dino_region_search/profiles/dinov3_vits16.yaml \
  --requests examples/requests.yaml --output responses.yaml
```

Windows 按实际 bin 位置使用 `.exe`。预设仅接受 `dinoConfigToYaml` 输出的新层次字段：`model`、`gallery_views`、`descriptors`、`query_features`、`coarse_scan`、`fine_match`、`decision`、`runtime`、`diagnostics`，以及展示用的 `preset_id`，不兼容旧 YAML。

将 `model.weights_file` 改为自己的权重/engine 路径。示例的 `model.weights_id` 使用对应 `.wts` 名称；替换为不同权重时必须设置不同的明确版本身份，仅迁移同一权重的路径则保持身份不变。修改模型身份、有效输入尺寸、图库视图或描述子配置后重新建库；外部改写索引后重启查询进程。

批量和精度分别配置在 `runtime.model_batch_size`、`runtime.model_precision`，默认截止时间在 `runtime.query_deadline_ms`。返回数量在 `coarse_scan.final_k`；精匹配权重在 `fine_match.score_weight_template`、`score_weight_coverage`、`score_weight_consistency`。阈值通过 `decision.enable_decision_threshold` 和 `decision.decision_threshold` 配置，不使用 `null`。

`tools/dino_region_devkit.py calibrate` 会生成同层次的权重变体并关闭阈值，以收集完整评分；`probe --threshold` 在临时预设中显式开启并覆盖阈值。工具输入同样必须是新预设，不会转换旧字段。

`include_self=false` 会排除查询源图整张图片；需要同图检索时设为 true。所有 bbox 是 EXIF 后原图浮点半开坐标。无判定阈值时只给排序，超时/失败不当成没有匹配。

批量输入是请求 YAML 数组；输出是依次以 `---` 分隔的响应文档。同进程顺序处理请求，复用模型、读取器和有限缓存。新增结果字段包括 localized_candidates、verification_candidates、verified_candidates 与 score_kind。
