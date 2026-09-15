# DINO 区域检索 v4 CLI

本目录使用已有 InferRT 示例目标，新增 `search-batch`。完整说明见 [开发与集成](../../../docs/04_development.md)、[Spec](../../../docs/02_spec.md)。

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

Windows 按实际 bin 位置使用 `.exe`。profile 中的模型路径须改为自己的权重/engine。修改建库描述配置后重新建库，外部改写索引后重启查询进程。

`include_self=false` 会排除查询源图整张图片；需要同图检索时设为 true。所有 bbox 是 EXIF 后原图浮点半开坐标。无判定阈值时只给排序，超时/失败不当成没有匹配。

批量输入是请求 YAML 数组；输出是依次以 `---` 分隔的响应文档。同进程顺序处理请求，复用模型、读取器和有限缓存。新增结果字段包括 localized_candidates、verification_candidates、verified_candidates 与 score_kind。

本包缺完整 InferRT 构建上下文，因此原工程集成尚需实际编译验证；根目录 CMake 只用于算法测试。
