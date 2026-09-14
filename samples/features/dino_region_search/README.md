# DINO 区域检索示例

`inferrt_sample_dino_region_search` 是冻结 DINO 骨干的区域检索 CLI：用一张图的标注区域（bbox 或 polygon）在未标注图库中检索外观与局部结构相似的区域，返回原图坐标、排序分数与来源通道。

实现位于 [`src/features/priv/dino`](../../../src/features/priv/dino)，公共入口是 [`DinoRegionSearch.hpp`](../../../src/features/include/inferrt/features/DinoRegionSearch.hpp)；方案、spec、tickets 与验收门槛见 [`docs/dino_region_search_v1`](../../../docs/dino_region_search_v1)。

## 构建

```bash
cmake --build build --config Release --target inferrt_sample_dino_region_search
```

## Profile

profile 使用 YAML 格式，本目录自带两个开发 profile：

```bash
samples/features/dino_region_search/profiles/dinov3_vits16.yaml
samples/features/dino_region_search/profiles/dinov2_vits14_reg4.yaml
```

关键点：

- `model.name` 决定骨干别名（`dinov3_vits16` 或 `dinov2_vits14_reg4`），`model.weights_path` 指向 `.wts` 或已构建的 `.engine`。
- `model.encoder_edge` 必须是 patch 边长的整数倍；DINOv3 为 16，DINOv2 为 14。
- `quantization.format` 为 `int8` 时使用 INT8 紧凑存储，`fp32` 为对照路径。
- `search.coarse_k` 决定粗选候选数，`fine.final_k` 不得超过 `coarse_k`。

## 用法

```bash
CLI=build/bin/inferrt_sample_dino_region_search.exe
PROFILE=samples/features/dino_region_search/profiles/dinov3_vits16.yaml

# 1) 建库
$CLI build --gallery assets/pics --index build/dino_region/index_dinov3 --profile $PROFILE

# 2) 查询：矩形 ROI，输出契约 YAML
$CLI search --index build/dino_region/index_dinov3 --profile $PROFILE \
  --query assets/pics/dog.jpg --roi 140,110,420,400 --top-k 5 --output result.yaml

# 3) 查询：多边形 ROI
$CLI search --index build/dino_region/index_dinov3 --profile $PROFILE \
  --query assets/pics/dog.jpg --polygon "140,110;420,120;420,400;150,395"

# 4) 自查询验证（允许返回与查询路径相同的图像）
$CLI search --index build/dino_region/index_dinov3 --profile $PROFILE \
  --query assets/pics/dog.jpg --roi 140,110,420,400 --include-self

# 5) 使用请求 YAML 文件查询
$CLI search --index build/dino_region/index_dinov3 --profile $PROFILE \
  --request query.yaml --output result.yaml
```

## 输入输出契约

- 请求 YAML 字段：`query_path`、可选 `request_id`、`profile_id`、`deadline_ms`，以及互斥的 `bbox`/`polygon`；可选 `include_self`、`top_k`。
- 响应 YAML 字段：`status`、`decision`、`results[]`（包含 `source_path`、`bbox`、`score`）。
- 所有坐标都在 canonical image 上，使用浮点半开区间 `[x0, y0, x1, y1)`。
- `status` 与 `decision` 语义：`completed + ranked_only` 表示排好序但未配置判定阈值；`incomplete` 表示超时或部分候选未处理完，此时结果是部分结果。
- stdout 只输出机器可读 YAML；进度写 stderr。

退出码：`0` 完成、`2` 参数/ROI/profile 非法、`3` 权重或索引缺失或不兼容、`4` 建库含失败文件、`5` 查询未完成、`6` 内部错误。

## 路径与编码契约

- 契约（profile / request / response）里的路径一律是 **UTF-8 文本**；实现内部一律是 `fs::path`。二者通过 [`DinoPaths.hpp`](../../../src/features/priv/dino/DinoPaths.hpp) 互转，统一使用 `/` 分隔符。
- 图片解码按 `fs::path` 读取原始字节后再 `cv::imdecode`，避免 Windows 上窄字符路径解码失败。
- 原图尺寸保留由 [`DinoIngest.cpp`](../../../src/features/priv/dino/DinoIngest.cpp) 的 `DinoImageLoader::load` 负责；profile 适用范围诊断见 [`DinoQuery.cpp`](../../../src/features/priv/dino/DinoQuery.cpp) 与 [`DinoGeometry.cpp`](../../../src/features/priv/dino/DinoGeometry.cpp)，不另设解码尺寸门禁。
