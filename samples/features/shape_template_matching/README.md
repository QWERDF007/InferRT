# 形状模板匹配示例

`shape_template_matching` 使用 `irt::features::v0::ShapeTemplateMatcher` 或
`irt::features::v1::ShapeTemplateMatcher` 在源图中查找相同边缘形状的目标。它不依赖深度学习模型，适合轮廓稳定、边缘清晰的零件、工件或标记定位。

示例按两个独立阶段运行：

1. `train`：从模板图（或大图中的矩形区域）训练旋转/缩放模板，并保存 YAML/XML 模板文件。
2. `match`：加载模板文件，在源图中搜索并保存带标注框的结果图。

## 构建

```powershell
cmake --build build --config Release --target inferrt_sample_shape_template_matching
```

可执行文件：

```text
build\bin\inferrt_sample_shape_template_matching.exe
```

查看按阶段分组的完整参数：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe --help
```

## 实现版本

两个版本实现相同的 `IShapeTemplateMatcher` 接口，模板 YAML/XML 格式和匹配结果完全兼容：

- `--version v0`：固定的 `shape_based_matching` 风格标量基线路径，不使用显式 SIMD intrinsic；适合结果对照和基准比较。
- `--version v1`：AVX2 加速实现，默认值；除训练候选点和方向量化外，还会批量评分相邻滑窗、保留阈值早停，并直接从量化标签查表。运行 v1 需要 AVX2 CPU；不支持时请选择 v0。

训练和匹配可以选择不同版本。例如可用 `v0` 训练、`v1` 匹配，或用 `v0` 对照验证结果：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe --mode train --version v0 --template "D:\data\part.png" --save-templates "D:\data\part_templates.yaml"
.\build\bin\inferrt_sample_shape_template_matching.exe --mode match --version v1 --load-templates "D:\data\part_templates.yaml" --source "D:\data\scene.png"
```

## v1 真实图优化验证

以下结果使用真实输入 `F:\data\shape_match\51661.png` 和已有的 2,000 个模板
`F:\data\shape_match\part_templates.yaml` 测得。模板配置为 96 features、`scan_step=1`、方向容差 1，匹配阈值为 85。每一行都使用：

```powershell
--warmup 0 --repeat 1
```

因此表格用于一次端到端功能验证，而非统计学性能结论；计时范围仍仅为 `match()`，不含文件读取、模板加载、绘制和结果写入。所有保留的 v1 步骤均返回同一结果：`class=part`、`template=1000`、`score=100`、`box=(1289,247,437,442)`。

| 阶段 | 保留的 v1 优化 | 单次 match | 相对上一步 | 相对初始 v1 |
| --- | --- | ---: | ---: | ---: |
| 基线 | 原逐窗口标量评分 | 95,225.506 ms | 1.00× | 1.00× |
| 1 | AVX2 16-lane 批量评分 + 每 4 特征早停 | 9,016.183 ms | 10.56× | 10.56× |
| 2 | 按源图平均响应由稀疏到稠密重排特征 | 6,136.396 ms | 1.47× | 15.52× |
| 3 | 早停比较保留在 AVX2 寄存器中 | 4,735.851 ms | 1.30× | 20.11× |
| 4 | 由量化标签 AVX2 shuffle 查表，改善缓存局部性 | 4,285.189 ms | 1.11× | 22.22× |
| 5 | v1 不再物化 8 张响应图，改用标签直方图计算排序均值 | 4,185.867 ms | 1.02× | 22.75× |
| 6 | 分子上界不超过 255 时使用 8-bit 累加和 32-lane AVX2 | 2,643.169 ms | 1.58× | 36.03× |

上表的“相对初始 v1”是所有已保留优化叠加后的累计收益；下表只展示每个新增优化相对其前一个版本的单项收益，因此不能将百分比直接相加：

| 新增优化 | 相对上一步 | 单步耗时降低 |
| --- | ---: | ---: |
| AVX2 16-lane 批量评分 + 每 4 特征早停 | 10.56× | 90.53% |
| 按源图平均响应由稀疏到稠密重排特征 | 1.47× | 31.94% |
| 早停比较保留在 AVX2 寄存器中 | 1.30× | 22.82% |
| 由量化标签 AVX2 shuffle 查表，改善缓存局部性 | 1.11× | 9.52% |
| v1 不再物化 8 张响应图，改用标签直方图计算排序均值 | 1.02× | 2.32% |
| 分子上界不超过 255 时使用 8-bit 累加和 32-lane AVX2 | 1.58× | 36.86% |

两个看似合理但在该真实场景中更慢的方案没有保留：完整得分图累加超过 180,000 ms 后超时；在少量存活 lane 时改回标量收尾为 13,845.349 ms（比阶段 2 慢 2.26×）。

最终同条件全图真实图对照为：v0 `100,158.746 ms`，v1 `2,675.392 ms`，v1 为 **37.44×** 更快（耗时降低 **97.33%**），且匹配字段一致。v2/AVX512 本轮未加入；当前评分内核已抽象为独立策略，后续可在不改动 v0、模板格式、训练流程或 NMS 的前提下单独实现。

## 本轮：精确优化与可选近似加速

本节也使用 `F:\data\shape_match\51661.png`、`part_templates.yaml` 和 2,000 个模板；所有数据都严格使用一次运行：

```powershell
--warmup 0 --repeat 1
```

因此数据用于功能与量级验证，不应替代多次重复的稳定性能基准。

### 以默认 v1 为 baseline 的整体耗时

这里的“默认 v1 baseline”指当前正常的全图精确调用：不传 `--search-mask`、`--template-stride 1`、`--scan-step 0`（模板文件实际为 `scan_step=1`）。其 `match()` 耗时为 **2,675.392 ms**。下表中的“相对 baseline”全部直接用这个数相除，不是相对上一行：

| 场景 / 开关 | 搜索范围 | `template-stride` | 有效 `scan-step` | 整体单次 match | 相对默认 v1 | 结果影响 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 默认 v1 baseline | 全图 | 1 | 1 | 2,675.392 ms | 1.00× | 精确：`1000` / `100` / `(1289,247,437,442)` |
| 仅模板抽样 | 全图 | 7 | 1 | 563.922 ms | 4.74× | ID 变为 `1001`，分数 91.6667，IoU 0.9776 |
| 仅空间抽样 | 全图 | 1 | 2 | 1,640.056 ms | 1.63× | 同 ID、同分数，框偏移 1 px，IoU 0.9909 |
| 两项近似同时开 | 全图 | 7 | 2 | **404.547 ms** | **6.61×** | ID `1001`，分数 91.6667，IoU 0.9776 |

表中只统计全图匹配；不把外部先验 ROI、裁剪或搜索范围限制计为算法优化。若上游能够确定目标所在区域，应由上游裁剪图像后再调用匹配器，并单独评估整个流水线的耗时与召回。

### 可能影响精度：运行时匹配选项

`ShapeTemplateMatchOptions` 不写入 YAML/XML，只影响本次 `match()` / `matchFile()` 调用。示例把它们暴露为以下默认关闭的参数：

- `--template-stride N`：只扫描模板 ID 可被 `N` 整除的变体。默认 `1`，扫描全部模板；`N > 1` 可能跳过最佳角度/尺度变体，从而影响召回、分数和框大小。
- `--scan-step N`：`0` 表示使用模板文件中的 `scan_step`；正数覆盖空间扫描步长。当前模板文件为 `scan_step=1`；增大到 `2` 会减少候选位置，可能产生像素级定位偏差或漏检。

上表中的 IoU 为当前 top-1 框相对精确 top-1 框的 IoU；所有配置均返回 1 个匹配。相对精确基线，三种近似配置的耗时分别降低 **78.92%**、**38.70%** 和 **84.88%**。真实图上 `template-stride=7` 仍检测到目标，但最佳变体从 ID 1000 变为 1001，分数降至 91.6667；这正是该选项的预期精度代价。`scan-step=2` 保持相同模板和分数，但定位偏移 1 像素。不同图像、阈值和角度/尺度采样密度下，近似配置也可能直接漏检，因此生产使用前应按目标数据集验证。

对应 API 用法：

```cpp
irt::features::ShapeTemplateMatchOptions options;
options.template_stride = 7; // 默认 1：精确
options.scan_step = 2;       // 默认 0：沿用模板文件配置
const auto matches = matcher.match(source, 85.0f, {"part"}, cv::Mat(), options);
```

## 第一阶段：训练模板

训练只需要模板输入图和可选的模板掩码，**不需要传入待匹配的大图**。训练完成后会写入 `--save-templates` 指定的 YAML/XML 文件。

### 从已裁剪的小图训练

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v1 `
  --template "D:\data\part.png" `
  --template-mask "D:\data\part_mask.png" `
  --class-id part `
  --angle-begin 0 `
  --angle-end 180 `
  --angle-step 15 `
  --scale-begin 0.9 `
  --scale-end 1.1 `
  --scale-step 0.1 `
  --features 96 `
  --warmup 3 `
  --repeat 10 `
  --save-templates "D:\data\part_templates.yaml"
```

### 从大图和矩形区域训练

`--template` 也可以是包含目标的大图。传入 `--template-roi x,y,width,height` 后，示例会先裁剪该矩形区域，再使用裁剪结果训练：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v1 `
  --template "D:\data\capture.png" `
  --template-roi "420,180,160,120" `
  --template-mask "D:\data\capture_mask.png" `
  --class-id part `
  --angle-begin 0 `
  --angle-end 180 `
  --angle-step 15 `
  --save-templates "D:\data\part_templates.yaml"
```

矩形格式为 `x,y,width,height`，坐标以 `--template` 输入图的左上角为原点，且矩形必须完全位于图像内。为支持旋转/缩放变体，建议矩形在目标周围保留适量背景空白，避免变换后裁掉目标边缘。

### 训练掩码 `--template-mask`

`--template-mask` 是训练阶段的**目标掩码**，不是匹配大图的 ROI：

- 非零像素参与模板特征训练；零像素会被忽略。
- 未使用 `--template-roi` 时，掩码必须与 `--template` 图像同尺寸。
- 使用 `--template-roi` 时，掩码可以与原始大图同尺寸（示例会按同一矩形自动裁剪），也可以直接与裁剪后小图同尺寸。

模板图背景干净时可以省略掩码；背景边缘复杂时，建议提供只覆盖目标本体的精确掩码。

## 第二阶段：匹配模板

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode match `
  --version v1 `
  --load-templates "D:\data\part_templates.yaml" `
  --source "D:\data\scene.png" `
  --class-filter part `
  --threshold 85 `
  --warmup 3 `
  --repeat 10 `
  --output "D:\data\result.png"
```

`match` 阶段不会重新训练。它只加载 `--load-templates` 中保存的模板与配置，然后在 `--source` 大图中搜索。

如需限制搜索区域，可传入与源图同尺寸的 `--search-mask`：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode match `
  --version v1 `
  --load-templates "D:\data\part_templates.yaml" `
  --source "D:\data\scene.png" `
  --search-mask "D:\data\scene_search_mask.png" `
  --output "D:\data\result.png"
```

`--search-mask` 非零区域允许搜索，零值区域会跳过；它与训练阶段的 `--template-mask` 用途不同。

需要更短的延迟、且允许量化的召回或定位损失时，可显式开启近似匹配选项，例如：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode match `
  --version v1 `
  --load-templates "D:\data\part_templates.yaml" `
  --source "D:\data\scene.png" `
  --template-stride 7 `
  --scan-step 2 `
  --output "D:\data\result.png"
```

省略上述两个参数等价于 `--template-stride 1 --scan-step 0`，保持完整模板集合和模板文件保存的空间扫描步长。

## 时间统计

`--warmup` 和 `--repeat` 对当前 `--mode` 指定的阶段生效，默认分别为 `3` 和 `10`：

- 预热执行不计入统计，用于降低首次调用、内存页和运行时初始化带来的偏差。
- 重复执行的每一轮都会采样并输出 `total`、`avg`、`median`、`min`、`max` 和 `stddev`，单位为毫秒。
- 训练计时仅覆盖 `addTemplateVariants()`；每次预热和重复都会使用新的匹配器，避免向同一模板库重复添加模板。
- 匹配计时仅覆盖 `match()`；模板文件、输入图像和掩码会在计时前加载。
- YAML/XML 写入、图像读取、ROI 裁剪、结果绘制和输出图片均不计入核心耗时。

需要只执行一次功能验证时，可使用：

```powershell
--warmup 0 --repeat 1
```

## 参数分组

### 计时参数（两个阶段通用）

- `--warmup`：不计入统计的预热次数，必须大于等于 `0`，默认 `3`。
- `--repeat`：计入统计的重复次数，必须大于 `0`，默认 `10`。

### 实现版本（两个阶段通用）

- `--version`：选择 `v0` 或 `v1`，默认 `v1`。两者可以读写同一模板文件；需要验证 SIMD 路径时，可在同一输入上分别运行两个版本并比较结果。

### 训练参数（`--mode train`）

- `--template`：训练输入图，可为目标小图或原始大图。
- `--template-roi`：可选裁剪矩形，格式 `x,y,width,height`。
- `--template-mask`：可选目标掩码。
- `--save-templates`：必填，训练输出的 YAML/XML 路径。
- `--class-id`：写入模板的类别 ID，默认 `part`。
- `--angle-begin`、`--angle-end`、`--angle-step`：离散训练旋转角度；端点包含在内。
- `--scale-begin`、`--scale-end`、`--scale-step`：离散训练缩放尺度。
- `--features`：每个模板最多保留的边缘特征点数，常用 `64` 到 `128`；更多特征通常更精细但更慢。
- `--max-results`、`--nms`：保存到模板文件的匹配默认配置。`--nms` 为同类别 NMS 的 IoU 阈值，负数表示关闭；`--max-results` 为最多输出的结果数。

### 匹配参数（`--mode match`）

- `--load-templates`：必填，训练阶段生成的 YAML/XML 文件。
- `--source`：必填，待搜索源图。
- `--search-mask`：可选源图搜索区域掩码，必须与源图同尺寸。
- `--class-filter`：可选类别过滤；省略时搜索模板文件中的全部类别。
- `--threshold`：本次匹配的分数阈值，范围 `[0,100]`；传负数或省略时使用模板文件中保存的默认值。
- `--template-stride`：本次匹配每隔多少个模板变体扫描一次，默认 `1`（精确）；大于 `1` 为可能影响召回和角度/尺度精度的近似模式。
- `--scan-step`：本次匹配覆盖空间扫描步长；默认 `0`，表示使用模板文件配置。大于保存值时通常更快，但可能产生定位偏差或漏检。
- `--output`：带匹配框的输出图，默认 `shape_template_matching_result.png`。

角度/尺度步长越小，模板数量和匹配耗时越高，但对相应变化的召回率通常更好。误检较多时提高 `--threshold`；漏检较多时适当降低它。



### cmd


#### train

F:\Projects\InferRT\build\bin\inferrt_sample_shape_template_matching.exe --mode train --template "F:\data\shape_match\51661.png" --template-roi "1285,245,445,448" --class-id part --angle-begin -20 --angle-end 20 --angle-step 0.1  --scale-begin 0.8 --scale-end 1.2 --scale-step 0.1 --save-templates "F:\data\shape_match\part_templates.yaml"

#### match

F:\Projects\InferRT\build\bin\inferrt_sample_shape_template_matching.exe --mode match --load-templates "F:\data\shape_match\part_templates.yaml" --source "F:\data\shape_match\51661.png" --class-filter part --threshold 85 --output "F:\data\shape_match\result.png" --version v1
