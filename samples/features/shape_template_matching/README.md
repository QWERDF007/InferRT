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

- `--version v0`：固定的 `shape_based_matching` 风格标量基线路径。训练时保留原始串行流程：逐变体变换、逐变体提取特征并立即写入模板库；不使用线程并行、工作缓冲复用或跳过掩膜规范化等 v1 优化。适合结果对照和基准比较。
- `--version v1`：AVX2 加速实现，默认值。匹配时会批量评分相邻滑窗、保留阈值早停，并直接从量化标签查表；训练时会并行处理相互独立的旋转/缩放变体、复用每个工作线程的中间缓冲，并按输入顺序统一提交模板。运行 v1 需要 AVX2 CPU；不支持时请选择 v0。

训练和匹配可以选择不同版本。例如可用 `v0` 训练、`v1` 匹配，或用 `v0` 对照验证结果：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe --mode train --version v0 --template "D:\data\part.png" --save-templates "D:\data\part_templates.yaml"
.\build\bin\inferrt_sample_shape_template_matching.exe --mode match --version v1 --load-templates "D:\data\part_templates.yaml" --source "D:\data\scene.png"
```

两种训练路径产生的模板格式和内容严格兼容。v1 的并行阶段不写模板库，最终按变体输入顺序提交，因此模板 ID、YAML 顺序和精确匹配结果均与 v0 保持一致。

## 模板文件格式：破坏性 v2 升级

新训练的 YAML/XML 模板文件使用 **v2** 格式。为去除每个特征重复的 `x`、`y`、`label`、`angle_degrees` 键，`features` 改为紧凑二维数组；每一行固定按如下顺序存储：

```yaml
features:
  - [x, y, label, angle_degrees]
  - [x, y, label, angle_degrees]
```

例如：

```yaml
features:
  - [12, 8, 3, 135.0]
  - [16, 8, 2, 90.0]
```

这是破坏性修改：旧版 `version: 1`、每个特征为 `{x: ..., y: ..., label: ..., angle_degrees: ...}` 的模板文件将被拒绝加载，必须重新执行 `train` 生成 v2 模板。当前 v0 和 v1 都读写同一 v2 格式，四列数量不是恰好 4 的文件也会被拒绝。

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
  --train-parallelism 0 `
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

- `--version`：选择 `v0` 或 `v1`，默认 `v1`。两者可以读写同一 v2 模板文件；需要验证 SIMD 路径时，可在同一输入上分别运行两个版本并比较结果。

### 训练参数（`--mode train`）

- `--template`：训练输入图，可为目标小图或原始大图。
- `--template-roi`：可选裁剪矩形，格式 `x,y,width,height`。
- `--template-mask`：可选目标掩码。
- `--save-templates`：必填，训练输出的 YAML/XML 路径。
- `--class-id`：写入模板的类别 ID，默认 `part`。
- `--angle-begin`、`--angle-end`、`--angle-step`：离散训练旋转角度；端点包含在内。
- `--scale-begin`、`--scale-end`、`--scale-step`：离散训练缩放尺度。
- `--features`：每个模板最多保留的边缘特征点数，常用 `64` 到 `128`；更多特征通常更精细但更慢。
- `--train-parallelism`：仅影响 v1 的训练工作线程数；`0`（默认）自动选择，`1` 强制 v1 串行。v0 忽略该项，始终保持原始串行训练。此运行时参数不写入模板文件，也不会改变后续匹配的线程设置。
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





## v1 真实图优化验证

以下结果使用真实输入 `F:\data\shape_match\51661.png` 和已有的 2,000 个模板
`F:\data\shape_match\part_templates.yaml` 测得。模板配置为 96 features、`scan_step=1`、方向容差 1，匹配阈值为 85。每一行都使用：

```powershell
--warmup 0 --repeat 1
```

因此表格用于一次端到端功能验证，而非统计学性能结论；计时范围仍仅为 `match()`，不含文件读取、模板加载、绘制和结果写入。阶段 0–7 是严格等价的 v1 实现优化，均返回同一结果：`class=part`、`template=1000`、`score=100`、`box=(1289,247,437,442)`；阶段 8a、8b、9 是默认关闭、可能影响精度的显式近似配置。

| 阶段 | v1 优化 / 显式配置                                      |     单次 match |        相对上一步 | 相对初始 v1 |
| ---- | ------------------------------------------------------- | -------------: | ----------------: | ----------: |
| 基线 | 原逐窗口标量评分                                        |  95,225.506 ms |             1.00× |       1.00× |
| 1    | AVX2 16-lane 批量评分 + 每 4 特征早停                   |   9,016.183 ms |            10.56× |      10.56× |
| 2    | 按源图平均响应由稀疏到稠密重排特征                      |   6,136.396 ms |             1.47× |      15.52× |
| 3    | 早停比较保留在 AVX2 寄存器中                            |   4,735.851 ms |             1.30× |      20.11× |
| 4    | 由量化标签 AVX2 shuffle 查表，改善缓存局部性            |   4,285.189 ms |             1.11× |      22.22× |
| 5    | v1 不再物化 8 张响应图，改用标签直方图计算排序均值      |   4,185.867 ms |             1.02× |      22.75× |
| 6    | 分子上界不超过 255 时使用 8-bit 累加和 32-lane AVX2     |   2,643.169 ms |             1.58× |      36.03× |
| 7    | 全图无掩膜快路径：跳过每个候选的掩膜读取与判断          |   2,451.975 ms |             1.08× |      38.84× |
| 8a   | 可选近似：仅模板抽样（`template-stride=7`，基于阶段 7） |     515.386 ms | 4.76×（对阶段 7） |     184.77× |
| 8b   | 可选近似：仅空间抽样（`scan-step=2`，基于阶段 7）       |   1,414.481 ms | 1.73×（对阶段 7） |      67.32× |
| 9    | 可选近似：两项同时开（8a 再启用 `scan-step=2`）         | **372.530 ms** |    1.38×（对 8a） |     255.62× |

上表的“相对初始 v1”均以原逐窗口标量评分 `95,225.506 ms` 计算。阶段 0–7 是线性叠加的精确优化；8a 与 8b 是从阶段 7 分出的替代近似分支，9 是在 8a 上叠加空间抽样。因此分支行的“相对上一步”明确标注其父阶段，不能将不同分支的收益相加。

两个看似合理但在该真实场景中更慢的方案没有保留：完整得分图累加超过 180,000 ms 后超时；在少量存活 lane 时改回标量收尾为 13,845.349 ms（比阶段 2 慢 2.26×）。

截至上述阶段 7 的同条件全图真实图对照为：v0 `100,158.746 ms`，v1 `2,451.975 ms`，v1 为 **40.85×** 更快（耗时降低 **97.55%**），且匹配字段一致。匹配器 v2/AVX512 本轮未加入；当前评分内核已抽象为独立策略，后续可在不改动 v0、当前模板文件格式、训练流程或 NMS 的前提下单独实现。

### 可选近似配置的整体耗时

下表是首表阶段 7、8a、8b、9 的聚焦视图。正常全图精确调用为阶段 7：不传 `--search-mask`、`--template-stride 1`、`--scan-step 0`（模板文件实际为 `scan_step=1`），单次 `match()` 为 **2,451.975 ms**。表中的“相对精确 v1”都直接相对阶段 7 计算，不是相对上一行：

| 场景 / 开关            | 搜索范围 | `template-stride` | 有效 `scan-step` | 整体单次 match | 相对精确 v1 | 结果影响                                    |
| ---------------------- | -------- | ----------------: | ---------------: | -------------: | ----------: | ------------------------------------------- |
| 阶段 7 精确 v1         | 全图     |                 1 |                1 |   2,451.975 ms |       1.00× | 精确：`1000` / `100` / `(1289,247,437,442)` |
| 阶段 8a，仅模板抽样    | 全图     |                 7 |                1 |     515.386 ms |       4.76× | ID 变为 `1001`，分数 91.6667，IoU 0.9776    |
| 阶段 8b，仅空间抽样    | 全图     |                 1 |                2 |   1,414.481 ms |       1.73× | 同 ID、同分数，框偏移 1 px，IoU 0.9909      |
| 阶段 9，两项近似同时开 | 全图     |                 7 |                2 | **372.530 ms** |   **6.58×** | ID `1001`，分数 91.6667，IoU 0.9776         |

表中只统计全图匹配；不把外部先验 ROI、裁剪或搜索范围限制计为算法优化。若上游能够确定目标所在区域，应由上游裁剪图像后再调用匹配器，并单独评估整个流水线的耗时与召回。

### 可能影响精度：运行时匹配选项

`ShapeTemplateMatchOptions` 不写入 YAML/XML，只影响本次 `match()` / `matchFile()` 调用。示例把它们暴露为以下默认关闭的参数：

- `--template-stride N`：只扫描模板 ID 可被 `N` 整除的变体。默认 `1`，扫描全部模板；`N > 1` 可能跳过最佳角度/尺度变体，从而影响召回、分数和框大小。
- `--scan-step N`：`0` 表示使用模板文件中的 `scan_step`；正数覆盖空间扫描步长。当前模板文件为 `scan_step=1`；增大到 `2` 会减少候选位置，可能产生像素级定位偏差或漏检。

上表中的 IoU 为当前 top-1 框相对精确 top-1 框的 IoU；所有配置均返回 1 个匹配。相对精确基线，三种近似配置的耗时分别降低 **78.98%**、**42.31%** 和 **84.81%**。真实图上 `template-stride=7` 仍检测到目标，但最佳变体从 ID 1000 变为 1001，分数降至 91.6667；这正是该选项的预期精度代价。`scan-step=2` 保持相同模板和分数，但定位偏移 1 像素。不同图像、阈值和角度/尺度采样密度下，近似配置也可能直接漏检，因此生产使用前应按目标数据集验证。

对应 API 用法：

```cpp
irt::features::ShapeTemplateMatchOptions options;
options.template_stride = 7; // 默认 1：精确
options.scan_step = 2;       // 默认 0：沿用模板文件配置
const auto matches = matcher.match(source, 85.0f, {"part"}, cv::Mat(), options);
```

## 全图精确快路径与 5 组 ROI 验证

v1 新增了严格等价的“全图无掩膜”快路径：当调用方未传 `search_mask`，或传入全非零掩膜时，匹配器不再为每一个滑窗候选重复读取和判断掩膜像素。扫描范围、候选顺序、评分、NMS 和返回结果均不变；带有零值的搜索掩膜仍走原有精确路径。v0 保持 `shape_based_matching` 风格的标量参考实现，不使用这项 v1 优化。

在已有 `part_templates.yaml`、2,000 个模板、全图精确匹配和一次运行（`--warmup 0 --repeat 1`）下，这项优化将 v1 从 `2,675.392 ms` 降至 `2,451.975 ms`，即 **1.09×**（耗时降低 **8.35%**），匹配字段完全相同。`ShapeTemplateMatcher*` 的 23 项测试也全部通过，其中包括 v2 紧凑特征格式、旧 v1 模板拒绝加载、特征四列校验、v0/v1 参数组合、全非零搜索掩膜，以及 v0 与 v1 串行/并行训练结果一致性测试。

随后以 `F:\data\shape_match\51661.png` 为训练图和待匹配图，按以下统一训练参数生成了 5 个相互独立的模板库：96 个特征、`angle=-20..20` / `step=0.1`、`scale=0.8..1.2` / `step=0.1`，每组均为 2,000 个变体。所有训练与匹配均只执行一次（`--warmup 0 --repeat 1`）；计时只覆盖 `addTemplateVariants()` 或 `match()`，不包含图像/YAML 读写、ROI 裁剪和结果图绘制。

### 训练输出：v0 原始路径与 v1 优化路径

v0 使用当前公开的 `--version v0` 原始串行训练路径；v1 使用当前公开的 `--version v1 --train-parallelism 0` 自动并行路径。为展示本轮 v1 训练优化的整体收益，表中还保留了改动前的 v1 串行基线；它不是第三个公开版本，只用于分离“原有 v1”与“本轮训练优化”的耗时。所有数据均为一次运行。

| 组 | 模板 ROI `(x,y,w,h)` | 变体数 | v0 原始串行 | 改动前 v1 串行基线 | v1 优化训练 | v0 → v1 | 串行 v1 → 优化 v1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| roi_01 | `(1262,232,486,476)` | 2,000 | 14,307.289 ms | 14,724.648 ms | 2,224.333 ms | 6.43× | 6.62× |
| roi_02 | `(2504,214,470,459)` | 2,000 | 12,300.875 ms | 12,289.469 ms | 1,940.889 ms | 6.34× | 6.33× |
| roi_03 | `(2528,1460,469,452)` | 2,000 | 10,424.353 ms | 11,037.027 ms | 1,752.932 ms | 5.95× | 6.30× |
| roi_04 | `(2549,2701,463,451)` | 2,000 | 11,629.561 ms | 11,755.439 ms | 1,805.868 ms | 6.44× | 6.51× |
| roi_05 | `(1932,2086,445,458)` | 2,000 | 10,950.867 ms | 11,042.227 ms | 1,650.729 ms | 6.63× | 6.69× |
| 合计 / 平均 | — | 10,000 | 59,612.945 / 11,922.589 ms | 60,848.810 / 12,169.762 ms | 9,374.751 / 1,874.950 ms | **6.36×** | **6.49×** |

每个 ROI 均输出了 `F:\data\shape_match\roi_0N_templates_v0.yaml` 和 `roi_0N_templates_v1_optimized.yaml`。同一 ROI 的 v0、改动前 v1 和优化后 v1 三份 YAML 的 SHA-256 完全相同；随后使用优化后 v1 YAML 的全图精确匹配结果图，也与此前对应的 v1 结果图 SHA-256 完全相同。换言之，上表的加速不改变模板内容或匹配精度。上述历史文件是 v1 格式；升级到当前代码后需重新训练为 v2，才能加载匹配。

### 全图精确：v0 与优化后 v1

两版均使用同一 YAML、`--threshold 85 --template-stride 1 --scan-step 0`，未传搜索掩膜。每组的 v0 和 v1 top-1 在类别、模板 ID、分数、框、角度和尺度上完全相同；下表列出共同的 top-1。5 对输出 PNG 的 SHA-256 也逐对相同。对应结果图已写入 `F:\data\shape_match\roi_0N_v0.png` 和 `roi_0N_v1.png`。

| ROI         |               v0 单次 match |             v1 单次 match |    v1 加速 | 共同 top-1 `(template / score / box)` | 精度结果               |
| ----------- | --------------------------: | ------------------------: | ---------: | ------------------------------------- | ---------------------- |
| roi_01      |               94,878.818 ms |              2,798.300 ms |     33.91× | `1000 / 100 / (1290,247,437,441)`     | 完全一致               |
| roi_02      |               95,798.364 ms |              2,594.704 ms |     36.92× | `1000 / 100 / (2518,222,441,439)`     | 完全一致               |
| roi_03      |               96,133.326 ms |              2,617.838 ms |     36.72× | `1000 / 100 / (2540,1463,441,438)`    | 完全一致               |
| roi_04      |               96,401.170 ms |              2,749.973 ms |     35.06× | `1000 / 100 / (2562,2705,439,430)`    | 完全一致               |
| roi_05      |               96,471.570 ms |              2,361.401 ms |     40.85× | `1000 / 100 / (1938,2096,438,427)`    | 完全一致               |
| 合计 / 平均 | 479,683.248 / 95,936.650 ms | 13,122.216 / 2,624.443 ms | **36.56×** | 5/5 一致                              | v1 耗时降低 **97.26%** |

### 可选近似：默认关闭的速度/精度取舍

以下不是 v1 默认行为，而是显式传入 `--template-stride 7 --scan-step 2` 的组合。它减少模板变体和空间候选，因而可能漏检、改变最佳角度/尺度或产生定位偏差；本组同图验证均检出目标，但不应据此推断其他图像的召回率。对应输出图为 `F:\data\shape_match\roi_0N_v1_approx.png`。

| ROI         |                   精确 v1 |            可选近似 v1 | 相对精确 v1 | 近似 top-1 `(template / score / box)`  | 相对精确框 IoU | 精度影响                          |
| ----------- | ------------------------: | ---------------------: | ----------: | -------------------------------------- | -------------: | --------------------------------- |
| roi_01      |              2,798.300 ms |             376.138 ms |       7.44× | `1001 / 88.5417 / (1264,230,484,474)`  |         0.8400 | 检出；分数 -11.4583，框和变体改变 |
| roi_02      |              2,594.704 ms |             381.325 ms |       6.80× | `1001 / 91.6667 / (2504,214,468,457)`  |         0.9052 | 检出；分数 -8.3333，框和变体改变  |
| roi_03      |              2,617.838 ms |             379.500 ms |       6.90× | `1001 / 95.3125 / (2530,1462,467,450)` |         0.9191 | 检出；分数 -4.6875，框和变体改变  |
| roi_04      |              2,749.973 ms |             377.531 ms |       7.28× | `1001 / 93.75 / (2548,2700,461,449)`   |         0.9120 | 检出；分数 -6.25，框和变体改变    |
| roi_05      |              2,361.401 ms |             378.302 ms |       6.24× | `1001 / 95.8333 / (1932,2086,443,456)` |         0.9218 | 检出；分数 -4.1667，框和变体改变  |
| 合计 / 平均 | 13,122.216 / 2,624.443 ms | 1,892.796 / 378.559 ms |   **6.93×** | 5/5 检出                               |              — | 相对精确 v1 耗时降低 **85.58%**   |

因此，默认精确模式已在这 5 组 ROI 上验证为与 v0 完全一致；若业务能够接受实际数据集验证后的召回、角度、尺度和定位损失，才应显式开启近似开关。ROI 仅用于训练模板，匹配仍是整张 `51661.png`，没有把已知目标区域作为算法内的搜索限制。
