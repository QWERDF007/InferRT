# 形状模板匹配示例

`shape_template_matching` 使用 `irt::features::v0::ShapeTemplateMatcher` 或
`irt::features::v1::ShapeTemplateMatcherFast` 或
`irt::features::v2::ShapeTemplateMatcherAvx512` 在源图中查找相同边缘形状的目标。它不依赖深度学习模型，适合轮廓稳定、边缘清晰的零件、工件或标记定位。

示例按两个独立阶段运行：

1. `train`：从模板图（或大图中的矩形区域）训练旋转/缩放模板，并保存标准 YAML 模板文件。
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

三个版本实现相同的 `IShapeTemplateMatcher` 接口，模板标准 YAML 格式和匹配结果完全兼容：

- `--version v0`：固定的 `shape_based_matching` 风格标量基线路径。训练时保留原始串行流程：逐变体变换、逐变体提取特征并立即写入模板库；不使用线程并行、工作缓冲复用或跳过掩膜规范化等 v1 优化。适合结果对照和基准比较。
- `--version v1`：AVX2 加速实现，默认值。匹配时会批量评分相邻滑窗、保留阈值早停，并直接从量化标签查表；训练时会并行处理相互独立的旋转/缩放变体、复用每个工作线程的中间缓冲，并按输入顺序统一提交模板。运行 v1 需要 AVX2 CPU；不支持时请选择 v0。
- `--version v2`：AVX512F/BW 加速实现。训练阶段以 16-lane 量化和 64-lane 候选筛选处理梯度；全图精确匹配在分子上界不超过 `255` 时使用 64-lane 8-bit 累加，较大但安全的上界使用 32-lane 16-bit 累加。`scan_step != 1` 或累计范围过大时回退到同语义的标量路径，不改变结果。运行 v2 必须同时支持 AVX512F 和 AVX512BW；不支持时构造会明确报错，不会隐式退回 v1。

训练和匹配可以选择不同版本。例如可用 `v0` 训练、`v1` 或 `v2` 匹配，或用 `v0` 对照验证结果：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe --mode train --version v0 --template "D:\data\part.png" --save-templates "D:\data\part_templates.yaml"
.\build\bin\inferrt_sample_shape_template_matching.exe --mode match --version v1 --load-templates "D:\data\part_templates.yaml" --source "D:\data\scene.png"
```

三种训练路径产生的模板格式和内容严格兼容。v1 的并行阶段不写模板库，最终按变体输入顺序提交，因此模板 ID、YAML 顺序和精确匹配结果均与 v0 保持一致。

v2 也按输入变体顺序统一提交模板，且其训练、匹配结果与 v0/v1 保持逐字段一致。注意：这里的“实现版本 v2”和下文“模板文件格式 v4”是两个独立概念；v0、v1、v2 实现都读写同一模板文件格式。

## 模板文件格式：破坏性 v4 升级

新训练的标准 YAML 模板文件使用 **v4** 格式，并由 `yaml-cpp` 读写。一个模板文件对应调用方外部定义的一类目标；文件内部不保存类别字段。每个模板还必须保存原始训练画布的 `template_width` 和 `template_height`，用于输出完整训练 ROI。为去除每个特征重复的 `x`、`y`、`label`、`angle_degrees` 键，`features` 改为紧凑二维数组；每一行固定按如下顺序存储：

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

这是破坏性修改：旧版 `version: 1`/`version: 2`/`version: 3` 模板文件，以及缺少 `template_width`/`template_height` 的文件，都会被拒绝加载，必须重新执行 `train` 生成 v4 模板。类别字段已从 YAML 和 API 移除；需要识别多类目标时，请由外部为每类维护独立模板文件并分别加载匹配。当前 v0、v1 和 v2 都读写同一标准 YAML v4 格式，四列数量不是恰好 4 的文件也会被拒绝。

## 第一阶段：训练模板

训练只需要模板输入图和可选的模板掩码，**不需要传入待匹配的大图**。训练完成后会写入 `--save-templates` 指定的标准 YAML 文件。

### 从已裁剪的小图训练

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v1 `
  --template "D:\data\part.png" `
  --template-mask "D:\data\part_mask.png" `
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

`--template` 也可以是包含目标的大图。`--template-roi x,y,width,height` 可重复传入；每个矩形都会从同一张大图裁剪出一个独立模板，再按同一组角度/尺度变体训练，并统一写入一个标准 YAML 模板库。未传该参数时，整张 `--template` 图像视为一个模板。

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v1 `
  --template "D:\data\capture.png" `
  --template-roi "420,180,160,120" `
  --template-mask "D:\data\capture_mask.png" `
  --angle-begin 0 `
  --angle-end 180 `
  --angle-step 15 `
  --save-templates "D:\data\part_templates.yaml"
```

一次训练多个模板时，重复添加参数即可：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v1 `
  --template "D:\data\capture.png" `
  --template-roi "420,180,160,120" `
  --template-roi "760,190,155,118" `
  --angle-begin -20 `
  --angle-end 20 `
  --angle-step 0.1 `
  --scale-begin 0.8 `
  --scale-end 1.2 `
  --scale-step 0.1 `
  --save-templates "D:\data\part_templates.yaml"
```

上例有两个 ROI、每个 ROI 2,000 个变体时，输出模板库共有 4,000 个模板；一次 `match` 会扫描它们，无需逐 ROI 分别匹配。v1/v2 会把全部 `ROI × 变体` 放入同一个全局任务队列，由 `--train-parallelism` 控制的工作线程统一调度并复用工作缓冲；完成后仍按 ROI 输入顺序、再按变体顺序写入模板。v0 保持原始的逐 ROI、逐变体串行路径。矩形格式为 `x,y,width,height`，坐标以 `--template` 输入图的左上角为原点，且每个矩形必须完全位于图像内。为支持旋转/缩放变体，建议矩形在目标周围保留适量背景空白，避免变换后裁掉目标边缘。

### 训练掩码 `--template-mask`

`--template-mask` 是训练阶段的**目标掩码**，不是匹配大图的 ROI：

- 非零像素参与模板特征训练；零像素会被忽略。
- 未使用 `--template-roi` 时，掩码必须与 `--template` 图像同尺寸。
- 只使用一个 `--template-roi` 时，掩码可以与原始大图同尺寸（示例会自动裁剪），也可以直接与该裁剪后小图同尺寸。
- 使用多个 `--template-roi` 时，掩码必须与原始大图同尺寸；示例会对每个 ROI 自动裁剪同一张掩码。多个不同 ROI 若需要不同掩码，应使用全图掩码一次性标出各目标，或通过 API 分别添加模板。

模板图背景干净时可以省略掩码；背景边缘复杂时，建议提供只覆盖目标本体的精确掩码。

## 第二阶段：匹配模板

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode match `
  --version v1 `
  --load-templates "D:\data\part_templates.yaml" `
  --source "D:\data\scene.png" `
  --threshold 85 `
  --warmup 3 `
  --repeat 10 `
  --output "D:\data\result.png"
```

`match` 阶段不会重新训练。它只加载 `--load-templates` 中保存的模板与配置，然后在 `--source` 大图中搜索。一个模板文件即由外部定义为一类目标；不同目标应由外部保存为不同的 YAML 模板文件，并分别创建匹配器执行匹配。

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

### 使用 v2 AVX512

在支持 AVX512F 和 AVX512BW 的机器上，将同一套训练、匹配命令中的 `--version` 改为 `v2` 即可：

```powershell
.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode train `
  --version v2 `
  --template "D:\data\part.png" `
  --angle-begin -20 `
  --angle-end 20 `
  --angle-step 0.1 `
  --scale-begin 0.8 `
  --scale-end 1.2 `
  --scale-step 0.1 `
  --train-parallelism 0 `
  --save-templates "D:\data\part_templates_v2.yaml"

.\build\bin\inferrt_sample_shape_template_matching.exe `
  --mode match `
  --version v2 `
  --load-templates "D:\data\part_templates_v2.yaml" `
  --source "D:\data\scene.png" `
  --threshold 85 `
  --output "D:\data\result_v2.png"
```

`v2` 不适合 AVX512 被禁用或不具备 AVX512BW 的 CPU。例如第 12 代桌面 Intel Core i7-12700K 没有可用的 AVX512F/BW，示例会返回 `IRT_ERROR_INVALID_OPERATION`；请在该机器上使用 v1。

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
- 训练计时仅覆盖核心训练调用：单 ROI 为 `addTemplateVariants()`，多个 ROI 为 `addTemplateVariantsBatch()`；每次预热和重复都会使用新的匹配器，避免向同一模板库重复添加模板。
- 匹配计时仅覆盖 `match()`；模板文件、输入图像和掩码会在计时前加载。
- YAML 写入、图像读取、ROI 裁剪、结果绘制和输出图片均不计入核心耗时。

需要只执行一次功能验证时，可使用：

```powershell
--warmup 0 --repeat 1
```

## 参数分组

### 计时参数（两个阶段通用）

- `--warmup`：不计入统计的预热次数，必须大于等于 `0`，默认 `3`。
- `--repeat`：计入统计的重复次数，必须大于 `0`，默认 `10`。

### 实现版本（两个阶段通用）

- `--version`：选择 `v0`、`v1` 或 `v2`，默认 `v1`。三者可以读写同一模板文件格式 v4；需要验证 SIMD 路径时，可在同一输入上分别运行各版本并比较结果。v2 需要 AVX512F/BW。

### 训练参数（`--mode train`）

- `--template`：训练输入图，可为目标小图或原始大图。
- `--template-roi`：可重复的裁剪矩形，格式 `x,y,width,height`；每个 ROI 都会训练为同一模板库中的一组模板。未传时训练整张输入图。
- `--template-mask`：可选目标掩码。
- `--save-templates`：必填，训练输出的标准 YAML 路径；一个文件由外部定义为一类目标。
- `--angle-begin`、`--angle-end`、`--angle-step`：离散训练旋转角度；端点包含在内。
- `--scale-begin`、`--scale-end`、`--scale-step`：离散训练缩放尺度。
- `--features`：每个模板最多保留的边缘特征点数，常用 `64` 到 `128`；更多特征通常更精细但更慢。
- `--weak-threshold`、`--strong-threshold`：弱梯度保留阈值和模板候选强梯度阈值；候选不足时会按既有流程回退到弱阈值。
- `--max-label-difference`：方向环形容差，范围 `[0,4]`；值越大方向稳定性越强但区分度降低。
- `--min-feature-distance`：模板特征点最小空间间距；`0` 按模板面积和特征数自动估计，v1 用等价空间网格加速选点。
- `--template-scan-step`：写入模板文件的精确空间扫描步长，通常保持 `1`；匹配时可用 `--scan-step` 临时覆盖，但大于 `1` 会进入近似模式。
- `--train-parallelism`：影响 v1/v2 的训练工作线程数；多个 ROI 时线程从同一个 `ROI × 变体` 全局任务队列领取工作。`0`（默认）自动选择，`1` 强制 SIMD 版本串行。v0 忽略该项，始终保持原始串行训练。此运行时参数不写入模板文件，也不会改变后续匹配的线程设置。
- `--gaussian-gradient`、`--edge-nms`、`--edge-connectivity`、`--orientation-histogram`、`--polarity-invariant`、`--spatial-spread`：v1 可选梯度/方向预处理，默认关闭；会改变特征和精度，训练后配置写入 YAML。
- `--reuse-base-features`：v1 可选旋转/缩放训练加速；每个 ROI 只抽取一次基础特征，默认关闭，可能改变最佳变体和定位。
- `--max-results`、`--nms`：保存到模板文件的匹配默认配置。`--nms` 为全模板库 NMS 的 IoU 阈值，负数表示关闭；`--max-results` 为最多输出的结果数。

### 匹配参数（`--mode match`）

- `--load-templates`：必填，训练阶段生成的标准 YAML 文件。
- `--source`：必填，待搜索源图。
- `--search-mask`：可选源图搜索区域掩码，必须与源图同尺寸。
- `--threshold`：本次匹配的分数阈值，范围 `[0,100]`；传负数或省略时使用模板文件中保存的默认值。
- `--template-stride`：本次匹配每隔多少个模板变体扫描一次，默认 `1`（精确）；大于 `1` 为可能影响召回和角度/尺度精度的近似模式。
- `--scan-step`：本次匹配覆盖空间扫描步长；默认 `0`，表示使用模板文件配置。大于保存值时通常更快，但可能产生定位偏差或漏检。
- `--output`：带匹配框的输出图，默认 `shape_template_matching_result.png`。

角度/尺度步长越小，模板数量和匹配耗时越高，但对相应变化的召回率通常更好。误检较多时提高 `--threshold`；漏检较多时适当降低它。

## v1 完整算法链路与优化边界

此前只打开 Gaussian 不能代表完整的 v1 优化。当前 v1 已覆盖从梯度到输出的整条链路；严格等价项默认启用，可能改变特征或召回率的项必须显式打开：

| 阶段 | v1 当前实现 | 默认/精度属性 |
| --- | --- | --- |
| 边缘提取 | OpenCV Sobel 计算 `grad_x/grad_y`；AVX2 批量完成幅值阈值筛选和 8-bin 标签写入 | 默认精确；`--gaussian-gradient`、`--edge-nms`、`--edge-connectivity` 为可选近似 |
| 方向量化 | 22.5° 边界的 8 方向量化；v1 使用 AVX2 8 像素批处理，尾部保持标量 | 默认精确 |
| 方向稳定性 | `max_label_difference` 的环形方向容差；按源图方向直方图排序特征访问顺序 | 容差和排序严格等价；`--orientation-histogram`、`--polarity-invariant` 会改变标签 |
| 空间容差 | 模板训练的 `min_feature_distance` 贪心选点；v1 使用等价空间网格减少邻点比较 | 默认精确；`--spatial-spread` 会扩散标签 |
| 旋转/缩放模板 | `warpAffine` 生成每个角度/尺度变体；模板保存原始训练画布和变体元数据 | 默认精确；`--reuse-base-features` 只变换基础特征，属于近似训练 |
| 多尺度 | `scale_begin/end/step` 离散生成完整尺度集合，并与角度组合 | 默认精确；步长越小模板越多、训练/匹配越慢 |
| 响应图 | v0 物化 8 张响应图；v1 只保留量化标签，一次匹配共享 8 组 SIMD shuffle 查表 | v1 严格等价 |
| 单点得分 | AVX2 8/16-bit 累加，按理论上界每 4 个特征早停；整数分子最后统一换算为百分比 | 严格等价 |
| 候选抑制 | 排序后执行 NMS；v1 使用空间网格只比较可能相交的框 | IoU、排序、`max_results` 语义严格等价 |
| 输出框 | 根据模板训练画布、特征包围盒、旋转/尺度元数据还原完整 ROI 框 | 严格等价 |
| 并行调度 | 训练使用 `ROI × 变体` 全局任务队列；匹配按模板分片；线程复用梯度/候选/扫描工作区 | 结果顺序和模板 ID 稳定，默认精确 |

### v1 配置开关

以下开关只允许 v1 使用，默认均为 `false`；训练开关会写入模板 YAML，加载时自动采用同一配置，必须重新训练模板才能改变：

- `--gaussian-gradient`：Sobel 前增加 5×5 GaussianBlur。
- `--edge-nms`：沿量化梯度方向保留局部极大边缘。
- `--edge-connectivity`：从强梯度种子保留 8 邻域连通的弱边缘。
- `--orientation-histogram`：3×3 方向多数滤波。
- `--polarity-invariant`：将相反梯度极性折叠到 4 个无符号方向。
- `--spatial-spread`：把邻域多数方向扩散到孤立弱像素。
- `--reuse-base-features`：每个 ROI 只提取一次基础特征，旋转/缩放变体只变换坐标和方向。

这些选项会改变模板特征、响应分数、最佳变体或定位，不能与 v0 的 `shape_based_matching` 基线混用。v1 默认关闭它们时，v0/v1 的模板和匹配结果继续逐字段一致。`--template-stride` 和 `--scan-step` 是匹配调用级近似开关，仍默认分别为 `1` 和 `0`。

注意：精确 v1 的耗时会随匹配阈值明显变化。阈值较高时，SIMD 早停可以在累加少量特征后淘汰绝大多数窗口；阈值降低后，更多窗口必须完成完整特征累加，因此全图匹配可能变慢，这不是 NMS 或可选预处理造成的回归。对比速度时请固定 `--parallelism`、`--warmup 0 --repeat 1` 和阈值；输出中的 `total` 是所有重复次数的总耗时，`avg` 才是单次平均耗时。若需要接近 `shapeMatchV2` 的金字塔粗筛，应显式使用 `--template-stride`/`--scan-step` 等近似开关并单独验证召回率，默认精确路径不会跳过候选位置。

本机对 `tmp_grid.yaml`（2,000 模板）的一次复测为：`threshold=85`、20 线程 **2,447.177 ms**，返回 1 个命中；`threshold=38`、相同线程 **8,817.666 ms**，返回 49 个命中。两次均为 `--warmup 0 --repeat 1`、全图 `scan-step=1`，top-1 为同一模板和框。该差异说明“变慢”主要来自阈值导致的评分工作量，而不是 v1 的输出或 NMS 回归。

### 真实图单次验证（`F:\data\shape_match\51661.png`）

以下是本次新增代码在同一 ROI `(1285,245,445,448)`、96 features、2,000 个变体、全图匹配上的一次验证，均为 `--warmup 0 --repeat 1`。训练和匹配使用自动并行；表格只比较同一机器、同一构建下的当前结果，不把历史运行波动当作收益：

| 配置 | 训练 | 匹配 | 相对精确 match | top-1 / 结果 |
| --- | ---: | ---: | ---: | --- |
| v1 精确（网格选点、标签查表、工作区、索引 NMS） | 1,471.751 ms（2,000 模板）/ 10.925 ms（单模板） | 2,403.661 ms（2,000 模板）/ 212.086 ms（单模板） | 1.00× | `template=1000 / score=100 / box=(1285,242,445,448)`（2,000 模板） |
| 单模板 + `--edge-nms` | 6.587 ms | 274.591 ms | 0.77× | `template=0 / score=100 / box=(1285,245,445,448)` |
| 单模板 + `--edge-connectivity` | 14.264 ms | 402.853 ms | 0.53× | `template=0 / score=100 / box=(1287,241,445,448)` |
| 单模板 + `--polarity-invariant` | 10.391 ms | 206.146 ms | 1.03× | `template=0 / score=100 / box=(1287,241,445,448)` |
| 单模板 + `--spatial-spread` | 11.343 ms | 309.835 ms | 0.68× | `template=0 / score=100 / box=(1287,241,445,448)` |
| v1 `--reuse-base-features`（近似训练） | 148.626 ms（2,000 模板） | 2,187.095 ms（2,000 模板） | 1.10× | `template=991 / score=100 / box=(1285,243,445,448)` |

`reuse-base-features` 让本组训练约 **9.91×** 加速，但最佳变体从 `1000` 变为 `991`、框发生像素偏移，因此默认关闭。单模板开关表只用于逐阶段验证预处理确实接入，单次测量受系统噪声影响；不同配置会改变特征图，不能把单模板耗时与 2,000 模板耗时直接相加。生产使用时应在自己的数据集上分别统计召回、分数、角度/尺度误差和框 IoU。

同一单模板默认配置的 v0 标量 match 为 `687.145 ms`，v1 为 `212.086 ms`，输出框和分数逐字段一致；该对照只用于说明精确 SIMD 路径没有牺牲结果，不把单模板比例外推到 2,000 模板库。



### cmd


#### train

F:\Projects\InferRT\build\bin\inferrt_sample_shape_template_matching.exe --mode train --template "F:\data\shape_match\51661.png" --template-roi "1285,245,445,448" --angle-begin -20 --angle-end 20 --angle-step 0.1  --scale-begin 0.8 --scale-end 1.2 --scale-step 0.1 --save-templates "F:\data\shape_match\part_templates.yaml"

#### match

F:\Projects\InferRT\build\bin\inferrt_sample_shape_template_matching.exe --mode match --load-templates "F:\data\shape_match\part_templates.yaml" --source "F:\data\shape_match\51661.png" --threshold 85 --output "F:\data\shape_match\result.png" --version v1





## v2 AVX512 真实图验证

本轮在真实图 `F:\data\shape_match\51661.png` 上重新训练并验证了当前模板文件格式。训练使用 ROI
`(1285,245,445,448)`、96 features、`angle=-20..20 / step=0.1`、`scale=0.8..1.2 / step=0.1`，共 2,000 个模板；
匹配使用全图精确搜索、`threshold=85`、`--warmup 0 --repeat 1`。计时仅覆盖 `addTemplateVariants()` 或 `match()`。

| 实现 | 本机指令集状态 | 单次训练 | 单次全图 match | 结果 |
| --- | --- | ---: | ---: | --- |
| v1 AVX2 | 可用 | 1,945.296 ms | 2,392.368 ms | `template=1000 / score=100 / (1289,247,437,442)` |
| v2 AVX512F/BW | 不可用 | — | — | `IRT_ERROR_INVALID_OPERATION`：CPU 缺少 AVX512F/BW |

本机 CPU 为 **12th Gen Intel Core i7-12700K**。该型号没有可用 AVX512F/BW，因此不能提供伪造的 v2 性能数据；v2 不会回退到 v1，以保证 `--version v2` 始终代表真实 AVX512 路径。已构建 v2、并加入与 v1 的条件奇偶测试；在具有 AVX512F/BW 的机器上该测试会自动执行训练和匹配逐字段对比。请使用上文的 v2 命令在支持的机器上重新测量，并将 v1/v2 的 `--warmup 0 --repeat 1` 输出按相同输入进行比较。

本次复核生成的 v1 训练产物为 `build\part_templates_v2_reverified.yaml`，它采用当前紧凑模板文件格式；旧的
`F:\data\shape_match\part_templates.yaml` 仍为 `version: 1`，会被当前程序按设计拒绝加载。

## v1 真实图优化验证

以下结果使用真实输入 `F:\data\shape_match\51661.png` 和已有的 2,000 个模板
`F:\data\shape_match\part_templates.yaml` 测得。模板配置为 96 features、`scan_step=1`、方向容差 1，匹配阈值为 85。每一行都使用：

```powershell
--warmup 0 --repeat 1
```

因此表格用于一次端到端功能验证，而非统计学性能结论；计时范围仍仅为 `match()`，不含文件读取、模板加载、绘制和结果写入。阶段 0–7 是严格等价的 v1 实现优化，均返回同一结果：`template=1000`、`score=100`、`box=(1289,247,437,442)`；阶段 8a、8b、9 是默认关闭、可能影响精度的显式近似配置。

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

截至上述阶段 7 的同条件全图真实图对照为：v0 `100,158.746 ms`，v1 `2,451.975 ms`，v1 为 **40.85×** 更快（耗时降低 **97.55%**），且匹配字段一致。这是 v0/v1 的历史优化对照；独立 v2 AVX512 实现及本机可用性见上方 [v2 AVX512 真实图验证](#v2-avx512-真实图验证)。

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

`ShapeTemplateMatchOptions` 不写入 YAML，只影响本次 `match()` / `matchFile()` 调用。示例把它们暴露为以下默认关闭的参数：

- `--template-stride N`：只扫描模板 ID 可被 `N` 整除的变体。默认 `1`，扫描全部模板；`N > 1` 可能跳过最佳角度/尺度变体，从而影响召回、分数和框大小。
- `--scan-step N`：`0` 表示使用模板文件中的 `scan_step`；正数覆盖空间扫描步长。当前模板文件为 `scan_step=1`；增大到 `2` 会减少候选位置，可能产生像素级定位偏差或漏检。

上表中的 IoU 为当前 top-1 框相对精确 top-1 框的 IoU；所有配置均返回 1 个匹配。相对精确基线，三种近似配置的耗时分别降低 **78.98%**、**42.31%** 和 **84.81%**。真实图上 `template-stride=7` 仍检测到目标，但最佳变体从 ID 1000 变为 1001，分数降至 91.6667；这正是该选项的预期精度代价。`scan-step=2` 保持相同模板和分数，但定位偏移 1 像素。不同图像、阈值和角度/尺度采样密度下，近似配置也可能直接漏检，因此生产使用前应按目标数据集验证。

对应 API 用法：

```cpp
irt::features::ShapeTemplateMatchOptions options;
options.template_stride = 7; // 默认 1：精确
options.scan_step = 2;       // 默认 0：沿用模板文件配置
const auto matches = matcher.match(source, 85.0f, cv::Mat(), options);
```

## 全图精确快路径与 5 组 ROI 验证

v1 新增了严格等价的“全图无掩膜”快路径：当调用方未传 `search_mask`，或传入全非零掩膜时，匹配器不再为每一个滑窗候选重复读取和判断掩膜像素。扫描范围、候选顺序、评分、NMS 和返回结果均不变；带有零值的搜索掩膜仍走原有精确路径。v0 保持 `shape_based_matching` 风格的标量参考实现，不使用这项 v1 优化。

在已有 `part_templates.yaml`、2,000 个模板、全图精确匹配和一次运行（`--warmup 0 --repeat 1`）下，这项优化将 v1 从 `2,675.392 ms` 降至 `2,451.975 ms`，即 **1.09×**（耗时降低 **8.35%**），匹配字段完全相同；当前回归测试共 27 项，其中 26 项通过，AVX512 条件奇偶测试因本机缺少该指令集跳过。其余测试覆盖紧凑特征格式、旧 v1 模板拒绝加载、特征四列校验、v0/v1 参数组合、全非零搜索掩膜、单 ROI 串行/并行训练，以及多输入全局训练队列与逐输入训练的一致性。

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

### 多 ROI 全局训练调度

上述表逐个 ROI 调用训练接口。为避免每个 ROI 分别创建线程和工作区，当前 sample 会把同一张真实图的 5 个 ROI 与各自 2,000 个变体合并为一个 10,000 任务的全局队列。v1/v2 工作线程跨 ROI 复用仿射、梯度和候选缓冲区；提交阶段仍固定为 ROI 输入顺序、再按变体顺序，因此不改变模板 ID 或内容。以下均为 `--warmup 0 --repeat 1` 的单次结果：

| 训练调度 | 模板数 | 单次核心训练 | 相对分 ROI 调度 | 模板与精度 |
| --- | ---: | ---: | ---: | --- |
| 旧 sample：逐 ROI 调度、每个 ROI 内部并行 | 10,000 | 9,712.363 ms | 1.00× | 基线 |
| 当前 sample：全局 `ROI × 变体` 任务队列 | 10,000 | **9,338.475 ms** | **1.04×** | YAML SHA-256 完全相同；全图 5 个命中字段和输出 PNG 完全相同 |

本机该真实工作负载的训练耗时降低 **3.85%**。收益主要来自只创建一次线程组、跨 ROI 复用工作区以及统一负载分配；每个 ROI 已有 2,000 个变体且 CPU 已接近满载，因此这是严格等价优化的实际收益，不把运行波动伪装成更大的加速。匹配路径没有改动；重新使用新的 10,000 模板库全图匹配仍返回 5 个 `score=100` 的目标。

### 本轮严格等价训练优化：候选缓冲复用

在全局队列基础上，v1/v2 现在让每个训练工作线程复用一个候选点 `std::vector`：AVX2/AVX512 内核只负责向该数组追加当前变体的候选，公共流程仍使用原有的稳定 `std::sort` 比较器和原有贪心选点规则。v0 不进入该路径，保持与 `shape_based_matching` 一致的原始逐变体实现。

以下使用同一张 `F:\data\shape_match\51661.png`、同一 5 组 ROI、每组 2,000 个变体（共 10,000 个模板）、Release 构建、`--warmup 0 --repeat 1` 的单次真实图结果。计时只覆盖 `addTemplateVariantsBatch()`，不包含读图、ROI 裁剪和 YAML 写入。

| 训练路径 | 单次核心训练 | 相对上一步 | 相对初始 v1 全局队列 | 模板与精度验证 |
| --- | ---: | ---: | ---: | --- |
| 初始 v1 全局 `ROI × 变体` 队列 | 9,326.378 ms | 1.00× | 1.00× | 基线 |
| v1：候选缓冲跨变体复用（当前） | **7,461.071 ms** | **1.250×** | **1.250×** | 10,000 模板 YAML 的 SHA-256 完全相同 |

本项单次节省 **1,865.307 ms**，训练核心耗时降低 **20.00%**。相对于上表中的 v0 原始串行 5 ROI 合计 `59,612.945 ms`，当前 v1 总体为约 **7.99×**。由于候选排序比较器、候选顺序、贪心选点和模板提交顺序均未改动，这是一项严格等价优化；新增的高密度候选图回归测试也会逐字段比较 v0/v1 的全部旋转、缩放模板。随后加载本轮 10,000 模板 YAML 做全图精确匹配，仍返回 5 个 `score=100` 命中，模板 ID 依次为 `1000/3000/5000/7000/9000`，与既有验证结果一致。

每个 ROI 均输出了 `F:\data\shape_match\roi_0N_templates_v0.yaml` 和 `roi_0N_templates_v1_optimized.yaml`。同一 ROI 的 v0、改动前 v1 和优化后 v1 三份 YAML 的 SHA-256 完全相同；随后使用优化后 v1 YAML 的全图精确匹配结果图，也与此前对应的 v1 结果图 SHA-256 完全相同。换言之，上表的加速不改变模板内容或匹配精度。上述历史文件是 v1 格式；升级到当前代码后需重新训练为 v2，才能加载匹配。

### 全图精确：v0 与优化后 v1

两版均使用同一 YAML、`--threshold 85 --template-stride 1 --scan-step 0`，未传搜索掩膜。每组的 v0 和 v1 top-1 在模板 ID、分数、框、角度和尺度上完全相同；下表列出共同的 top-1。5 对输出 PNG 的 SHA-256 也逐对相同。对应结果图已写入 `F:\data\shape_match\roi_0N_v0.png` 和 `roi_0N_v1.png`。

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

## `tmp.log` 外部真实图：本轮 v1 优化分阶段验证

下面的验证使用 `tmp.log` 下半段的真实图
`E:\works\用户\wzy\picture_16_2026_06_22_15_00_22_298.png`，训练 ROI 为
`(2034,1306,297,296)`，角度 `-5..5 / 0.1`、尺度 `1`、64 个特征，共 101 个模板；匹配为全图、阈值 60、
`template-stride=1`、`scan-step=1`、固定 20 个线程。所有计时均为 `--warmup 0 --repeat 1` 的一次
`match()`，不含读图、加载 YAML、绘制和写图。该图肉眼确认的正确结果为 144 个命中。

### 严格等价优化（默认启用）

“原始 v1”是本轮改动前的 AVX2 扫描路径；工作区复用只重用线程内的结果数组和特征访问数组，方向评分、扫描坐标、早停判定、NMS 和返回排序均不变。两行都返回 144 个结果，top-1 均为
`template=49 / score=100 / box=(2034,1306,297,296) / angle=-0.100003 / scale=1`。

| 阶段 | v1 精确改动 | 单次 match | 相对上一步 | 相对原始 v1 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| E0 | 原始 v1：每模板创建临时结果/特征 vector | 1,608.519 ms | 1.00× | 1.00× | 144，top-1 一致 |
| E1 | 线程级扫描工作区复用（当前默认） | **1,461.139 ms** | **1.10×** | **1.10×** | 144，逐字段一致 |

E1 相对 E0 节省 147.380 ms（9.16%）。这是内存分配/释放优化，不改变模板文件或匹配精度；不同机器的线程数和内存带宽会影响绝对耗时。v0 没有接入该工作区路径，仍保持原始标量基线。
按 `tmp.log` 中 shapeMatchV2 约 1,700 ms 的同图单次结果，E1 约为 **1.16×**；两者参数和内部梯度流程并非完全相同，因此该数值只作工程参考，不能替代逐参数精度对照。

当前 v1 还将 AVX2 方向量化改为一次向量乘加/整数转换，并仅在 22.5° 等量化边界附近回退到参考公式；边界回退保证与 v0 的方向标签一致，不改变模板或匹配结果。该优化属于预处理常数项，主要扫描耗时仍由模板数量、窗口数量和阈值决定，不能抵消低阈值下完整评分的成本。

### 影响精度的 v1 预处理（显式开启，默认关闭）

这两项参考 `shapeMatchV2` 的梯度预处理，但会改变量化标签、模板特征、分数或最佳变体，因此不会默认启用，也不对 v0/v2 开放：

- `--gaussian-gradient`：Sobel 前执行 5×5 `GaussianBlur`。
- `--orientation-histogram`：对量化方向做 3×3 邻域多数滤波（至少 5 个有效邻居同向才替换）。

训练时的开关写入模板 YAML，匹配时自动使用相同设置；必须重新训练模板，不能把开关追加到已有的不同设置模板上。命令示例：

```powershell
build\bin\inferrt_sample_shape_template_matching.exe --mode train --version v1 `
  --template "E:\works\用户\wzy\picture_16_2026_06_22_15_00_22_298.png" `
  --template-roi "2034,1306,297,296" --angle-begin -5 --angle-end 5 --angle-step 0.1 `
  --features 64 --gaussian-gradient --orientation-histogram `
  --save-templates "v1_approx_pre.yaml" --warmup 0 --repeat 1
```

真实图单次结果如下；E1 是精确父基线，近似行的“相对 E1”不是与其他近似项叠加的承诺。

| 配置 | 训练耗时 | 单次 match | 相对 E1 | 命中数 | top-1 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| E1：默认（两项关闭） | 约 54.0 ms | 1,461.139 ms | 1.00× | 144 | `49 / 100 / (2034,1306,297,296)` | 精确基线 |
| 仅 Gaussian | 54.359 ms | 1,557.788 ms | 0.94× | 144 | `49 / 100 / (2034,1306,297,296)` | 结果数/top-1 相同，其他分数可能变化 |
| 仅方向直方图 | 53.363 ms | 1,616.390 ms | 0.90× | 144 | `49 / 100 / (2034,1306,297,296)` | 结果数/top-1 相同，其他分数/变体可能变化 |
| Gaussian + 方向直方图 | 56.632 ms | 1,713.718 ms | 0.85× | 144 | `49 / 100 / (2034,1306,297,296)` | 本图未漏检，但分数分布和部分变体已变化 |

本组数据表明，近似预处理在这张图上没有带来速度收益（Gaussian 还增加预处理成本），它们的价值是可能改善噪声场景的召回/稳定性，而不是无条件加速。生产环境应将这些开关作为数据集级配置单独评估；不传参数即采用 E1 的默认精确路径。
