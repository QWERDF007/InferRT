# ShapeTemplateMatcher benchmark

该测试比较 `irt::features::v1::ShapeTemplateMatcherFast`（AVX2）与
`irt::features::v0::ShapeTemplateMatcher`（固定标量基线）。两者使用相同的梯度量化、模板格式、配置和输入；计时范围是 `match()`，不含模板训练。

## 场景

- 源图：640×480 灰度图，固定随机背景和一个 128×128 目标
- 模板：96 个梯度方向特征
- `scan_step=1`，阈值 80，NMS IoU 阈值 0.3
- 命令：`build\benchmark\shape_template\inferrt_benchmark_shape_template.exe --benchmark_min_time=1s`

## 报告

执行基准后，重点比较 `AVX2` 与 `Scalar` 的 `real_time`，加速比为 `Scalar real_time / AVX2 real_time`。此前记录的 128-feature、`scan_step=2` 数据不再适用于当前 v1 评分内核，已移除；请以此版本重新测量。

四模板场景将同一模板注册为四个不同类别，以覆盖模板级并行扫描。结果会受 CPU 型号、编译优化级别、OpenCV 构建、睿频和内存带宽影响，应在同一机器、同一构建配置内比较。真实 2,000 模板单次验证与逐步优化收益见 [示例 README](../../samples/features/shape_template_matching/README.md#v1-真实图优化验证)。
### 本轮优化

v0 保持参考 `shape_based_matching` 风格的标量响应图与逐窗口评分。v1 将评分下沉到 AVX2 kernel：相邻候选批量累加、每 4 个特征以理论上界早停、按源图稀疏响应排序特征，并直接从量化标签用 shuffle 查表。96 features、方向容差 1 的常用配置满足 8-bit 累加上界，v1 因此一次处理 32 个候选；更大配置使用 16-bit 路径。`scan_step=2` 的空间抽样使用 AVX2 shuffle 压缩间隔候选，其他正步长使用 AVX2 gather，只有累计范围超过安全上限时才精确回退到标量路径。多模板匹配仍按 `max_parallelism` 并行扫描（`0` 为自动，`1` 为串行），合并后再执行统一的稳定排序和 NMS。

AVX512/v2 暂未实现；后续仅需新增内核与版本包装器，不需要复制训练、序列化或 NMS。
