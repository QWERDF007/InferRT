# ShapeTemplateMatcher benchmark

该测试比较 `irt::features::ShapeTemplateMatcher`（AVX2）与 `irt::features::scalar::ShapeTemplateMatcher`（不使用 SIMD intrinsic）。两者使用相同的梯度量化、模板格式、配置和输入；计时范围是 `match()`，不含模板训练。

## 场景

- 源图：640×480 灰度图，固定随机背景和一个 128×128 目标
- 模板：128 个梯度方向特征
- `scan_step=2`，阈值 80，NMS IoU 阈值 0.3
- 命令：`build\benchmark\shape_template\inferrt_benchmark_shape_template.exe --benchmark_min_time=1s`

## 报告

执行基准后，将 Google Benchmark 输出粘贴在此处；重点比较 `AVX2` 与 `Scalar` 的 `real_time`，加速比为 `Scalar real_time / AVX2 real_time`。

| 场景 | 实现 | real_time | CPU time | 相对 AVX2 |
| --- | --- | ---: | ---: | ---: |
| 单模板 | AVX2 | 5.15 ms | 5.08 ms | 1.00× |
| 单模板 | Scalar | 8.59 ms | 8.41 ms | 1.67× slower |
| 四模板、自动并行 | AVX2 | 5.04 ms | 1.89 ms | 1.00× |
| 四模板、自动并行 | Scalar | 9.11 ms | 5.84 ms | 1.81× slower |

测量于 2026-07-12（Release、MSVC 19.44、OpenCV 4.8.0、28 个逻辑 CPU），每个基准以 `--benchmark_min_time=1s` 重复 3 次，表中取 Google Benchmark 的均值。四模板场景将同一模板注册为四个不同类别，以覆盖模板级并行扫描；它相对于串行线性外推的收益会受 CPU 调度、睿频和内存带宽影响，应以同机重复测量为准。
### 本轮优化

参考 `shape_based_matching` 的候选过滤思想：评分过程中计算未访问特征可达到的理论上界；上界低于阈值即停止。通过阈值的窗口仍完成全部特征累加，因此匹配分数与原始公式一致。此外，仅为当前类别过滤后的模板实际使用的方向生成响应图。AVX2 与 Scalar 共享这两项算法优化，响应图生成继续分别使用 AVX2 与标量内核。多模板匹配会按 `max_parallelism` 并行扫描（`0` 为自动，`1` 为串行），合并后再执行统一的稳定排序和 NMS。

结果会受 CPU 型号、编译优化级别、OpenCV 构建和睿频策略影响，报告仅应在同一机器、同一构建配置内比较。