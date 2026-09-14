# DINO 区域检索流程

本文维护「模块边界 + 唯一信息索引」。算法、字段与门槛的权威定义在：对外契约见
[`docs/dino_region_search_v1/02_spec.md`](../../docs/dino_region_search_v1/02_spec.md)，质量与验收见
[`docs/dino_region_search_v1/03_test_acceptance.md`](../../docs/dino_region_search_v1/03_test_acceptance.md)。

## 1. 入口与边界

- 公共 API 与对外类型：[`DinoRegionSearch.hpp`](include/inferrt/features/DinoRegionSearch.hpp)。
  核心入口为 `build` 与 `search`。
- 门面转发：[`DinoRegionSearch.cpp`](DinoRegionSearch.cpp)。
- 编排层：[`priv/dino/DinoEngine.hpp`](priv/dino/DinoEngine.hpp)。
- CLI：[`samples/features/dino_region_search`](../../samples/features/dino_region_search/README.md)。
- 契约解析与序列化（YAML profile / query / result / report）：
  [`priv/dino/DinoContracts.hpp`](priv/dino/DinoContracts.hpp) 与 [`priv/dino/DinoProfile.hpp`](priv/dino/DinoProfile.hpp)。

## 2. 模块职责

| 模块 | 职责 | 关键不变量 |
| --- | --- | --- |
| [`DinoTypes.hpp`](priv/dino/DinoTypes.hpp) | 矩形/仿射/视图/网格/描述/候选基础类型 | 坐标为半开区间；缩放与偏移一律经矩阵换算 |
| [`DinoGeometry.*`](priv/dino/DinoGeometry.hpp) | 面积、交集、IoU、ROI 校验、patch 权重、NMS | bbox 用矩形覆盖面积，polygon 用相交面积，不按中心点判定 |
| [`DinoViews.*`](priv/dino/DinoViews.hpp) | 多尺度图库视图与查询上下文视图的规划与光栅渲染 | 每个视图存 `canonical_to_input`、`valid_input_rect`、patch 网格；padding 不参与描述 |
| [`DinoBackbone.*`](priv/dino/DinoBackbone.hpp) | 冻结骨干适配与进程级模型复用 | 排除 CLS/register；patch 数等于 `Hf*Wf`；配置变化替换缓存实例 |
| [`DinoDescriptors.*`](priv/dino/DinoDescriptors.hpp) | 多尺度窗口池化、四叉相邻合并、INT8 编码 | 合并距离在原始 FP32 归一化特征上度量；叶节点受最大边长约束；不丢弃有效证据 |
| [`DinoQuery.*`](priv/dino/DinoQuery.hpp) | 查询视图、ROI 平均描述、格子局部选择 | 每格最多两条描述且只计一份格子权重；无效描述不伪造 |
| [`DinoTopK.*`](priv/dino/DinoTopK.hpp) | 有界 Top-K 收集 | 内存与 K 成正比，不构造全量分数矩阵 |
| [`DinoScan.*`](priv/dino/DinoScan.hpp) | 分块穷举扫描、区域/局部双通道与在线候选归约 | 不使用 ANN；紧凑描述按有界块读取；GPU 只返回块内视图/token 最大值，宿主端维护有界 Top-K |
| [`DinoSimilarity.*`](priv/dino/DinoSimilarity.hpp) | 描述子块与查询 token 的相似度归约 | 标量/AVX2 为 CPU 参考与回退；CUDA 在 kernel 内解码 INT8 并计算点积 |
| [`DinoFusion.*`](priv/dino/DinoFusion.hpp) | 双路配额、轮流取用与空间去重 | 重复只计来源；同图不同位置不合并 |
| [`DinoFineMatch.*`](priv/dino/DinoFineMatch.hpp) | 候选内多尺度模板匹配、峰值细化、覆盖率与打分 | 覆盖率取自原始 ROI 模板的互为近邻；尺度按最终分数择优 |
| [`DinoIndexStore.*`](priv/dino/DinoIndexStore.hpp) | 本地紧凑数组读写与流式读取 | 直接写入目标目录，生成 `index.yaml` 与紧凑向量/元数据数组；无 manifest/generation 冗余 |
| [`DinoStorageFormat.*`](priv/dino/DinoStorageFormat.hpp) | 小端数组、npy 容器、32 B packed meta | meta 为 packed 布局；npy 容器头带 shape |
| [`DinoPaths.*`](priv/dino/DinoPaths.hpp) | `fs::path` 与契约文本之间的唯一转换入口 | 契约路径一律 UTF-8；统一用 `/` 分隔 |
| [`DinoTime.hpp`](priv/dino/DinoTime.hpp) | 截止时间与阶段计时 | 阶段边界与批次之间检查 deadline |
| [`DinoFeatureCache.*`](priv/dino/DinoFeatureCache.hpp) | 候选密集特征的字节 LRU | key 由图像身份、extractor signature 与 crop 组成 |
| [`DinoSearchCache.*`](priv/dino/DinoSearchCache.hpp) | 同一 extractor 下进程级缓存 bundle 生命周期 | 图像缓存与特征缓存独立受固定进程级字节上限约束
| [`DinoProfile.*`](priv/dino/DinoProfile.hpp) | YAML profile 解析、序列化与校验 | 权威配置入口，校验语义合法性，无 JSON 双轨 |
| [`DinoContracts.*`](priv/dino/DinoContracts.hpp) | 请求/响应 YAML 序列化与枚举字符串转换 | 请求支持 bbox/polygon 二选一，响应输出统一 YAML 格式 |

## 3. 磁盘布局

本地索引根目录包含：
- `index.yaml`：图像身份列表、特征维度与量化标记。
- `views.npy`：每个视图的几何规划参数。
- `offsets.npy`：每个视图对应的区域描述与局部描述起始偏移。
- `region_vectors.i8` / `region_scales.f32`（量化 profile；FP32 profile 使用 `region_vectors.f32`）：紧凑区域描述向量与缩放因子。
- `region_meta.npy`：区域描述空间元数据（32 字节 packed meta）。
- `local_vectors.i8` / `local_scales.f32`（量化 profile；FP32 profile 使用 `local_vectors.f32`）：紧凑局部描述向量与缩放因子。
- `local_meta.npy`：局部描述空间元数据（32 字节 packed meta）。

## 4. 端到端顺序

1. `build`：扫描图库 → 解码与计算身份 → 规划视图 → 骨干提取 patch token → 生成区域与局部描述 → 量化 → 写入本地索引数组与 `index.yaml`。
2. `search`：解码查询与 ROI（矩形或多边形）→ 构造查询视图 → 区域通道扫描 + 局部通道视图级聚合 → 视图窗口重打分与配额融合 → 候选按来源图分组，命中缓存复用网格，否则批量精匹配 → 同图 NMS → 按 profile 判定语义输出统一 YAML。
   deadline 到期返回 `incomplete`，不伪装为 `completed` 或 `no_match`。

## 5. 验证入口

- 单元测试：[`tests/features/TestDinoRegionSearch.cpp`](../../tests/features/TestDinoRegionSearch.cpp)
  （契约、几何、视图覆盖、池化、合并、INT8、相似度归约、Top-K、融合、NMS、缓存）。
- 开发集与阈值标定：[`tools/dino_region_devkit.py`](../../tools/dino_region_devkit.py)。
- 真实资源冒烟：见 [`samples/features/dino_region_search/README.md`](../../samples/features/dino_region_search/README.md)。
