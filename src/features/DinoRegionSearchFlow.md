# DINO 区域检索 v4 代码流程

方案与契约见 [方案](../../docs/01_design.md)、[Spec](../../docs/02_spec.md)。

| 阶段 | 实际入口/模块 | 数据约定 |
|---|---|---|
| 建库 | Engine → Views → Backbone → Descriptors → IndexStore | 冻结原维度提取，10区域/64局部，投影d维，再INT8存储 |
| 查询 | Engine → Query → RetrievalCore::select | 两档上下文，面积证据，原始D与粗选d分开 |
| 区域粗选 | Scan → SimilarityReducer | 分块精确整体向量比较 |
| 局部粗选 | Scan → matchViewGroup → RetrievalCore::vote | 全部视图先空间打分，再全局Top-K |
| 融合 | Fusion | 各半额度；不并集扩大框；保留对应来源 |
| 定位 | FineMatch → SparseMatcher | 原始D的S图只计算一次，连续ROI几何搜索 |
| 复核 | FineMatch → Backbone | 有界紧裁候选，整体和4×4布局最终打分 |
| 输出 | Engine → Contracts | 原图框、score_kind、分阶段候选、完成状态 |
| 批量 | Sample search-batch | 顺序请求，共用模型、当前索引读取器和缓存 |

`DinoRetrievalCore.hpp` 为生产算法公共实现，独立测试直接编译它。GPU top-2 仍使用实际 `DinoSimilarityCuda.cu`，需要在原工程的 CUDA 环境中验证。

原 `local_view_pool` 不再参与淘汰，旧模板整数短边参数不再控制连续ROI搜索。兼容旧字段不代表旧算法仍在默认主路径中。
