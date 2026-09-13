# 资料来源与设计归属

查阅日期：2026-09-12。以下资料用于确认可用组件和技术背景，不为本项目的速度、存储或准确率目标背书。

| 资料 | 用途 | 本包没有据此声称的内容 |
|---|---|---|
| [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3) | 预训练骨干、patch 特征和模型接口 | 本项目区域召回率、无需验证即可上线 |
| [DINOv3 ViT-S 官方模型卡](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | patch=16、D=384、register 与输入形状 | 固定分辨率能保留任意小目标 |
| [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2) | ViT-S/14 register 对照骨干 | DINOv2/v3 向量空间可混用 |
| [DINOv3 原始论文](https://arxiv.org/abs/2508.10104) | 冻结视觉表示与稠密特征的背景 | 本包压缩与候选融合已得到论文验证 |
| [区域聚合与定位论文](https://arxiv.org/abs/1511.05879) | 复用一张特征图生成区域描述、检索后定位的技术背景 | 直接照搬 CNN pooling 即保证 DINO 的业务质量 |
| [Faiss 索引文档](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) | 理解 Flat、HNSW、SQ 等存储和搜索开销 | 首版必须依赖 Faiss，或任意索引天然零召回损失 |

本包自行设计、需要在真实数据上验证的部分：局部四叉合并、最大半径阈值、两路候选配额、查询格子采样、局部 Top-200 视图筛选与精匹配评分。合并和 INT8 应与未合并 FP32 对照，观察召回、precision 和定位 IoU；速度和存储只记录实测值，不作发布门禁。

最大距离对应点积误差的界来自 Cauchy–Schwarz：单位 q 有 `|q·(f-r)| <= ||f-r||2`。这个关系不约束 Top-K 排序翻转，也不证明 DINO 编码了业务上关注的差异。
