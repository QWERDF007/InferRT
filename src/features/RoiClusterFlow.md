# RoiCluster 文档索引

RoiCluster 的方案、运行时设计、实现总结和源码包范围集中在 [RoiCluster 文档](../../docs/roi_cluster/README.md)。本文件不复制第二套算法说明。

- [方案与当前状态](../../docs/roi_cluster/01_plan.md)
- [设计与数学契约](../../docs/roi_cluster/02_design.md)
- [实现、测试范围与限制](../../docs/roi_cluster/03_implementation.md)
- [源码包范围](../../docs/roi_cluster/04_package.md)

公共入口：[RoiCluster.hpp](include/inferrt/features/RoiCluster.hpp)；实现：[RoiCluster.cpp](RoiCluster.cpp)、[RoiClusterImpl.cpp](priv/RoiClusterImpl.cpp)；共享 ROI 特征：[RoiFeatureExtractor.cpp](priv/RoiFeatureExtractor.cpp)；聚类算子：[HDBSCAN.cpp](../ops/HDBSCAN.cpp)。

字段和默认值以公共头文件为准。本文档没有声明本次任务执行过端到端模型或聚类质量测试。
