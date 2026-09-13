# DINO 区域搜图：质量优先轻量方案

## 目标与边界
输入查询图像和 ROI，在本地图库返回相似区域、原图坐标、分数和预览。召回与定位质量优先；轻量化只删除发布生命周期防护，不删搜索证据。保留矩形与多边形 ROI、多尺度视图、区域/局部双路粗选、多尺度候选精匹配。

## 复用的实现骨干
- 图像与坐标：`src/features/priv/dino/DinoGeometry.*`, `DinoQuery.*`
- DINOv3 backbone 与多尺度视图：`DinoBackbone.*`, `DinoViews.*`
- 描述与索引：`DinoDescriptors.*`, `DinoIndexStore.*`, `DinoStorageFormat.*`
- 扫描/融合/精匹配：`DinoScan.*`, `DinoFusion.*`, `DinoFineMatch.*`, `DinoSimilarity.*`
- CLI 接线：`samples/features/dino_region_search/SampleDinoRegionSearch.cpp`

## 方案
1. 读取查询图和 ROI（矩形或多边形），以原图像素、半开区间表达坐标，并拒绝非法输入。
2. 对图库生成全图及多尺度切片视图；一次 backbone forward 产出空间 patch token。
3. 写出区域描述、局部描述（按 profile 的 merge 设置）、向量与原图坐标。向量为紧凑二进制数组，元数据和请求/响应为 YAML 文本。
4. 区域和局部通道独立扫描，各保留有限候选，去重后进入精匹配。
5. 对候选执行多尺度模板/局部匹配、边界处理与坐标恢复，排序并 NMS，返回 Top-K。

## 轻量索引生命周期
索引是实现私有的普通本地文件集合；profile 或编码配置变化时删除旧目录并重新 `build`。不引入 SHA-256、内容/配置摘要、manifest、generation、原子发布、增量事务或第二套 JSON/schema 双轨。

## 质量原则
不得用轻量化降低多尺度覆盖、局部通道、候选预算、精匹配或多边形支持。质量必须用真实正例与困难负例评估召回、precision、定位 IoU，并覆盖不同尺度、边界、遮挡、重复纹理和空候选；小规模手测只能作烟测，不能作为唯一验收。

## 不做
HTTP、数据库、分布式检索、训练、登录权限、资源预算发布门禁及 SHA/manifest 发布体系。性能记录实际观察值，不宣称固定 SLA。
