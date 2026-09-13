# DINO 区域搜图开发包

本目录是本地质量优先的轻量搜图方案：DINOv3 backbone、多尺度视图、矩形/多边形 ROI、区域与局部双路粗选、多尺度候选精匹配及 CLI。

## 单一索引
- [设计](01_design.md)：目标、流程、边界与质量原则
- [规格](02_spec.md)：YAML 请求/响应、二进制向量和错误契约
- [验证](03_test_acceptance.md)：真实数据质量与功能验收
- [Tickets](tickets/README.md)：T01–T07 执行顺序与源码落点
- [示例](examples/README.md)：YAML 文本及向量格式说明

YAML 是配置、请求、响应的唯一文本格式；向量使用紧凑二进制数组。不保留 JSON/schema 双轨，也不引入 SHA-256、manifest、generation、摘要、增量事务或发布门禁。profile/编码变化直接删除旧索引后重新 `build`；这不表示运行时已经迁移。

轻量化不降低质量：多尺度双路检索、矩形和多边形支持、精匹配必须通过真实正例/困难负例的召回、precision 和定位 IoU 验证。小规模手测只能作为 smoke check。
