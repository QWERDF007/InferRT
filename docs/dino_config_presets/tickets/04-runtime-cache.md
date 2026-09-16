# 04: 运行参数调整与缓存隔离

**What to build:** 部署方在同进程调整运行资源并复用图库索引，推理实例与缓存随真实依赖正确变化。

**Blocked by:** 02：硬参数变化明确要求重建

**Status:** ready-for-agent

## Acceptance criteria

- [ ] batch/device/backend/precision 调整不因运行参数本身要求图库重建，必要时创建合适推理实例。
- [ ] 实际推理批次反映 batch 设置，而不是仅回显新参数。
- [ ] scan_backend 为调用级强类型配置，迁移 CLI/测试并删除全局 override。
- [ ] 不同索引相同业务 ID 不串图像/特征缓存，extractor 改变后不复用不匹配特征。
- [ ] 真实支持的后端和批量组合完成连续查询；索引未重写，数值使用已有容差，运行错误不静默降级。

## Testing decisions

先用既有公共 Interface 编写可失败的行为测试，再实现；使用下一层真实依赖，真实资源烟测验证用户路径。不为字段复制或内部调用次数编写测试。每项独立完成相关文档、消费者与验证，实施进度只更新项目唯一工作账本。

## Contract

依照 [功能 Spec](../04_feature_spec.md) 与 [参数合同](../02_specs.md)，不另行定义覆盖优先级、字段类别或异常语义。本项待实施，ready-for-agent 不代表依赖已完成；仅在阻塞项完成后领取。
