# 05: 控制器按需重建并执行查询

**What to build:** DeepLearningTool 用户修改设置后，控制器复用兼容索引或按需先建库再查询，并保持任务可控。

**Blocked by:** 02：硬参数变化明确要求重建

**Status:** ready-for-agent

## Acceptance criteria

- [ ] loadConfig 在预设之后应用用户设置并冻结 job 快照，不吞错退回默认预设。
- [ ] 公共 needsRebuild 与现有图库 scope 条件合并；硬参数改变触发 worker 建库再查，动态参数不触发重建。
- [ ] 控制器不解析索引内部结构、不维护另一份硬参数清单、不无限重试 NOT_READY。
- [ ] 业务 image_id、白名单、resolver、取消、进度和错误展示继续可用。
- [ ] 使用新的安装头文件与库构建消费者，并从实际控制器/UI 验证动态调整、硬参数重建、取消和错误展示。

## Testing decisions

先用既有公共 Interface 编写可失败的行为测试，再实现；使用下一层真实依赖，真实资源烟测验证用户路径。不为字段复制或内部调用次数编写测试。每项独立完成相关文档、消费者与验证，实施进度只更新项目唯一工作账本。

## Contract

依照 [功能 Spec](../04_feature_spec.md) 与 [参数合同](../02_specs.md)，不另行定义覆盖优先级、字段类别或异常语义。本项待实施，ready-for-agent 不代表依赖已完成；仅在阻塞项完成后领取。
