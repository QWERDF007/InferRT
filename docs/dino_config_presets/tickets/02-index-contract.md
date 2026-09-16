# 02: 硬参数变化明确要求重建

**What to build:** 使用相同公共合同完成建库、只读重建检查和直接查询，让调用方明确知道何时必须重建。

**Blocked by:** 01：分类配置与预设自由覆盖

**Status:** ready-for-agent

## Acceptance criteria

- [ ] 新索引持久化有效硬参数和合同版本；动态参数不参与兼容条件。
- [ ] 模型/权重身份、有效光栅、切片或描述子变化使 needsRebuild 返回 true，直接 search 抛 NOT_READY 并指出差异。
- [ ] 自动光栅与其显式有效值兼容；相同权重身份仅迁移路径不要求重建。
- [ ] reader 缓存命中仍校验；旧格式要求重建，损坏和 I/O 错误保留实际错误。
- [ ] build、预检查、search 共用唯一合同；search 不隐式建库；用真实索引通过公共 Interface 验证。

## Testing decisions

先用既有公共 Interface 编写可失败的行为测试，再实现；使用下一层真实依赖，真实资源烟测验证用户路径。不为字段复制或内部调用次数编写测试。每项独立完成相关文档、消费者与验证，实施进度只更新项目唯一工作账本。

## Contract

依照 [功能 Spec](../04_feature_spec.md) 与 [参数合同](../02_specs.md)，不另行定义覆盖优先级、字段类别或异常语义。本项待实施，ready-for-agent 不代表依赖已完成；仅在阻塞项完成后领取。
