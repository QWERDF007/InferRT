# DINO 参数预设与索引兼容 Spec

## Problem Statement

调用方需要针对不同图库、缺陷类型和模型持续调优 DINO 区域检索。预设只能提供经过特定数据验证的起点，不能锁定参数。当前平铺配置、索引兼容条件与运行资源复用缺少清晰分工，容易把调整 batch、阈值等操作误判为需要重建，或让真正影响索引特征的变化未经检查就进入查询。

DeepLearningTool 需要直接使用公共配置和兼容检查，而不是自行复制硬参数清单或读取索引内部结构。

## Solution

提供可直接读写的分类配置结构体。默认值、可选 YAML 预设、调用方修改和请求覆盖组成明确的优先级。Manifest 只比较影响索引的有效硬参数；动态参数在下一次查询生效，继续复用索引。

提供统一的只读重建检查；实际查询仍独立执行相同合同校验。控制器掌握图库输入，负责必要时先重建再搜索。索引、推理实例和特征缓存分别决定复用，不相互混淆。

## User Stories

1. As an SDK 调用方, I want 默认构造可修改的配置, so that 不依赖 YAML 也能执行检索。
2. As an 算法调优者, I want 加载经过验证的预设, so that 从可复现基线开始实验。
3. As an SDK 调用方, I want 覆盖预设中的所有有效字段, so that 参数适用于自己的数据。
4. As an SDK 调用方, I want 按职责分类的强类型子结构体, so that 不必阅读长参数列表。
5. As an SDK 调用方, I want 唯一的默认值来源, so that C++ 和 YAML 缺省行为一致。
6. As an 算法调优者, I want 导出最终有效配置, so that 能复现覆盖后的实验。
7. As an SDK 调用方, I want 缺省预设与损坏预设区别处理, so that 配置错误不会被静默默认值掩盖。
8. As an 算法调优者, I want 调整候选数量而不重建, so that 可以比较召回成本。
9. As an 业务调用方, I want 调整阈值而不重建, so that 可以选择业务判定标准。
10. As an 算法调优者, I want 调整评分权重而不重建, so that 可以比较排序效果。
11. As an 算法调优者, I want 调整复核数量而不重建, so that 可以比较复核收益与延迟。
12. As an 业务调用方, I want 请求 Top-K 明确覆盖配置默认值, so that 不受隐藏上限截断。
13. As an 业务调用方, I want 请求 deadline 明确覆盖配置默认值, so that 可以控制本次等待时间。
14. As an 部署维护者, I want 调整 batch 而复用图库索引, so that 可以适配设备容量。
15. As an 部署维护者, I want 切换支持的推理运行时或扫描后端, so that 无需仅为运行资源变化重建索引。
16. As an 部署维护者, I want 移动相同权重文件而复用索引, so that 资源路径不绑定建库机器。
17. As an 模型维护者, I want 独立声明权重版本身份, so that 更换权重会明确要求重建。
18. As an SDK 调用方, I want 模型、切片或描述子硬参数变化时被拒绝, so that 不会查询不兼容的索引。
19. As an SDK 调用方, I want 不兼容错误指出具体差异, so that 可以直接决定重建或修正配置。
20. As an SDK 调用方, I want 缓存命中的索引仍检查当前配置, so that 连续调用不会绕过合同。
21. As an SDK 调用方, I want 不同索引中相同业务 ID 的缓存互不污染, so that 不会使用错误图像特征。
22. As an SDK 调用方, I want 旧索引明确要求重建, so that 不必维护历史迁移逻辑。
23. As an SDK 调用方, I want 损坏索引暴露实际错误, so that 不会被误认为普通参数差异。
24. As an 控制器维护者, I want 公共重建预检查, so that 不必了解索引内部存储。
25. As an DeepLearningTool 用户, I want 动态参数修改后直接查询, so that 调优不用重复建库。
26. As an DeepLearningTool 用户, I want 硬参数或图库范围改变后按需建库再查询, so that 使用正确索引。
27. As an DeepLearningTool 用户, I want 取消、进度和错误信息保持可用, so that 能控制长任务。
28. As an SDK 调用方, I want 每次任务使用配置快照, so that 调参不改变正在运行的请求。
29. As an 算法调优者, I want 关闭紧裁复核仍保留候选定位, so that 消融不会悄悄变成粗选结果。
30. As an 模块维护者, I want 新字段明确归入硬参数或动态参数, so that 算法扩展不会产生遗漏的兼容条件。
31. As an 模块维护者, I want 删除无效旋钮和旧字段别名, so that 对外暴露的参数都有真实用途。
32. As an 实验评估者, I want 预设标明适用证据而不是普遍有效承诺, so that 不会把局部验证误当作全数据保证。

## Implementation Decisions

- DinoRegionSearchConfig 使用 model、gallery_views、descriptors、query_features、coarse_scan、fine_match、decision、runtime、diagnostics 分类值结构；不保留旧平铺字段别名。字段级合同以现有参数规范为唯一明细来源。
- Profile 定位为 Preset；preset_id 只用于展示和溯源，不参与索引兼容或缓存身份。
- 沿用 dinoConfigFromYaml，增加 dinoConfigToYaml 输出最终配置。默认值只定义一次，不额外维护具名预设工厂中的数值副本。
- 覆盖顺序为默认值、YAML、调用方配置修改、请求显式 Top-K/deadline。请求零值使用配置默认，非零 Top-K 不再被默认 final_k 隐藏截断。
- 模型身份、有效光栅、图库视图及描述子编码属于硬参数；查询、评分、阈值和运行资源属于动态参数。权重身份与权重位置分离，不使用绝对路径判定特征空间。
- IndexStore 在现有索引元数据中保存单一 Manifest；内部 IndexContract 集中派生、序列化和比较。格式版本与特征语义版本独立于预设标签，不使用整份配置 hash。
- needsRebuild 为只读公共 Interface；缺失、旧格式或不兼容索引要求重建，非法输入和实际损坏错误抛出。search 自行校验，不依赖调用方先预检查，不隐式建库。
- batch、device、backend、precision 可以触发推理实例变化，但不能仅因此要求图库重建。运行时必须保持相同模型与特征语义；不承诺不同精度逐位一致。
- 扫描后端迁移为调用级配置，删除进程全局 override。索引和 extractor 切换清理相关缓存，动态请求参数不得被旧结果覆盖。
- RegionSearchController 合并兼容性检查与现有图库 scope 条件，保存任务配置快照，由 worker 组织建库及查询。业务 ID、resolver、取消和进度沿用。
- fine_verify_k 为零仅关闭紧裁复核，候选定位仍使用原图，不新增纯粗选替代精排路径。
- 广泛配置结构重构按一次干净切换交付，迁移全部消费者并保持交付可编译，不引入 expand–contract 双轨状态。

## Testing Decisions

- 用户已确认主要测试 seam 为 DinoRegionSearch 公共 Interface：配置加载、建库、重建检查与查询。复用现有 seam，不为私有比较器或缓存实现新增公共测试入口。
- 好测试断言调用方可观察的结果、错误、重建需求和任务状态；不测试字段转发、私有函数调用次序或源代码文字。
- 参考现有 DINO 专项 Google Test 的请求合同、索引写读、过滤和数值比较方式；使用真实索引资源。仅更深层昂贵依赖才允许替身，真实模型烟测保留。
- 不兼容测试覆盖不同硬参数类别、自动值与有效显式值等价、缓存命中后配置改变，以及旧格式与损坏错误的区别。
- 动态调参用同进程连续查询，观察阈值两侧的结果、返回上限、定位/复核阶段和真实批次；验证索引未重写，不以字段回显代替行为。
- 隔离测试使用两个含相同业务 ID 但不同图像的索引；运行后端比较沿用已有数值容差。
- Controller seam 仅验证用户设置覆盖、重建调度、取消、进度和错误展示，不复制 Manifest 算法测试。
- 每个 ticket 自带定向测试、真实烟测及文档收口；不能等待最后一个 ticket 才证明前面工作可用。

## Out of Scope

- 修改区域检索算法目标、替换模型、训练或宣称新的召回质量。
- 内容哈希、索引迁移、增量建库、发布代际、通用配置插件系统。
- 自动重试、吞错默认值、search 中隐式重建。
- 支持修改正在执行的配置对象，或保留新旧字段的永久兼容层。
- 一次性为每个算法参数制作 UI 控件；所有有效参数必须通过 C++ 与 YAML 可访问。
- 发布远程 issue 或建立第二份进度账本。

## Further Notes

用户已批准公共 Interface 测试 seam 和五项 ticket 的粒度与依赖。文档只定义待实施合同，不代表实现或测试已完成。权重身份由调用方正确维护，不宣称能检测同 ID 偷换文件。预设适用范围取决于实际验证数据。

本 Spec 负责用户需求与决策；字段、错误语义及验收矩阵继续集中在 [参数合同](02_specs.md)，不在此复制第二张字段表。任务入口见 [Tickets](03_tickets.md)。本地交付不要求 issue tracker 初始化。
