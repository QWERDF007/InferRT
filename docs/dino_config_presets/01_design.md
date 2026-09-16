# DINO 参数与预设设计

本目录是待实施的参数重构规范，不代表功能已实现。

- [规范与字段归属](02_specs.md)：唯一的目标参数合同、Manifest 规则和验收标准。
- [实施任务](03_tickets.md)：五项纵向切片、依赖、交付与验证。
- [功能 Spec](04_feature_spec.md)：问题、用户故事、实现与测试决策，按 to-spec 模板组织。

## 设计决策

Profile 是预设模板，不是不可变配置，也不是索引身份。默认构造、YAML 加载和调用方覆盖最终都产生一个普通的 `DinoRegionSearchConfig` 值；build/search 只使用这个值，不再次加载 profile 或覆盖调用方设置。

按职责提供强类型子结构体；不保留旧平铺字段、引用别名或双向同步访问器。接口仍接收一个完整配置对象，不为每个字段增加方法参数，不引入参数注册框架或继承树。

参数组织与索引兼容分类是两件事。例如 `model` 既含决定特征空间的模型身份，也含可改变的运行资源位置。不能简单把某个子结构体整体序列化后比较。

## 与当前实现的关系

现有公共配置已经允许直接赋值，并非封闭黑盒。本次重点是分类、完整的索引硬参数合同、覆盖优先级与实际运行期生效。

源码入口：

- [公共配置](../../src/features/include/inferrt/features/DinoRegionSearch.hpp)
- [YAML 与校验](../../src/features/priv/dino/DinoProfile.cpp)
- [建库与查询](../../src/features/priv/dino/DinoEngine.cpp)
- [模型实例复用](../../src/features/priv/dino/DinoBackbone.cpp)
- [索引持久化](../../src/features/priv/dino/DinoIndexStore.cpp)
- [ImageSearch Manifest](../../src/features/priv/ImageSearchImpl.cpp)
- [RoiSearch Manifest](../../src/features/priv/RoiSearchImpl.cpp)

借鉴后两者的元数据比较方式，但不照搬字段清单：ImageSearch 当前也记录 batch、运行后端和路径，不适合作为 DINO 的硬参数清单。

## 深度封装

DinoRegionSearch 模块对外暴露配置、建库、查询和一个兼容性检查入口；内部负责有效参数解析、索引合同、运行实例和缓存复用。控制器不读取 index.yaml，不复制参数差异判断。

单一 index.yaml 同时承载现有索引元数据和新的 manifest 节点，不创建旁路 manifest 文件。保存有效硬参数并直接比较，不引入完整配置 hash 或“版本哈希”。格式版本与特征算法版本只表达实际存储/算法变化。

硬参数不一致时抛出 NOT_READY，告知重建。search 不隐式建库：它没有完整图库输入，也不应在交互查询中执行不可见的长任务。DeepLearningTool 在任务启动前检查，由现有 worker 组织重建。

## 预设管理

现有 samples 下的 YAML 是具名预设来源，不再写一套 `presetIndustrialDefect()` 数值副本。直接 C++ 构造获得合理默认值；加载 YAML 叠加缺省值；随后调用方自由覆盖。只有实际有对应数据验证证据的组合才能标注“已验证”，不能暗示适用所有图库。

调参或算法扩展应更新目标预设和实验记录，而不是修改已经存在的索引身份。实验报告可以保存最终有效配置作为运行产物，不用另建配置管理系统。

## 消费者

集成目标：`F:/Projects/DeepLearningTool/src/feature/RegionSearchController.cpp`。

当前该文件已有 `loadConfig()`、`DinoRegionSearch::build/search`、业务 ID 白名单和 resolver 接入；本次不是重新设计 ID 映射。目标流程：加载可选预设 → 应用设置/UI 覆盖 → 在 job 中保存配置快照 → 检查索引合同 → 必要时重建 → 搜索。

动态修改指下一次调用使用新配置，不是跨线程原地修改正在运行的 job。图库内容/范围变更继续由现有业务 scope 管理，不能把“参数兼容”误认为“图库完整且最新”。

## 不做的事情

不新增自动重试、无声降级、完整文件哈希、发布代际、旧格式迁移、通用插件系统或无限兼容层。保留必要参数合法性校验；错误直接抛出，不吞错回退到默认 profile。
