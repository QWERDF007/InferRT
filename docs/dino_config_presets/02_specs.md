# 参数与索引兼容规范

## S1 配置 Interface

目标主结构体由下列值类型组合，均允许外部直接读写；嵌套结构均使用 `Dino` 前缀的公共强类型。build/search 保持完整配置参数，不展开子结构体为长参数列表。

```cpp
struct DinoRegionSearchConfig {
    std::string preset_id{}; // 展示/溯源标签，不参与兼容判断
    DinoModelConfig model;
    DinoGalleryViewsConfig gallery_views;
    DinoDescriptorConfig descriptors;
    DinoQueryFeaturesConfig query_features;
    DinoCoarseScanConfig coarse_scan;
    DinoFineMatchConfig fine_match;
    DinoDecisionConfig decision;
    DinoRuntimeConfig runtime;
    DinoDiagnosticsConfig diagnostics;
    void validate() const;
};
```

沿用现有公开加载函数 `dinoConfigFromYaml`；新增 `dinoConfigToYaml(const DinoRegionSearchConfig &)` 输出有效配置文本，不增加同义入口。YAML 键与目标结构层次对应；结构体成员初始化是默认值唯一来源，解析器只覆盖出现的字段。加载器检查语法和字段类型，最终 build/search 对合并后的有效配置执行必要语义校验，允许先加载不含本机权重路径的预设再补路径。

```cpp
auto config = dinoConfigFromYaml(yaml_text);
config.model.weights_file = weights_path;
config.model.weights_id = "dinov3-industrial-weights-01";
config.coarse_scan.coarse_k = 120;
config.fine_match.fine_verify_k = 32;
config.decision.enable_decision_threshold = true;
config.decision.decision_threshold = 0.82;
config.runtime.model_batch_size = 2;
// build/search 接收 config；不依据 preset_id 恢复预设值。
```


## S2 字段归属与生命周期

以下列出现有字段的完整迁移分类。H 表示持久化索引硬参数，D 表示不参与索引兼容；字段名除明确删除/替换外保持原名，只移动到子结构体。

| 子结构体 | 字段 | 类别 |
|---|---|---|
| model | model_name、encoder_edge（比较解析后的值） | H |
| model | 新增 weights_id：调用方声明的权重版本身份 | H |
| model | weights_file：本机权重或 engine 路径 | D |
| gallery_views | gallery_tile_edges、view_overlap | H |
| descriptors | region_window_ratios、region_window_stride_ratio、region_min_valid_fraction、coarse_dimension、local_representatives、merge_enabled、merge_epsilon、max_leaf_side_patches、quantize_int8 | H |
| query_features | query_roi_target_lengths、query_local_cells、query_local_max_per_cell、query_min_local_evidence | D |
| coarse_scan | region_topk、channel_candidate_limit、coarse_k、coarse_dedup_iou、coarse_dedup_area_ratio、final_k | D |
| fine_match | fine_verify_k、fine_candidate_expand、fine_template_scale_step、fine_template_max_sizes、fine_peaks_per_candidate、fine_refinement_rounds、fine_match_cosine_threshold、fine_nms_iou、consistency_mode | D |
| fine_match | score_weight_template、score_weight_coverage、score_weight_consistency | D |
| decision | enable_decision_threshold、decision_threshold | D |
| runtime | model_runtime、model_precision、model_batch_size、query_deadline_ms、region_scan_block | D |
| runtime | 新增强类型 scan_backend：Auto/Cpu/Cuda | D |
| diagnostics | validated_min_image_edge、validated_max_image_edge、validated_min_target_short_px、validated_max_target_aspect | D，仅诊断 |
| 主结构体 | profile_id 改为 preset_id | D |

删除当前注释已说明无效的 `local_view_topk`、`fine_template_min_short_patches`、`fine_position_tolerance`：实施前检查实际调用，迁移解析器、预设、测试和消费者，不把无效参数包装成可调参数。其他字段也须以实际算法消费为准：发现仅被解析但没有作用的字段，删除而非假装支持。

所有 D 参数都可不重建索引地变化；改变后必须真正影响其所属阶段。运行后端、精度变化允许数值差异，不承诺排序逐位一致。不同运行时必须实现相同模型、预处理和输出 token 语义，否则属于特征合同变化而非“硬件替换”。

权重路径不等于权重身份。`weights_id` 在正式 build/search 时非空，同一权重迁移路径保持 ID，不同权重必须改 ID；不计算大文件 hash，也不声称能检测调用方同 ID 偷换权重。新模型字段默认、示例配置和 Controller 设置必须共同覆盖该项。

## S3 请求覆盖

DinoSearchRequest 保留 query_path、query_image_id、ROI、allowed_image_ids、include_self、image_resolver 等请求数据，不再携带重复的 preset/profile 身份。

保留已有请求 top_k/deadline_ms 便捷覆盖：0 表示使用 config；非零 top_k 直接作为本次返回上限，不再被 config.final_k 隐藏截断；实际结果仍受候选数量约束。非零 deadline_ms 覆盖 runtime.query_deadline_ms。

优先级：C++ 默认值 < YAML 预设 < 调用方修改配置 < 请求中这两个显式覆盖。没有其他隐式覆盖源。

## S4 索引 Manifest

index.yaml 增加 `manifest` 映射，由 IndexStore 写入并读取一个内部强类型 IndexContract。比较逻辑唯一，建库、search 和预检查共享，不序列化/比较整个 config。

Manifest 包含：

- schema_version：本次新增合同格式版本；
- feature_version：预处理、token 选取、投影或描述子计算语义的版本；
- S2 所有 H 字段；
- 从实际模型/预处理派生的 patch_size、token_dimension、有效 encoder_edge 与预处理描述。

不写入权重绝对路径、模型运行时、batch、preset_id、阈值或整份动态配置作为兼容条件。现有索引中 dimensions/quantization 等元数据与合同使用同一有效值来源，不各算一份。

encoder_edge=0 与显式填写其解析值相等。浮点按序列化可往返的数值比较；数组保留顺序，不擅自排序改变算法行为。为简单可预测，全部 H 字段均参与比较，即使某参数在当前算法分支未启用也要求一致。

缺失 Manifest 的旧索引直接要求重建，不从数组猜测模型身份。硬参数或版本不一致：NOT_READY，错误包含差异字段、索引值、请求值和“重建索引”；底层损坏/读取失败保留实际错误，不冒充普通参数差异。

search 每次使用当前 config 校验合同，即使 reader 是缓存命中也必须校验；不得只在首次打开索引时比较。

提供一个公共预检查 Interface：

```cpp
static bool needsRebuild(
    const std::filesystem::path &index_root,
    const DinoRegionSearchConfig &config);
```

不存在索引、旧格式或硬参数不一致返回 true；参数非法、文件损坏及实际 I/O 错误抛异常。只读，不创建索引、不提取图库特征。有效参数派生复用模型描述逻辑，不通过完整图库推理获取合同。search 仍必须自行校验，不能依赖调用方一定预检查。

build 保持显式重建语义，不偷偷跳过构建。是否重建由掌握图库输入的控制器决定。

## S5 运行实例与缓存

索引复用、推理实例复用、特征缓存复用分开处理：

- batch/device/backend/precision 改变：索引复用；按现有 DinoBackboneRegistry 创建匹配实例，不复用容量不合适的 engine。
- query/coarse/fine/decision 改变：下一次 search 使用当前值，不能复用旧请求的候选、分数或最终结果。
- 图像/特征缓存保留有界实现；缓存键覆盖它所缓存结果的真实依赖，不能用 preset_id 作为键。
- extractor 或索引切换时重置相关缓存，避免两个索引相同业务 ID 互相污染；不引入全局缓存框架。
- 将扫描后端选择从进程级 override 迁入 runtime.scan_backend，迁移 CLI/测试调用，删除旧全局入口，避免一个调用方影响另一个。
- 输入配置按调用快照使用；禁止修改正在运行的配置对象。硬参数热改意味着后续需要重建，不是原地变更现有索引。

`fine_verify_k=0` 只关闭紧裁复核，候选定位仍需加载原图并提取特征，因此正常 search 始终需要候选 resolver。不能通过关闭复核跳过定位或返回粗选冒充精排结果。

## S6 消费者与扩展

RegionSearchController 的 loadConfig 保留预设加载，但移除“加载失败吞错后使用默认值”。设置/UI 覆盖一次性应用到配置快照，缺省预设与显式损坏预设必须区分。通过 needsRebuild 合并现有图库 scope 变化条件，再由 worker 建库；NOT_READY 不做无条件无限重试。业务 image_id、resolver、取消和进度链路保持。

新增字段流程：先确定实际消费者与 H/D 类别，再定义唯一默认值、YAML 映射和消费者设置。H 字段进入 IndexContract；D 字段证明更改后索引复用且新值生效。算法语义变化更新 feature_version，纯默认值或预设调整不自动升级算法版本。没有当前用途的扩展接口不预建。

## S7 验收矩阵

1. 默认构造和加载预设均能被外部修改；配置 YAML 往返保留有效值，遗漏字段来自 C++ 默认值。
2. 使用真实索引逐类修改模型身份、encoder、切片、描述子与量化参数，预检查要求重建；直接 search 抛 NOT_READY 并指出具体差异。
3. 保持 weights_id 只迁移权重文件路径，索引可复用；更换 weights_id 要求重建。
4. 同进程连续修改 top_k、阈值、verify_k、分数权重、batch 和扫描后端；索引文件未重写，响应/模型调用体现当前参数。threshold 使用已知分数两侧的阈值，batch 观察实际推理批次而非字段转发。
5. reader 已缓存时再改硬参数，仍拒绝；切换相同业务 ID 的不同索引不会取到对方缓存。
6. verify_k=0 仍有定位结果，verify_k>0 完成紧裁复核；缺少原图不能伪装为 completed。
7. 新配置安装给 DeepLearningTool 后真实执行“改动态参数直接查、改硬参数先建后查”；取消与进度正常。
8. 旧索引要求重建，损坏索引暴露实际错误，无新旧字段双轨兼容。

永久测试仅覆盖可出错的合同和缓存边界；真实模型/Controller 验证使用已有资源，不引入纯字段复制测试。数值比较允许既有 CPU/CUDA 容差；不能以性能参数热调要求所有分数逐位不变。
