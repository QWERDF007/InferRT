# YAML 配置、请求与结果样例

本目录说明[轻量规格](../02_spec.md)中已实现的 YAML 契约，并提供可直接改路径使用的样例。样例中的绝对路径须替换为本地路径，分数和框是虚构示例；真实质量结论见[验收记录](../acceptance.yaml)。

| 文件 | 用途 |
|---|---|
| [profile.yaml](profile.yaml) | 开发配置；保留多尺度、区域/局部双路粗选与候选精匹配，参数仍需真实数据验证 |
| [query.yaml](query.yaml) | 矩形请求，`bbox` 为原图像素半开区间 `[x0, y0, x1, y1]` |
| [query_polygon.yaml](query_polygon.yaml) | 多边形请求，顶点采用同一原图坐标系；与 `bbox` 二选一 |
| [result.yaml](result.yaml) | 完成查询后的排序结果；相似度分数不是概率 |
| [result_incomplete.yaml](result_incomplete.yaml) | 截止时间到达时的部分结果；不得解释为全库无匹配 |

配置、请求和响应只保留 YAML 文本格式；索引向量仍使用紧凑二进制数组。无需为使用这些样例运行专用文档校验器。

示例采用 DINOv3-S，patch=16、输入长边 512。更换 DINOv2-S register 对照时应同时改模型名、patch=14、输入长边 518 和权重路径，并重建独立索引；不同骨干的向量不能混用。模型依据见[参考资料](../references/sources.md)。

保留既有合并和 INT8 开发参数，不以轻量化改变搜索算法。质量对照分别保存实际 YAML 配置：关闭合并并使用 FP32；启用合并但仍用 FP32；启用合并和 INT8。其余视图、候选和精匹配参数保持一致，比较召回、precision、定位 IoU，不能仅凭文件更小判定通过。完整要求见[真实质量验收](../03_test_acceptance.md)。

`decision.threshold: null` 表示仅排序；只有真实开发集完成阈值选择后，完成查询才可报告 `matches` 或 `no_match`。未完成查询始终报告 `incomplete`，不能用空结果或未测阈值宣称无匹配。
