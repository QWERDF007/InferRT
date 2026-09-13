# DINO 区域搜图：轻量规格

## 输入与坐标
请求 YAML 包含 `query_path`、`bbox: [x0,y0,x1,y1]` 或 `polygon: [[x,y],...]`（二选一）、`top_k`、可选 `include_self` 与 `deadline_ms`。坐标为原图像素、半开区间；多边形按有效覆盖区域参与池化和匹配。

## 配置与格式
profile 使用 YAML，由现有 yaml-cpp 入口解析。配置包括模型、`views`（全图/切片/重叠）、`regions`、区域与局部候选配额、`fine`（扩边和模板尺度）及查询截止时间。请求和响应也是 YAML 文本；向量及数组使用紧凑二进制格式，不维护 JSON、schema 或摘要包副本。编码配置变化后删除索引并重新建库，不计算 SHA 或配置摘要。

## 建库契约
每张图片生成全图和多尺度视图；一次 backbone 提取空间 patch token，写出区域描述、局部描述（按 profile 的 merge 设置）、视图/图片标识及原图坐标。索引缺失或不可用由 CLI 报错，不能静默返回空结果。

## 查询契约
区域与局部两路独立扫描并保留有限候选；候选合并去重后执行多尺度精匹配，处理边界与坐标恢复，再排序和 NMS。分数是相似度，不是概率。响应 YAML 约定为：

```yaml
status: completed # 或 incomplete / error
decision: ranked_only # 校准 threshold 后可为 matches / no_match；incomplete 时只能为 incomplete
results:
  - source_path: /path/image.jpg
    bbox: [x0, y0, x1, y1]
    score: 0.0
```

只有 `completed` 且使用开发集校准过 `decision.threshold` 时，`decision` 才能表达 `matches` 或 `no_match`；`incomplete` 绝不能伪装成 `no_match`。非法参数、图片解码失败、ROI 越界、索引缺失返回 `error` 及明确错误信息，不用空数组掩盖错误。`include_self` 按请求明确排除或允许查询自身图片。

## 验证契约
质量优先于存储与发布便利。必须以真实标注正例和困难负例验证：区域召回率、precision、定位 IoU；覆盖矩形/多边形、尺度变化、边界、遮挡、重复纹理、多区域和空候选。至少报告数据划分、样本数量、阈值/Top-K、聚合方式及失败案例。3～10 张图片、3 个 ROI 的手测仅是 smoke check，不构成验收；不设置 SHA、manifest、generation 或发布门禁。

## 实现约束
复用 `src/features/priv/dino/` 中现有图像读取、坐标变换、backbone、描述、扫描、融合、精匹配和 YAML 组件；相同逻辑只保留一份，不在每个文件堆 helpers。CLI 入口为 `samples/features/dino_region_search/SampleDinoRegionSearch.cpp`。
