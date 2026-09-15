# DINO 区域检索系统设计与执行流程

本文档是 DINO 区域检索（DINO Region Search）的完整设计方案、算法原理与执行流程说明。

---

## 1. 架构目标与核心设计

DINO 区域检索面向工业缺陷、高分辨率大图及复杂外观形态下的局部区域检索，核心解决高分辨率整图直接前向开销过大、单一全局特征无法感知微小缺陷、以及纯局部特征缺乏空间结构一致性约束的问题。

系统采用**有界两阶段检索架构**：
1. **第一阶段：离线多尺度切片建库与轻量双路粗选**
   - **多尺度分块切片**：不依赖外部目标标注，利用整图 + 512 / 1024 / 2048 像素切片覆盖不同尺度目标与上下文；
   - **紧凑特征表达**：每视图提炼最多 10 条区域池化描述子与最多 64 条局部代表描述子；
   - **无训练确定性投影与量化**：采用确定性伪随机 Hadamard 变换将原始维度（如 D=384）降维至粗选维度（d=96），并以 INT8 紧凑流式存储，将索引体积较稠密图库压缩 45 倍；
   - **双路粗选与全视图姿态投票**：在线检索时通过区域通道（全局语义匹配）与局部通道（CUDA Top-2 相似度归约 + 空间尺度投票）双路召回，保留各半配额生成紧凑粗选候选（默认 64 个）。
2. **第二阶段：原维度连续定位与原图紧裁复核**
   - **标量图连续几何定位**：在候选区域以原始特征维度（D=384）单次计算点积相似度图，驱动连续 ROI 空间搜索（SparseMatcher），直接输出精确连续边界，克服 Patch 包围框引起的 IoU 下降；
   - **原图紧裁加权复核**：对定位排位前列的候选（默认 128 个）从原图提取精确紧裁特征，比对全局面积加权向量与 4×4 网格局部布局特征，实现最终高置信度打分与重排。

---

## 2. 坐标空间与几何契约

1. **统一图像坐标**：
   - 外部传入与输出的矩形一律采用应用 EXIF 后的规范化图像像素坐标（Canonical Image Coordinates）。
   - 矩形坐标统一使用浮点半开区间 `[x0, y0, x1, y1)`，满足 `x0 < x1` 且 `y0 < y1`。
2. **多边形与矩形支持**：
   - 支持轴对齐矩形（Bbox）与任意多边形（Polygon），两者严格互斥。
   - 所有 Patch 证据与特征池化计算均基于几何实际相交面积（Intersection Area），禁止以 Patch 中心点是否落在 ROI 内这种离散方式判定，保证极细长缺陷（长宽比 ≥ 25:1）与狭窄边界不会丢失特征证据。
3. **自身排除语义**：
   - `include_self = false`：排除与查询图像来源相同的整个图像文件；
   - `include_self = true`：允许同图检索，用于检索同图内的其他相似缺陷实例；评估时结合 `exclude_roi` 滤除查询自身的 ROI。

---

## 3. 离线建库流程 (Offline Index Build)

建库流程将工业大图转为轻量级分块可检索索引，执行阶段如下：

```
[原始图像文件]
       │
       ▼
[多尺度视图规划 (DinoViewPlanner)] ── 整图 + 512/1024/2048 切片 (Overlap 25%)
       │
       ▼
[冻结骨干网络前向 (DinoBackbone)] ── 提取原始 D 维 Patch Tokens (NCHW)
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
[区域描述子提炼 (regionWindows)]        [局部代表描述子采样 (select)]
 - 最小有效支持网格                      - 划分最多 4×4 空间格
 - 1 个完整有效窗口 + 3×3 重叠半窗       - 格中心代表 + 最远特征差异代表
 - 面积加权池化 (最多 10 条/视图)        - 最多 64 条/视图 (每格最多 4 条)
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
           [确定性阿达马投影 (Projection)]
            - 补零至 2 的幂 (如 384 → 512)
            - 伪随机正负符号变换与行置乱
            - 快速 Walsh-Hadamard 蝶形变换
            - 子采样截取 d 维 (如 96 维) 并 L2 归一化
                          │
                          ▼
           [INT8 对称量化与元数据打包]
            - 每向量: d 字节 int8 + 8 字节量化缩放 + 32 字节打包元数据
                          │
                          ▼
              [写入磁盘紧凑索引 (IndexStore)]
            - index.yaml, views.npy, offsets.npy, region_vectors.i8, local_vectors.i8
```

### 3.1 区域描述子生成
- 在有效网格内，先提取覆盖全部有效区域的一个最大窗口，再提取 3×3 个半宽、半高的重叠滑动窗口（最多 10 个）。
- 对窗口覆盖的所有 Patch Token 按照其有效像素面积进行加权平均池化，生成能代表局部区域宏观外观的描述子。

### 3.2 局部代表描述子采样
- 为避免纹理复杂区域特征膨胀，将每个视图的 Patch 网格划分为最多 4×4 个空间格。
- 格内第一个代表选取物理位置最靠近格子中心的 Patch；
- 后续代表采用最远点采样（FPS）：$\arg\max_{p} \min_{s \in \text{selected}} \|f(p) - f(s)\|^2$，保留格内特征差异最大的 Patch，每格最多 4 个，单视图局部代表总数固定不超过 64 个。

### 3.3 确定性无码本投影 (Hadamard Projection)
- 粗选降维不依赖外部训练或离线拟合的 PCA / 聚类码本。
- 将原始 D=384 维向量补齐至 512 维，通过预设随机种子的确定性符号反转与行置乱，执行无乘法快速 Walsh-Hadamard 变换，子采样至 d=96 维并归一化。
- 投影仅作用于粗选层；精排定位与复核始终使用原始 D 维特征。

---

## 4. 在线检索全流程 (Online Retrieval)

在线查询接收单个查询图像与查询 ROI，依次执行以下流水线：

```
[查询输入: Query Image + ROI]
       │
       ▼
[1. 查询视图构建 (DinoQuery)]
 - 规划 256 与 448 像素两档长边上下文切片
 - 面积加权选取局部证据 (Evidence Points & Cells)
 - 生成粗选 d 维与原维 D 两套查询特征
       │
       ▼
[2. 分块双路粗选 (DinoScan)]
 ├── 区域通道: 区域描述子分块点积扫描 → 收集 Top-400
 └── 局部通道:
      ├─ CUDA 核函数计算每视图各 Token 的 Top-2 对应代表 (DinoSimilarityCuda)
      ├─ 三档尺度假设 (1/√2, 1, √2)
      ├─ 空间中心累加与空间投票 (Spatial & Scale Voting)
      └─ 计算姿态置信度: 0.60×外观 + 0.25×支持度 + 0.15×中心一致性
       │
       ▼
[3. 候选融合与配额去重 (DinoFusion)]
 - 区域与局部通道各分配一半额度，一路不足由另一路补齐
 - IoU ≥ 0.85 且面积比 ≤ 1.25 的候选合并来源，生成最多 64 个粗选候选
       │
       ▼
[4. 原始维度连续几何定位 (DinoFineMatch::SparseMatcher)]
 - 候选原图裁剪扩边 (1.2×)，单次模型前向提取 D 维特征图 (共享缓存复用)
 - 计算 Query Token 与候选 Patch 的点积相似度标量图 S[t, p]
 - 连续几何尺度扫描 (12 档尺度 × 3 档长宽比 × 1 Patch 步长)
 - 综合定位分数: F_loc = 0.60×外观 + 0.25×覆盖率 + 0.15×空间一致性
 - 保留空间峰值并执行 81 假设网格连续微调 (Refinement)
 - 同图 NMS (0.85)，选取前 verify_k (默认 128) 进入复核
       │
       ▼
[5. 原图紧裁结构复核 (DinoFineMatch::tightScore)]
 - 对查询与定位候选分别以 1.04 扩边紧裁提取精确特征图
 - 计算 ROI 面积加权全局向量与 4×4 局部网格池化特征
 - 最终置信度打分: F_final = 0.60×全局余弦 + 0.40×加权网格余弦
 - 最终结果 NMS (0.50)，截取 Top-K 输出
```

---

## 5. 核心算法细节

### 5.1 空间与尺度姿态投票 (Spatial Pose Voting)
设查询局部证据在原图的坐标为 $x$，匹配到的图库代表在原图的坐标为 $y$，查询 ROI 中心为 $r$。
图库与查询视图的每 Patch 像素分辨率比值定义基准尺度：
$$s_{\text{base}} = \frac{\sqrt{g_{px} \cdot g_{py}}}{\sqrt{q_{px} \cdot q_{py}}}$$
系统评估三档尺度假说 $s \in s_{\text{base}} \times \{2^{-0.25}, 1, 2^{0.25}\}$。每个特征匹配对预测目标中心：
$$c = y + s \cdot (r - x)$$
空间投票在离散量化网格（Bin 边长约 2 个 Patch 尺寸）内累积：
- **外观相似度 (Appearance)**：当前 Bin 内所有命中证据的相似度加权均值；
- **空间支持度 (Support)**：当前 Bin 覆盖的查询格总权重占比；
- **中心一致性 (Agreement)**：各证据预测中心相对于加权均值中心的均方根偏差；
综合姿态评分计算公式：
$$\text{Score}_{\text{pose}} = 0.60 \times \text{Appearance} + 0.25 \times \text{Support} + 0.15 \times \text{Support} \times \text{Agreement}$$

### 5.2 稀疏定位器 (SparseMatcher) 与连续 ROI 搜索
传统定位直接输出 Patch 块整数边界，对于细长目标会导致 IoU 严重下降。v4 引入连续 ROI 搜索：
1. **标量图预计算**：每个查询视图仅计算一次 $S[t, p] = q_t \cdot g_p$ 矩阵，并在 CPU 端构建 $3 \times 3$ 邻域最大响应图与位置索引。
2. **多尺度与长宽比搜索**：
   - 连续短边从 $0.75 \text{ patch}$ 起步，按 $\sqrt{2}$ 递增，扫描最多 12 档尺度；
   - 长宽比在 ROI 原始比例基础上叠加 $\{1/\sqrt{2}, 1, \sqrt{2}\}$；
   - 滑动步长为 1 个 Patch，包含最后一行和最后一列的合法边界。
3. **空间极值细化 (Refinement)**：
   - 对前列峰值假设，尝试中心偏移 $\Delta x, \Delta y \in \{-0.5, 0, 0.5\}$ 与宽高缩放系数 $\{2^{-0.25}, 1, 2^{0.25}\}$ 共 81 种连续几何扰动，寻找完整评分最高的连续矩形。

### 5.3 原图紧裁复核 (Tight-Crop Verification)
在定位阶段，高重叠候选或背景相似区域可能会排在前面。复核阶段对定位输出执行“精准手术”式复核：
1. **紧裁提取**：候选框以极小扩边系数（1.04×）从原图提取，直接对齐目标边界，消除多余背景干扰。
2. **两级特征比对**：
   - **全局特征**：ROI 内部 Patch 面积加权池化向量点积；
   - **4×4 网格布局**：将 ROI 空间划分为 16 个小格，分别计算格内面积加权特征，评估内部结构排列的一致性。
3. **最终打分**：
   $$F_{\text{final}} = 0.60 \times \cos_{01}(\text{Global}) + 0.40 \times \sum_{i=1}^{16} w_i \cos_{01}(\text{Cell}_i)$$
   若设置 `verify_k = 0`，系统执行消融模式，直接以定位阶段分数 $F_{\text{loc}}$ 输出。

---

## 6. 代码模块与代码落点索引

| 流程环节 | 核心类 / 函数 | 源码文件 | 职责与关键逻辑 |
| :--- | :--- | :--- | :--- |
| **对外接口** | [`DinoRegionSearch`](include/inferrt/features/DinoRegionSearch.hpp) | [`include/inferrt/features/DinoRegionSearch.hpp`](include/inferrt/features/DinoRegionSearch.hpp) | `build`、`search`、`searchBatch` 顶层公共门面 |
| **编排核心** | [`DinoEngine`](priv/dino/DinoEngine.cpp) | [`priv/dino/DinoEngine.cpp`](priv/dino/DinoEngine.cpp) | 串联查询提取、扫描、融合、定位与复核全生命周期 |
| **算法内核** | [`Projection`](priv/dino/DinoRetrievalCore.hpp#L36), [`select`](priv/dino/DinoRetrievalCore.hpp#L91), [`vote`](priv/dino/DinoRetrievalCore.hpp#L159), [`SparseMatcher`](priv/dino/DinoRetrievalCore.hpp#L224) | [`priv/dino/DinoRetrievalCore.hpp`](priv/dino/DinoRetrievalCore.hpp) | 纯算法实现（阿达马投影、代表采样、姿态投票、连续定位） |
| **CUDA 加速**| `dinoSimilarityReduceViewGroupCuda` | [`priv/dino/DinoSimilarityCuda.cu`](priv/dino/DinoSimilarityCuda.cu) | GPU 视图分组 Top-2 相似度归约核函数 |
| **多尺度规划**| [`DinoViewPlanner`](priv/dino/DinoViews.cpp) | [`priv/dino/DinoViews.cpp`](priv/dino/DinoViews.cpp) | 图库切片与查询上下文切片规划及坐标双向映射 |
| **扫描调度** | [`dinoScan`](priv/dino/DinoScan.cpp) | [`priv/dino/DinoScan.cpp`](priv/dino/DinoScan.cpp) | 区域扫描、局部块级读取与全视图多线程姿态投票 |
| **双路融合** | [`dinoFuseCandidates`](priv/dino/DinoFusion.cpp) | [`priv/dino/DinoFusion.cpp`](priv/dino/DinoFusion.cpp) | 双通道配额分配、去重合并与来源标记 |
| **定位复核** | [`dinoFineMatch`](priv/dino/DinoFineMatch.cpp), `describeTight` | [`priv/dino/DinoFineMatch.cpp`](priv/dino/DinoFineMatch.cpp) | 原维标量图定位、批量紧裁前向与 4×4 网格复核打分 |
| **紧凑存储** | [`DinoIndexWriter`](priv/dino/DinoIndexStore.cpp), [`DinoIndexReader`](priv/dino/DinoIndexStore.cpp) | [`priv/dino/DinoIndexStore.cpp`](priv/dino/DinoIndexStore.cpp) | 流式写入/读取 `.i8` 向量、量化参数与 Packed 元数据 |
| **配置契约** | [`DinoRegionSearchConfig`](priv/dino/DinoProfile.cpp) | [`priv/dino/DinoProfile.cpp`](priv/dino/DinoProfile.cpp) | YAML Profile 解析、合法性校验与序列化 |
| **CLI 入口**  | `SampleDinoRegionSearch` | [`samples/features/dino_region_search/SampleDinoRegionSearch.cpp`](../../samples/features/dino_region_search/SampleDinoRegionSearch.cpp) | `build`、`search`、`search-batch` 命令行入口 |

---

## 7. 资源控制与运行时预算

1. **显存预算 (GPU VRAM)**：
   - 骨干网络（DINOv3 ViT-S/16 或 DINOv2 ViT-S/14）常驻 GPU，前向批量分别为 4 或 1。
   - CUDA 扫描核函数仅分配视图块归约缓冲，显存峰值控制在 4 MiB 以内，无整库 GPU 常驻开销。
2. **主机内存预算 (Host RAM)**：
   - 索引数据采用按块流式读取（默认每次 32,768 个描述子），避免一次性将全库解压进内存；
   - 内置图像缓存（256 MiB）与候选密集特征缓存（256 MiB），基于 LRU 策略自动回收。
3. **超时控制 (Deadline)**：
   - 查询请求支持 `deadline_ms`。在粗选分块扫描、定位每档尺度扫描及复核批次边界严格检查截止时间，超时即安全熔断并标记 `status: Incomplete`，杜绝服务挂死。
