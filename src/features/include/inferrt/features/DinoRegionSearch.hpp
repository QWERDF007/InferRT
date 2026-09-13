#pragma once

/**
 * @file DinoRegionSearch.hpp
 * @brief 冻结 DINO 骨干的区域检索公共 API：配置、请求/响应契约与四个核心入口。
 *
 * 该模块用标注区域在未标注图库中检索外观与局部结构相似的区域。索引一次建立、
 * 增量更新，查询只计算查询图与有限候选，不训练或微调网络。
 *
 * 坐标约定：所有对外坐标都在 **EXIF 已应用的 canonical image** 上，使用浮点半开区间
 * ``[x0, y0, x1, y1)``。入口函数失败时抛出 ``irt::Exception``；返回值中的 status/decision
 * 表达“是否完成”与“是否有匹配”的语义，二者不可互相代替。
 */

#include <inferrt/features/Export.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/ModelRuntime.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/** @brief 查询完成状态。 */
enum class DinoSearchStatus
{
    Completed,  ///< 全部候选完成精算。
    Incomplete, ///< 超时或部分候选失败；结果只是部分结果。
    Failed,     ///< 查询整体失败，没有可用结果。
};

/** @brief 判定语义；阈值未配置时不得输出有/无匹配结论。 */
enum class DinoSearchDecision
{
    RankedOnly, ///< 未配置有效判定阈值，只给出排序。
    Matches,    ///< 至少一个结果超过已冻结阈值。
    NoMatch,    ///< 在本次索引与已验证 profile 内未检出超阈值结果。
    Incomplete, ///< 未完成，无法给出判定。
};

/** @brief 一致性模式，决定精匹配阶段的一致性严格程度。 */
enum class DinoConsistencyMode
{
    Appearance, ///< 外观模式：宽松一致性，强调外观近似。
    Instance,   ///< 实例模式：更严格的局部空间一致性。
};

/** @brief CLI 退出码契约。 */
enum class DinoExitCode : int
{
    Success          = 0, ///< 完成。
    InvalidArgument  = 2, ///< 参数、ROI 或 profile 错误。
    ResourceMissing  = 3, ///< 权重或索引缺失、不兼容。
    BuildFailed      = 4, ///< 建库含失败文件或预算超限。
    QueryIncomplete  = 5, ///< 查询未完成。
    InternalError    = 6, ///< 内部错误。
};

/** @brief 建库/更新阶段。 */
enum class DinoBuildStage
{
    Unknown,             ///< 未初始化。
    ScanningImages,      ///< 扫描图库并计算内容身份。
    LoadingModel,        ///< 加载冻结骨干。
    ExtractingViews,     ///< 逐批次提取视图特征并生成描述。
    Quantizing,          ///< INT8 量化与误差统计。
    WritingIndex,        ///< 写入 staging generation。
    Finalizing,          ///< 校验摘要并原子切换 active。
};

/** @brief 建库进度事件。 */
struct DinoBuildProgress
{
    DinoBuildStage stage{DinoBuildStage::Unknown};
    size_t         processed_count{0};
    size_t         total_count{0};
    std::string    message{};
};

using DinoBuildProgressCallback = std::function<void(const DinoBuildProgress &)>;

/** @brief 查询阶段。 */
enum class DinoSearchStage
{
    Unknown,
    Decode,
    QueryExtract,
    RegionScan,
    LocalScan,
    LocalWindowRescore,
    Fusion,
    FineExtract,
    FineMatch,
    Output,
};

/** @brief 查询进度事件；阶段边界必须可观测。 */
struct DinoSearchProgress
{
    DinoSearchStage stage{DinoSearchStage::Unknown};
    size_t          processed_count{0};
    size_t          total_count{0};
};

using DinoSearchProgressCallback = std::function<void(const DinoSearchProgress &)>;

/** @brief 矩形 ROI，单位 canonical 像素。 */
struct DinoSearchRect
{
    float x0{0.0F};
    float y0{0.0F};
    float x1{0.0F};
    float y1{0.0F};
};

/** @brief 多边形顶点。 */
struct DinoSearchPoint
{
    float x{0.0F};
    float y{0.0F};
};

/** @brief 查询区域：必须恰好提供 bbox 或多边形之一。 */
struct DinoSearchRoi
{
    bool                        has_bbox{false};
    bool                        has_polygon{false};
    DinoSearchRect              bbox{};
    std::vector<DinoSearchPoint> polygon{};
};

/**
 * @brief 区域检索 profile（配置）唯一来源。
 *
 * 该结构同时是索引签名与查询配置。任何影响建库编码的字段变化都会改变 profile hash，
 * 必须重建索引；纯查询字段变化只改变检索 profile hash，可在同一 generation 上引用并记录。
 */
struct INFERRT_FEATURES_API DinoRegionSearchConfig
{
    /// profile 名称，用于报告与发布记录。
    std::string profile_id{"development"};

    /// 冻结骨干名称；DINOv3 与 DINOv2 各自建立独立索引，不能混用向量。
    std::string model_name{"dinov3_vits16"};

    /// 骨干权重或已构建 engine 文件路径。
    std::filesystem::path weights_file{};

    /// 期望的权重文件 SHA-256；非空时在加载骨干前校验，不匹配直接拒绝。
    std::string weights_sha256{};

    /// 推理运行目标（后端 + 设备）。
    irt::model::ModelRuntime model_runtime{};

    /// 骨干构建精度；量化误差单独统计，不依赖该字段。
    irt::model::ModelPrecision model_precision{irt::model::ModelPrecision::FP32};

    /// 骨干输入光栅边长；0 表示按 patch 推导为 spec DEFAULT 的 512/518。
    int encoder_edge{0};

    /// 骨干推理批量上限；建库与精匹配共用。
    size_t model_batch_size{4};

    /// 标准适用范围：图像边长下限。
    int validated_min_image_edge{256};

    /// 标准适用范围：图像边长上限。
    int validated_max_image_edge{4096};

    /// 标准适用范围：目标短边下限（canonical 像素）。
    double validated_min_target_short_px{64.0};

    /// 标准适用范围：目标长宽比上限（1:4～4:1）。
    double validated_max_target_aspect{4.0};

    /// 图库切片源边长尺度。
    std::vector<int> gallery_tile_edges{512, 1024, 2048};

    /// 切片重叠率。
    double view_overlap{0.25};

    /// 区域整体描述的窗口边长比例（相对视图短边）。
    std::vector<double> region_window_ratios{1.0, 0.5, 0.25};

    /// 区域窗口步长比例（相对窗口边长）。
    double region_window_stride_ratio{0.5};

    /// 窗口有效面积占比下限，低于该值的窗口不入库。
    double region_min_valid_fraction{0.5};

    /// 是否启用空间相邻合并；false 为不合并基线。
    bool merge_enabled{true};

    /// 合并误差上限（原始 FP32 归一化特征上的最大替换距离）。
    double merge_epsilon{0.10};

    /// 叶节点最大边长（patch 数），防止大范围同质背景用单点代表。
    int max_leaf_side_patches{4};

    /// 是否使用 INT8 紧凑存储；false 时使用 FP32 参考。
    bool quantize_int8{true};

    /// 区域通道候选额度。
    size_t region_topk{400};

    /// 局部通道每个视图保留的标量分数数量。
    size_t local_view_topk{200};

    /// 每个通道进入融合的最大候选数。
    size_t channel_candidate_limit{400};

    /// 粗选候选总数。
    size_t coarse_k{100};

    /// 粗选去重：空间 IoU 阈值。
    double coarse_dedup_iou{0.85};

    /// 粗选去重：面积比上限。
    double coarse_dedup_area_ratio{1.25};

    /// 最终返回结果数上限。
    size_t final_k{20};

    /// 查询 ROI 在模型输入中的目标长边个数。
    std::vector<double> query_roi_target_lengths{128.0, 256.0, 448.0};

    /// 查询 ROI 划分的格子边长（4 表示 4x4）。
    int query_local_cells{4};

    /// 每格最多贡献的局部描述数。
    int query_local_max_per_cell{2};

    /// 有效局部描述数量下限；低于该值标记局部证据不足。
    int query_min_local_evidence{4};

    /// 候选裁剪扩边比例。
    double fine_candidate_expand{1.2};

    /// 模板短边起始 patch 数。
    int fine_template_min_short_patches{2};

    /// 模板尺寸档位递增比例（DEFAULT sqrt(2)）。
    double fine_template_scale_step{1.4142135623730951};

    /// 单一候选的模板尺寸上限。
    int fine_template_max_sizes{12};

    /// 每候选保留的空间峰值数量。
    int fine_peaks_per_candidate{3};

    /// 峰值细化的有界轮数。
    int fine_refinement_rounds{1};

    /// 模板匹配余弦阈值（开发集冻结值）。
    double fine_match_cosine_threshold{0.55};

    /// 归一化位置容差（开发集冻结值）。
    double fine_position_tolerance{0.25};

    /// 最终框 NMS 的 IoU 阈值。
    double fine_nms_iou{0.5};

    /// template_similarity 权重。
    double score_weight_template{0.60};

    /// query_coverage 权重。
    double score_weight_coverage{0.25};

    /// spatial_consistency 权重。
    double score_weight_consistency{0.15};

    /// 一致性模式。
    DinoConsistencyMode consistency_mode{DinoConsistencyMode::Appearance};

    /// 是否启用最终判定阈值；false 时为 ranked_only。
    bool enable_decision_threshold{false};

    /// 判定阈值；仅在 ``enable_decision_threshold`` 为 true 时生效。
    double decision_threshold{0.0};

    /// 判定阈值标定 ID；必须能在发布记录中追溯。
    std::string decision_calibration_id{};

    /// 索引稳态字节预算。
    uint64_t index_budget_bytes{4ULL * 1024ULL * 1024ULL * 1024ULL};

    /// 单查询 wall deadline（毫秒）。
    int64_t query_deadline_ms{30000};

    /// 紧凑向量扫描块大小。
    size_t region_scan_block{32768};

    /// 候选密集特征缓存上限。
    uint64_t dense_feature_cache_bytes{256ULL * 1024ULL * 1024ULL};

    /// 原图解码缓存上限。
    uint64_t image_cache_bytes{256ULL * 1024ULL * 1024ULL};


    /// 评测：粗选命中要求的最小真值覆盖率。
    double evaluation_coarse_min_gt_coverage{0.90};

    /// 评测：粗选命中允许的最大面积比。
    double evaluation_coarse_max_area_ratio{16.0};

    /// 评测：最终结果框的 IoU 命中阈值。
    double evaluation_final_iou_threshold{0.5};

    /// 评测：需要报告召回率的候选数档位。
    std::vector<int> evaluation_ks{20, 50, 100};

    /// 评测：可复现随机种子。
    int64_t evaluation_seed{20260912};

    /**
     * @brief 校验配置的语义合法性。
     * @throws irt::Exception 组合非法或数值越界时抛出。
     */
    void validate() const;

    /** @brief 返回解析后的骨干输入边长（考虑 ``encoder_edge`` 的自动取值）。 */
    int resolvedEncoderEdge(int patch_size) const;

};

/** @brief 查询请求。 */
struct DinoSearchRequest
{
    std::string              request_id{};
    std::filesystem::path    query_path{};
    DinoSearchRoi            roi{};
    size_t                   top_k{0};             ///< 0 表示使用 profile 的 ``final_k``。
    bool                     include_self{false};  ///< 是否允许返回同内容副本。
    std::string              profile_id{};         ///< 为空时使用 profile 自身的 ``profile_id``。
};

/** @brief 单条检索结果。 */
struct DinoSearchResult
{
    std::string   image_id{};      ///< 稳定内容身份 ID。
    std::string   source_path{};   ///< 图库图像路径。
    DinoSearchRect bbox{};         ///< canonical 空间结果框。
    float         score{0.0F};     ///< final_score，非概率。
    float         template_similarity{0.0F};
    float         query_coverage{0.0F};
    float         spatial_consistency{0.0F};
    std::vector<std::string> coarse_sources{}; ///< 命中的粗选通道来源。
};

/** @brief 阶段耗时（毫秒）。 */
struct DinoSearchTimings
{
    double decode_ms{0.0};
    double query_extract_ms{0.0};
    double region_scan_ms{0.0};
    double local_scan_ms{0.0};
    double local_window_rescore_ms{0.0};
    double fusion_ms{0.0};
    double fine_extract_ms{0.0};
    double fine_match_ms{0.0};
    double output_ms{0.0};
    double queue_ms{0.0};
    double wall_ms{0.0};
};

/** @brief 查询诊断计数。 */
struct DinoSearchDiagnostics
{
    size_t   scanned_region_descriptors{0};
    size_t   scanned_local_descriptors{0};
    size_t   retained_local_views{0};
    size_t   region_candidates{0};
    size_t   local_candidates{0};
    size_t   fused_candidates{0};
    size_t   view_count{0};
    size_t   model_forwards{0};
    size_t   cache_hits{0};
    size_t   query_local_descriptors{0};
    size_t   query_valid_cells{0};
    bool     low_local_evidence{false};
    bool     outside_validated_profile{false};
    uint64_t peak_rss_bytes{0};
    uint64_t gpu_allocated_peak_bytes{0};   ///< 相似度归约器占用的设备显存峰值。
    uint64_t gpu_reserved_delta_peak_bytes{0}; ///< 进程显存占用相对查询起点的峰值增量。
    std::string similarity_backend{};       ///< 局部通道实际使用的归约后端。
    bool     local_scan_compact{false};     ///< 是否走紧凑（INT8）扫描路径。
    std::string query_local_selection{};
};

/** @brief Candidate exposed for recall evaluation without private index identifiers. */
struct DinoCoarseCandidate
{
    std::filesystem::path source_path{};
    DinoSearchRect        bbox{};
};

/** @brief Query response. */
struct DinoSearchResponse
{
    std::string                   request_id{};
    DinoSearchStatus              status{DinoSearchStatus::Failed};
    DinoSearchDecision            decision{DinoSearchDecision::Incomplete};
    size_t                        completed_candidates{0};
    size_t                        total_candidates{0};
    std::vector<DinoSearchResult> results{};
    std::vector<DinoCoarseCandidate> region_candidates{};
    std::vector<DinoCoarseCandidate> local_candidates{};
    std::vector<DinoCoarseCandidate> coarse_candidates{};
    DinoSearchTimings             timings{};
    DinoSearchDiagnostics         diagnostics{};
    std::string                   message{};
};

/** @brief 建库报告中的单图记录。 */
struct DinoImageRecord
{
    std::string    image_id{};
    std::string    source_path{};
    int            width{0};
    int            height{0};
    int            view_count{0};
    int            region_descriptor_count{0};
    int            local_descriptor_count{0};
    uint64_t       original_patch_count{0};
    uint64_t       quantized_bytes{0};
};

/** @brief 建库/更新报告。 */
struct DinoBuildReport
{
    bool          ready_with_errors{false};
    size_t        image_count{0};
    size_t        failed_image_count{0};
    size_t        view_count{0};
    size_t        region_descriptor_count{0};
    size_t        local_descriptor_count{0};
    uint64_t      original_patch_count{0};
    uint64_t      index_bytes{0};
    uint64_t      temporary_peak_bytes{0};
    double        duration_ms{0.0};
    std::vector<DinoImageRecord> images{};
    std::vector<std::string>     failed_files{};
    std::vector<std::string>     messages{};
};

/** @brief 评测模式，决定指标的可解释范围。 */
enum class DinoEvaluationMode
{
    FullAnnotation, ///< 完整标注子图库，可报告完整区域 recall。
    JudgedPool,     ///< 只核验过候选池，只能报告 judged-pool 指标。
};

/** @brief 单个门槛的结果；未测时 value 为 null 且 verified 为 false。 */
struct DinoGateResult
{
    std::string name{};
    bool        measured{false};
    bool        passed{false};
    double      value{0.0};
    double      threshold{0.0};
    std::string note{};
};


/**
 * @brief 强制指定局部通道的相似度归约后端，用于 T13 的参考/优化差异报告与对照实验。
 *
 * @param backend ``"auto"``（默认，有可用 GPU 用 GPU）、``"cpu"`` 或 ``"cuda"``；
 *                非 ``"auto"`` 时请求的后端不可用会退回 CPU。
 * @throws irt::Exception 名称非法时抛出。
 */
INFERRT_FEATURES_API void dinoSetScanBackendOverride(const std::string &backend);

/** @brief Parse documented YAML profile text. */
INFERRT_FEATURES_API DinoRegionSearchConfig dinoConfigFromYaml(const std::string &text);

/** @brief Parse YAML search request text. */
INFERRT_FEATURES_API DinoSearchRequest dinoSearchRequestFromYaml(const std::string &text);

/** @brief Serialize search response as YAML. */
INFERRT_FEATURES_API std::string dinoSearchResponseToYaml(const DinoSearchResponse &response);

/** @brief Serialize build report as YAML. */
INFERRT_FEATURES_API std::string dinoBuildReportToYaml(const DinoBuildReport &report);

/**
 * @brief 冻结 DINO 骨干的区域检索引擎（无请求状态，进程级共享模型与缓存）。
 *
 * 四个入口共享同一份 profile、同一套坐标与错误语义；模型、图像与候选特征缓存按配置隔离，CLI 与可选 HTTP
 * 适配都调用这里。
 */
class INFERRT_FEATURES_API DinoRegionSearch
{
public:
    /**
     * @brief 从图库目录建立新 generation 索引。
     *
     * @param gallery_root 图库根目录；递归扫描受支持图片。
     * @param config 检索 profile。
     * @param index_root 索引根目录；为空时使用 ``gallery_root`` 的上层默认目录。
     * @param progress_callback 可选进度回调。
     * @return 建库报告；含失败文件时 ``ready_with_errors`` 为 true，抛出前先写出诊断。
     * @throws irt::Exception 权重缺失、配置非法或索引预算超限时抛出。
     */
    static DinoBuildReport build(const std::filesystem::path &gallery_root, const DinoRegionSearchConfig &config,
                                 const std::filesystem::path &index_root = {},
                                 const DinoBuildProgressCallback &progress_callback = {});


    /**
     * @brief 在索引中检索与查询区域相似的区域。
     *
     * @param index_root 索引根目录。
     * @param request 查询请求。
     * @param config 检索 profile（查询字段参与本次检索并计入检索 profile hash）。
     * @param progress_callback 可选进度回调。
     * @return 查询响应；未完成时 status 为 ``Incomplete`` 且保留部分结果。
     * @throws irt::Exception 索引缺失、profile 不兼容或请求非法时抛出。
     */
    static DinoSearchResponse search(const std::filesystem::path &index_root, const DinoSearchRequest &request,
                                     const DinoRegionSearchConfig &config,
                                     const DinoSearchProgressCallback &progress_callback = {});

};

} // namespace irt::features
