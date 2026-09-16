#pragma once

/**
 * @file DinoRegionSearch.hpp
 * @brief 冻结 DINO 骨干的区域检索公共 API：配置、请求/响应契约与四个核心入口。
 *
 * 该模块用标注区域在未标注图库中检索外观与局部结构相似的区域。索引一次建立，查询只计算查询图与有限候选，
 * 不训练或微调网络。
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
#include <optional>
#include <string>
#include <vector>

namespace irt::features {

/** @brief 协作式取消与运行控制。 */
struct DinoOperationControl
{
    std::function<bool()> cancelled{};
};

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
    Matches,    ///< 至少一个结果超过已配置阈值。
    NoMatch,    ///< 完成查询但没有超过已配置阈值的结果。
    Incomplete, ///< 未完成，无法给出判定。
    Error,      ///< 查询输入或索引错误。
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
    BuildFailed      = 4, ///< 建库含失败文件。
    QueryIncomplete  = 5, ///< 查询未完成。
    InternalError    = 6, ///< 内部错误。
};

/** @brief 建库阶段。 */
enum class DinoBuildStage
{
    Unknown,             ///< 未初始化。
    ScanningImages,      ///< 扫描图库并计算图像身份。
    LoadingModel,        ///< 加载冻结骨干。
    ExtractingViews,     ///< 提取视图特征并生成描述。
    Quantizing,          ///< INT8 量化与误差统计。
    WritingIndex,        ///< 写入本地索引数组。
    Finalizing,          ///< 完成 index.yaml 与数组写入。
};

/** @brief 建库进度事件（对齐 ImageSearch / RoiSearch 批次规范）。 */
struct DinoBuildProgress
{
    DinoBuildStage stage{DinoBuildStage::Unknown}; ///< 当前构建阶段。
    size_t         batch_index{0};                 ///< 从 0 开始的已完成批次编号。
    size_t         batch_begin{0};                 ///< 当前批次第一张图库图片的下标。
    size_t         batch_count{0};                 ///< 当前批次包含的图库图片数量。
    size_t         processed_count{0};             ///< 当前阶段累计已处理的工作单元数量。
    size_t         total_count{0};                 ///< 当前阶段需要处理的工作单元总数；不可度量时为 0。
    std::string    message{};                      ///< 阶段或文件描述信息。
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

/** @brief 查询进度事件（阶段与批次边界可观测）。 */
struct DinoSearchProgress
{
    DinoSearchStage stage{DinoSearchStage::Unknown}; ///< 当前查询阶段。
    size_t          batch_index{0};                  ///< 当前阶段内从 0 开始的批次编号。
    size_t          batch_begin{0};                  ///< 当前批次起始下标。
    size_t          batch_count{0};                  ///< 当前批次包含的项目/候选数量。
    size_t          processed_count{0};              ///< 当前阶段累计已处理数量。
    size_t          total_count{0};                  ///< 当前阶段总工作单元数量；不可度量时为 0。
    std::string     message{};                       ///< 附加信息。
};

using DinoSearchProgressCallback = std::function<void(const DinoSearchProgress &)>;

/**
 * @brief DINO 区域检索图库输入条目。
 */
struct DinoImageItem
{
    int64_t               image_id{0}; ///< 调用方提供的图像唯一 ID。
    std::filesystem::path image_path;  ///< 图像文件路径。
};

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


/** @brief 局部通道与紧凑向量扫描的计算后端。 */
enum class DinoScanBackend
{
    Auto, ///< 自动选择：优先使用 CUDA，不可用时回退 CPU。
    Cpu,  ///< 强制使用 CPU 归约。
    Cuda, ///< 强制使用 CUDA 归约；无可用设备时回退 CPU。
};

/** @brief 骨干模型与输入光栅配置。 */
struct DinoModelConfig
{
    std::string           model_name{"dinov3_vits16"}; ///< 冻结骨干名称（如 dinov3_vits16 / dinov2_vits14_reg4）。
    std::filesystem::path weights_file{};             ///< 骨干权重或 engine 路径（动态路径，不影响索引兼容）。
    std::string           weights_id{"default"};      ///< 调用方声明的权重版本身份（参与索引契约）。
    int                   encoder_edge{0};            ///< 骨干输入光栅边长；0 表示按 patch 推导。
};

/** @brief 图库切片与多尺度视图配置。 */
struct DinoGalleryViewsConfig
{
    std::vector<int> gallery_tile_edges{512, 1024, 2048}; ///< 切片尺度源边长。
    double           view_overlap{0.25};                  ///< 切片重叠率。
};

/** @brief 描述子生成与量化配置。 */
struct DinoDescriptorConfig
{
    std::vector<double> region_window_ratios{1.0, 0.5}; ///< 区域窗口边长比例。
    double              region_window_stride_ratio{0.5}; ///< 区域窗口步长比例。
    double              region_min_valid_fraction{0.5};  ///< 窗口有效面积占比下限。
    int                 coarse_dimension{96};            ///< 粗排描述子维度。
    int                 local_representatives{64};       ///< 局部代表 token 上限。
    bool                merge_enabled{true};             ///< 是否启用空间相邻合并。
    double              merge_epsilon{0.10};             ///< 合并误差上限。
    int                 max_leaf_side_patches{4};        ///< 叶节点最大边长（patch 数）。
    bool                quantize_int8{true};             ///< 是否使用 INT8 紧凑量化存储。
};

/** @brief 查询特征提取配置。 */
struct DinoQueryFeaturesConfig
{
    std::vector<double> query_roi_target_lengths{256.0, 448.0}; ///< 查询 ROI 目标边长。
    int                 query_local_cells{4};                  ///< 查询局部空间网格划分。
    int                 query_local_max_per_cell{2};           ///< 每格局部描述上限。
    int                 query_min_local_evidence{4};           ///< 有效局部描述数量下限。
};

/** @brief 粗排扫描与候选融合配置。 */
struct DinoCoarseScanConfig
{
    size_t region_topk{400};             ///< 区域通道候选额度。
    size_t channel_candidate_limit{400}; ///< 每通道进入融合的最大候选数。
    size_t coarse_k{64};                 ///< 粗选候选总数。
    double coarse_dedup_iou{0.85};       ///< 粗选空间去重 IoU。
    double coarse_dedup_area_ratio{1.25};///< 粗选空间去重面积比上限。
    size_t final_k{20};                  ///< 最终返回结果数默认上限。
};

/** @brief 精匹配与紧裁复核配置。 */
struct DinoFineMatchConfig
{
    size_t              fine_verify_k{64};                   ///< 紧裁复核候选数上限（0 表示关闭紧裁复核）。
    double              fine_candidate_expand{1.2};          ///< 候选裁剪扩边比例。
    double              fine_template_scale_step{1.4142135623730951}; ///< 模板尺寸递增步长。
    int                 fine_template_max_sizes{12};         ///< 模板尺寸档位上限。
    int                 fine_peaks_per_candidate{3};         ///< 每候选空间峰值数。
    int                 fine_refinement_rounds{1};           ///< 峰值细化轮数。
    double              fine_match_cosine_threshold{0.55};   ///< 模板匹配余弦阈值。
    double              fine_nms_iou{0.5};                   ///< 最终结果 NMS IoU 阈值。
    DinoConsistencyMode consistency_mode{DinoConsistencyMode::Appearance}; ///< 一致性模式。
    double              score_weight_template{0.60};         ///< 模板相似度权重。
    double              score_weight_coverage{0.25};         ///< 查询覆盖率权重。
    double              score_weight_consistency{0.15};      ///< 空间一致性权重。
};

/** @brief 判定与业务阈值配置。 */
struct DinoDecisionConfig
{
    bool   enable_decision_threshold{false}; ///< 是否启用最终判定阈值。
    double decision_threshold{0.0};          ///< 判定阈值。
};

/** @brief 运行资源与推理环境配置。 */
struct DinoRuntimeConfig
{
    irt::model::ModelRuntime   model_runtime{};                             ///< 推理运行目标（后端与设备）。
    irt::model::ModelPrecision model_precision{irt::model::ModelPrecision::FP32}; ///< 推理精度。
    size_t                     model_batch_size{4};                         ///< 推理批量大小。
    int64_t                    query_deadline_ms{30000};                    ///< 单查询截止时间（毫秒）。
    size_t                     region_scan_block{32768};                    ///< 紧凑扫描块大小。
    DinoScanBackend            scan_backend{DinoScanBackend::Auto};         ///< 扫描计算后端。
};

/** @brief 诊断与适用范围规格配置。 */
struct DinoDiagnosticsConfig
{
    int    validated_min_image_edge{256};        ///< 适用图像边长下限。
    int    validated_max_image_edge{4096};       ///< 适用图像边长上限。
    double validated_min_target_short_px{64.0};  ///< 适用目标短边下限。
    double validated_max_target_aspect{4.0};     ///< 适用目标长宽比上限。
};

/**
 * @brief 区域检索配置主结构体（强类型子结构体组合）。
 */
struct INFERRT_FEATURES_API DinoRegionSearchConfig
{
    std::string             preset_id{"development"}; ///< 预设/展示标签（不参与索引兼容判断）。
    DinoModelConfig         model{};
    DinoGalleryViewsConfig  gallery_views{};
    DinoDescriptorConfig    descriptors{};
    DinoQueryFeaturesConfig query_features{};
    DinoCoarseScanConfig    coarse_scan{};
    DinoFineMatchConfig     fine_match{};
    DinoDecisionConfig      decision{};
    DinoRuntimeConfig       runtime{};
    DinoDiagnosticsConfig   diagnostics{};

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
    std::string                         request_id{};
    std::filesystem::path               query_path{};
    int64_t                             query_image_id{-1};  ///< 可选：查询图像自身 ID（用于 include_self: false 时排除自身，-1 表示无匹配 ID）。
    DinoSearchRoi                       roi{};
    size_t                              top_k{0};            ///< 0 表示使用 config 的 ``coarse_scan.final_k``。
    bool                                include_self{false}; ///< 是否允许返回查询自身对应的图像。
    std::optional<std::vector<int64_t>> allowed_image_ids{std::nullopt}; ///< 允许参与检索的图像 ID 白名单；nullopt 表示不过滤；空集合表示范围为空。
    int64_t                             deadline_ms{0};      ///< 请求级 wall deadline；0 表示使用 runtime 配置。
    std::function<std::filesystem::path(int64_t)> image_resolver{}; ///< 可选：图像 ID 到文件路径的解析函数（用于精排与紧裁复核阶段加载候选原图）。
};

/** @brief 单条检索结果。 */
struct DinoSearchResult
{
    int64_t        image_id{0};     ///< 命中的图库图像 ID（调用方提供的 int64_t 标识）。
    DinoSearchRect bbox{};          ///< canonical 空间结果框。
    float          score{0.0F};     ///< final_score，非概率。
    float          template_similarity{0.0F};
    float          query_coverage{0.0F};
    float          spatial_consistency{0.0F};
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

/** @brief 阶段候选结果（去路径化，仅携带图像 ID 与边界框）。 */
struct DinoCoarseCandidate
{
    int64_t        image_id{0};    ///< 候选图像 ID。
    DinoSearchRect bbox{};
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
    std::vector<DinoCoarseCandidate> localized_candidates{};
    std::vector<DinoCoarseCandidate> verification_candidates{};
    std::string score_kind{"localization"};
    size_t verified_candidates{0};
    DinoSearchTimings             timings{};
    DinoSearchDiagnostics         diagnostics{};
    std::string                   message{};
};

/** @brief 建库报告中的单图记录。 */
struct DinoImageRecord
{
    int64_t        image_id{0};    ///< 图像 ID。
    int            width{0};
    int            height{0};
    int            view_count{0};
    int            region_descriptor_count{0};
    int            local_descriptor_count{0};
    uint64_t       original_patch_count{0};
    uint64_t       quantized_bytes{0};
};

/** @brief 建库报告。 */
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
    double        duration_ms{0.0};
    std::vector<DinoImageRecord> images{};
    std::vector<std::string>     failed_files{};
    std::vector<std::string>     messages{};
};

/** @brief Parse documented YAML profile/config text. */
INFERRT_FEATURES_API DinoRegionSearchConfig dinoConfigFromYaml(const std::string &text);

/** @brief Serialize configuration to YAML text. */
INFERRT_FEATURES_API std::string dinoConfigToYaml(const DinoRegionSearchConfig &config);

/** @brief Parse YAML search request text. */
INFERRT_FEATURES_API DinoSearchRequest dinoSearchRequestFromYaml(const std::string &text);

/** @brief Parse a YAML sequence of requests for a persistent-process batch. */
INFERRT_FEATURES_API std::vector<DinoSearchRequest> dinoSearchRequestsFromYaml(const std::string &text);

/** @brief Serialize search response as YAML. */
INFERRT_FEATURES_API std::string dinoSearchResponseToYaml(const DinoSearchResponse &response);

/** @brief Serialize build report as YAML. */
INFERRT_FEATURES_API std::string dinoBuildReportToYaml(const DinoBuildReport &report);

/**
 * @brief 冻结 DINO 骨干的区域检索引擎（无请求状态，进程级共享模型与缓存）。
 *
 * 入口共享同一份 profile、同一套坐标与错误语义；模型、图像与候选特征缓存使用进程级默认预算。
 */
class INFERRT_FEATURES_API DinoRegionSearch
{
public:
    /**
     * @brief 预先检查索引是否存在、格式是否有效且与当前配置硬参数兼容。
     *
     * @param index_root 索引根目录。
     * @param config 检索配置。
     * @return true 表示需要（重新）构建索引；false 表示可直接复用现有索引。
     * @throws irt::Exception 参数非法、索引文件严重损坏或 I/O 错误时抛出。
     */
    static bool needsRebuild(const std::filesystem::path &index_root, const DinoRegionSearchConfig &config);

    /**
     * @brief 从显式图像条目列表建立本地索引（核心入口）。
     *
     * @param items 待加入索引的图像路径和外部图像 ID 列表。
     * @param config 检索 profile。
     * @param index_root 索引根目录。
     * @param progress_callback 可选进度回调。
     * @param control 可选协作取消控制。
     * @return 建库报告；含失败文件时 ready_with_errors 为 true。
     */
    static DinoBuildReport build(const std::vector<DinoImageItem> &items,
                                 const DinoRegionSearchConfig &config,
                                 const std::filesystem::path &index_root = {},
                                 const DinoBuildProgressCallback &progress_callback = {},
                                 const DinoOperationControl &control = {});

    /**
     * @brief 从图库目录建立本地索引（便捷重载，自动生成 0..N-1 连续递增 image_id）。
     *
     * @param gallery_root 图库根目录；递归扫描受支持图片。
     * @param config 检索 profile。
     * @param index_root 索引根目录；为空时使用 ``gallery_root`` 的上层默认目录。
     * @param progress_callback 可选进度回调。
     * @param control 可选协作取消控制。
     * @return 建库报告；含失败文件时 ``ready_with_errors`` 为 true。
     * @throws irt::Exception 权重缺失、配置非法或输入目录不可用时抛出。
     */
    static DinoBuildReport build(const std::filesystem::path &gallery_root, const DinoRegionSearchConfig &config,
                                 const std::filesystem::path &index_root = {},
                                 const DinoBuildProgressCallback &progress_callback = {},
                                 const DinoOperationControl &control = {});

    /**
     * @brief 在索引中检索与查询区域相似的区域。
     *
     * @param index_root 索引根目录。
     * @param request 查询请求；请求级 deadline 覆盖 profile deadline。
     * @param config 检索 profile。
     * @param progress_callback 可选进度回调。
     * @return 查询响应；未完成时 status 为 ``Incomplete`` 且保留部分结果。
     * @throws irt::Exception 索引缺失、profile 不兼容或请求非法时抛出。
     */
    static DinoSearchResponse search(const std::filesystem::path &index_root, const DinoSearchRequest &request,
                                     const DinoRegionSearchConfig &config,
                                     const DinoSearchProgressCallback &progress_callback = {});

    /**
     * @brief 支持协作取消与白名单过滤的区域检索。
     *
     * @param index_root 索引根目录。
     * @param request 查询请求。
     * @param config 检索 profile。
     * @param progress_callback 可选进度回调。
     * @param control 协作控制。
     * @return 查询响应。
     */
    static DinoSearchResponse search(const std::filesystem::path &index_root, const DinoSearchRequest &request,
                                     const DinoRegionSearchConfig &config,
                                     const DinoSearchProgressCallback &progress_callback,
                                     const DinoOperationControl &control);

    /**
     * @brief 释放当前进程持有的 DINO 区域检索模型、缓存与索引 reader。
     *
     * @param release_backbone 是否同时释放冻结 backbone 实例。
     */
    static void releaseRuntime(bool release_backbone = true);
};

} // namespace irt::features

