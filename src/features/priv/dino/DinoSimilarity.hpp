#pragma once

/**
 * @file DinoSimilarity.hpp
 * @brief 紧凑描述块对查询 token 的相似度归约：CPU 参考实现与 GPU 实现。
 *
 * 描述子与 token 都是 L2 归一化向量，相似度即点积。查询路径上有两个随图库规模线性
 * 增长的热点，都由本模块承担：
 *
 * - 局部通道：每个索引视图上、每个查询 token 的最大相似度（组合所有查询视图的
 *   token 后一次扫描完成，索引数据每查询只搬一次）；
 * - 区域通道：每条区域描述子对每个查询 ROI 向量的分数（供宿主端在线 Top-K）。
 *
 * 语义只有一份：标量实现是参考实现，其余实现只允许改变浮点累加顺序。归约是逐元素
 * 最大值或逐描述子分数，与累加顺序无关，因此各后端给出同一组结果（浮点尾差在内）。
 * 紧凑块（INT8 码 + 每条描述子的还原因子）是唯一输入形态：GPU 后端在 kernel 内解码，
 * 不经过宿主端 FP32 展开，每次查询的上传量就是紧凑索引自身的大小。
 */

#include "DinoIndexStoreTypes.hpp"
#include "DinoRetrievalCore.hpp"

#include <inferrt/features/Export.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace irt::features::priv {

/** @brief 相似度归约后端。 */
enum class DinoSimilarityBackend
{
    Cpu,  ///< 宿主端实现（内部按 CPU 能力选择标量或 AVX2 内核）。
    Cuda, ///< 设备端实现；不可用时自动回退到 CPU。
};

/** @brief 后端名称，用于查询诊断（例如 ``cpu:avx2`` / ``cuda``）。 */
INFERRT_FEATURES_API const char *dinoSimilarityBackendName(DinoSimilarityBackend backend) noexcept;

/** @brief 按本机能力选择后端：有可用 GPU 用 GPU，否则 CPU。 */
INFERRT_FEATURES_API DinoSimilarityBackend dinoSelectSimilarityBackend() noexcept;

/** @brief 覆盖后端选择（诊断与对照用）；``enable`` 为 false 时恢复自动选择。 */
INFERRT_FEATURES_API void dinoOverrideSimilarityBackend(bool enable, DinoSimilarityBackend backend) noexcept;

/** @brief 紧凑描述块的 CPU 归约内核；``tokens`` 为行优先 ``token_count x dimension`` 矩阵。 */
INFERRT_FEATURES_API void dinoAccumulateSimilarityMaxima(const float *descriptors, std::size_t descriptor_count,
                                                         const float *tokens, int token_count, int dimension,
                                                         const int *slot_of_token, float *maxima);

/** @brief 后端实现接口（模块内部）；CPU 与 CUDA 各有一个编译单元实现它。 */
class DinoSimilarityEngine
{
public:
    virtual ~DinoSimilarityEngine() = default;

    /**
     * @brief 对一组连续索引视图的局部描述做一次完整归约。
     *
     * @param block        覆盖这组视图全部描述的紧凑块（描述在块内连续排列）。
     * @param view_offsets 每个视图在块内的起始偏移；长度 ``view_count``。
     * @param view_counts  每个视图的描述数；长度 ``view_count``，均为正。
     * @param tokens       行优先 token 矩阵（所有查询视图拼接），``token_count x dimension``。
     * @param out          输出 ``view_count x token_count`` 的最大相似度，行优先。
     */
    virtual void reduceViewGroup(const DinoCompactBlock &block, const std::size_t *view_offsets,
                                 const std::size_t *view_counts, std::size_t view_count, const float *tokens,
                                 int token_count, float *out) = 0;

    /**
     * @brief 区域通道：每条描述子对每个查询 ROI 向量的分数。
     * @param roi_vectors  行优先 ``query_view_count x dimension`` 的 ROI 描述。
     * @param out          输出 ``block.count x query_view_count`` 的分数，行优先。
     */
    virtual void reduceRegionScores(const DinoCompactBlock &block, const float *roi_vectors, int query_view_count,
                                    float *out) = 0;

    // Two distinct local descriptors per query token. Indices are relative to each view.
    virtual void matchViewGroup(const DinoCompactBlock &block, const std::size_t *offsets,
                                const std::size_t *counts, std::size_t views,
                                const float *tokens, int token_count, retrieval::Pair *out);

    /** @brief 该后端占用的设备显存峰值（字节）；纯 CPU 后端为 0。 */
    virtual uint64_t deviceAllocatedBytes() const noexcept = 0;

    /** @brief 进程显存占用相对首个查询起点的峰值增量（字节）；纯 CPU 后端为 0。 */
    virtual uint64_t deviceReservedDeltaBytes() const noexcept = 0;

    /** @brief 已上传的设备字节数。 */
    virtual uint64_t uploadedBytes() const noexcept = 0;
};

/** @brief 构造 CPU 后端；``dimension`` 是描述子维度。 */
INFERRT_FEATURES_API std::unique_ptr<DinoSimilarityEngine> dinoMakeCpuSimilarityEngine(int dimension);

/** @brief 构造 CUDA 后端；无可用设备时抛出异常。 */
INFERRT_FEATURES_API std::unique_ptr<DinoSimilarityEngine> dinoMakeCudaSimilarityEngine(int dimension);

/** @brief 本机是否能构造 CUDA 后端。 */
INFERRT_FEATURES_API bool dinoCudaSimilarityAvailable() noexcept;

/**
 * @brief 相似度归约器：后端选择的唯一门面。
 *
 * 请求 CUDA 但设备不可用时按契约回退 CPU；实际后端经 ``backend()`` 查询。
 * 生命周期：每次查询构造一个，跨视图分组多次调用归约方法，方法内部完成数据搬运与同步。
 */
class INFERRT_FEATURES_API DinoSimilarityReducer
{
public:
    explicit DinoSimilarityReducer(DinoSimilarityBackend backend, int dimension);
    ~DinoSimilarityReducer();

    DinoSimilarityReducer(const DinoSimilarityReducer &) = delete;
    DinoSimilarityReducer &operator=(const DinoSimilarityReducer &) = delete;

    void reduceViewGroup(const DinoCompactBlock &block, const std::size_t *view_offsets,
                         const std::size_t *view_counts, std::size_t view_count, const float *tokens,
                         int token_count, float *out);

    void reduceRegionScores(const DinoCompactBlock &block, const float *roi_vectors, int query_view_count,
                            float *out);

    void matchViewGroup(const DinoCompactBlock &block, const std::size_t *offsets,
                        const std::size_t *counts, std::size_t views,
                        const float *tokens, int token_count, retrieval::Pair *out);

    DinoSimilarityBackend backend() const noexcept { return backend_; }
    uint64_t             deviceAllocatedBytes() const noexcept;
    uint64_t             deviceReservedDeltaBytes() const noexcept;
    uint64_t             uploadedBytes() const noexcept;

private:
    DinoSimilarityBackend                 backend_{DinoSimilarityBackend::Cpu};
    std::unique_ptr<DinoSimilarityEngine> engine_{};
};

} // namespace irt::features::priv
