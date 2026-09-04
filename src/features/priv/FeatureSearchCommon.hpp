#pragma once

/**
 * @file FeatureSearchCommon.hpp
 * @brief 图像级检索与 ROI 检索共用的配置校验、特征归一化与 Faiss 索引辅助函数。
 */

#include "ImageSearchFaissIndex.hpp"

#include <inferrt/core/Tensor.hpp>
#include <inferrt/features/ImageSearch.hpp>
#include <inferrt/features/RoiFeature.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#include <faiss/gpu/StandardGpuResources.h>
#pragma warning(pop)

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace irt::features::priv {

/** Validate the finite, positively ordered coordinates shared by ROI modules. */
void validateRoi(const RoiFeatureBox &roi);

/**
 * @brief 一次特征批处理完成后的进度信息。
 */
struct FeatureBatchProgress
{
    size_t batch_index{0};
    size_t batch_begin{0};
    size_t batch_count{0};
    size_t processed_count{0};
    size_t total_count{0};
};

using FeatureBatchLoader   = std::function<std::vector<float>(size_t, size_t)>;
using FeatureBatchConsumer = std::function<void(size_t, size_t, const std::vector<float> &)>;
using FeatureBatchProgressCallback = std::function<void(const FeatureBatchProgress &)>;

/**
 * @brief 按统一批量策略提取并消费一组特征。
 *
 * 该函数只负责批边界、结果尺寸校验和进度推进；图像、ROI 和聚类的特征后处理由 loader 负责。
 */
void processFeatureBatches(size_t item_count, size_t batch_size, int feature_dim,
                           const FeatureBatchLoader &loader, const FeatureBatchConsumer &consumer,
                           const FeatureBatchProgressCallback &progress_callback = {});

/**
 * @brief 校验并规范化一个图像文件路径。
 */
std::filesystem::path normalizeImageFilePath(const std::filesystem::path &image_path, const char *owner_name);

/**
 * @brief 返回构建阶段使用的临时特征文件路径。
 */
std::filesystem::path featureStorePath(const std::filesystem::path &index_path);

/**
 * @brief 构建阶段的临时特征存储。
 *
 * 特征先按模型 batch 写入临时文件，Faiss 建库阶段按需读取，避免图库特征在内存中完整驻留，
 * 同时让“特征提取”和“索引构建”成为两个独立的进度阶段。
 */
class FeatureStore
{
public:
    FeatureStore(std::filesystem::path path, size_t item_count, int feature_dim);
    ~FeatureStore();

    FeatureStore(const FeatureStore &)            = delete;
    FeatureStore &operator=(const FeatureStore &) = delete;

    void writeBatch(size_t begin, size_t count, const std::vector<float> &features);
    /**
     * @brief 将一批特征写入任意连续位置。
     *
     * ROI 搜索会按图像分组提取特征，但 Faiss ID 仍需保持调用方传入的 ROI 顺序，
     * 因此建库阶段允许分组结果回写到临时特征文件的原始位置。
     */
    void writeBatchAt(size_t begin, size_t count, const std::vector<float> &features);
    void finishWriting();

    int featureDim() const noexcept
    {
        return feature_dim_;
    }

    std::vector<float> read(size_t index) const;
    std::vector<float> readBatch(size_t begin, size_t count) const;
    std::vector<float> readBatch(const std::vector<size_t> &indices) const;

private:
    void validateRange(size_t begin, size_t count) const;

    std::filesystem::path path_;
    size_t                item_count_{0};
    int                   feature_dim_{0};
    size_t                next_write_index_{0};
    size_t                written_count_{0};
    bool                  writing_finished_{false};
    std::ofstream         output_;
    std::vector<unsigned char> written_;
};

/**
 * @brief Faiss 索引及其可选 GPU 资源的生命周期包。
 *
 * GPU Faiss 索引依赖 ``StandardGpuResources``，因此二者必须一起移动和销毁。
 */
struct FaissIndexBundle
{
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources; ///< GPU Faiss 资源，可为空。
    std::unique_ptr<faiss::Index>                     index;         ///< 可搜索的 Faiss 索引。
};

/**
 * @brief 判断特征提取模型是否使用 TensorRT 后端。
 * @param config 检索配置。
 * @return 使用 TensorRT 时返回 true。
 */
bool usesTensorRtModelBackend(const ImageSearchConfig &config) noexcept;

/**
 * @brief 判断是否使用 CPU 磁盘 IVF 索引。
 * @param config 检索配置。
 * @return CPU Faiss 且索引存储为 Disk 时返回 true。
 */
bool useCpuDiskIndex(const ImageSearchConfig &config) noexcept;

/**
 * @brief 校验图像/ROI 检索共用配置。
 * @param config 待校验配置。
 * @param owner_name 错误消息中使用的模块名称。
 */
void validateFeatureSearchConfig(ImageSearchConfig &config, const char *owner_name);

/** @brief 将预处理后端枚举序列化为元数据字符串。 */
const char *preprocessBackendName(ImageSearchPreprocessBackend backend) noexcept;

/** @brief 将特征归一化枚举序列化为元数据字符串。 */
const char *featureNormName(ImageSearchFeatureNorm norm) noexcept;

/** @brief 将 Faiss 后端枚举序列化为元数据字符串。 */
const char *faissBackendName(ImageSearchFaissBackend backend) noexcept;

/** @brief 将索引存储枚举序列化为元数据字符串。 */
const char *indexStorageName(ImageSearchIndexStorage storage) noexcept;

/** @brief 返回当前配置对应的索引实现标识。 */
const char *indexKindName(const ImageSearchConfig &config) noexcept;

/**
 * @brief 对特征向量执行原地归一化。
 * @param values 特征数组起始地址。
 * @param count 特征分量数量。
 * @param norm 归一化策略。
 */
void normalizeFeature(float *values, size_t count, ImageSearchFeatureNorm norm);

/**
 * @brief 对完整特征向量执行原地归一化。
 * @param values 特征向量。
 * @param norm 归一化策略。
 */
void normalizeFeature(std::vector<float> &values, ImageSearchFeatureNorm norm);

/**
 * @brief 将 CPU Faiss 索引按配置保留在 CPU 或迁移到 GPU。
 * @param cpu_index 已构建或已加载的 CPU 索引。
 * @param backend 目标 Faiss 后端。
 * @param device_id GPU Faiss 使用的设备编号。
 * @return 迁移后的索引包。
 */
FaissIndexBundle moveCpuIndexToConfiguredBackend(std::unique_ptr<faiss::Index> cpu_index,
                                                 ImageSearchFaissBackend       backend, int device_id);

/**
 * @brief 按配置构建 Faiss 索引。
 *
 * 该函数统一封装 RAM IVF-PQ 与 CPU 磁盘 IVF 两条路径，供图像级检索和 ROI 检索复用。
 *
 * @param vector_count 向量数量。
 * @param feature_dim 单条向量维度。
 * @param index_path 输出索引路径。
 * @param config 检索配置。
 * @param load_feature 单条特征加载回调。
 * @param load_feature_batch 连续区间批量特征加载回调。
 * @param load_feature_index_batch 任意下标批量特征加载回调。
 * @param progress_callback 构建进度回调。
 * @return 构建完成的索引包。
 */
FaissIndexBundle buildConfiguredFaissIndex(size_t vector_count, int feature_dim,
                                           const std::filesystem::path &index_path, const ImageSearchConfig &config,
                                           const LoadFeatureCallback              &load_feature,
                                           const LoadFeatureBatchCallback         &load_feature_batch,
                                           const LoadFeatureIndexedBatchCallback  &load_feature_index_batch,
                                           const ImageSearchBuildProgressCallback &progress_callback);

/**
 * @brief 按配置从磁盘加载 Faiss 索引。
 * @param index_path Faiss 索引路径。
 * @param config 检索配置。
 * @return 可搜索的索引包。
 */
FaissIndexBundle loadConfiguredFaissIndex(const std::filesystem::path &index_path, const ImageSearchConfig &config);

} // namespace irt::features::priv
