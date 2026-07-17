#pragma once

/**
 * @file FeatureSearchCommon.hpp
 * @brief 图像级检索与 ROI 检索共用的配置校验、特征归一化与 Faiss 索引辅助函数。
 */

#include "ImageSearchFaissIndex.hpp"

#include <inferrt/features/ImageSearch.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#include <faiss/gpu/StandardGpuResources.h>
#pragma warning(pop)

#include <filesystem>
#include <memory>
#include <vector>

namespace irt::features::priv {

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
void validateFeatureSearchConfig(const ImageSearchConfig &config, const char *owner_name);

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
