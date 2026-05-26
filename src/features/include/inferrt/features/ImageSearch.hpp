#pragma once

/**
 * @file ImageSearch.hpp
 * @brief 图像检索公共 API、配置类型与 ``ImageSearch`` 类声明。
 */

#include <inferrt/features/Export.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/// 默认检索模型名称（ResNet-18）。
inline constexpr const char *kDefaultImageSearchModelName = "resnet18";

/// 默认导出的中间特征层名称。
inline constexpr const char *kDefaultImageSearchFeatureName = "layer4";

/// CPU 磁盘索引构建时的默认特征批大小。
inline constexpr size_t kDefaultImageSearchDiskBuildBatchSize = 256;

/**
 * @brief 图像检索结果。
 */
struct ImageSearchResult
{
    ///< 与查询图片的相似度分数；当前实现使用配置归一化后的特征内积。
    float score{0.0f};

    ///< 命中的图库图片路径。
    std::filesystem::path image_path;
};

/**
 * @brief 图像预处理执行后端。
 */
enum class ImageSearchPreprocessBackend
{
    CPU, ///< 使用 OpenCV 与 CPU 完成图像预处理。
    GPU, ///< 预留 GPU 预处理配置；当前尚未实现。
};

/**
 * @brief 推理特征归一化方式。
 */
enum class ImageSearchFeatureNorm
{
    None, ///< 不对推理特征做归一化。
    L1,   ///< 对推理特征做 L1 归一化。
    L2,   ///< 对推理特征做 L2 归一化。
};

/**
 * @brief Faiss 索引执行后端。
 */
enum class ImageSearchFaissBackend
{
    CPU, ///< 使用 CPU Faiss 索引。
    GPU, ///< 使用 GPU Faiss 索引。
};

/**
 * @brief Faiss 索引在搜索阶段的存储位置。
 */
enum class ImageSearchIndexStorage
{
    RAM,  ///< 搜索时索引常驻内存。
    Disk, ///< CPU Faiss 搜索时按需从磁盘读取索引。
};

/**
 * @brief 图像检索流程的配置项集合。
 */
struct ImageSearchConfig
{
    ///< 内置分类、ViT 或 DINO 模型名称。
    std::string model_name{kDefaultImageSearchModelName};

    ///< 用作检索向量的中间特征名。
    std::string feature_name{kDefaultImageSearchFeatureName};

    ///< 预处理执行后端。
    ImageSearchPreprocessBackend preprocess_backend{ImageSearchPreprocessBackend::CPU};

    ///< 推理特征归一化方式。
    ImageSearchFeatureNorm norm{ImageSearchFeatureNorm::L2};

    ///< Faiss 索引执行后端。
    ImageSearchFaissBackend faiss_backend{ImageSearchFaissBackend::CPU};

    ///< Faiss 索引搜索存储位置；GPU Faiss 当前始终使用 RAM。
    ImageSearchIndexStorage index_storage{ImageSearchIndexStorage::RAM};

    ///< CPU disk index build batch size.
    size_t disk_build_batch_size{kDefaultImageSearchDiskBuildBatchSize};
};

/**
 * @brief 基于 InferRT 中间特征与 Faiss 的图像检索器。
 *
 * 该类负责从分类、ViT 或 DINO 模型的指定中间层提取特征，按配置做归一化，
 * 构建或加载 Faiss 内积索引，并对查询图片返回 Top-K 相似图片。索引会伴随保存路径映射文件，
 * 因此后续运行可直接加载已有索引。
 */
class INFERRT_FEATURES_API ImageSearch
{
public:
    /// 默认检索模型名称。
    static constexpr const char *kDefaultModelName = kDefaultImageSearchModelName;

    /// 默认导出的中间特征名。
    static constexpr const char *kDefaultFeatureName = kDefaultImageSearchFeatureName;

    /// 默认返回的相似图片数量。
    static constexpr int kDefaultTopK = 5;

    /**
     * @brief 构造图像检索器。
     * @param config 检索流程配置。
     */
    explicit ImageSearch(ImageSearchConfig config = {});

    /**
     * @brief 析构图像检索器。
     */
    ~ImageSearch();

    /** @brief 禁止拷贝构造。 */
    ImageSearch(const ImageSearch &) = delete;

    /** @brief 禁止拷贝赋值。 */
    ImageSearch &operator=(const ImageSearch &) = delete;

    /**
     * @brief 移动构造图像检索器。
     * @param other 被移动对象。
     */
    ImageSearch(ImageSearch &&other) noexcept;

    /**
     * @brief 移动赋值图像检索器。
     * @param other 被移动对象。
     * @return 当前对象引用。
     */
    ImageSearch &operator=(ImageSearch &&other) noexcept;

    /**
     * @brief 构建或加载图库索引。
     *
     * 当 `rebuild_index` 为 false 且索引文件及路径映射文件均存在时，会直接加载；
     * 否则递归扫描图库目录并重建索引。
     *
     * @param weights_file 模型 `.wts` 权重文件路径。
     * @param gallery_dir 图库目录。
     * @param index_file Faiss 索引文件路径；为空时使用默认路径。
     * @param rebuild_index 是否强制重建索引。
     */
    void buildOrLoad(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
                     const std::filesystem::path &index_file = {}, bool rebuild_index = false);

    /**
     * @brief 查询单张图片的 Top-K 相似图片。
     * @param query_image 查询图片路径。
     * @param top_k 返回数量。
     * @return 按相似度从高到低排列的结果列表。
     */
    std::vector<ImageSearchResult> search(const std::filesystem::path &query_image, int top_k = kDefaultTopK);

    /**
     * @brief 判断索引是否已经加载或构建完成。
     * @return 可搜索时返回 true。
     */
    bool isReady() const noexcept;

    /**
     * @brief 获取当前检索流程配置。
     * @return 构造时传入的配置。
     */
    const ImageSearchConfig &config() const noexcept;

    /**
     * @brief 获取当前索引路径。
     * @return 索引文件路径；尚未 buildOrLoad 时为空。
     */
    const std::filesystem::path &indexPath() const noexcept;

    /**
     * @brief 获取当前图库图片列表。
     * @return 图库图片路径列表。
     */
    std::vector<std::filesystem::path> galleryImages() const;

    /**
     * @brief 获取当前特征向量维度。
     * @return 特征维度；尚未加载索引时返回 0。
     */
    int featureDim() const noexcept;

    /**
     * @brief 判断路径是否为支持的图片文件。
     * @param path 待判断路径。
     * @return 是图片文件返回 true。
     */
    static bool isImageFile(const std::filesystem::path &path);

    /**
     * @brief 递归收集图库目录下的图片路径。
     * @param gallery_dir 图库根目录。
     * @return 排序后的绝对图片路径列表。
     */
    static std::vector<std::filesystem::path> collectGalleryImages(const std::filesystem::path &gallery_dir);

    /**
     * @brief 生成默认索引文件路径。
     * @param gallery_dir 图库目录。
     * @param model_name 模型名称。
     * @param feature_name 特征名称。
     * @return `<gallery_dir>/<model>_<feature>.faiss`。
     */
    static std::filesystem::path defaultIndexPath(const std::filesystem::path &gallery_dir,
                                                  const std::string &model_name, const std::string &feature_name);

private:
    class Impl;

    /// 私有实现，隐藏 Faiss、CUDA、OpenCV 与 TensorRT 细节。
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features
