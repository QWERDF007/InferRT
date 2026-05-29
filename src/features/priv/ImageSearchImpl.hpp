#pragma once

/**
 * @file ImageSearchImpl.hpp
 * @brief ``ImageSearch::Impl`` PIMPL 声明。
 */

#include <inferrt/features/ImageSearch.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
struct Index;
namespace gpu {
class StandardGpuResources;
} // namespace gpu
} // namespace faiss

namespace irt::features {

namespace priv {
/** @brief 图像检索特征提取器（定义于 ``ImageSearchImpl.cpp``）。 */
class ImageSearchFeatureExtractor;
} // namespace priv

/**
 * @brief ``ImageSearch`` 的 PIMPL 实现。
 *
 * 持有 Faiss 内积索引、图库路径映射及可选的特征提取器；负责索引的构建、
 * 加载与 Top-K 检索。具体逻辑见 ``ImageSearchImpl.cpp``。
 */
class ImageSearch::Impl
{
public:
    /**
     * @brief 构造实现对象。
     * @param config 检索流程配置。
     */
    explicit Impl(ImageSearchConfig config);

    /** @brief 析构实现对象。 */
    ~Impl();

    /** @brief 禁止拷贝构造。 */
    Impl(const Impl &) = delete;

    /** @brief 禁止拷贝赋值。 */
    Impl &operator=(const Impl &) = delete;

    /**
     * @brief 构建或加载图库索引。
     *
     * 当 ``rebuild_index`` 为 false 且索引文件及 ``.paths.txt`` 映射文件均存在时，
     * 直接加载 Faiss 索引；否则扫描图库、提取特征并按配置归一化后重建索引。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_dir 图库根目录。
     * @param index_file Faiss 索引文件路径；为空时使用 ``ImageSearch::defaultIndexPath``。
     * @param rebuild_index 是否强制重建索引。
     */
    void buildOrLoad(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
                     const std::filesystem::path &index_file, bool rebuild_index,
                     ImageSearchBuildProgressCallback progress_callback);

    /**
     * @brief 从图库目录构建图像检索索引。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_dir 图库目录。
     * @param index_file Faiss 索引文件路径；为空时使用 ``ImageSearch::defaultIndexPath``。
     */
    void build(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
               const std::filesystem::path &index_file, ImageSearchBuildProgressCallback progress_callback);

    /**
     * @brief 从显式图片路径列表构建图像检索索引。
     *
     * 向量顺序决定 Faiss id 与图片路径的映射关系。由于无图库目录可用于推导默认索引路径，
     * 必须显式指定 ``index_file``。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_images 待加入索引的图片路径列表。
     * @param index_file Faiss 索引文件路径；不可为空。
     */
    void build(const std::filesystem::path &weights_file,
               const std::vector<std::filesystem::path> &gallery_images,
               const std::filesystem::path &index_file, ImageSearchBuildProgressCallback progress_callback);

    /**
     * @brief 为图库目录加载已有图像检索索引。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_dir 用于校验元数据的图库目录。
     * @param index_file Faiss 索引文件路径；为空时使用 ``ImageSearch::defaultIndexPath``。
     */
    void load(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
              const std::filesystem::path &index_file);

    /**
     * @brief 对查询图片执行 Top-K 相似检索。
     * @param query_image 查询图片路径。
     * @param top_k 返回数量。
     * @return 按相似度从高到低排列的结果列表。
     */
    std::vector<ImageSearchResult> search(const std::filesystem::path &query_image, int top_k);

    /**
     * @brief 判断索引是否已加载或构建完成。
     * @return 可执行 ``search`` 时返回 true。
     */
    bool isReady() const noexcept;

    /**
     * @brief 获取当前检索流程配置。
     * @return 构造时传入的配置副本。
     */
    const ImageSearchConfig &config() const noexcept;

    /**
     * @brief 获取当前索引文件路径。
     * @return 最近一次 ``buildOrLoad`` 使用的索引路径；尚未构建时可能为空。
     */
    const std::filesystem::path &indexPath() const noexcept;

    /**
     * @brief 获取当前图库图片路径列表。
     * @return 与 Faiss 索引向量一一对应的图库图片路径。
     */
    std::vector<std::filesystem::path> galleryImages() const;

    /**
     * @brief 获取特征向量维度。
     * @return 特征维度；索引未就绪时返回 0。
     */
    int featureDim() const noexcept;

private:
    /**
     * @brief 从已确定的图库图片列表构建索引并更新内部状态。
     *
     * ``build`` 两个重载的公共实现：提取特征、构建 Faiss 索引、保存 ``.meta.txt``，
     * 并将索引实例与路径映射写入成员变量。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_dir 图库根目录；显式路径列表构建时为空。
     * @param gallery_images 与 Faiss id 一一对应的图库图片路径（调用方负责扫描或规范化）。
     * @param index_path 已解析的 Faiss 索引文件路径。
     * @param metadata_gallery_value 写入 ``gallery_dir`` 元数据字段的值（目录 canonical 路径或占位哨兵）。
     */
    void buildWithImages(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
                         std::vector<std::filesystem::path> gallery_images,
                         const std::filesystem::path &index_path, const std::string &metadata_gallery_value,
                         ImageSearchBuildProgressCallback progress_callback);

    /**
     * @brief 按需懒加载特征提取器。
     *
     * 从磁盘加载索引时不会保留 ``extractor_``；首次 ``search`` 前根据
     * 已保存的 ``weights_file_`` 创建提取器。
     */
    void ensureExtractor();

    ImageSearchConfig config_{};

    ///< 模型权重路径。
    std::filesystem::path weights_file_;

    ///< 图库根目录。
    std::filesystem::path gallery_dir_;

    ///< Faiss 索引文件路径。
    std::filesystem::path index_path_;

    ///< 图库图片路径，与索引向量顺序一致。
    std::vector<std::filesystem::path> gallery_images_;

    ///< GPU Faiss 资源；必须比 GPU 索引生命周期更长。
    std::unique_ptr<faiss::gpu::StandardGpuResources> faiss_gpu_resources_;

    ///< Faiss 内积索引（RAM ``IndexIVFPQ``/GPU clone 或 CPU 磁盘 IVF）。
    std::unique_ptr<faiss::Index> index_;

    ///< 查询侧特征提取器；加载索引后可能为空直至首次检索。
    std::unique_ptr<priv::ImageSearchFeatureExtractor> extractor_;

    int feature_dim_{0}; ///< 特征向量维度。
};

} // namespace irt::features
