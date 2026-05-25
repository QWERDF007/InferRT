#pragma once

#include <inferrt/features/ImageSearch.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
struct Index;
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
     * @param model_name 内置分类、ViT 或 DINO 模型名称。
     * @param feature_name 用作检索向量的中间特征名。
     */
    Impl(std::string model_name, std::string feature_name);

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
     * 直接加载 Faiss 索引；否则扫描图库、提取 L2 归一化特征并重建索引。
     *
     * @param weights_file 模型 ``.wts`` 权重文件路径。
     * @param gallery_dir 图库根目录。
     * @param index_file Faiss 索引文件路径；为空时使用 ``ImageSearch::defaultIndexPath``。
     * @param rebuild_index 是否强制重建索引。
     */
    void buildOrLoad(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
                     const std::filesystem::path &index_file, bool rebuild_index);

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
     * @brief 获取当前模型名称。
     * @return 构造时传入的模型名称。
     */
    const std::string &modelName() const noexcept;

    /**
     * @brief 获取当前特征名称。
     * @return 构造时传入的中间特征名。
     */
    const std::string &featureName() const noexcept;

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
     * @brief 按需懒加载特征提取器。
     *
     * 从磁盘加载索引时不会保留 ``extractor_``；首次 ``search`` 前根据
     * 已保存的 ``weights_file_`` 创建提取器。
     */
    void ensureExtractor();

    ///< 检索模型名称。
    std::string model_name_;

    ///< 中间特征层名称。
    std::string feature_name_;

    ///< 模型权重路径。
    std::filesystem::path weights_file_;

    ///< 图库根目录。
    std::filesystem::path gallery_dir_;

    ///< Faiss 索引文件路径。
    std::filesystem::path index_path_;

    ///< 图库图片路径，与索引向量顺序一致。
    std::vector<std::filesystem::path> gallery_images_;

    ///< Faiss 内积索引（``IndexFlatIP``）。
    std::unique_ptr<faiss::Index> index_;

    ///< 查询侧特征提取器；加载索引后可能为空直至首次检索。
    std::unique_ptr<priv::ImageSearchFeatureExtractor> extractor_;

    int feature_dim_{0}; ///< 特征向量维度。
};

} // namespace irt::features
