#pragma once

#include <inferrt/features/Export.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

/**
 * @brief 图像检索结果。
 */
struct ImageSearchResult
{
    /// 与查询图片的相似度分数；当前实现使用 L2 归一化特征的内积。
    float score{0.0f};

    /// 命中的图库图片路径。
    std::filesystem::path image_path;
};

/**
 * @brief 基于 InferRT 中间特征与 Faiss 的图像检索器。
 *
 * 该类负责从分类、ViT 或 DINO 模型的指定中间层提取 L2 归一化特征，构建或加载 Faiss
 * 内积索引，并对查询图片返回 Top-K 相似图片。索引会伴随保存路径映射文件，
 * 因此后续运行可直接加载已有索引。
 */
class INFERRT_FEATURES_API ImageSearch
{
public:
    /// 默认检索模型名称。
    static constexpr const char *kDefaultModelName = "resnet18";

    /// 默认导出的中间特征名。
    static constexpr const char *kDefaultFeatureName = "layer4";

    /// 默认返回的相似图片数量。
    static constexpr int kDefaultTopK = 5;

    /**
     * @brief 构造图像检索器。
     * @param model_name 内置分类、ViT 或 DINO 模型名称。
     * @param feature_name 用作检索向量的中间特征名。
     */
    explicit ImageSearch(std::string model_name = kDefaultModelName,
                         std::string feature_name = kDefaultFeatureName);

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
     * @brief 获取当前模型名称。
     * @return 模型名称。
     */
    const std::string &modelName() const noexcept;

    /**
     * @brief 获取当前特征名称。
     * @return 特征名称。
     */
    const std::string &featureName() const noexcept;

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
                                                  const std::string &model_name,
                                                  const std::string &feature_name);

private:
    class Impl;

    /// 私有实现，隐藏 Faiss、CUDA、OpenCV 与 TensorRT 细节。
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::features
