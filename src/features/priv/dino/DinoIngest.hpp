#pragma once

/**
 * @file DinoIngest.hpp
 * @brief 图像接入：解码、EXIF 方向规范化、内容身份与解码缓存。
 */

#include "DinoTypes.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace irt::features::priv {

/** @brief 解码后的 canonical 图像与其身份记录。 */
struct DinoCanonicalImage
{
    cv::Mat         image{};    ///< EXIF 已应用的 8 位三通道 BGR 图像。
    DinoImageIdentity record{};
};

/** @brief 按字节预算约束的解码图像 LRU 缓存。 */
class DinoImageCache
{
public:
    explicit DinoImageCache(uint64_t budget_bytes);

    /** @brief 命中时移动该条目到最近使用端。 */
    std::shared_ptr<const DinoCanonicalImage> find(const std::string &key);

    /** @brief 插入或替换条目；单张超过预算时直接丢弃。 */
    void insert(const std::string &key, std::shared_ptr<const DinoCanonicalImage> image);
    size_t   hits() const noexcept;
    size_t   misses() const noexcept;
    uint64_t bytes() const noexcept;

private:
    void evict();

    struct Entry
    {
        std::shared_ptr<const DinoCanonicalImage> image{};
        uint64_t                                  bytes{0};
        std::list<std::string>::iterator          order{};
    };

    uint64_t                                     budget_bytes_{0};
    uint64_t                                     bytes_{0};
    size_t                                       hits_{0};
    size_t                                       misses_{0};
    mutable std::mutex                           mutex_{};
    std::list<std::string>                       order_{};
    std::unordered_map<std::string, Entry>       entries_{};
};

/** @brief 图库图像接入器。 */
class DinoImageLoader
{
public:
    /** @brief 允许的标准输入格式判定，复用图像检索模块的扩展名规则。 */
    static bool isSupportedFile(const std::filesystem::path &path);

    /** @brief 递归收集图库目录下的受支持图片，按路径排序。 */
    static std::vector<std::filesystem::path> collectGalleryImages(const std::filesystem::path &gallery_root);

    /** @brief 解码一张图片为 canonical 图像并计算身份（可选指定外部 image_id）。 */
    static DinoCanonicalImage load(const std::filesystem::path &path, int64_t image_id = 0);

    /** @brief 只计算文件身份（大小与修改时间），不解码像素。 */
    static DinoImageIdentity statIdentity(int64_t image_id, const std::filesystem::path &path);

    static std::shared_ptr<const DinoCanonicalImage> loadCached(const std::filesystem::path &path,
                                                               DinoImageCache &cache,
                                                               int64_t image_id = 0);
};

} // namespace irt::features::priv
