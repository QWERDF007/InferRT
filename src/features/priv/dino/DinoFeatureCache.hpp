#pragma once

/**
 * @file DinoFeatureCache.hpp
 * @brief 按字节预算缓存候选图像的冻结特征网格。
 */

#include "DinoTypes.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace irt::features::priv {

/**
 * @brief 候选特征网格的线程安全 LRU。
 *
 * key 必须由调用方按“图像身份 + 文件状态 + extractor 配置 + crop”构造，缓存隔离图像、骨干与裁剪。
 */
class DinoFeatureGridCache final
{
public:
    explicit DinoFeatureGridCache(uint64_t budget_bytes);

    /** @brief 返回命中项并提升到 MRU；未命中会计数。 */
    std::shared_ptr<const DinoFeatureGrid> find(const std::string &key);

    /** @brief 插入网格；单项超过字节预算时丢弃。 */
    void insert(const std::string &key, std::shared_ptr<const DinoFeatureGrid> grid);

    size_t hits() const noexcept;
    size_t misses() const noexcept;
    uint64_t bytes() const noexcept;

private:
    void evictLocked();

    struct Entry
    {
        std::shared_ptr<const DinoFeatureGrid> grid{};
        uint64_t                               bytes{0};
        std::list<std::string>::iterator       order{};
    };

    const uint64_t                             budget_bytes_{0};
    mutable std::mutex                         mutex_{};
    uint64_t                                   bytes_{0};
    size_t                                     hits_{0};
    size_t                                     misses_{0};
    std::list<std::string>                     order_{};
    std::unordered_map<std::string, Entry>     entries_{};
};

/**
 * @brief 构造包含图像身份、骨干配置和裁剪框的稳定缓存键。
 */
std::string dinoFeatureCacheKey(const std::string &image_identity, const std::string &extractor_signature,
                                const DinoRect &crop);

} // namespace irt::features::priv
