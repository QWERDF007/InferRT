#pragma once

/**
 * @file DinoSearchCache.hpp
 * @brief 查询进程级的图像与候选特征缓存生命周期。
 */

#include "DinoFeatureCache.hpp"
#include "DinoIngest.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace irt::features::priv {

/** @brief 一次 profile/extractor 身份下共享的两个独立字节 LRU。 */
struct DinoSearchCaches final
{
    DinoSearchCaches(uint64_t image_budget_bytes, uint64_t feature_budget_bytes)
        : images(image_budget_bytes), features(feature_budget_bytes)
    {
    }

    DinoImageCache      images;
    DinoFeatureGridCache features;
};

/**
 * @brief 获取当前进程的缓存 bundle。
 *
 * bundle key 包含 extractor hash 和两个预算。切换 profile 时旧 bundle 由正在使用它的查询持有，
 * 新查询不会串用旧骨干特征；同一 profile 的后续查询复用已有 LRU。
 */
std::shared_ptr<DinoSearchCaches> dinoAcquireSearchCaches(const std::string &extractor_signature,
                                                          uint64_t image_budget_bytes,
                                                          uint64_t feature_budget_bytes);

} // namespace irt::features::priv
