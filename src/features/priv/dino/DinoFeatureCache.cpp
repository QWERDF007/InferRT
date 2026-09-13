/**
 * @file DinoFeatureCache.cpp
 * @brief 候选特征网格缓存实现。
 */

#include "DinoFeatureCache.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace irt::features::priv {

namespace {

uint64_t vectorBytes(const std::vector<float> &values) noexcept
{
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    const auto     capacity = static_cast<uint64_t>(values.capacity());
    return capacity > maximum / sizeof(float) ? maximum : capacity * sizeof(float);
}

uint64_t gridBytes(const DinoFeatureGrid &grid) noexcept
{
    constexpr auto maximum    = std::numeric_limits<uint64_t>::max();
    const auto     token_bytes = vectorBytes(grid.tokens);
    const auto     area_bytes  = vectorBytes(grid.valid_area);
    return token_bytes > maximum - area_bytes ? maximum : token_bytes + area_bytes;
}

} // namespace

DinoFeatureGridCache::DinoFeatureGridCache(const uint64_t budget_bytes)
    : budget_bytes_(budget_bytes)
{
}

std::shared_ptr<const DinoFeatureGrid> DinoFeatureGridCache::find(const std::string &key)
{
    std::lock_guard lock(mutex_);
    const auto      it = entries_.find(key);
    if (it == entries_.end())
    {
        ++misses_;
        return nullptr;
    }
    order_.erase(it->second.order);
    order_.push_front(key);
    it->second.order = order_.begin();
    ++hits_;
    return it->second.grid;
}

void DinoFeatureGridCache::insert(const std::string &key, std::shared_ptr<const DinoFeatureGrid> grid)
{
    if (grid == nullptr)
    {
        return;
    }
    const auto item_bytes = gridBytes(*grid);
    if (item_bytes > budget_bytes_)
    {
        return;
    }

    std::lock_guard lock(mutex_);
    const auto      existing = entries_.find(key);
    if (existing != entries_.end())
    {
        order_.erase(existing->second.order);
        bytes_ -= existing->second.bytes;
        entries_.erase(existing);
    }
    order_.push_front(key);
    entries_.emplace(key, Entry{std::move(grid), item_bytes, order_.begin()});
    bytes_ += item_bytes;
    evictLocked();
}

size_t DinoFeatureGridCache::hits() const noexcept
{
    std::lock_guard lock(mutex_);
    return hits_;
}

size_t DinoFeatureGridCache::misses() const noexcept
{
    std::lock_guard lock(mutex_);
    return misses_;
}

uint64_t DinoFeatureGridCache::bytes() const noexcept
{
    std::lock_guard lock(mutex_);
    return bytes_;
}

void DinoFeatureGridCache::evictLocked()
{
    while (bytes_ > budget_bytes_ && !order_.empty())
    {
        const auto &victim = order_.back();
        const auto  it     = entries_.find(victim);
        if (it != entries_.end())
        {
            bytes_ -= it->second.bytes;
            entries_.erase(it);
        }
        order_.pop_back();
    }
}

std::string dinoFeatureCacheKey(const std::string &image_identity, const std::string &extractor_signature,
                                const DinoRect &crop)
{
    std::ostringstream stream;
    stream << image_identity.size() << ':' << image_identity << '|';
    stream << extractor_signature.size() << ':' << extractor_signature << '|';
    stream << std::setprecision(17) << crop.x0 << ',' << crop.y0 << ',' << crop.x1 << ',' << crop.y1;
    return stream.str();
}

} // namespace irt::features::priv
