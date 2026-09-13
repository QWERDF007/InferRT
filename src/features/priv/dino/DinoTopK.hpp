#pragma once

/**
 * @file DinoTopK.hpp
 * @brief 有界 Top-K 收集器：保持内存有界，不构造全量分数矩阵。
 */

#include <inferrt/features/Export.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace irt::features::priv {

/** @brief 维护容量固定的最大 K 个 (score, index)。 */
class INFERRT_FEATURES_API DinoTopK
{
public:
    explicit DinoTopK(size_t capacity);

    void push(float score, size_t index);

    /** @brief 按分数从高到低返回 (score, index)；同分按 index 升序保证稳定。 */
    std::vector<std::pair<float, size_t>> sorted() const;

    size_t capacity() const noexcept
    {
        return capacity_;
    }

private:
    size_t                                capacity_{0};
    std::vector<std::pair<float, size_t>> heap_{};
};

} // namespace irt::features::priv
