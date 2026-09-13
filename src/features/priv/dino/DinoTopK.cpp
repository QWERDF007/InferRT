/**
 * @file DinoTopK.cpp
 * @brief 有界 Top-K 收集器实现。
 */

#include "DinoTopK.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <functional>

namespace irt::features::priv {

DinoTopK::DinoTopK(const size_t capacity)
    : capacity_(capacity)
{
    if (capacity_ == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Top-K capacity must be positive");
    }
    heap_.reserve(capacity_ + 1U);
}

void DinoTopK::push(const float score, const size_t index)
{
    if (heap_.size() < capacity_)
    {
        heap_.emplace_back(score, index);
        std::push_heap(heap_.begin(), heap_.end(), std::greater<>());
        return;
    }
    // 堆顶是当前最小值；只有更大分数才替换，保持内存有界。
    if (score <= heap_.front().first)
    {
        return;
    }
    std::pop_heap(heap_.begin(), heap_.end(), std::greater<>());
    heap_.back() = std::make_pair(score, index);
    std::push_heap(heap_.begin(), heap_.end(), std::greater<>());
}

std::vector<std::pair<float, size_t>> DinoTopK::sorted() const
{
    auto result = heap_;
    std::sort(result.begin(), result.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b)
              {
                  if (a.first != b.first)
                  {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });
    return result;
}

} // namespace irt::features::priv
