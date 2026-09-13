#pragma once

/**
 * @file DinoTime.hpp
 * @brief 查询截止时间与阶段计时的统一来源。
 */

#include <chrono>
#include <cstdint>

namespace irt::features::priv {

/** @brief 单调时钟截止时间检查器；阶段边界与批次之间调用。 */
class DinoDeadline
{
public:
    /** @param budget_ms wall deadline 预算；<= 0 表示不限制。 */
    explicit DinoDeadline(int64_t budget_ms)
        : budget_ms_(budget_ms)
        , start_(std::chrono::steady_clock::now())
    {
    }

    /** @brief 是否已超时。 */
    bool expired() const
    {
        if (budget_ms_ <= 0)
        {
            return false;
        }
        return elapsedMs() >= static_cast<double>(budget_ms_);
    }

    /** @brief 已用毫秒。 */
    double elapsedMs() const
    {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    /** @brief 剩余预算毫秒；未设预算时返回 0。 */
    int64_t remainingMs() const
    {
        if (budget_ms_ <= 0)
        {
            return 0;
        }
        const auto remaining = static_cast<double>(budget_ms_) - elapsedMs();
        return remaining > 0.0 ? static_cast<int64_t>(remaining) : 0;
    }

private:
    int64_t                                budget_ms_{0};
    std::chrono::steady_clock::time_point  start_{};
};

/** @brief 单调时钟毫秒读数，用于阶段计时。 */
inline double dinoNowMs()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace irt::features::priv
