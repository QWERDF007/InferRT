#pragma once

#include <inferrt/util/Export.h>

#include <chrono>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace irt::util {

using TimingClock = std::chrono::steady_clock;

struct TimingStats
{
    double total_ms{0.0};
    double avg_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

INFERRT_UTIL_API double elapsedMs(TimingClock::time_point start, TimingClock::time_point end);

INFERRT_UTIL_API TimingStats summarizeTimings(const std::vector<double> &values);

/**
 * @brief 输出统一格式的计时统计。
 *
 * @param prefix 输出名称前的文本，例如 `", "` 或 `"  "`。
 */
INFERRT_UTIL_API void printTimingStats(std::ostream &output, std::string_view name, const TimingStats &stats,
                                       std::string_view prefix = ", ");

} // namespace irt::util
