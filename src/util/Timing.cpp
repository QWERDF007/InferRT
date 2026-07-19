#include <inferrt/util/Timing.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace irt::util {

double elapsedMs(TimingClock::time_point start, TimingClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

TimingStats summarizeTimings(const std::vector<double> &values)
{
    if (values.empty())
    {
        return {};
    }

    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const double total          = std::accumulate(values.begin(), values.end(), 0.0);
    return {total, total / static_cast<double>(values.size()), *min_it, *max_it};
}

void printTimingStats(std::ostream &output, std::string_view name, const TimingStats &stats, std::string_view prefix)
{
    output << prefix << name << "_total=" << stats.total_ms << " ms, " << name << "_avg=" << stats.avg_ms
           << " ms, " << name << "_min=" << stats.min_ms << " ms, " << name << "_max=" << stats.max_ms
           << " ms";
}

} // namespace irt::util
