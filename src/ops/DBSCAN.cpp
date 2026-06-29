#include "priv/ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/DBSCAN.hpp>

#include <cmath>
#include <vector>

namespace irt::ops {
namespace {

void validateInputs(const float *samples, int64_t num_samples, int64_t num_features, const DBSCANConfig &config)
{
    detail::validateSampleMatrix(samples, num_samples, num_features);
    detail::validateNeighborSearchConfig(config.algorithm, config.leaf_size);
    if (!std::isfinite(config.eps) || config.eps <= 0.0f)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "eps must be positive and finite");
    }
    if (config.min_samples < 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "min_samples must be positive, got %lld",
                        static_cast<long long>(config.min_samples));
    }
}

} // namespace

DBSCANResult dbscan(const float *samples, int64_t num_samples, int64_t num_features, const DBSCANConfig &config)
{
    validateInputs(samples, num_samples, num_features, config);
    if (num_samples == 0)
    {
        return {};
    }

    const auto neighborhoods = detail::radiusNeighborhoods(samples, num_samples, num_features, config.eps,
                                                           config.algorithm, config.leaf_size);

    std::vector<uint8_t> is_core(static_cast<size_t>(num_samples), uint8_t{0});
    DBSCANResult         result;
    result.labels.assign(static_cast<size_t>(num_samples), int64_t{-1});

    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (static_cast<int64_t>(neighborhoods[static_cast<size_t>(sample)].size()) >= config.min_samples)
        {
            is_core[static_cast<size_t>(sample)] = uint8_t{1};
            result.core_sample_indices.push_back(sample);
        }
    }

    int64_t              label_num = 0;
    std::vector<int64_t> stack;
    stack.reserve(static_cast<size_t>(num_samples));
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (result.labels[static_cast<size_t>(sample)] != -1 || !is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        int64_t current = sample;
        while (true)
        {
            if (result.labels[static_cast<size_t>(current)] == -1)
            {
                result.labels[static_cast<size_t>(current)] = label_num;
                if (is_core[static_cast<size_t>(current)])
                {
                    for (const int64_t neighbor : neighborhoods[static_cast<size_t>(current)])
                    {
                        if (result.labels[static_cast<size_t>(neighbor)] == -1)
                        {
                            stack.push_back(neighbor);
                        }
                    }
                }
            }

            if (stack.empty())
            {
                break;
            }
            current = stack.back();
            stack.pop_back();
        }
        ++label_num;
    }

    return result;
}

} // namespace irt::ops
