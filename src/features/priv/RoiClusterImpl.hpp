#pragma once

/**
 * @file RoiClusterImpl.hpp
 * @brief ``RoiCluster::Impl`` PIMPL 声明。
 */

#include <inferrt/features/RoiCluster.hpp>

namespace irt::features {

/**
 * @brief ``RoiCluster`` 的私有实现。
 */
class RoiCluster::Impl
{
public:
    explicit Impl(RoiClusterConfig config);
    ~Impl();

    Impl(const Impl &)            = delete;
    Impl &operator=(const Impl &) = delete;

    RoiClusterResult cluster(const std::filesystem::path &weights_file, const std::vector<RoiClusterItem> &items,
                             RoiClusterProgressCallback progress_callback);

    const RoiClusterConfig &config() const noexcept;
    int                     featureDim() const noexcept;

private:
    RoiClusterResult clusterFeatures(const float *features, const int64_t *roi_ids, size_t count, int feature_dim,
                                     RoiClusterProgressCallback progress_callback);

    RoiClusterConfig config_{};
    int              feature_dim_{0};
};

} // namespace irt::features
