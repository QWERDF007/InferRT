/**
 * @file RoiCluster.cpp
 * @brief ``RoiCluster`` 公共 API 的 PIMPL 转发实现。
 */

#include "priv/RoiClusterImpl.hpp"

#include <utility>

namespace irt::features {

RoiCluster::RoiCluster(RoiClusterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

RoiCluster::~RoiCluster() = default;

RoiCluster::RoiCluster(RoiCluster &&other) noexcept = default;

RoiCluster &RoiCluster::operator=(RoiCluster &&other) noexcept = default;

RoiClusterResult RoiCluster::cluster(const std::filesystem::path &weights_file,
                                     const std::vector<RoiClusterItem> &items,
                                     RoiClusterProgressCallback progress_callback)
{
    return impl_->cluster(weights_file, items, std::move(progress_callback));
}

const RoiClusterConfig &RoiCluster::config() const noexcept
{
    return impl_->config();
}

int RoiCluster::featureDim() const noexcept
{
    return impl_->featureDim();
}

} // namespace irt::features
