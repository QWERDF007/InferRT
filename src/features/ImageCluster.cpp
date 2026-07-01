/**
 * @file ImageCluster.cpp
 * @brief ``ImageCluster`` 公共 API 的 PIMPL 转发实现。
 */

#include "priv/ImageClusterImpl.hpp"

#include <utility>

namespace irt::features {

ImageCluster::ImageCluster(ImageClusterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

ImageCluster::~ImageCluster() = default;

ImageCluster::ImageCluster(ImageCluster &&other) noexcept = default;

ImageCluster &ImageCluster::operator=(ImageCluster &&other) noexcept = default;

ImageClusterResult ImageCluster::cluster(const std::filesystem::path         &weights_file,
                                         const std::vector<ImageClusterItem> &items,
                                         ImageClusterProgressCallback         progress_callback)
{
    return impl_->cluster(weights_file, items, std::move(progress_callback));
}

const ImageClusterConfig &ImageCluster::config() const noexcept
{
    return impl_->config();
}

int ImageCluster::featureDim() const noexcept
{
    return impl_->featureDim();
}

} // namespace irt::features
