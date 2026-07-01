#pragma once

/**
 * @file ImageClusterImpl.hpp
 * @brief ``ImageCluster::Impl`` PIMPL 声明。
 */

#include <inferrt/features/ImageCluster.hpp>

namespace irt::features {

/**
 * @brief ``ImageCluster`` 的私有实现。
 */
class ImageCluster::Impl
{
public:
    explicit Impl(ImageClusterConfig config);
    ~Impl();

    Impl(const Impl &)            = delete;
    Impl &operator=(const Impl &) = delete;

    ImageClusterResult cluster(const std::filesystem::path &weights_file, const std::vector<ImageClusterItem> &items,
                               ImageClusterProgressCallback progress_callback);

    const ImageClusterConfig &config() const noexcept;
    int                       featureDim() const noexcept;

private:
    ImageClusterConfig config_{};
    int                feature_dim_{0};
};

} // namespace irt::features
