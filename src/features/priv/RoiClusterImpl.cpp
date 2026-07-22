/**
 * @file RoiClusterImpl.cpp
 * @brief ROI 聚类 PIMPL：共享图像特征图、ROIAlign 与 HDBSCAN。
 */

#include "RoiClusterImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "RoiFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

void reportProgress(const RoiClusterProgressCallback &callback, RoiClusterStage stage, size_t batch_index = 0,
                    size_t batch_begin = 0, size_t batch_count = 0, size_t processed_count = 0,
                    size_t total_count = 0)
{
    if (!callback)
    {
        return;
    }
    callback(RoiClusterProgress{stage, batch_index, batch_begin, batch_count, processed_count, total_count});
}

RoiClusterItem normalizeItem(const RoiClusterItem &item)
{
    validateRoi(item.roi);
    return RoiClusterItem{item.roi_id, priv::normalizeImageFilePath(item.image_path, "RoiCluster"), item.roi};
}

std::vector<RoiClusterItem> normalizeItems(const std::vector<RoiClusterItem> &items)
{
    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI cluster item list must not be empty");
    }

    std::vector<RoiClusterItem> normalized;
    normalized.reserve(items.size());
    for (const auto &item : items)
    {
        normalized.push_back(normalizeItem(item));
    }
    return normalized;
}

} // namespace

RoiCluster::Impl::Impl(RoiClusterConfig config)
    : config_(std::move(config))
{
    if (!irt::model::isSupportedModel(config_.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", config_.model_name.c_str());
    }
    if (config_.feature_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "feature name must not be empty");
    }
    if (config_.pooled_height <= 0 || config_.pooled_width <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI pooled size must be positive");
    }
    if (config_.sampling_ratio < -1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI sampling ratio must be -1 or non-negative");
    }
    if (config_.pca_dim < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be non-negative");
    }
    if (config_.use_pca && config_.pca_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be positive when PCA is enabled");
    }
    priv::validateFeatureSearchConfig(config_, "RoiCluster");
}

RoiCluster::Impl::~Impl() = default;

RoiClusterResult RoiCluster::Impl::cluster(const fs::path &weights_file,
                                           const std::vector<RoiClusterItem> &items,
                                           RoiClusterProgressCallback progress_callback)
{
    const auto normalized_items = normalizeItems(items);

    reportProgress(progress_callback, RoiClusterStage::LoadingModel, 0, 0, 0, 0, 1);
    priv::RoiFeatureExtractor extractor(config_, weights_file);
    feature_dim_ = extractor.featureDim();
    reportProgress(progress_callback, RoiClusterStage::LoadingModel, 0, 0, 0, 1, 1);

    reportProgress(progress_callback, RoiClusterStage::ExtractingFeatures, 0, 0, 0, 0, normalized_items.size());
    const auto features = extractor.extractAll(
        normalized_items,
        [&](size_t batch_index, size_t batch_begin, size_t batch_count, size_t processed_count)
        {
            reportProgress(progress_callback, RoiClusterStage::ExtractingFeatures, batch_index, batch_begin,
                           batch_count, processed_count, normalized_items.size());
        });

    const auto expected_size = normalized_items.size() * static_cast<size_t>(feature_dim_);
    if (features.size() != expected_size)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ROI cluster feature matrix size mismatch");
    }

    reportProgress(progress_callback, RoiClusterStage::Clustering, 0, 0, 0, 0, 1);
    const auto hdbscan = irt::ops::hdbscan(features.data(), static_cast<int64_t>(normalized_items.size()),
                                           static_cast<int64_t>(feature_dim_), config_.hdbscan);
    reportProgress(progress_callback, RoiClusterStage::Clustering, 0, 0, 0, 1, 1);

    if (hdbscan.labels.size() != normalized_items.size()
        || hdbscan.probabilities.size() != normalized_items.size())
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ROI cluster HDBSCAN result size mismatch");
    }

    RoiClusterResult result;
    result.feature_dim = feature_dim_;
    result.assignments.reserve(normalized_items.size());
    std::set<int64_t> cluster_ids;
    for (size_t i = 0; i < normalized_items.size(); ++i)
    {
        const auto label = hdbscan.labels[i];
        if (label < 0)
        {
            ++result.noise_count;
        }
        else
        {
            cluster_ids.insert(label);
        }
        result.assignments.push_back({normalized_items[i].roi_id, label, hdbscan.probabilities[i]});
    }
    result.cluster_count = static_cast<int64_t>(cluster_ids.size());
    return result;
}

const RoiClusterConfig &RoiCluster::Impl::config() const noexcept
{
    return config_;
}

int RoiCluster::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

} // namespace irt::features
