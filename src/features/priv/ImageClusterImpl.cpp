/**
 * @file ImageClusterImpl.cpp
 * @brief 图像聚类 PIMPL：特征提取、可选本地 PCA 与 HDBSCAN 聚类。
 */

#include "ImageClusterImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <set>
#include <system_error>
#include <utility>
#include <vector>


namespace fs = std::filesystem;

namespace irt::features {
namespace {

void reportProgress(const ImageClusterProgressCallback &callback, ImageClusterStage stage, size_t batch_index = 0,
                    size_t batch_begin = 0, size_t batch_count = 0, size_t processed_count = 0, size_t total_count = 0)
{
    if (!callback)
    {
        return;
    }
    callback(ImageClusterProgress{stage, batch_index, batch_begin, batch_count, processed_count, total_count});
}

ImageClusterItem normalizeItem(const ImageClusterItem &item)
{
    return ImageClusterItem{item.image_id, priv::normalizeImageFilePath(item.image_path, "ImageCluster")};
}

std::vector<ImageClusterItem> normalizeItems(const std::vector<ImageClusterItem> &items)
{
    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster item list must not be empty");
    }

    std::vector<ImageClusterItem> normalized;
    normalized.reserve(items.size());
    for (const auto &item : items)
    {
        normalized.push_back(normalizeItem(item));
    }
    return normalized;
}

std::vector<fs::path> itemPaths(const std::vector<ImageClusterItem> &items)
{
    std::vector<fs::path> paths;
    paths.reserve(items.size());
    for (const auto &item : items)
    {
        paths.push_back(item.image_path);
    }
    return paths;
}

std::vector<int64_t> itemIds(const std::vector<ImageClusterItem> &items)
{
    std::vector<int64_t> ids;
    ids.reserve(items.size());
    for (const auto &item : items)
    {
        ids.push_back(item.image_id);
    }
    return ids;
}

/**
 * @brief 使用 OpenCV 对本图局部通道向量执行 PCA。
 */
std::vector<float> projectLocalPca(const std::vector<float> &rows, size_t row_count, int input_dim, int output_dim)
{
    if (row_count == 0 || input_dim <= 0 || output_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster PCA requires non-empty features");
    }
    if (row_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster PCA feature count is too large");
    }
    if (output_dim > input_dim || output_dim > static_cast<int>(row_count))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageCluster PCA dim must not exceed channel count or local sample count");
    }
    const auto expected_input_elements = irt::checkedSizeMul(row_count, static_cast<size_t>(input_dim),
                                                             "ImageCluster PCA input elements");
    if (rows.size() != expected_input_elements)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster PCA input size mismatch");
    }

    cv::Mat       input(static_cast<int>(row_count), input_dim, CV_32F, const_cast<float *>(rows.data()));
    const cv::PCA pca(input, cv::Mat(), cv::PCA::DATA_AS_ROW, output_dim);
    cv::Mat       output;
    pca.project(input, output);
    if (output.rows != static_cast<int>(row_count) || output.cols != output_dim || output.type() != CV_32F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster PCA output shape mismatch");
    }

    const auto projected_elements = irt::checkedSizeMul(row_count, static_cast<size_t>(output_dim),
                                                        "ImageCluster PCA output elements");
    std::vector<float> projected(projected_elements);
    if (output.isContinuous())
    {
        std::memcpy(projected.data(), output.ptr<float>(),
                    irt::checkedSizeMul(projected.size(), sizeof(float), "ImageCluster PCA output bytes"));
    }
    else
    {
        for (size_t row = 0; row < row_count; ++row)
        {
            std::memcpy(projected.data() + row * static_cast<size_t>(output_dim),
                        output.ptr<float>(static_cast<int>(row)),
                        irt::checkedSizeMul(static_cast<size_t>(output_dim), sizeof(float),
                                            "ImageCluster PCA row bytes"));
        }
    }
    return projected;
}

/**
 * @brief 聚类特征抽取器，负责可选的每图本地 PCA。
 */
class ClusterFeatureExtractor
{
public:
    ClusterFeatureExtractor(const ImageClusterConfig &config, const fs::path &weights_file)
        : config_(config)
        , image_extractor_(config.model_name, config.feature_name, weights_file, ImageSearchConfig(config))
    {
        if (!config_.use_pca)
        {
            feature_dim_ = image_extractor_.featureDim();
            return;
        }

        const auto dims = image_extractor_.featureTensorShape();
        if (dims.nbDims == 4 && dims.d[1] > 0 && dims.d[2] > 0 && dims.d[3] > 0)
        {
            layout_           = Layout::Nchw;
            local_samples_    = irt::checkedSizeToInt(
                irt::checkedSizeMul(static_cast<size_t>(dims.d[2]), static_cast<size_t>(dims.d[3]),
                                    "ImageCluster local feature samples"),
                "ImageCluster local feature samples");
            feature_channels_ = static_cast<int>(dims.d[1]);
        }
        else if (dims.nbDims == 3 && dims.d[1] > 0 && dims.d[2] > 0)
        {
            layout_           = Layout::RowsChannels;
            local_samples_    = static_cast<int>(dims.d[1]);
            feature_channels_ = static_cast<int>(dims.d[2]);
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageCluster PCA requires NCHW feature map or BxSamplesxChannels tensor");
        }

        if (config_.pca_dim <= 0 || config_.pca_dim > feature_channels_ || config_.pca_dim > local_samples_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageCluster PCA dim must be positive and not exceed channel/sample count");
        }
        feature_dim_ = irt::checkedSizeToInt(
            irt::checkedSizeMul(static_cast<size_t>(config_.pca_dim), static_cast<size_t>(local_samples_),
                                "ImageCluster feature dimension"),
            "ImageCluster feature dimension");
    }

    int featureDim() const noexcept
    {
        return feature_dim_;
    }

    size_t maxBatchSize() const noexcept
    {
        return image_extractor_.maxBatchSize();
    }

    std::vector<float> extractBatch(const std::vector<fs::path> &image_paths, size_t begin, size_t count)
    {
        if (!config_.use_pca)
        {
            return image_extractor_.extractBatch(image_paths, begin, count);
        }

        if (count == 0)
        {
            return {};
        }
        if (count > image_extractor_.maxBatchSize())
        {
            std::vector<float> features;
            features.reserve(irt::checkedSizeMul(count, static_cast<size_t>(feature_dim_),
                                                 "ImageCluster batch feature elements"));
            for (size_t offset = 0; offset < count; offset += image_extractor_.maxBatchSize())
            {
                const size_t chunk_count = std::min(image_extractor_.maxBatchSize(), count - offset);
                auto         chunk       = extractBatch(image_paths, begin + offset, chunk_count);
                features.insert(features.end(), chunk.begin(), chunk.end());
            }
            return features;
        }

        const auto tensor = image_extractor_.extractFeatureTensorBatch(image_paths, begin, count);
        if (tensor.dims.nbDims <= 0 || tensor.dims.d[0] != static_cast<int32_t>(count))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster feature batch shape mismatch");
        }

        std::vector<float> features;
        features.reserve(irt::checkedSizeMul(count, static_cast<size_t>(feature_dim_),
                                             "ImageCluster batch feature elements"));
        for (size_t b = 0; b < count; ++b)
        {
            auto sample = layout_ == Layout::Nchw ? extractPcaNchwSample(tensor, b) : extractPcaRowsSample(tensor, b);
            priv::normalizeFeature(sample, config_.norm);
            features.insert(features.end(), sample.begin(), sample.end());
        }
        return features;
    }

private:
    enum class Layout
    {
        Nchw,
        RowsChannels,
    };

    std::vector<float> extractPcaNchwSample(const priv::FeatureTensorBatch &tensor, size_t batch_index) const
    {
        const int batch    = static_cast<int>(tensor.dims.d[0]);
        const int channels = static_cast<int>(tensor.dims.d[1]);
        const int height   = static_cast<int>(tensor.dims.d[2]);
        const int width    = static_cast<int>(tensor.dims.d[3]);
        const auto sample_count = irt::checkedSizeMul(static_cast<size_t>(height), static_cast<size_t>(width),
                                                      "ImageCluster feature map samples");
        if (batch_index >= static_cast<size_t>(batch) || channels != feature_channels_
            || sample_count != static_cast<size_t>(local_samples_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster NCHW feature shape mismatch");
        }

        const auto sample_size  = irt::checkedSizeMul(static_cast<size_t>(channels), sample_count,
                                                      "ImageCluster feature map sample elements");
        const auto base         = tensor.data.data() + batch_index * sample_size;

        std::vector<float> rows(sample_size);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const auto row = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                for (int c = 0; c < channels; ++c)
                {
                    const auto src = (static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(y))
                                       * static_cast<size_t>(width)
                                   + static_cast<size_t>(x);
                    rows[row * static_cast<size_t>(channels) + static_cast<size_t>(c)] = base[src];
                }
            }
        }

        const auto projected = projectLocalPca(rows, sample_count, channels, config_.pca_dim);

        std::vector<float> flattened(irt::checkedSizeMul(static_cast<size_t>(config_.pca_dim), sample_count,
                                                         "ImageCluster projected feature elements"));
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const auto row = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                for (int c = 0; c < config_.pca_dim; ++c)
                {
                    const auto dst = (static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(y))
                                       * static_cast<size_t>(width)
                                   + static_cast<size_t>(x);
                    flattened[dst] = projected[row * static_cast<size_t>(config_.pca_dim) + static_cast<size_t>(c)];
                }
            }
        }
        return flattened;
    }

    std::vector<float> extractPcaRowsSample(const priv::FeatureTensorBatch &tensor, size_t batch_index) const
    {
        const int batch    = static_cast<int>(tensor.dims.d[0]);
        const int rows     = static_cast<int>(tensor.dims.d[1]);
        const int channels = static_cast<int>(tensor.dims.d[2]);
        if (batch_index >= static_cast<size_t>(batch) || rows != local_samples_ || channels != feature_channels_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageCluster BxSamplesxChannels feature shape mismatch");
        }

        const auto         sample_count = static_cast<size_t>(rows);
        const auto         sample_size  = irt::checkedSizeMul(sample_count, static_cast<size_t>(channels),
                                                               "ImageCluster row feature elements");
        const auto         base         = tensor.data.data() + batch_index * sample_size;
        std::vector<float> local_rows(base, base + sample_size);
        return projectLocalPca(local_rows, sample_count, channels, config_.pca_dim);
    }

    ImageClusterConfig          config_{};
    priv::ImageFeatureExtractor image_extractor_;
    Layout                      layout_{Layout::Nchw};
    int                         local_samples_{0};
    int                         feature_channels_{0};
    int                         feature_dim_{0};
};

} // namespace

ImageCluster::Impl::Impl(ImageClusterConfig config)
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
    if (config_.pca_dim < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageCluster PCA dim must be non-negative");
    }
    if (config_.use_pca && config_.pca_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageCluster PCA dim must be positive when PCA is enabled");
    }
    priv::validateFeatureSearchConfig(config_, "ImageCluster");
}

ImageCluster::Impl::~Impl() = default;

ImageClusterResult ImageCluster::Impl::cluster(const fs::path &weights_file, const std::vector<ImageClusterItem> &items,
                                               ImageClusterProgressCallback progress_callback)
{
    auto       normalized_items = normalizeItems(items);
    const auto image_paths      = itemPaths(normalized_items);
    const auto image_ids        = itemIds(normalized_items);

    reportProgress(progress_callback, ImageClusterStage::LoadingModel, 0, 0, 0, 0, 1);
    ClusterFeatureExtractor extractor(config_, weights_file);
    feature_dim_ = extractor.featureDim();
    reportProgress(progress_callback, ImageClusterStage::LoadingModel, 0, 0, 0, 1, 1);

    std::vector<float> features;
    features.reserve(irt::checkedSizeMul(normalized_items.size(), static_cast<size_t>(feature_dim_),
                                         "ImageCluster feature elements"));
    reportProgress(progress_callback, ImageClusterStage::ExtractingFeatures, 0, 0, 0, 0, image_paths.size());
    priv::processFeatureBatches(
        image_paths.size(), extractor.maxBatchSize(), feature_dim_,
        [&](size_t begin, size_t count) { return extractor.extractBatch(image_paths, begin, count); },
        [&](size_t, size_t, const std::vector<float> &batch) { features.insert(features.end(), batch.begin(), batch.end()); },
        [&](const priv::FeatureBatchProgress &progress)
        {
            reportProgress(progress_callback, ImageClusterStage::ExtractingFeatures, progress.batch_index,
                           progress.batch_begin, progress.batch_count, progress.processed_count, progress.total_count);
        });

    reportProgress(progress_callback, ImageClusterStage::Clustering, 0, 0, 0, 0, 1);
    const auto hdbscan = irt::ops::hdbscan(features.data(), static_cast<int64_t>(normalized_items.size()), feature_dim_,
                                           config_.hdbscan);
    reportProgress(progress_callback, ImageClusterStage::Clustering, 0, 0, 0, 1, 1);

    if (hdbscan.labels.size() != normalized_items.size() || hdbscan.probabilities.size() != normalized_items.size())
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ImageCluster HDBSCAN result size mismatch");
    }

    ImageClusterResult result;
    result.feature_dim = feature_dim_;
    result.assignments.reserve(normalized_items.size());

    std::set<int64_t> cluster_ids;
    for (size_t i = 0; i < normalized_items.size(); ++i)
    {
        const int64_t label = hdbscan.labels[i];
        if (label < 0)
        {
            ++result.noise_count;
        }
        else
        {
            cluster_ids.insert(label);
        }
        result.assignments.push_back({image_ids[i], label, hdbscan.probabilities[i]});
    }
    result.cluster_count = static_cast<int64_t>(cluster_ids.size());

    return result;
}

const ImageClusterConfig &ImageCluster::Impl::config() const noexcept
{
    return config_;
}

int ImageCluster::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

} // namespace irt::features
