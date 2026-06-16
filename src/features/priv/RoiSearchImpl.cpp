/**
 * @file RoiSearchImpl.cpp
 * @brief ROI 搜索匹配 PIMPL 实现。
 */

#include "RoiSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/ops/RoIAlign.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#pragma warning(pop)

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

/**
 * @brief 将 ROI 配置切片为图像检索共用配置。
 */
ImageSearchConfig featureSearchConfig(const RoiSearchConfig &config)
{
    const ImageSearchConfig &base = config;
    return base;
}

/**
 * @brief 判断 ROI 坐标是否合法。
 */
bool isValidRoi(const RoiSearchBox &roi) noexcept
{
    return std::isfinite(roi.x1) && std::isfinite(roi.y1) && std::isfinite(roi.x2) && std::isfinite(roi.y2)
        && roi.x2 > roi.x1 && roi.y2 > roi.y1;
}

/**
 * @brief 校验 ROI 坐标，失败时抛出异常。
 */
void validateRoi(const RoiSearchBox &roi)
{
    if (!isValidRoi(roi))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI must be finite and satisfy x2 > x1, y2 > y1");
    }
}

/**
 * @brief 校验并规范化单个 ROI 条目。
 */
RoiSearchItem normalizeItem(const RoiSearchItem &item)
{
    if (item.image_path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI image path must not be empty");
    }

    std::error_code ec;
    const bool      exists          = fs::exists(item.image_path, ec);
    const bool      is_regular_file = exists && fs::is_regular_file(item.image_path, ec);
    if (!is_regular_file)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI image path does not exist: %s",
                             item.image_path.string().c_str());
    }
    if (!ImageSearch::isImageFile(item.image_path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ROI image file: %s",
                             item.image_path.string().c_str());
    }
    validateRoi(item.roi);
    return RoiSearchItem{fs::absolute(item.image_path), item.roi};
}

/**
 * @brief 校验并规范化 ROI 特征库条目。
 */
std::vector<RoiSearchItem> normalizeItems(const std::vector<RoiSearchItem> &items)
{
    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI item list must not be empty");
    }

    std::vector<RoiSearchItem> normalized;
    normalized.reserve(items.size());
    for (const auto &item : items)
    {
        normalized.push_back(normalizeItem(item));
    }
    return normalized;
}

/**
 * @brief 由索引路径推导 ROI 映射文件路径。
 */
fs::path roiMappingPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".rois.txt";
}

/**
 * @brief 保存 Faiss ID 到 ROI 条目的映射。
 */
void saveRoiMapping(const fs::path &mapping_path, const std::vector<RoiSearchItem> &items)
{
    std::ofstream output(mapping_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI mapping file: %s",
                             mapping_path.string().c_str());
    }

    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto &item : items)
    {
        output << item.image_path.generic_string() << '\t' << item.roi.x1 << '\t' << item.roi.y1 << '\t' << item.roi.x2
               << '\t' << item.roi.y2 << '\n';
    }
}

/**
 * @brief 从映射文件加载 ROI 条目列表。
 */
std::vector<RoiSearchItem> loadRoiMapping(const fs::path &mapping_path)
{
    std::ifstream input(mapping_path);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI mapping file: %s",
                             mapping_path.string().c_str());
    }

    std::vector<RoiSearchItem> items;
    std::string                line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> fields;
        size_t                   begin = 0;
        while (begin <= line.size())
        {
            const auto end = line.find('\t', begin);
            fields.push_back(line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos)
            {
                break;
            }
            begin = end + 1;
        }
        if (fields.size() != 5)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI mapping line: %s", line.c_str());
        }

        RoiSearchItem item;
        item.image_path = fields[0];
        item.roi.x1     = std::stof(fields[1]);
        item.roi.y1     = std::stof(fields[2]);
        item.roi.x2     = std::stof(fields[3]);
        item.roi.y2     = std::stof(fields[4]);
        validateRoi(item.roi);
        items.push_back(std::move(item));
    }
    return items;
}

/**
 * @brief 判断两个 ROI 框是否近似相等。
 */
bool roiNearlyEquals(const RoiSearchBox &lhs, const RoiSearchBox &rhs) noexcept
{
    constexpr float eps = 1.0e-5f;
    return std::abs(lhs.x1 - rhs.x1) <= eps && std::abs(lhs.y1 - rhs.y1) <= eps && std::abs(lhs.x2 - rhs.x2) <= eps
        && std::abs(lhs.y2 - rhs.y2) <= eps;
}

/**
 * @brief 判断 ROI 条目列表是否与映射文件内容一致。
 */
bool itemsMatch(const std::vector<RoiSearchItem> &lhs, const std::vector<RoiSearchItem> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i].image_path != rhs[i].image_path || !roiNearlyEquals(lhs[i].roi, rhs[i].roi))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief 序列化布尔值。
 */
const char *boolName(bool value) noexcept
{
    return value ? "true" : "false";
}

/**
 * @brief 保存 ROI 搜索索引元数据。
 */
void saveRoiMetadata(const fs::path &metadata_path, const RoiSearchConfig &config)
{
    std::ofstream output(metadata_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI metadata file: %s",
                             metadata_path.string().c_str());
    }

    output << "kind=roi_search\n";
    output << "model=" << config.model_name << "\n";
    output << "feature=" << config.feature_name << "\n";
    output << "model_backend=" << irt::model::modelBackendName(config.model_backend) << "\n";
    output << "model_device=" << irt::model::modelDeviceName(config.model_device) << "\n";
    output << "preprocess_backend=" << priv::preprocessBackendName(config.preprocess_backend) << "\n";
    output << "norm=" << priv::featureNormName(config.norm) << "\n";
    output << "faiss_backend=" << priv::faissBackendName(config.faiss_backend) << "\n";
    output << "index_storage=" << priv::indexStorageName(config.index_storage) << "\n";
    output << "disk_build_batch_size=" << config.disk_build_batch_size << "\n";
    output << "model_batch_size=" << config.model_batch_size << "\n";
    output << "index_kind=" << priv::indexKindName(config) << "\n";
    output << "roi_pooled_height=" << config.pooled_height << "\n";
    output << "roi_pooled_width=" << config.pooled_width << "\n";
    output << "roi_sampling_ratio=" << config.sampling_ratio << "\n";
    output << "roi_aligned=" << boolName(config.aligned) << "\n";
}

/**
 * @brief 加载 key=value 元数据。
 */
std::unordered_map<std::string, std::string> loadMetadata(const fs::path &metadata_path)
{
    std::ifstream input(metadata_path);
    if (!input)
    {
        return {};
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string                                  line;
    while (std::getline(input, line))
    {
        const auto separator = line.find('=');
        if (separator == std::string::npos)
        {
            continue;
        }
        metadata.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return metadata;
}

/**
 * @brief 判断元数据字段是否与期望值一致。
 */
bool metadataEquals(const std::unordered_map<std::string, std::string> &metadata, const std::string &key,
                    const std::string &expected)
{
    const auto it = metadata.find(key);
    return it != metadata.end() && it->second == expected;
}

/**
 * @brief 判断磁盘上的 ROI 索引是否匹配当前配置。
 */
bool existingRoiIndexMatchesConfig(const fs::path &index_path, const RoiSearchConfig &config)
{
    const auto metadata = loadMetadata(priv::metadataPathFromIndex(index_path));
    if (metadata.empty())
    {
        return false;
    }

    return metadataEquals(metadata, "kind", "roi_search") && metadataEquals(metadata, "model", config.model_name)
        && metadataEquals(metadata, "feature", config.feature_name)
        && metadataEquals(metadata, "model_backend", irt::model::modelBackendName(config.model_backend))
        && metadataEquals(metadata, "model_device", irt::model::modelDeviceName(config.model_device))
        && metadataEquals(metadata, "preprocess_backend", priv::preprocessBackendName(config.preprocess_backend))
        && metadataEquals(metadata, "norm", priv::featureNormName(config.norm))
        && metadataEquals(metadata, "faiss_backend", priv::faissBackendName(config.faiss_backend))
        && metadataEquals(metadata, "index_storage", priv::indexStorageName(config.index_storage))
        && metadataEquals(metadata, "index_kind", priv::indexKindName(config))
        && metadataEquals(metadata, "roi_pooled_height", std::to_string(config.pooled_height))
        && metadataEquals(metadata, "roi_pooled_width", std::to_string(config.pooled_width))
        && metadataEquals(metadata, "roi_sampling_ratio", std::to_string(config.sampling_ratio))
        && metadataEquals(metadata, "roi_aligned", boolName(config.aligned));
}

} // namespace

namespace priv {

/**
 * @brief ROI 特征抽取器：模型特征图 + ROIAlign + 展平归一化。
 */
class RoiFeatureExtractor
{
public:
    RoiFeatureExtractor(const RoiSearchConfig &config, const fs::path &weights_file)
        : config_(config)
        , image_extractor_(config.model_name, config.feature_name, weights_file, featureSearchConfig(config))
    {
        const auto dims = image_extractor_.featureTensorShape();
        if (dims.nbDims != 4 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "RoiSearch feature tensor must be NCHW spatial feature map");
        }
        feature_dim_ = static_cast<int>(dims.d[1]) * config_.pooled_height * config_.pooled_width;
    }

    /**
     * @brief 获取 ROI 特征向量维度。
     */
    int featureDim() const noexcept
    {
        return feature_dim_;
    }

    /**
     * @brief 提取单个 ROI 的归一化特征。
     */
    std::vector<float> extract(const RoiSearchItem &item)
    {
        validateRoi(item.roi);
        const std::vector<fs::path> paths{item.image_path};
        const auto                  tensor = image_extractor_.extractFeatureTensorBatch(paths, 0, 1);
        return roiAlignAndFlatten(tensor, item.roi, 0);
    }

    /**
     * @brief 批量提取 ROI 特征。
     */
    std::vector<float> extractBatch(const std::vector<RoiSearchItem> &items, size_t begin, size_t count)
    {
        if (begin > items.size() || count > items.size() - begin)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch range is invalid");
        }

        std::vector<float> features;
        features.reserve(count * static_cast<size_t>(feature_dim_));
        for (size_t i = 0; i < count; ++i)
        {
            auto feature = extract(items[begin + i]);
            features.insert(features.end(), feature.begin(), feature.end());
        }
        return features;
    }

    /**
     * @brief 按任意下标批量提取 ROI 特征。
     */
    std::vector<float> extractBatch(const std::vector<RoiSearchItem> &items, const std::vector<size_t> &indices)
    {
        std::vector<float> features;
        features.reserve(indices.size() * static_cast<size_t>(feature_dim_));
        for (const auto index : indices)
        {
            if (index >= items.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch index is invalid");
            }
            auto feature = extract(items[index]);
            features.insert(features.end(), feature.begin(), feature.end());
        }
        return features;
    }

private:
    /**
     * @brief 将原图 ROI 映射到特征图坐标并执行 ROIAlign。
     */
    std::vector<float> roiAlignAndFlatten(const FeatureTensorBatch &tensor, const RoiSearchBox &roi,
                                          size_t batch_index) const
    {
        if (tensor.dims.nbDims != 4 || tensor.dims.d[1] <= 0 || tensor.dims.d[2] <= 0 || tensor.dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch requires NCHW feature tensor");
        }
        if (batch_index >= tensor.original_sizes.size() || batch_index >= static_cast<size_t>(tensor.dims.d[0]))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI batch index is invalid");
        }

        const auto image_size = tensor.original_sizes[batch_index];
        if (image_size.width <= 0 || image_size.height <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid source image size for ROI mapping");
        }

        const auto feature_height = static_cast<float>(tensor.dims.d[2]);
        const auto feature_width  = static_cast<float>(tensor.dims.d[3]);
        const auto scale_x        = feature_width / static_cast<float>(image_size.width);
        const auto scale_y        = feature_height / static_cast<float>(image_size.height);

        const std::array<float, 5> mapped_roi{
            static_cast<float>(batch_index), roi.x1 * scale_x, roi.y1 * scale_y, roi.x2 * scale_x, roi.y2 * scale_y,
        };
        const int64_t input_shape[4]{
            tensor.dims.d[0],
            tensor.dims.d[1],
            tensor.dims.d[2],
            tensor.dims.d[3],
        };

        const irt::ops::RoIAlign roi_align({config_.pooled_height, config_.pooled_width}, 1.0f, config_.sampling_ratio,
                                           config_.aligned);
        auto                     feature = roi_align.forward(tensor.data.data(), input_shape, mapped_roi.data(), 1);
        if (feature.size() != static_cast<size_t>(feature_dim_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ROI feature size");
        }
        normalizeFeature(feature, config_.norm);
        return feature;
    }

    RoiSearchConfig       config_{};        ///< ROI 检索配置。
    ImageFeatureExtractor image_extractor_; ///< 共用图像特征图抽取器。
    int                   feature_dim_{0};  ///< ROI 特征向量维度。
};

} // namespace priv

RoiSearch::Impl::Impl(RoiSearchConfig config)
    : config_(std::move(config))
{
    if (config_.faiss_backend == ImageSearchFaissBackend::GPU)
    {
        config_.index_storage = ImageSearchIndexStorage::RAM;
    }
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
    priv::validateFeatureSearchConfig(config_, "RoiSearch");
}

RoiSearch::Impl::~Impl() = default;

void RoiSearch::Impl::buildOrLoad(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                                  const fs::path &index_file, bool rebuild_index,
                                  RoiSearchBuildProgressCallback progress_callback)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());

    if (!rebuild_index && fs::exists(index_file) && fs::exists(roiMappingPathFromIndex(index_file))
        && existingRoiIndexMatchesConfig(index_file, config_))
    {
        const auto mapped_items = loadRoiMapping(roiMappingPathFromIndex(index_file));
        if (itemsMatch(normalized_items, mapped_items))
        {
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
            load(weights_file, index_file);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0,
                                      gallery_items_.size(), gallery_items_.size());
            return;
        }
    }

    buildWithItems(weights_file, std::move(normalized_items), index_file, std::move(progress_callback));
}

void RoiSearch::Impl::build(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                            const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());
    buildWithItems(weights_file, std::move(normalized_items), index_file, std::move(progress_callback));
}

void RoiSearch::Impl::load(const fs::path &weights_file, const fs::path &index_file)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }
    if (!fs::exists(index_file))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index file does not exist: %s",
                             index_file.string().c_str());
    }
    if (!fs::exists(roiMappingPathFromIndex(index_file)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI mapping file does not exist: %s",
                             roiMappingPathFromIndex(index_file).string().c_str());
    }
    if (!existingRoiIndexMatchesConfig(index_file, config_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index metadata does not match current config: %s",
                             priv::metadataPathFromIndex(index_file).string().c_str());
    }

    auto items = loadRoiMapping(roiMappingPathFromIndex(index_file));
    auto faiss = priv::loadConfiguredFaissIndex(index_file, config_);
    if (static_cast<faiss::idx_t>(items.size()) != faiss.index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index size (%lld) does not match mapping size (%zu)",
                             static_cast<long long>(faiss.index->ntotal), items.size());
    }

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    index_path_          = index_file;
    faiss_gpu_resources_ = std::move(faiss.gpu_resources);
    index_               = std::move(faiss.index);
    gallery_items_       = std::move(items);
    extractor_.reset();
    feature_dim_ = static_cast<int>(index_->d);
}

void RoiSearch::Impl::buildWithItems(const fs::path &weights_file, std::vector<RoiSearchItem> gallery_items,
                                     const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 0, 1);
    auto extractor = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 1, 1);

    if (!index_file.parent_path().empty())
    {
        fs::create_directories(index_file.parent_path());
    }

    auto faiss = priv::buildConfiguredFaissIndex(
        gallery_items.size(), extractor->featureDim(), index_file, config_,
        [&](size_t index) { return extractor->extract(gallery_items[index]); }, [&](size_t begin, size_t count)
        { return extractor->extractBatch(gallery_items, begin, count); }, [&](const std::vector<size_t> &indices)
        { return extractor->extractBatch(gallery_items, indices); }, progress_callback);
    saveRoiMapping(roiMappingPathFromIndex(index_file), gallery_items);

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    index_path_          = index_file;
    faiss_gpu_resources_ = std::move(faiss.gpu_resources);
    index_               = std::move(faiss.index);
    gallery_items_       = std::move(gallery_items);
    feature_dim_         = extractor->featureDim();
    extractor_           = std::move(extractor);

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 0, 1);
    saveRoiMetadata(priv::metadataPathFromIndex(index_path_), config_);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 1, 1);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0, gallery_items_.size(),
                              gallery_items_.size());
}

std::vector<RoiSearchResult> RoiSearch::Impl::search(const fs::path &query_image, const RoiSearchBox &roi, int top_k)
{
    if (!index_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "RoiSearch index is not ready");
    }
    if (top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }
    if (index_->ntotal <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "RoiSearch index is empty");
    }

    const auto query_item = normalizeItem(RoiSearchItem{query_image, roi});
    ensureExtractor();
    const auto query_feature = extractor_->extract(query_item);
    if (query_feature.size() != static_cast<size_t>(index_->d))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI query feature dimension mismatch");
    }

    const int                 result_count = std::min(top_k, static_cast<int>(index_->ntotal));
    std::vector<faiss::idx_t> indices(result_count);
    std::vector<float>        distances(result_count);
    index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());

    std::vector<RoiSearchResult> results;
    results.reserve(static_cast<size_t>(result_count));
    for (int i = 0; i < result_count; ++i)
    {
        if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_items_.size())
        {
            continue;
        }
        const auto item_index = static_cast<size_t>(indices[i]);
        results.push_back(
            {distances[i], gallery_items_[item_index].image_path, gallery_items_[item_index].roi, item_index});
    }
    return results;
}

bool RoiSearch::Impl::isReady() const noexcept
{
    return index_ != nullptr;
}

const RoiSearchConfig &RoiSearch::Impl::config() const noexcept
{
    return config_;
}

const fs::path &RoiSearch::Impl::indexPath() const noexcept
{
    return index_path_;
}

std::vector<RoiSearchItem> RoiSearch::Impl::galleryItems() const
{
    return gallery_items_;
}

int RoiSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void RoiSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        extractor_   = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
