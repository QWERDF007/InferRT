/**
 * @file RoiSearchImpl.cpp
 * @brief ROI 搜索匹配 PIMPL 实现。
 */

#include "RoiSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "RoiFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/util/FileManifest.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#pragma warning(pop)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace fs = std::filesystem;

namespace irt::features {
namespace {

/**
 * @brief 校验并规范化单个 ROI 条目。
 */
RoiSearchItem normalizeItem(const RoiSearchItem &item)
{
    priv::validateRoi(item.roi);
    return RoiSearchItem{item.roi_id, priv::normalizeImageFilePath(item.image_path, "RoiSearch"), item.roi};
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
 * @brief 解析 ROI Faiss 索引输出路径；空路径使用当前工作目录下的时间戳文件名。
 */
fs::path resolveRoiIndexPath(const fs::path &index_file)
{
    return irt::util::resolveOutputFilePath(index_file, {}, ".faiss");
}

/**
 * @brief 生成绝对路径形式的 manifest 值。
 */
std::string absolutePathManifestValue(const fs::path &path)
{
    return path.empty() ? std::string{} : fs::absolute(path).generic_string();
}

/**
 * @brief 获取写入元数据的有效 PCA 维度。
 */
int effectivePcaDim(const RoiSearchConfig &config) noexcept
{
    return config.use_pca ? config.pca_dim : 0;
}

const char *roiPcaModeName(const RoiSearchConfig &config) noexcept
{
    return config.use_pca ? "local_image" : "none";
}

/**
 * @brief 提取 ROI 条目的外部 ID。
 */
std::vector<int64_t> roiItemIds(const std::vector<RoiSearchItem> &items)
{
    std::vector<int64_t> ids;
    ids.reserve(items.size());
    for (const auto &item : items)
    {
        ids.push_back(item.roi_id);
    }
    return ids;
}

/**
 * @brief 判断 ROI 条目列表是否与 manifest ID 序列一致。
 */
bool roiIdsMatch(const std::vector<RoiSearchItem> &items, const std::vector<int64_t> &ids)
{
    if (items.size() != ids.size())
    {
        return false;
    }
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].roi_id != ids[i])
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
 * @brief 构造 ROI 搜索 manifest 条目。
 */
irt::util::ManifestEntries roiSearchManifestEntries(const fs::path &index_path, const RoiSearchConfig &config,
                                                    const std::vector<int64_t> &roi_ids)
{
    irt::util::ManifestEntries entries{
        {           "version",                                                    "1"},
        {              "kind",                                           "roi_search"},
        {        "index_file",                  absolutePathManifestValue(index_path)},
        {             "model",                                      config.model_name},
        {           "feature",                                    config.feature_name},
        {     "model_runtime",       config.model_runtime.toString()},
        {   "model_precision",  irt::model::modelPrecisionName(config.model_precision)},
        {"preprocess_backend", priv::preprocessBackendName(config.preprocess_backend)},
        {              "norm",                     priv::featureNormName(config.norm)},
        {     "faiss_backend",           priv::faissBackendName(config.faiss_backend)},
        {     "index_storage",           priv::indexStorageName(config.index_storage)},
        {  "model_batch_size",                std::to_string(config.model_batch_size)},
        {        "index_kind",                            priv::indexKindName(config)},
        { "roi_pooled_height",                   std::to_string(config.pooled_height)},
        {  "roi_pooled_width",                    std::to_string(config.pooled_width)},
        {"roi_sampling_ratio",                  std::to_string(config.sampling_ratio)},
        {       "roi_aligned",                               boolName(config.aligned)},
        {       "roi_use_pca",                               boolName(config.use_pca)},
        {      "roi_pca_mode",                                 roiPcaModeName(config)},
        {       "roi_pca_dim",                std::to_string(effectivePcaDim(config))},
    };
    if (priv::useCpuDiskIndex(config))
    {
        entries.emplace_back("ivf_data_file", absolutePathManifestValue(priv::cpuOnDiskIvfDataPath(index_path)));
    }

    entries.emplace_back("roi_count", std::to_string(roi_ids.size()));
    for (size_t i = 0; i < roi_ids.size(); ++i)
    {
        const auto prefix = "roi." + std::to_string(i);
        entries.emplace_back(prefix + ".id", std::to_string(roi_ids[i]));
    }
    return entries;
}

/**
 * @brief 写入 ROI 搜索 manifest。
 */
void saveRoiManifest(const fs::path &index_path, const RoiSearchConfig &config, const std::vector<int64_t> &roi_ids)
{
    irt::util::writeYamlManifest(irt::util::manifestPathForDataFile(index_path),
                                 roiSearchManifestEntries(index_path, config, roi_ids));
}

/**
 * @brief 解析 manifest 中的 ROI ID 列表。
 */
std::vector<int64_t> loadRoiIdsFromManifest(const fs::path &index_path)
{
    const auto manifest = irt::util::loadYamlManifest(irt::util::manifestPathForDataFile(index_path));
    if (manifest.empty() || !irt::util::manifestValueEquals(manifest, "kind", "roi_search"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI manifest is missing or invalid: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    const auto count_text = irt::util::manifestValue(manifest, "roi_count");
    if (count_text.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI manifest has no roi_count: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    size_t roi_count{0};
    try
    {
        roi_count = static_cast<size_t>(std::stoull(count_text));
    }
    catch (const std::exception &)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid roi_count in ROI manifest: %s",
                             count_text.c_str());
    }

    std::vector<int64_t> ids;
    ids.reserve(roi_count);
    for (size_t i = 0; i < roi_count; ++i)
    {
        const auto prefix = "roi." + std::to_string(i);
        const auto value  = irt::util::manifestValue(manifest, prefix + ".id");
        if (value.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI manifest is missing %s.id", prefix.c_str());
        }
        try
        {
            ids.push_back(std::stoll(value));
        }
        catch (const std::exception &)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI ID in manifest entry %s",
                                 prefix.c_str());
        }
    }
    return ids;
}

/**
 * @brief 判断磁盘上的 ROI 索引是否匹配当前配置。
 */
bool existingRoiIndexMatchesConfig(const fs::path &index_path, const RoiSearchConfig &config)
{
    const auto manifest_path = irt::util::manifestPathForDataFile(index_path);
    if (!fs::exists(manifest_path))
    {
        return false;
    }

    const auto manifest = irt::util::loadYamlManifest(manifest_path);
    if (manifest.empty())
    {
        return false;
    }

    return irt::util::manifestValueEquals(manifest, "kind", "roi_search")
        && irt::util::manifestValueEquals(manifest, "index_file", absolutePathManifestValue(index_path))
        && irt::util::manifestValueEquals(manifest, "model", config.model_name)
        && irt::util::manifestValueEquals(manifest, "feature", config.feature_name)
        && irt::util::manifestValueEquals(manifest, "model_runtime", config.model_runtime.toString())
        && irt::util::manifestValueEquals(manifest, "model_precision",
                                          irt::model::modelPrecisionName(config.model_precision))
        && irt::util::manifestValueEquals(manifest, "preprocess_backend",
                                          priv::preprocessBackendName(config.preprocess_backend))
        && irt::util::manifestValueEquals(manifest, "faiss_backend", priv::faissBackendName(config.faiss_backend))
        && irt::util::manifestValueEquals(manifest, "model_batch_size", std::to_string(config.model_batch_size))
        && irt::util::manifestValueEquals(manifest, "norm", priv::featureNormName(config.norm))
        && irt::util::manifestValueEquals(manifest, "index_storage", priv::indexStorageName(config.index_storage))
        && irt::util::manifestValueEquals(manifest, "index_kind", priv::indexKindName(config))
        && irt::util::manifestValueEquals(manifest, "roi_pooled_height", std::to_string(config.pooled_height))
        && irt::util::manifestValueEquals(manifest, "roi_pooled_width", std::to_string(config.pooled_width))
        && irt::util::manifestValueEquals(manifest, "roi_sampling_ratio", std::to_string(config.sampling_ratio))
        && irt::util::manifestValueEquals(manifest, "roi_aligned", boolName(config.aligned))
        && irt::util::manifestValueEquals(manifest, "roi_use_pca", boolName(config.use_pca))
        && irt::util::manifestValueEquals(manifest, "roi_pca_mode", roiPcaModeName(config))
        && irt::util::manifestValueEquals(manifest, "roi_pca_dim", std::to_string(effectivePcaDim(config)));
}

} // namespace


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
    if (config_.pca_dim < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be non-negative");
    }
    if (config_.use_pca && config_.pca_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be positive when PCA is enabled");
    }
    priv::validateFeatureSearchConfig(config_, "RoiSearch");
}

RoiSearch::Impl::~Impl() = default;

void RoiSearch::Impl::buildOrLoad(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                                  const fs::path &index_file, bool rebuild_index,
                                  RoiSearchBuildProgressCallback progress_callback)
{
    const auto resolved_index_path = resolveRoiIndexPath(index_file);
    auto       normalized_items    = normalizeItems(gallery_items);

    if (!rebuild_index && fs::exists(resolved_index_path)
        && fs::exists(irt::util::manifestPathForDataFile(resolved_index_path))
        && existingRoiIndexMatchesConfig(resolved_index_path, config_)
        && roiIdsMatch(normalized_items, loadRoiIdsFromManifest(resolved_index_path)))
    {
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
        load(weights_file, resolved_index_path);
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
        return;
    }

    buildWithItems(weights_file, std::move(normalized_items), resolved_index_path, std::move(progress_callback));
}

void RoiSearch::Impl::build(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                            const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    const auto resolved_index_path = resolveRoiIndexPath(index_file);
    auto       normalized_items    = normalizeItems(gallery_items);
    buildWithItems(weights_file, std::move(normalized_items), resolved_index_path, std::move(progress_callback));
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
    if (!fs::exists(irt::util::manifestPathForDataFile(index_file)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI manifest file does not exist: %s",
                             irt::util::manifestPathForDataFile(index_file).string().c_str());
    }
    if (!existingRoiIndexMatchesConfig(index_file, config_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index manifest does not match current config: %s",
                             irt::util::manifestPathForDataFile(index_file).string().c_str());
    }

    auto ids   = loadRoiIdsFromManifest(index_file);
    auto faiss = priv::loadConfiguredFaissIndex(index_file, config_);
    if (config_.use_pca)
    {
        const auto expected_dim
            = static_cast<faiss::idx_t>(config_.pca_dim * config_.pooled_height * config_.pooled_width);
        if (!faiss.index || faiss.index->d != expected_dim)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI local PCA dim does not match Faiss index dim");
        }
    }
    if (!faiss.index || static_cast<faiss::idx_t>(ids.size()) != faiss.index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index size (%lld) does not match ID mapping size (%zu)",
                             faiss.index ? static_cast<long long>(faiss.index->ntotal) : 0LL, ids.size());
    }

    const int feature_dim = static_cast<int>(faiss.index->d);
    installIndex(weights_file, index_file, std::move(faiss), std::move(ids), feature_dim, {});
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

    const int final_feature_dim = extractor->featureDim();
    priv::FeatureStore feature_store(priv::featureStorePath(index_file), gallery_items.size(), final_feature_dim);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::ExtractingFeatures, 0, 0, 0, 0,
                              gallery_items.size());

    struct RoiImageGroup
    {
        fs::path           image_path;
        std::vector<size_t> roi_indices;
    };
    std::vector<RoiImageGroup> groups;
    std::map<fs::path, size_t> group_by_path;
    groups.reserve(gallery_items.size());
    for (size_t index = 0; index < gallery_items.size(); ++index)
    {
        const auto [it, inserted] = group_by_path.emplace(gallery_items[index].image_path, groups.size());
        if (inserted)
        {
            groups.push_back(RoiImageGroup{gallery_items[index].image_path, {}});
        }
        groups[it->second].roi_indices.push_back(index);
    }

    size_t              processed_count{0};
    size_t              batch_index{0};
    std::vector<size_t> packed_roi_indices;
    size_t              packed_image_count{0};
    auto                 process_packed = [&]
    {
        if (packed_roi_indices.empty())
        {
            return;
        }

        std::vector<RoiSearchItem> batch_items;
        batch_items.reserve(packed_roi_indices.size());
        for (const auto roi_index : packed_roi_indices)
        {
            batch_items.push_back(gallery_items[roi_index]);
        }
        const auto features = extractor->extractItems(batch_items);
        const auto expected_feature_elements
            = irt::checkedSizeMul(batch_items.size(), static_cast<size_t>(final_feature_dim),
                                  "ROI feature batch elements");
        if (features.size() != expected_feature_elements)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch size mismatch");
        }

        const auto feature_dim = static_cast<size_t>(final_feature_dim);
        size_t     run_begin   = 0;
        while (run_begin < packed_roi_indices.size())
        {
            size_t run_count = 1;
            while (run_begin + run_count < packed_roi_indices.size()
                   && packed_roi_indices[run_begin + run_count] == packed_roi_indices[run_begin + run_count - 1] + 1)
            {
                ++run_count;
            }
            std::vector<float> run_features(
                features.begin() + static_cast<std::ptrdiff_t>(irt::checkedSizeMul(run_begin, feature_dim,
                                                                                    "ROI feature run offset")),
                features.begin() + static_cast<std::ptrdiff_t>(irt::checkedSizeMul(
                                                                    irt::checkedSizeAdd(run_begin, run_count,
                                                                                        "ROI feature run range"),
                                                                    feature_dim, "ROI feature run end")));
            feature_store.writeBatchAt(packed_roi_indices[run_begin], run_count, run_features);
            run_begin += run_count;
        }

        processed_count += packed_roi_indices.size();
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::ExtractingFeatures, batch_index++,
                                  packed_roi_indices.front(), packed_roi_indices.size(), processed_count,
                                  gallery_items.size());
        packed_roi_indices.clear();
        packed_image_count = 0;
    };

    for (const auto &group : groups)
    {
        if (!packed_roi_indices.empty() && packed_image_count >= extractor->maxBatchSize())
        {
            process_packed();
        }
        packed_roi_indices.insert(packed_roi_indices.end(), group.roi_indices.begin(), group.roi_indices.end());
        ++packed_image_count;
    }
    process_packed();
    feature_store.finishWriting();

    const size_t build_total = priv::useCpuDiskIndex(config_)
                                 ? irt::checkedSizeMul(gallery_items.size(), 2U, "ROI search build progress")
                                 : gallery_items.size();
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, 0, 0, 0, 0, build_total);
    auto built = priv::buildConfiguredFaissIndex(
        gallery_items.size(), final_feature_dim, index_file, config_,
        [&](size_t index) { return feature_store.read(index); },
        [&](size_t begin, size_t count) { return feature_store.readBatch(begin, count); },
        [&](const std::vector<size_t> &indices) { return feature_store.readBatch(indices); }, progress_callback);

    installIndex(weights_file, index_file, std::move(built), roiItemIds(gallery_items), final_feature_dim,
                 std::move(extractor));
    saveRoiManifest(index_path_, config_, gallery_ids_);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, 0, 0, 0, build_total,
                              build_total);
}

void RoiSearch::Impl::installIndex(const fs::path &weights_file, const fs::path &index_path,
                                   priv::FaissIndexBundle bundle, std::vector<int64_t> gallery_ids, int feature_dim,
                                   std::unique_ptr<priv::RoiFeatureExtractor> extractor)
{
    if (!bundle.index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot install an empty RoiSearch index");
    }

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    index_path_          = index_path;
    faiss_gpu_resources_ = std::move(bundle.gpu_resources);
    index_               = std::move(bundle.index);
    gallery_ids_         = std::move(gallery_ids);
    extractor_           = std::move(extractor);
    feature_dim_         = feature_dim > 0 ? feature_dim : static_cast<int>(index_->d);
}

std::vector<RoiSearchResult> RoiSearch::Impl::search(const fs::path &query_image, const RoiSearchBox &roi, int top_k)
{
    if (!index_)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "RoiSearch index is not ready");
    }
    if (top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }
    if (index_->ntotal <= 0)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "RoiSearch index is empty");
    }

    const auto query_item = normalizeItem(RoiSearchItem{0, query_image, roi});
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
        if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_ids_.size())
        {
            continue;
        }
        results.push_back({distances[i], gallery_ids_[static_cast<size_t>(indices[i])]});
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

std::vector<int64_t> RoiSearch::Impl::galleryIds() const
{
    return gallery_ids_;
}

int RoiSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void RoiSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        auto       extractor     = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file_);
        const auto roi_align_dim = extractor->featureDim();
        if (index_ && index_->d != static_cast<faiss::idx_t>(roi_align_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI index feature dimension (%lld) does not match ROIAlign feature dimension (%d); "
                                 "rebuild the ROI index",
                                 static_cast<long long>(index_->d), roi_align_dim);
        }
        extractor_   = std::move(extractor);
        feature_dim_ = roi_align_dim;
    }
}

} // namespace irt::features
