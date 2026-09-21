/**
 * @file RoiSearchImpl.cpp
 * @brief ROI 搜索匹配 PIMPL 实现。
 */

#include "RoiSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "RoiFeatureExtractor.hpp"
#include "RoiEmbeddingCore.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/FileManifest.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#pragma warning(pop)

#include <algorithm>
#include <cstddef>
#include <iostream>
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
    if (item.polygon.empty()) priv::validateRoi(item.roi);
    auto normalized = item;
    normalized.image_path = priv::normalizeImageFilePath(item.image_path, "RoiSearch");
    return normalized;
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
        {        "index_kind",                            (config.exact_search ? "flat_ip" : priv::indexKindName(config))},
        { "roi_pooled_height",                   std::to_string(config.pooled_height)},
        {  "roi_pooled_width",                    std::to_string(config.pooled_width)},
        {"roi_sampling_ratio",                  std::to_string(config.sampling_ratio)},
        {       "roi_aligned",                               boolName(config.aligned)},
        {       "roi_use_pca",                               boolName(config.use_pca)},
        {      "roi_pca_mode",                                 roiPcaModeName(config)},
        {       "roi_pca_dim",                std::to_string(effectivePcaDim(config))},
    };
    entries.emplace_back("roi_mode", config.mode == RoiFeatureMode::CropMaskedMean ? "crop_masked_mean" : "legacy_roi_align");
    entries.emplace_back("roi_patch_size", std::to_string(config.patch_size));
    entries.emplace_back("roi_crop_margin", std::to_string(config.crop_margin));
    entries.emplace_back("roi_background_keep", std::to_string(config.background_keep));
    entries.emplace_back("roi_spatial_weight", std::to_string(config.spatial_weight));
    entries.emplace_back("roi_max_detail_views", std::to_string(config.max_detail_views));
    entries.emplace_back("roi_detail_weight", std::to_string(config.detail_weight));
    entries.emplace_back("roi_exact_search", boolName(config.exact_search));
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
    const auto entries = roiSearchManifestEntries(index_path, config, roi_ids);
    irt::util::writeYamlManifest(irt::util::manifestPathForDataFile(index_path), entries);
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

    // Match descriptor semantics only. Batch size, path spelling and preprocessing
    // execution backend do not define a new ROI representation.
    auto eq = [&](const char *key, const std::string &value) {
        return irt::util::manifestValueEquals(manifest, key, value);
    };
    if (!eq("kind", "roi_search") || !eq("model", config.model_name) || !eq("feature", config.feature_name)
        || !eq("norm", priv::featureNormName(config.norm))
        || !eq("roi_exact_search", boolName(config.exact_search))
        || !eq("roi_mode", config.mode == RoiFeatureMode::CropMaskedMean ? "crop_masked_mean" : "legacy_roi_align"))
        return false;
    if (config.mode == RoiFeatureMode::CropMaskedMean)
        return eq("roi_patch_size", std::to_string(config.patch_size))
            && eq("roi_crop_margin", std::to_string(config.crop_margin))
            && eq("roi_background_keep", std::to_string(config.background_keep))
            && eq("roi_spatial_weight", std::to_string(config.spatial_weight))
            && eq("roi_max_detail_views", std::to_string(config.max_detail_views))
            && eq("roi_detail_weight", std::to_string(config.detail_weight));
    return eq("roi_pooled_height", std::to_string(config.pooled_height))
        && eq("roi_pooled_width", std::to_string(config.pooled_width))
        && eq("roi_sampling_ratio", std::to_string(config.sampling_ratio))
        && eq("roi_aligned", boolName(config.aligned))
        && eq("roi_use_pca", boolName(config.use_pca))
        && eq("roi_pca_dim", std::to_string(effectivePcaDim(config)));
}

} // namespace


RoiSearch::Impl::Impl(RoiSearchConfig config)
    : config_(std::move(config))
{
    if (config_.exact_search)
    {
        config_.index_storage = ImageSearchIndexStorage::RAM;
    }
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
    if (config_.mode == RoiFeatureMode::LegacyRoiAlign)
    {
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

    const auto load_index_path = index_file;
    const auto load_weights_path = weights_file;
    releaseState();
    auto ids   = loadRoiIdsFromManifest(load_index_path);

    priv::FaissIndexBundle faiss_bundle;
    if (config_.exact_search)
    {
        auto cpu_flat = std::unique_ptr<faiss::IndexFlat>(
            dynamic_cast<faiss::IndexFlat *>(faiss::read_index(load_index_path.string().c_str())));
        if (!cpu_flat)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Failed to load exact ROI Faiss IndexFlat: %s",
                                 load_index_path.string().c_str());
        }
        if (config_.faiss_backend == ImageSearchFaissBackend::GPU)
        {
            faiss_bundle = priv::cloneCpuIndexToGpu(cpu_flat.get(), config_.model_runtime.deviceId(),
                                                    config_.model_precision);
            host_flat_index_ = std::move(cpu_flat);
        }
        else
        {
            faiss_bundle.index = std::move(cpu_flat);
        }
    }
    else
    {
        faiss_bundle = priv::loadConfiguredFaissIndex(load_index_path, config_);
    }

    if (config_.use_pca)
    {
        const auto expected_dim
            = static_cast<faiss::idx_t>(config_.pca_dim * config_.pooled_height * config_.pooled_width);
        if (!faiss_bundle.index || faiss_bundle.index->d != expected_dim)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI local PCA dim does not match Faiss index dim");
        }
    }
    if (!faiss_bundle.index || static_cast<faiss::idx_t>(ids.size()) != faiss_bundle.index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index size (%lld) does not match ID mapping size (%zu)",
                             faiss_bundle.index ? static_cast<long long>(faiss_bundle.index->ntotal) : 0LL, ids.size());
    }

    const int feature_dim = static_cast<int>(faiss_bundle.index->d);
    installIndex(load_weights_path, load_index_path, std::move(faiss_bundle), std::move(ids), feature_dim, {});
}

void RoiSearch::Impl::releaseState()
{
    // Explicit rebuild/load invalidates borrowed views. Caller serializes these operations.
    // Release the old GPU model and vectors before allocating their replacements.
    last_query_feature_.clear();
    host_flat_index_.reset();
    index_.reset();
    faiss_gpu_resources_.reset();
    extractor_.reset();
    gallery_ids_.clear();
    feature_dim_ = 0;
    weights_file_.clear();
    index_path_.clear();
}

void RoiSearch::Impl::buildWithItems(const fs::path &weights_file, std::vector<RoiSearchItem> gallery_items,
                                     const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    releaseState();
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 0, 1);
    auto extractor = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 1, 1);

    if (!index_file.parent_path().empty())
    {
        fs::create_directories(index_file.parent_path());
    }

    const int final_feature_dim = extractor->featureDim();
    if (config_.exact_search)
    {
        std::vector<float> gallery_features(
            irt::checkedSizeMul(gallery_items.size(), static_cast<size_t>(final_feature_dim), "ROI gallery features"));

        size_t written = 0;
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::ExtractingFeatures, 0, 0, 0, 0,
                                  gallery_items.size());
        extractor->extractTo(
            gallery_items,
            [&](const std::vector<size_t> &rows, const std::vector<float> &values) {
                embedding::scatterRows(rows, values, static_cast<size_t>(final_feature_dim), gallery_features.data(),
                                       gallery_items.size());
                written += rows.size();
            },
            [&](size_t batch, size_t begin, size_t count, size_t processed) {
                priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::ExtractingFeatures, batch, begin,
                                          count, processed, gallery_items.size());
            });
        if (written != gallery_items.size())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "ROI extraction did not finish every index row");
        }

        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, 0, 0, 0, 0, 1);
        auto flat = std::make_unique<faiss::IndexFlatIP>(final_feature_dim);
        flat->add(static_cast<faiss::idx_t>(gallery_items.size()), gallery_features.data());
        faiss::write_index(flat.get(), index_file.string().c_str());
        flat.reset();
        auto cpu_flat = std::unique_ptr<faiss::IndexFlat>(
            dynamic_cast<faiss::IndexFlat *>(faiss::read_index(index_file.string().c_str())));
        if (!cpu_flat)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to reload exact Faiss IndexFlat: %s",
                                 index_file.string().c_str());
        }

        priv::FaissIndexBundle built;
        if (config_.faiss_backend == ImageSearchFaissBackend::GPU)
        {
            built = priv::cloneCpuIndexToGpu(cpu_flat.get(), config_.model_runtime.deviceId(), config_.model_precision);
            host_flat_index_ = std::move(cpu_flat);
        }
        else
        {
            built.index = std::move(cpu_flat);
        }
        installIndex(weights_file, index_file, std::move(built), roiItemIds(gallery_items), final_feature_dim,
                     std::move(extractor));
        saveRoiManifest(index_path_, config_, gallery_ids_);
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, 0, 0, 0, 1, 1);
        return;
    }
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

    last_query_feature_.clear();
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
    return search(RoiSearchItem{0, query_image, roi}, top_k);
}

std::vector<RoiSearchResult> RoiSearch::Impl::search(const RoiSearchItem &query, int top_k)
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

    const auto query_item = normalizeItem(query);
    ensureExtractor();
    auto query_feature = extractor_->extract(query_item);
    if (query_feature.size() != static_cast<size_t>(index_->d))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI query feature dimension mismatch");
    }

    last_query_feature_ = std::move(query_feature);
    return searchVector(last_query_feature_.data(),top_k);
}

RoiFeatureWorkStats RoiSearch::Impl::featureWorkStats() const noexcept
{
    if(extractor_)return extractor_->workStats();
    RoiFeatureWorkStats stats;stats.available=config_.mode==RoiFeatureMode::CropMaskedMean;
    return stats; // a disk-loaded library has performed no image inference
}

const faiss::IndexFlat *RoiSearch::Impl::getHostFlatIndex() const noexcept
{
    if (host_flat_index_)
    {
        return host_flat_index_.get();
    }
    return dynamic_cast<const faiss::IndexFlat *>(index_.get());
}

std::vector<RoiSearchResult> RoiSearch::Impl::searchByRoiId(int64_t roi_id, int top_k)
{
    const auto *flat = getHostFlatIndex();
    if (!flat)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "searchByRoiId requires an exact ROI index");
    }
    const auto it = std::find(gallery_ids_.begin(), gallery_ids_.end(), roi_id);
    if (it == gallery_ids_.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI ID is not in this feature library");
    }
    const size_t row = static_cast<size_t>(it - gallery_ids_.begin());
    const float *vec = flat->get_xb() + row * flat->d;
    last_query_feature_.assign(vec, vec + flat->d);
    return searchVector(last_query_feature_.data(), top_k);
}

std::vector<RoiSearchResult> RoiSearch::Impl::repeatSearch(int top_k)
{
    if(last_query_feature_.empty())throw irt::Exception(irt::Status::INVALID_OPERATION,"No previous ROI query");
    return searchVector(last_query_feature_.data(),top_k);
}

std::vector<RoiSearchResult> RoiSearch::Impl::searchVector(const float* query,int top_k)
{
    if(!index_||index_->ntotal<=0)throw irt::Exception(irt::Status::INVALID_OPERATION,"ROI index is not ready");
    if(top_k<=0)throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,"top_k must be positive");
    if(faiss_gpu_resources_)
    {
        irt::model::setCudaDevice(config_.model_runtime.deviceId());
    }
    const int                 result_count = std::min(top_k, static_cast<int>(index_->ntotal));
    std::vector<faiss::idx_t> indices(result_count);
    std::vector<float>        distances(result_count);
    index_->search(1, query, result_count, distances.data(), indices.data());

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
        const auto descriptor_dim = extractor->featureDim();
        if (index_ && index_->d != static_cast<faiss::idx_t>(descriptor_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI index feature dimension (%lld) does not match descriptor dimension (%d); "
                                 "rebuild the ROI index",
                                 static_cast<long long>(index_->d), descriptor_dim);
        }
        extractor_   = std::move(extractor);
        feature_dim_ = descriptor_dim;
    }
}

} // namespace irt::features
