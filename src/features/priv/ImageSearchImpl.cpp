/**
 * @file ImageSearchImpl.cpp
 * @brief 图像检索 PIMPL：模型加载、特征抽取、Faiss 特征库构建/加载与 Top-K 搜索。
 */

#include "ImageSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"
#include "ImageSearchFaissIndex.hpp"

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#pragma warning(pop)

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/FileManifest.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

fs::path resolveIndexPath(const fs::path &gallery_dir, const fs::path &index_file)
{
    return irt::util::resolveOutputFilePath(index_file, gallery_dir, ".faiss");
}

fs::path resolveExplicitIndexPath(const fs::path &index_file)
{
    return irt::util::resolveOutputFilePath(index_file, {}, ".faiss");
}

std::string absolutePathManifestValue(const fs::path &path)
{
    return path.empty() ? std::string{} : fs::absolute(path).generic_string();
}

std::string galleryDirectoryMetadataValue(const fs::path &gallery_dir)
{
    return absolutePathManifestValue(gallery_dir);
}

std::string explicitPathListMetadataValue()
{
    return "<explicit_path_list>";
}

std::vector<ImageSearchItem> normalizeImageItems(const std::vector<ImageSearchItem> &gallery_items)
{
    if (gallery_items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image item list must not be empty");
    }

    std::vector<ImageSearchItem> normalized;
    normalized.reserve(gallery_items.size());
    for (const auto &item : gallery_items)
    {
        normalized.push_back(
            ImageSearchItem{item.image_id, priv::normalizeImageFilePath(item.image_path, "ImageSearch")});
    }
    return normalized;
}

std::vector<ImageSearchItem> makeImageItemsFromPaths(const std::vector<fs::path> &image_paths)
{
    std::vector<ImageSearchItem> items;
    items.reserve(image_paths.size());
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        items.push_back(ImageSearchItem{static_cast<int64_t>(i), image_paths[i]});
    }
    return items;
}

std::vector<fs::path> imageItemPaths(const std::vector<ImageSearchItem> &items)
{
    std::vector<fs::path> paths;
    paths.reserve(items.size());
    for (const auto &item : items)
    {
        paths.push_back(item.image_path);
    }
    return paths;
}

std::vector<int64_t> imageItemIds(const std::vector<ImageSearchItem> &items)
{
    std::vector<int64_t> ids;
    ids.reserve(items.size());
    for (const auto &item : items)
    {
        ids.push_back(item.image_id);
    }
    return ids;
}

bool imageIdsMatch(const std::vector<ImageSearchItem> &items, const std::vector<int64_t> &ids)
{
    if (items.size() != ids.size())
    {
        return false;
    }
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].image_id != ids[i])
        {
            return false;
        }
    }
    return true;
}

irt::util::ManifestEntries imageSearchManifestEntries(const fs::path &index_path, const std::string &gallery_value,
                                                       const ImageSearchConfig    &config,
                                                       const std::vector<int64_t> &image_ids)
{
    irt::util::ManifestEntries entries{
        {           "version",                                              "1"},
        {              "kind",                                   "image_search"},
        {        "index_file",            absolutePathManifestValue(index_path)},
        {             "model",                                config.model_name},
        {           "feature",                              config.feature_name},
        {       "gallery_dir",                                    gallery_value},
        {     "model_runtime",           config.model_runtime.toString()},
        {   "model_precision", irt::model::modelPrecisionName(config.model_precision)},
        {"preprocess_backend", priv::preprocessBackendName(config.preprocess_backend)},
        {              "norm",                     priv::featureNormName(config.norm)},
        {     "faiss_backend",           priv::faissBackendName(config.faiss_backend)},
        {     "index_storage",           priv::indexStorageName(config.index_storage)},
        {  "model_batch_size",          std::to_string(config.model_batch_size)},
        {        "index_kind",                            priv::indexKindName(config)},
    };
    if (priv::useCpuDiskIndex(config))
    {
        entries.emplace_back("ivf_data_file", absolutePathManifestValue(priv::cpuOnDiskIvfDataPath(index_path)));
    }
    entries.emplace_back("image_count", std::to_string(image_ids.size()));
    for (size_t i = 0; i < image_ids.size(); ++i)
    {
        entries.emplace_back("image." + std::to_string(i) + ".id", std::to_string(image_ids[i]));
    }
    return entries;
}

void saveImageSearchManifest(const fs::path &index_path, const std::string &gallery_value,
                             const ImageSearchConfig &config, const std::vector<int64_t> &image_ids)
{
    irt::util::writeYamlManifest(irt::util::manifestPathForDataFile(index_path),
                                 imageSearchManifestEntries(index_path, gallery_value, config, image_ids));
}

std::vector<int64_t> loadImageIdsFromManifest(const fs::path &index_path)
{
    const auto manifest = irt::util::loadYamlManifest(irt::util::manifestPathForDataFile(index_path));
    if (manifest.empty() || !irt::util::manifestValueEquals(manifest, "kind", "image_search"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest is missing or invalid: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    const auto count_text = irt::util::manifestValue(manifest, "image_count");
    if (count_text.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest has no image_count: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    size_t image_count{0};
    try
    {
        image_count = static_cast<size_t>(std::stoull(count_text));
    }
    catch (const std::exception &)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid image_count in ImageSearch manifest: %s",
                             count_text.c_str());
    }

    std::vector<int64_t> image_ids;
    image_ids.reserve(image_count);
    for (size_t i = 0; i < image_count; ++i)
    {
        const auto key   = "image." + std::to_string(i) + ".id";
        const auto value = irt::util::manifestValue(manifest, key);
        if (value.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest is missing %s", key.c_str());
        }
        try
        {
            image_ids.push_back(std::stoll(value));
        }
        catch (const std::exception &)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid image ID in ImageSearch manifest: %s",
                                 value.c_str());
        }
    }
    return image_ids;
}

bool existingIndexMatchesConfig(const fs::path &index_path, const fs::path &gallery_dir,
                                const ImageSearchConfig &config)
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

    const bool gallery_matches
        = gallery_dir.empty() || irt::util::manifestValueEquals(manifest, "gallery_dir",
                                                                 galleryDirectoryMetadataValue(gallery_dir))
        || irt::util::manifestValueEquals(manifest, "gallery_dir", explicitPathListMetadataValue());

    return irt::util::manifestValueEquals(manifest, "kind", "image_search")
        && irt::util::manifestValueEquals(manifest, "index_file", absolutePathManifestValue(index_path))
        && irt::util::manifestValueEquals(manifest, "model", config.model_name)
        && irt::util::manifestValueEquals(manifest, "feature", config.feature_name) && gallery_matches
        && irt::util::manifestValueEquals(manifest, "model_runtime", config.model_runtime.toString())
        && irt::util::manifestValueEquals(manifest, "model_precision",
                                          irt::model::modelPrecisionName(config.model_precision))
        && irt::util::manifestValueEquals(manifest, "preprocess_backend",
                                          priv::preprocessBackendName(config.preprocess_backend))
        && irt::util::manifestValueEquals(manifest, "faiss_backend", priv::faissBackendName(config.faiss_backend))
        && irt::util::manifestValueEquals(manifest, "model_batch_size", std::to_string(config.model_batch_size))
        && irt::util::manifestValueEquals(manifest, "norm", priv::featureNormName(config.norm))
        && irt::util::manifestValueEquals(manifest, "index_storage", priv::indexStorageName(config.index_storage))
        && irt::util::manifestValueEquals(manifest, "index_kind", priv::indexKindName(config));
}

void reportStage(const ImageSearchBuildProgressCallback &callback, ImageSearchBuildStage stage, size_t processed,
                 size_t total)
{
    priv::reportBuildProgress(callback, stage, 0, 0, 0, processed, total);
}

fs::path featureStorePath(const fs::path &index_path)
{
    return index_path.string() + ".features.tmp";
}

void extractFeaturesToStore(const std::vector<fs::path> &image_paths, priv::ImageFeatureExtractor &extractor,
                            priv::FeatureStore &store, const ImageSearchBuildProgressCallback &progress_callback)
{
    const size_t image_count = image_paths.size();
    reportStage(progress_callback, ImageSearchBuildStage::ExtractingFeatures, 0, image_count);

    priv::processFeatureBatches(
        image_count, extractor.maxBatchSize(), extractor.featureDim(),
        [&](size_t begin, size_t count) { return extractor.extractBatch(image_paths, begin, count); },
        [&](size_t begin, size_t count, const std::vector<float> &features) { store.writeBatch(begin, count, features); },
        [&](const priv::FeatureBatchProgress &progress)
        {
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::ExtractingFeatures,
                                      progress.batch_index, progress.batch_begin, progress.batch_count,
                                      progress.processed_count, progress.total_count);
        });
    store.finishWriting();
}

priv::FaissIndexBundle buildIndexFromStore(const std::vector<ImageSearchItem> &gallery_items,
                                           const fs::path &index_path, const ImageSearchConfig &config,
                                           priv::FeatureStore &store,
                                           const ImageSearchBuildProgressCallback &progress_callback)
{
    const size_t item_count = gallery_items.size();
    const size_t total_work = priv::useCpuDiskIndex(config) ? item_count * 2 : item_count;
    reportStage(progress_callback, ImageSearchBuildStage::BuildingIndex, 0, total_work);

    auto load_feature = [&](size_t index) { return store.read(index); };
    auto load_feature_batch = [&](size_t begin, size_t count) { return store.readBatch(begin, count); };
    auto load_feature_index_batch = [&](const std::vector<size_t> &indices) { return store.readBatch(indices); };

    auto built = priv::buildConfiguredFaissIndex(item_count, store.featureDim(), index_path, config, load_feature,
                                                 load_feature_batch, load_feature_index_batch, progress_callback);
    return built;
}

} // namespace

ImageSearch::Impl::Impl(ImageSearchConfig config)
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
    priv::validateFeatureSearchConfig(config_, "ImageSearch");
}

ImageSearch::Impl::~Impl() = default;

void ImageSearch::Impl::buildOrLoad(const fs::path &weights_file, const fs::path &gallery_dir,
                                    const fs::path &index_file, bool rebuild_index,
                                    ImageSearchBuildProgressCallback progress_callback)
{
    const fs::path resolved_index_path = resolveIndexPath(gallery_dir, index_file);
    if (!rebuild_index && fs::exists(resolved_index_path)
        && fs::exists(irt::util::manifestPathForDataFile(resolved_index_path))
        && existingIndexMatchesConfig(resolved_index_path, gallery_dir, config_))
    {
        reportStage(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 1);
        load(weights_file, resolved_index_path);
        reportStage(progress_callback, ImageSearchBuildStage::LoadingIndex, 1, 1);
        return;
    }

    build(weights_file, gallery_dir, resolved_index_path, std::move(progress_callback));
}

void ImageSearch::Impl::build(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file,
                              ImageSearchBuildProgressCallback progress_callback)
{
    auto images = ImageSearch::collectGalleryImages(gallery_dir);
    buildWithImages(weights_file, gallery_dir, makeImageItemsFromPaths(images), resolveIndexPath(gallery_dir, index_file),
                    galleryDirectoryMetadataValue(gallery_dir), std::move(progress_callback));
}

void ImageSearch::Impl::buildOrLoad(const fs::path &weights_file, const std::vector<ImageSearchItem> &gallery_items,
                                    const fs::path &index_file, bool rebuild_index,
                                    ImageSearchBuildProgressCallback progress_callback)
{
    const fs::path resolved_index_path = resolveExplicitIndexPath(index_file);
    auto            normalized_items   = normalizeImageItems(gallery_items);

    if (!rebuild_index && fs::exists(resolved_index_path)
        && fs::exists(irt::util::manifestPathForDataFile(resolved_index_path))
        && existingIndexMatchesConfig(resolved_index_path, {}, config_)
        && imageIdsMatch(normalized_items, loadImageIdsFromManifest(resolved_index_path)))
    {
        reportStage(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 1);
        load(weights_file, resolved_index_path);
        reportStage(progress_callback, ImageSearchBuildStage::LoadingIndex, 1, 1);
        return;
    }

    buildWithImages(weights_file, {}, std::move(normalized_items), resolved_index_path,
                    explicitPathListMetadataValue(), std::move(progress_callback));
}

void ImageSearch::Impl::build(const fs::path &weights_file, const std::vector<ImageSearchItem> &gallery_items,
                              const fs::path &index_file, ImageSearchBuildProgressCallback progress_callback)
{
    auto normalized_items = normalizeImageItems(gallery_items);
    buildWithImages(weights_file, {}, std::move(normalized_items), resolveExplicitIndexPath(index_file),
                    explicitPathListMetadataValue(), std::move(progress_callback));
}

void ImageSearch::Impl::load(const fs::path &weights_file, const fs::path &index_file)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageSearch index_file must not be empty when loading");
    }
    if (!fs::exists(index_file))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index file does not exist: %s",
                             index_file.string().c_str());
    }
    if (!fs::exists(irt::util::manifestPathForDataFile(index_file)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index manifest file does not exist: %s",
                             irt::util::manifestPathForDataFile(index_file).string().c_str());
    }
    if (!existingIndexMatchesConfig(index_file, {}, config_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index manifest does not match current ImageSearch config: %s",
                             irt::util::manifestPathForDataFile(index_file).string().c_str());
    }

    auto loaded     = priv::loadConfiguredFaissIndex(index_file, config_);
    const int dim   = static_cast<int>(loaded.index->d);
    auto      ids   = loadImageIdsFromManifest(index_file);
    if (static_cast<faiss::idx_t>(ids.size()) != loaded.index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index size (%lld) does not match ID mapping size (%zu)",
                             static_cast<long long>(loaded.index->ntotal), ids.size());
    }
    installIndex(weights_file, {}, index_file, std::move(loaded), std::move(ids), dim, {});
}

void ImageSearch::Impl::buildWithImages(const fs::path &weights_file, const fs::path &gallery_dir,
                                        std::vector<ImageSearchItem> gallery_items, const fs::path &index_path,
                                        const std::string &metadata_gallery_value,
                                        ImageSearchBuildProgressCallback progress_callback)
{
    reportStage(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 1);
    auto extractor = std::make_unique<priv::ImageFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                    weights_file, config_);
    reportStage(progress_callback, ImageSearchBuildStage::LoadingModel, 1, 1);

    if (!index_path.parent_path().empty())
    {
        fs::create_directories(index_path.parent_path());
    }

    const auto gallery_paths = imageItemPaths(gallery_items);
    const auto gallery_ids   = imageItemIds(gallery_items);
    priv::FeatureStore feature_store(priv::featureStorePath(index_path), gallery_items.size(), extractor->featureDim());
    extractFeaturesToStore(gallery_paths, *extractor, feature_store, progress_callback);

    auto built = buildIndexFromStore(gallery_items, index_path, config_, feature_store, progress_callback);
    installIndex(weights_file, gallery_dir, index_path, std::move(built), gallery_ids, extractor->featureDim(),
                 std::move(extractor));
    saveImageSearchManifest(index_path_, metadata_gallery_value, config_, gallery_ids_);

    const size_t total_work = priv::useCpuDiskIndex(config_) ? gallery_ids_.size() * 2 : gallery_ids_.size();
    reportStage(progress_callback, ImageSearchBuildStage::BuildingIndex, total_work, total_work);
}

void ImageSearch::Impl::installIndex(const fs::path &weights_file, const fs::path &gallery_dir,
                                     const fs::path &index_path, priv::FaissIndexBundle bundle,
                                     std::vector<int64_t> gallery_ids, int feature_dim,
                                     std::unique_ptr<priv::ImageFeatureExtractor> extractor)
{
    if (!bundle.index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot install an empty ImageSearch index");
    }

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    gallery_dir_         = gallery_dir;
    index_path_          = index_path;
    faiss_gpu_resources_ = std::move(bundle.gpu_resources);
    index_               = std::move(bundle.index);
    gallery_ids_         = std::move(gallery_ids);
    extractor_           = std::move(extractor);
    feature_dim_         = feature_dim > 0 ? feature_dim : static_cast<int>(index_->d);
}

std::vector<ImageSearchResult> ImageSearch::Impl::search(const fs::path &query_image, int top_k)
{
    if (!index_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is not ready");
    }
    if (top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }
    if (index_->ntotal <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is empty");
    }

    ensureExtractor();
    const auto query_feature = extractor_->extract(query_image);
    const int  result_count  = std::min(top_k, static_cast<int>(index_->ntotal));

    std::vector<faiss::idx_t> indices(result_count);
    std::vector<float>        distances(result_count);
    index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());

    std::vector<ImageSearchResult> results;
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

bool ImageSearch::Impl::isReady() const noexcept
{
    return index_ != nullptr;
}

const ImageSearchConfig &ImageSearch::Impl::config() const noexcept
{
    return config_;
}

const fs::path &ImageSearch::Impl::indexPath() const noexcept
{
    return index_path_;
}

std::vector<int64_t> ImageSearch::Impl::galleryIds() const
{
    return gallery_ids_;
}

int ImageSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void ImageSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        extractor_ = std::make_unique<priv::ImageFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                    weights_file_, config_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
