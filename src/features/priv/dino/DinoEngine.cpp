/**
 * @file DinoEngine.cpp
 * @brief 本地索引建库与区域查询编排。
 */

#include "DinoEngine.hpp"
#include "DinoDescriptors.hpp"
#include "DinoFineMatch.hpp"
#include "DinoFusion.hpp"
#include "DinoGeometry.hpp"
#include "DinoPaths.hpp"
#include "DinoProfile.hpp"
#include "DinoScan.hpp"
#include "DinoSearchCache.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <sstream>
#include <mutex>
#include <utility>
#include <new>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace fs = std::filesystem;

namespace irt::features::priv {

namespace {

std::timed_mutex model_mutex;
std::unique_ptr<DinoIndexReader> active_reader;
fs::path active_index_path;
constexpr const char *kDefaultIndexDirectoryName = "dino_region_index";
constexpr uint64_t kDefaultDenseFeatureCacheBytes = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t kDefaultImageCacheBytes        = 256ULL * 1024ULL * 1024ULL;

void reportProgress(const DinoBuildProgressCallback &callback, const DinoBuildStage stage, const size_t processed,
                    const size_t total, const std::string &message)
{
    if (!callback)
    {
        return;
    }
    DinoBuildProgress progress;
    progress.stage           = stage;
    progress.processed_count = processed;
    progress.total_count     = total;
    progress.message         = message;
    callback(progress);
}

void reportSearchProgress(const DinoSearchProgressCallback &callback, const DinoSearchStage stage,
                          const size_t processed, const size_t total)
{
    if (!callback)
    {
        return;
    }
    DinoSearchProgress progress;
    progress.stage           = stage;
    progress.processed_count = processed;
    progress.total_count     = total;
    callback(progress);
}

DinoSearchResponse runSearchPipeline(const DinoIndexReader &reader, const DinoCanonicalImage &query_image,
                                   const DinoRoi &roi, const DinoSearchRequest &request,
                                   const DinoRegionSearchConfig &config,
                                   const DinoSearchProgressCallback &progress_callback, const DinoDeadline &deadline,
                                   const DinoOperationControl &control)
{
    DinoSearchResponse detail;
    auto             backbone_holder = dinoAcquireBackbone(config);
    DinoBackbone    &backbone        = *backbone_holder;
    DinoViewPlanner  planner(backbone.patchSize(), backbone.encoderEdge(), config.view_overlap,
                            config.gallery_tile_edges, config.query_roi_target_lengths);

    if (reader.descriptorDim() != static_cast<size_t>(config.coarse_dimension))
    {
        throw irt::Exception(irt::Status::NOT_READY,
                             "Index coarse dimension differs from profile; rebuild the index");
    }
    const auto extractor_key = backbone.signature().cacheKey();
    const auto caches = dinoAcquireSearchCaches(extractor_key, kDefaultImageCacheBytes,
                                                kDefaultDenseFeatureCacheBytes);
    const size_t cache_hits_before = caches->images.hits() + caches->features.hits();
    reportSearchProgress(progress_callback, DinoSearchStage::QueryExtract, 0, 0);
    size_t model_forwards = 0;
    const auto query_started = dinoNowMs();
    auto   query          = dinoBuildQuery(query_image, roi, backbone, planner, config, model_forwards);
    detail.timings.query_extract_ms = dinoNowMs() - query_started;

    if (control.cancelled && control.cancelled())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
    }

    reportSearchProgress(progress_callback, DinoSearchStage::RegionScan, 0, 0);
    const int64_t excluded_image_id = (!request.include_self && request.query_image_id >= 0)
                                          ? request.query_image_id
                                          : -1;
    auto scan = dinoScan(reader, query, config, deadline, excluded_image_id, request.allowed_image_ids, control);
    detail.timings.region_scan_ms          = scan.region_scan_ms;
    detail.timings.local_scan_ms           = scan.local_scan_ms;
    detail.timings.local_window_rescore_ms = scan.window_rescore_ms;

    if (control.cancelled && control.cancelled())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
    }

    reportSearchProgress(progress_callback, DinoSearchStage::Fusion, 0, 0);
    const auto fusion_started = dinoNowMs();
    const auto fused = dinoFuseCandidates(scan.region_candidates, scan.local_candidates, config);
    detail.timings.fusion_ms = dinoNowMs() - fusion_started;
    detail.total_candidates = fused.candidates.size();
    const std::vector<DinoCandidate> *candidate_groups[] = {
        &scan.region_candidates, &scan.local_candidates, &fused.candidates};
    std::vector<DinoCoarseCandidate> *reported_groups[] = {
        &detail.region_candidates, &detail.local_candidates, &detail.coarse_candidates};
    for (size_t group = 0; group < 3; ++group)
    {
        auto &reported = *reported_groups[group];
        reported.reserve(candidate_groups[group]->size());
        for (const auto &candidate : *candidate_groups[group])
        {
            const auto image_index = reader.imageIndexById(candidate.image_id);
            if (image_index < 0)
                continue;
            const auto &image = reader.images()[static_cast<size_t>(image_index)];
            if (!request.include_self && request.query_image_id >= 0 && image.image_id == request.query_image_id)
                continue;
            const auto crop = dinoExpandRect(candidate.source_bbox, config.fine_candidate_expand,
                                             image.width, image.height);
            reported.push_back({image.image_id,
                {static_cast<float>(crop.x0), static_cast<float>(crop.y0),
                 static_cast<float>(crop.x1), static_cast<float>(crop.y1)}});
        }
    }

    if (control.cancelled && control.cancelled())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
    }

    reportSearchProgress(progress_callback, DinoSearchStage::FineMatch, 0, fused.candidates.size());
    auto fine = dinoFineMatch(reader, query_image, query, fused.candidates, config, backbone, planner,
                              caches->images, caches->features, extractor_key, deadline,
                              request.image_resolver);
    detail.score_kind = config.fine_verify_k > 0 ? "tight_global_and_grid" : "localization";
    detail.verified_candidates = config.fine_verify_k > 0 ? fine.results.size() : 0;
    const auto report_boxes = [](const std::vector<DinoMatchResult> &source, std::vector<DinoCoarseCandidate> &target) {
        for (const auto &item : source) target.push_back({item.image_id,
            {static_cast<float>(item.bbox.x0), static_cast<float>(item.bbox.y0), static_cast<float>(item.bbox.x1), static_cast<float>(item.bbox.y1)}});
    };
    report_boxes(fine.localized_results, detail.localized_candidates);
    report_boxes(fine.verification_input, detail.verification_candidates);
    detail.timings.fine_extract_ms = fine.extract_ms;
    detail.timings.fine_match_ms   = fine.match_ms;

    std::vector<DinoMatchResult> filtered;
    filtered.reserve(fine.results.size());
    for (const auto &match : fine.results)
    {
        if (!request.include_self && request.query_image_id >= 0)
        {
            if (match.image_id == request.query_image_id)
            {
                continue;
            }
        }
        filtered.push_back(match);
    }

    dinoNmsWithinImages(filtered, config.fine_nms_iou);

    // 阈值只作用于最终结果集合，不改变粗选覆盖。
    if (config.enable_decision_threshold)
    {
        filtered.erase(std::remove_if(filtered.begin(), filtered.end(),
                                      [&](const DinoMatchResult &match)
                                      { return match.score < config.decision_threshold; }),
                       filtered.end());
    }

    const size_t top_k = request.top_k > 0U ? std::min(request.top_k, config.final_k) : config.final_k;
    if (filtered.size() > top_k)
    {
        filtered.resize(top_k);
    }

    const auto output_started = dinoNowMs();
    for (const auto &match : filtered)
    {
        DinoSearchResult result;
        result.image_id             = match.image_id;
        result.bbox                 = DinoSearchRect{static_cast<float>(match.bbox.x0), static_cast<float>(match.bbox.y0),
                                     static_cast<float>(match.bbox.x1), static_cast<float>(match.bbox.y1)};
        result.score                = match.score;
        result.template_similarity  = match.template_similarity;
        result.query_coverage       = match.query_coverage;
        result.spatial_consistency  = match.spatial_consistency;
        if (match.from_region_channel)
        {
            result.coarse_sources.push_back("region");
        }
        if (match.from_local_channel)
        {
            result.coarse_sources.push_back("local");
        }
        detail.results.push_back(std::move(result));
    }

    detail.timings.output_ms = dinoNowMs() - output_started;
    detail.completed_candidates = fine.completed_candidates;
    detail.diagnostics.scanned_region_descriptors  = scan.scanned_region_descriptors;
    detail.diagnostics.scanned_local_descriptors    = scan.scanned_local_descriptors;
    detail.diagnostics.retained_local_views         = scan.retained_local_views;
    detail.diagnostics.region_candidates            = scan.region_candidates.size();
    detail.diagnostics.local_candidates             = scan.local_candidates.size();
    detail.diagnostics.fused_candidates             = fused.candidates.size();
    detail.diagnostics.view_count                   = reader.views().size();
    detail.diagnostics.model_forwards               = model_forwards + fine.model_forwards;
    detail.diagnostics.cache_hits                   = caches->images.hits() + caches->features.hits()
                                                              - cache_hits_before;
    detail.diagnostics.query_local_descriptors = 0;
    for (const auto &view : query.views) detail.diagnostics.query_local_descriptors += view.tokens.size();
    detail.diagnostics.query_valid_cells            = static_cast<size_t>(query.valid_cell_count);
    detail.diagnostics.low_local_evidence           = query.low_local_evidence;
    detail.diagnostics.outside_validated_profile    = query.outside_validated_profile;
    detail.diagnostics.peak_rss_bytes               = dinoPeakRssBytes();
    detail.diagnostics.gpu_allocated_peak_bytes     = scan.device_allocated_bytes;
    detail.diagnostics.gpu_reserved_delta_peak_bytes = scan.device_reserved_delta_bytes;
    detail.diagnostics.similarity_backend           = scan.similarity_backend;
    detail.diagnostics.local_scan_compact           = scan.compact_scan;
    detail.diagnostics.query_local_selection
        = query.selection_note + ";similarity_backend=" + scan.similarity_backend;

    if (deadline.expired() || scan.incomplete || fine.incomplete)
    {
        detail.status   = DinoSearchStatus::Incomplete;
        detail.decision = DinoSearchDecision::Incomplete;
        detail.message
            = "Query exceeded the wall deadline or some candidates failed; results are partial.";
    }
    else
    {
        detail.status = DinoSearchStatus::Completed;
        if (config.enable_decision_threshold)
        {
            detail.decision
                = filtered.empty() ? DinoSearchDecision::NoMatch : DinoSearchDecision::Matches;
            detail.message = "Configured decision threshold applied.";
        }
        else
        {
            detail.decision = DinoSearchDecision::RankedOnly;
            detail.message  = "No frozen decision threshold is configured; results are ranked only.";
        }
    }
    if (!query.profile_note.empty())
    {
        detail.message += (detail.message.empty() ? "" : " ") + query.profile_note;
    }
    return detail;
}

} // namespace

uint64_t dinoPeakRssBytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &counters, sizeof(counters)))
    {
        return static_cast<uint64_t>(counters.PeakWorkingSetSize);
    }
    return 0U;
#else
    std::ifstream status("/proc/self/status");
    std::string   line;
    while (std::getline(status, line))
    {
        if (line.rfind("VmHWM:", 0) == 0)
        {
            std::istringstream stream(line.substr(6));
            uint64_t           kilobytes = 0;
            stream >> kilobytes;
            return kilobytes * 1024U;
        }
    }
    return 0U;
#endif
}

DinoBuildReport dinoBuildItems(const std::vector<DinoImageItem> &items, const DinoRegionSearchConfig &config,
                               const fs::path &index_root, const DinoBuildProgressCallback &progress_callback,
                               const DinoOperationControl &control)
{
    const auto started = dinoNowMs();
    dinoValidateConfig(config);
    std::unique_lock lock(model_mutex, std::defer_lock);
    while (!lock.owns_lock())
    {
        if (control.cancelled && control.cancelled())
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
        }
        if (lock.try_lock_for(std::chrono::milliseconds(50)))
        {
            break;
        }
    }
    active_reader.reset(); // release file handles before rebuilding; no freshness protocol
    active_index_path.clear();
    dinoResetSearchCaches();

    DinoBuildReport report;
    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery items list is empty");
    }

    std::unordered_set<int64_t> seen_ids;
    for (const auto &item : items)
    {
        if (!seen_ids.insert(item.image_id).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Duplicate image_id in build items: %lld",
                                 static_cast<long long>(item.image_id));
        }
    }

    reportProgress(progress_callback, DinoBuildStage::LoadingModel, 0, 0, "loading frozen backbone");
    auto             backbone_holder = dinoAcquireBackbone(config);
    DinoBackbone    &backbone        = *backbone_holder;
    DinoViewPlanner  planner(backbone.patchSize(), backbone.encoderEdge(), config.view_overlap,
                            config.gallery_tile_edges, config.query_roi_target_lengths);
    DinoDescriptorBuildConfig descriptor_config;
    descriptor_config.coarse_dimension = config.coarse_dimension;
    descriptor_config.local_representatives = config.local_representatives;
    descriptor_config.window.ratios = config.region_window_ratios;
    descriptor_config.window.stride_ratio = config.region_window_stride_ratio;
    descriptor_config.window.min_valid_fraction = config.region_min_valid_fraction;
    descriptor_config.merge_enabled = config.merge_enabled;
    descriptor_config.merge_epsilon = config.merge_epsilon;
    descriptor_config.max_leaf_side_patches = config.max_leaf_side_patches;

    const auto root = fs::absolute(index_root).lexically_normal();
    DinoIndexWriter writer(root, static_cast<size_t>(config.coarse_dimension), config.quantize_int8);
    size_t processed = 0;
    for (const auto &item : items)
    {
        if (control.cancelled && control.cancelled())
        {
            writer.abort();
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
        }
        if (processed < 4U || processed % 16U == 0U)
        {
            reportProgress(progress_callback, DinoBuildStage::ExtractingViews, processed, items.size(),
                           dinoPathToUtf8(item.image_path.filename()));
        }
        try
        {
            const auto canonical = DinoImageLoader::load(item.image_path, item.image_id);
            writer.addImage(canonical.record);
            const auto image_index = writer.imageIdentities().size() - 1U;
            const auto plans = planner.planGalleryViews(canonical.record.width, canonical.record.height);
            const auto spec = dinoViewPreprocessSpec(planner.encoderEdge(), planner.patchSize());
            const auto batch = std::max<size_t>(1U, backbone.maxBatchSize());

            for (size_t begin = 0; begin < plans.size(); begin += batch)
            {
                if (control.cancelled && control.cancelled())
                {
                    writer.abort();
                    throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
                }
                const size_t count = std::min(batch, plans.size() - begin);
                std::vector<DinoViewRaster> rasters;
                rasters.reserve(count);
                for (size_t index = 0; index < count; ++index)
                {
                    rasters.push_back(dinoRenderView(canonical.image, plans[begin + index], spec));
                }
                const auto grids = backbone.extract(rasters);
                for (size_t index = 0; index < grids.size(); ++index)
                {
                    const auto view_id = writer.addView(static_cast<int>(image_index), plans[begin + index]);
                    const auto descriptors = dinoBuildViewDescriptors(view_id, grids[index], descriptor_config);
                    for (size_t region = 0; region < descriptors.region_meta.size(); ++region)
                    {
                        writer.addRegion(view_id, descriptors.region_meta[region], descriptors.region_vectors[region]);
                    }
                    for (size_t local = 0; local < descriptors.local_meta.size(); ++local)
                    {
                        writer.addLocal(view_id, descriptors.local_meta[local], descriptors.local_vectors[local]);
                    }
                }
            }
        }
        catch (const irt::Exception &error)
        {
            if (error.code() == irt::Status::INVALID_OPERATION)
            {
                writer.abort();
                throw;
            }
            report.failed_files.push_back(dinoPathToUtf8(fs::absolute(item.image_path).lexically_normal()));
            report.messages.push_back(std::string("failed: ") + dinoPathToUtf8(item.image_path.filename()) + " -> "
                                      + error.what());
        }
        catch (const std::exception &error)
        {
            report.failed_files.push_back(dinoPathToUtf8(fs::absolute(item.image_path).lexically_normal()));
            report.messages.push_back(std::string("failed: ") + dinoPathToUtf8(item.image_path.filename()) + " -> "
                                      + error.what());
        }
        ++processed;
    }
    reportProgress(progress_callback, DinoBuildStage::WritingIndex, items.size(), items.size(), "writing index");

    if (writer.imageIdentities().empty())
    {
        writer.abort();
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "No gallery image could be indexed; all inputs failed");
    }

    if (control.cancelled && control.cancelled())
    {
        writer.abort();
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
    }

    reportProgress(progress_callback, DinoBuildStage::Quantizing, 0, 0, "finalizing compact arrays");
    const auto finish_report = writer.finish();

    report.image_count             = writer.imageIdentities().size();
    report.failed_image_count      = report.failed_files.size();
    report.ready_with_errors       = !report.failed_files.empty();
    report.view_count              = finish_report.view_count;
    report.region_descriptor_count = finish_report.region_descriptor_count;
    report.local_descriptor_count  = finish_report.local_descriptor_count;
    report.original_patch_count    = finish_report.original_patch_count;
    report.index_bytes             = finish_report.index_bytes;
    report.duration_ms             = dinoNowMs() - started;
    report.messages.insert(report.messages.end(), finish_report.messages.begin(), finish_report.messages.end());

    for (size_t index = 0; index < writer.imageIdentities().size(); ++index)
    {
        const auto &identity = writer.imageIdentities()[index];
        DinoImageRecord record;
        record.image_id                 = identity.image_id;
        record.width                    = identity.width;
        record.height                   = identity.height;
        record.view_count               = static_cast<int>(writer.imageViewCount(index));
        record.region_descriptor_count  = static_cast<int>(writer.imageRegionCount(index));
        record.local_descriptor_count   = static_cast<int>(writer.imageLocalCount(index));
        record.original_patch_count     = writer.imagePatchCount(index);
        report.images.push_back(std::move(record));
    }

    reportProgress(progress_callback, DinoBuildStage::Finalizing, report.image_count, report.image_count,
                   "index complete");
    return report;
}

DinoBuildReport dinoBuildIndex(const fs::path &gallery_root, const DinoRegionSearchConfig &config,
                               const fs::path &index_root, const DinoBuildProgressCallback &progress_callback,
                               const DinoOperationControl &control)
{
    if (!fs::exists(gallery_root))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery root does not exist: %s",
                             gallery_root.string().c_str());
    }

    const auto images = DinoImageLoader::collectGalleryImages(gallery_root);
    reportProgress(progress_callback, DinoBuildStage::ScanningImages, 0, images.size(), "collecting gallery images");
    if (images.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery root contains no supported image: %s",
                             gallery_root.string().c_str());
    }

    std::vector<DinoImageItem> items;
    items.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        items.push_back(DinoImageItem{static_cast<int64_t>(i), images[i]});
    }

    const auto root = fs::absolute(index_root.empty() ? gallery_root / ".." / kDefaultIndexDirectoryName : index_root)
                         .lexically_normal();
    return dinoBuildItems(items, config, root, progress_callback, control);
}

DinoSearchResponse dinoSearchIndex(const fs::path &index_root, const DinoSearchRequest &request,
                                   const DinoRegionSearchConfig &config,
                                   const DinoSearchProgressCallback &progress_callback,
                                   const DinoOperationControl &control)
{
    dinoValidateConfig(config);
    if (request.deadline_ms < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Request deadline must be zero or positive");
    }
    const int64_t     query_deadline_ms = request.deadline_ms > 0 ? request.deadline_ms : config.query_deadline_ms;
    const DinoDeadline deadline(query_deadline_ms);
    const auto        wall_started = dinoNowMs();
    std::unique_lock lock(model_mutex, std::defer_lock);
    while (!lock.owns_lock())
    {
        if (control.cancelled && control.cancelled())
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
        }
        if (query_deadline_ms > 0 && deadline.expired())
        {
            DinoSearchResponse response;
            response.request_id = request.request_id;
            response.status = DinoSearchStatus::Incomplete;
            response.decision = DinoSearchDecision::Incomplete;
            response.message = "Query deadline expired while waiting for the model.";
            response.timings.wall_ms = dinoNowMs() - wall_started;
            reportSearchProgress(progress_callback, DinoSearchStage::Output, 0, 0);
            return response;
        }
        const int64_t wait_chunk = query_deadline_ms > 0 ? std::clamp<int64_t>(deadline.remainingMs(), 1, 50) : 50;
        if (lock.try_lock_for(std::chrono::milliseconds(wait_chunk)))
        {
            break;
        }
    }
    const double queue_ms = dinoNowMs() - wall_started;

    const auto incompleteResponse = [&](const std::string &message)
    {
        DinoSearchResponse detail;
        detail.request_id   = request.request_id;
        detail.status       = DinoSearchStatus::Incomplete;
        detail.decision     = DinoSearchDecision::Incomplete;
        detail.message      = message;
        detail.timings.queue_ms = queue_ms;
        detail.timings.wall_ms  = dinoNowMs() - wall_started;
        reportSearchProgress(progress_callback, DinoSearchStage::Output, 0, 0);
        return detail;
    };

    if (!lock.owns_lock())
    {
        return incompleteResponse("Query deadline expired while waiting for the model.");
    }

    if (control.cancelled && control.cancelled())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Cancelled");
    }

    const auto path = fs::absolute(index_root).lexically_normal();
    if (!active_reader || active_index_path != path) {
        active_reader = std::make_unique<DinoIndexReader>(path);
        active_index_path = path;
    }
    const auto &reader = *active_reader;
    if (deadline.expired())
    {
        return incompleteResponse("Query deadline expired while loading the index; no candidates were processed.");
    }

    reportSearchProgress(progress_callback, DinoSearchStage::Decode, 0, 0);
    const auto decode_started = dinoNowMs();
    if (request.query_path.empty() || !fs::exists(request.query_path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Query image does not exist: %s",
                             request.query_path.string().c_str());
    }
    const auto query_image = DinoImageLoader::load(request.query_path);
    const auto roi         = dinoToInternalRoi(request.roi, query_image.record.width, query_image.record.height);
    const auto decode_ms   = dinoNowMs() - decode_started;
    if (deadline.expired())
    {
        auto detail = incompleteResponse("Query deadline expired while decoding the query image; no candidates were processed.");
        detail.timings.decode_ms = decode_ms;
        return detail;
    }

    try
    {
        auto detail = runSearchPipeline(reader, query_image, roi, request, config, progress_callback, deadline, control);
        detail.request_id        = request.request_id;
        detail.timings.decode_ms = decode_ms;
        detail.timings.queue_ms = queue_ms;
        detail.timings.wall_ms   = dinoNowMs() - wall_started;
        reportSearchProgress(progress_callback, DinoSearchStage::Output, detail.results.size(),
                             detail.results.size());
        return detail;
    }
    catch (const irt::Exception &error)
    {
        if (error.code() == irt::Status::INVALID_OPERATION)
        {
            throw;
        }
        if (error.code() != irt::Status::ERROR_OUT_OF_MEMORY)
        {
            throw;
        }
        auto detail = incompleteResponse("GPU resource exhausted; the query was not completed and no result is final.");
        detail.timings.decode_ms = decode_ms;
        return detail;
    }
    catch (const std::bad_alloc &)
    {
        auto detail = incompleteResponse("Host memory exhausted; the query was not completed and no result is final.");
        detail.timings.decode_ms = decode_ms;
        return detail;
    }
}

void dinoReleaseRuntime(bool release_backbone)
{
    std::lock_guard lock(model_mutex);
    active_reader.reset();
    active_index_path.clear();
    dinoResetSearchCaches();
    if (release_backbone)
    {
        dinoResetBackbones();
    }
}

} // namespace irt::features::priv
