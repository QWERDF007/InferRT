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
constexpr const char *kDefaultIndexDirectoryName = "dino_region_index";

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




DinoSearchDetail runSearchPipeline(const DinoIndexReader &reader, const DinoCanonicalImage &query_image,
                                   const DinoRoi &roi, const DinoSearchRequest &request,
                                   const DinoRegionSearchConfig &config,
                                   const DinoSearchProgressCallback &progress_callback, const DinoDeadline &deadline)
{
    DinoSearchDetail detail;
    auto             backbone_holder = dinoAcquireBackbone(config);
    DinoBackbone    &backbone        = *backbone_holder;
    DinoViewPlanner  planner(backbone.patchSize(), backbone.encoderEdge(), config.view_overlap,
                            config.gallery_tile_edges, config.query_roi_target_lengths);

    if (reader.descriptorDim() != static_cast<size_t>(backbone.channels()))
    {
        throw irt::Exception(irt::Status::NOT_READY,
                             "Index descriptor dimension does not match the configured backbone");
    }
    const auto extractor_key = backbone.signature().cacheKey();
    const auto caches = dinoAcquireSearchCaches(extractor_key, config.image_cache_bytes,
                                                config.dense_feature_cache_bytes);
    const size_t cache_hits_before = caches->images.hits() + caches->features.hits();
    reportSearchProgress(progress_callback, DinoSearchStage::QueryExtract, 0, 0);
    size_t model_forwards = 0;
    const auto query_started = dinoNowMs();
    auto   query          = dinoBuildQuery(query_image, roi, backbone, planner, config, model_forwards);
    detail.response.timings.query_extract_ms = dinoNowMs() - query_started;

    reportSearchProgress(progress_callback, DinoSearchStage::RegionScan, 0, 0);
    auto scan = dinoScan(reader, query, config, deadline);
    detail.response.timings.region_scan_ms          = scan.region_scan_ms;
    detail.response.timings.local_scan_ms           = scan.local_scan_ms;
    detail.response.timings.local_window_rescore_ms = scan.window_rescore_ms;

    reportSearchProgress(progress_callback, DinoSearchStage::Fusion, 0, 0);
    const auto fusion_started = dinoNowMs();
    const auto fused = dinoFuseCandidates(scan.region_candidates, scan.local_candidates, config);
    detail.response.timings.fusion_ms = dinoNowMs() - fusion_started;
    detail.response.total_candidates = fused.candidates.size();
    const std::vector<DinoCandidate> *candidate_groups[] = {
        &scan.region_candidates, &scan.local_candidates, &fused.candidates};
    std::vector<DinoCoarseCandidate> *reported_groups[] = {
        &detail.response.region_candidates, &detail.response.local_candidates, &detail.response.coarse_candidates};
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
            if (!request.include_self && image.image_id == query_image.record.image_id)
                continue;
            const auto crop = dinoExpandRect(candidate.source_bbox, config.fine_candidate_expand,
                                             image.width, image.height);
            reported.push_back({dinoPathFromUtf8(image.source_path),
                {static_cast<float>(crop.x0), static_cast<float>(crop.y0),
                 static_cast<float>(crop.x1), static_cast<float>(crop.y1)}});
        }
    }

    reportSearchProgress(progress_callback, DinoSearchStage::FineMatch, 0, fused.candidates.size());
    auto fine = dinoFineMatch(reader, query_image, query, fused.candidates, config, backbone, planner,
                              caches->images, caches->features, extractor_key, deadline);
    detail.response.timings.fine_extract_ms = fine.extract_ms;
    detail.response.timings.fine_match_ms   = fine.match_ms;

    // 自身按规范源路径排除；不同路径的图片各自作为图库条目。
    std::vector<DinoMatchResult> filtered;
    filtered.reserve(fine.results.size());
    for (const auto &match : fine.results)
    {
        if (!request.include_self)
        {
            if (match.image_id == query_image.record.image_id)
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
        result.source_path          = match.source_path;
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
        detail.response.results.push_back(std::move(result));
    }

    detail.response.timings.output_ms = dinoNowMs() - output_started;
    detail.response.completed_candidates = fine.completed_candidates;
    detail.response.diagnostics.scanned_region_descriptors  = scan.scanned_region_descriptors;
    detail.response.diagnostics.scanned_local_descriptors    = scan.scanned_local_descriptors;
    detail.response.diagnostics.retained_local_views         = scan.retained_local_views;
    detail.response.diagnostics.region_candidates            = scan.region_candidates.size();
    detail.response.diagnostics.local_candidates             = scan.local_candidates.size();
    detail.response.diagnostics.fused_candidates             = fused.candidates.size();
    detail.response.diagnostics.view_count                   = reader.views().size();
    detail.response.diagnostics.model_forwards               = model_forwards + fine.model_forwards;
    detail.response.diagnostics.cache_hits                   = caches->images.hits() + caches->features.hits()
                                                              - cache_hits_before;
    detail.response.diagnostics.query_local_descriptors      = query.views.empty() ? 0U : query.views.front().tokens.size();
    detail.response.diagnostics.query_valid_cells            = static_cast<size_t>(query.valid_cell_count);
    detail.response.diagnostics.low_local_evidence           = query.low_local_evidence;
    detail.response.diagnostics.outside_validated_profile    = query.outside_validated_profile;
    detail.response.diagnostics.peak_rss_bytes               = dinoPeakRssBytes();
    detail.response.diagnostics.gpu_allocated_peak_bytes     = scan.device_allocated_bytes;
    detail.response.diagnostics.gpu_reserved_delta_peak_bytes = scan.device_reserved_delta_bytes;
    detail.response.diagnostics.similarity_backend           = scan.similarity_backend;
    detail.response.diagnostics.local_scan_compact           = scan.compact_scan;
    detail.response.diagnostics.query_local_selection
        = query.selection_note + ";similarity_backend=" + scan.similarity_backend;

    detail.query_image_id = query_image.record.image_id;

    if (deadline.expired() || scan.incomplete || fine.incomplete)
    {
        detail.response.status   = DinoSearchStatus::Incomplete;
        detail.response.decision = DinoSearchDecision::Incomplete;
        detail.response.message
            = "Query exceeded the wall deadline or some candidates failed; results are partial.";
    }
    else
    {
        detail.response.status = DinoSearchStatus::Completed;
        if (config.enable_decision_threshold)
        {
            detail.response.decision
                = filtered.empty() ? DinoSearchDecision::NoMatch : DinoSearchDecision::Matches;
            detail.response.message = "Configured decision threshold applied.";
        }
        else
        {
            detail.response.decision = DinoSearchDecision::RankedOnly;
            detail.response.message  = "No frozen decision threshold is configured; results are ranked only.";
        }
    }
    if (!query.profile_note.empty())
    {
        detail.response.message += (detail.response.message.empty() ? "" : " ") + query.profile_note;
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

DinoBuildReport dinoBuildIndex(const fs::path &gallery_root, const DinoRegionSearchConfig &config,

                               const fs::path &index_root, const DinoBuildProgressCallback &progress_callback)
{
    const auto started = dinoNowMs();
    dinoValidateConfig(config);
    std::lock_guard lock(model_mutex);
    if (!fs::exists(gallery_root))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery root does not exist: %s",
                             gallery_root.string().c_str());
    }

    DinoBuildReport report;
    const auto      images = DinoImageLoader::collectGalleryImages(gallery_root);
    reportProgress(progress_callback, DinoBuildStage::ScanningImages, 0, images.size(), "collecting gallery images");
    if (images.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery root contains no supported image: %s",
                             gallery_root.string().c_str());
    }

    reportProgress(progress_callback, DinoBuildStage::LoadingModel, 0, 0, "loading frozen backbone");
    auto             backbone_holder = dinoAcquireBackbone(config);
    DinoBackbone    &backbone        = *backbone_holder;
    DinoViewPlanner  planner(backbone.patchSize(), backbone.encoderEdge(), config.view_overlap,
                            config.gallery_tile_edges, config.query_roi_target_lengths);
    DinoDescriptorBuildConfig descriptor_config;
    descriptor_config.window.ratios = config.region_window_ratios;
    descriptor_config.window.stride_ratio = config.region_window_stride_ratio;
    descriptor_config.window.min_valid_fraction = config.region_min_valid_fraction;
    descriptor_config.merge_enabled = config.merge_enabled;
    descriptor_config.merge_epsilon = config.merge_epsilon;
    descriptor_config.max_leaf_side_patches = config.max_leaf_side_patches;

    const auto root = fs::absolute(index_root.empty() ? gallery_root / ".." / kDefaultIndexDirectoryName : index_root).lexically_normal();
    DinoIndexWriter writer(root, config, static_cast<size_t>(backbone.channels()), config.quantize_int8);

    size_t                processed = 0;
    for (const auto &path : images)
    {
        if (processed < 4U || processed % 16U == 0U)
        {
            reportProgress(progress_callback, DinoBuildStage::ExtractingViews, processed, images.size(),
                           dinoPathToUtf8(path.filename()));
        }
        try
        {
            const auto canonical = DinoImageLoader::load(path);
    writer.addImage(canonical.record);
    const auto image_index = writer.imageIdentities().size() - 1U;
    const auto plans       = planner.planGalleryViews(canonical.record.width, canonical.record.height);
    const auto spec        = dinoViewPreprocessSpec(planner.encoderEdge(), planner.patchSize());
    const auto batch       = std::max<size_t>(1U, backbone.maxBatchSize());

    for (size_t begin = 0; begin < plans.size(); begin += batch)
    {
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
        catch (const std::exception &error)
        {
            report.failed_files.push_back(dinoPathToUtf8(fs::absolute(path).lexically_normal()));
            report.messages.push_back(std::string("failed: ") + dinoPathToUtf8(path.filename()) + " -> "
                                              + error.what());
        }
        ++processed;
    }
    reportProgress(progress_callback, DinoBuildStage::WritingIndex, images.size(), images.size(), "writing index");

    if (writer.imageIdentities().empty())
    {
        writer.abort();
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "No gallery image could be indexed; all inputs failed");
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
        record.source_path              = identity.source_path;
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


DinoSearchDetail dinoSearchIndex(const fs::path &index_root, const DinoSearchRequest &request,
                                 const DinoRegionSearchConfig &config,
                                 const DinoSearchProgressCallback &progress_callback)
{
    dinoValidateConfig(config);
    const DinoDeadline deadline(config.query_deadline_ms);
    const auto        wall_started = dinoNowMs();
    std::unique_lock lock(model_mutex, std::defer_lock);
    if (config.query_deadline_ms > 0)
    {
        lock.try_lock_for(std::chrono::milliseconds(deadline.remainingMs()));
    }
    else
    {
        lock.lock();
    }
    const double queue_ms = dinoNowMs() - wall_started;

    const auto incompleteResponse = [&](const std::string &message)
    {
        DinoSearchDetail detail;
        detail.response.request_id   = request.request_id;
        detail.response.status       = DinoSearchStatus::Incomplete;
        detail.response.decision     = DinoSearchDecision::Incomplete;
        detail.response.message      = message;
        detail.response.timings.queue_ms = queue_ms;
        detail.response.timings.wall_ms  = dinoNowMs() - wall_started;
        reportSearchProgress(progress_callback, DinoSearchStage::Output, 0, 0);
        return detail;
    };

    if (!lock.owns_lock())
    {
        return incompleteResponse("Query deadline expired while waiting for the model.");
    }

    DinoIndexReader reader(index_root);
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
        detail.response.timings.decode_ms = decode_ms;
        return detail;
    }

    try
    {
        auto detail = runSearchPipeline(reader, query_image, roi, request, config, progress_callback, deadline);
        detail.response.request_id        = request.request_id;
        detail.response.timings.decode_ms = decode_ms;
        detail.response.timings.queue_ms = queue_ms;
        detail.response.timings.wall_ms   = dinoNowMs() - wall_started;
        reportSearchProgress(progress_callback, DinoSearchStage::Output, detail.response.results.size(),
                             detail.response.results.size());
        return detail;
    }
    catch (const irt::Exception &error)
    {
        if (error.code() != irt::Status::ERROR_OUT_OF_MEMORY)
        {
            throw;
        }
        auto detail = incompleteResponse("GPU resource exhausted; the query was not completed and no result is final.");
        detail.response.timings.decode_ms = decode_ms;
        return detail;
    }
    catch (const std::bad_alloc &)
    {
        auto detail = incompleteResponse("Host memory exhausted; the query was not completed and no result is final.");
        detail.response.timings.decode_ms = decode_ms;
        return detail;
    }
}


} // namespace irt::features::priv
