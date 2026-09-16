/**
 * @file DinoRegionSearch.cpp
 * @brief Public API implementation for DINO region search.
 */

#include <inferrt/features/DinoRegionSearch.hpp>

#include "priv/dino/DinoEngine.hpp"
#include "priv/dino/DinoIndexStore.hpp"

namespace irt::features {

bool DinoRegionSearch::needsRebuild(const std::filesystem::path &index_root, const DinoRegionSearchConfig &config)
{
    return priv::dinoIndexNeedsRebuild(index_root, config);
}

DinoBuildReport DinoRegionSearch::build(const std::vector<DinoImageItem> &items,
                                        const DinoRegionSearchConfig &config,
                                        const std::filesystem::path &index_root,
                                        const DinoBuildProgressCallback &progress_callback,
                                        const DinoOperationControl &control)
{
    return priv::dinoBuildItems(items, config, index_root, progress_callback, control);
}

DinoBuildReport DinoRegionSearch::build(const std::filesystem::path &gallery_root,
                                        const DinoRegionSearchConfig &config,
                                        const std::filesystem::path &index_root,
                                        const DinoBuildProgressCallback &progress_callback,
                                        const DinoOperationControl &control)
{
    return priv::dinoBuildIndex(gallery_root, config, index_root, progress_callback, control);
}

DinoSearchResponse DinoRegionSearch::search(const std::filesystem::path &index_root,
                                            const DinoSearchRequest &request,
                                            const DinoRegionSearchConfig &config,
                                            const DinoSearchProgressCallback &progress_callback)
{
    return priv::dinoSearchIndex(index_root, request, config, progress_callback);
}

DinoSearchResponse DinoRegionSearch::search(const std::filesystem::path &index_root,
                                            const DinoSearchRequest &request,
                                            const DinoRegionSearchConfig &config,
                                            const DinoSearchProgressCallback &progress_callback,
                                            const DinoOperationControl &control)
{
    return priv::dinoSearchIndex(index_root, request, config, progress_callback, control);
}

void DinoRegionSearch::releaseRuntime(bool release_backbone)
{
    priv::dinoReleaseRuntime(release_backbone);
}

} // namespace irt::features
