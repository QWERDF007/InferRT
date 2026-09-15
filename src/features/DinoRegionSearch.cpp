/**
 * @file DinoRegionSearch.cpp
 * @brief Public API implementation for DINO region search.
 */

#include <inferrt/features/DinoRegionSearch.hpp>

#include "priv/dino/DinoEngine.hpp"
#include "priv/dino/DinoSimilarity.hpp"

namespace irt::features {

void dinoSetScanBackendOverride(const std::string &backend)
{
    if (backend == "auto")
    {
        priv::dinoOverrideSimilarityBackend(false, priv::DinoSimilarityBackend::Cpu);
        return;
    }
    if (backend == "cpu")
    {
        priv::dinoOverrideSimilarityBackend(true, priv::DinoSimilarityBackend::Cpu);
        return;
    }
    if (backend == "cuda")
    {
        priv::dinoOverrideSimilarityBackend(true, priv::DinoSimilarityBackend::Cuda);
        return;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "Unsupported scan backend '%s'; expected auto, cpu or cuda", backend.c_str());
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
