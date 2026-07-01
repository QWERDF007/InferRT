/**
 * @file RoiSearch.cpp
 * @brief ``RoiSearch`` 公共 API 实现。
 */

#include "priv/RoiSearchImpl.hpp"

#include <inferrt/util/FileManifest.hpp>

#include <utility>

namespace fs = std::filesystem;

namespace irt::features {

RoiSearch::RoiSearch(RoiSearchConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

RoiSearch::~RoiSearch() = default;

RoiSearch::RoiSearch(RoiSearch &&other) noexcept = default;

RoiSearch &RoiSearch::operator=(RoiSearch &&other) noexcept = default;

void RoiSearch::buildOrLoad(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                            const fs::path &index_file, bool rebuild_index,
                            RoiSearchBuildProgressCallback progress_callback)
{
    impl_->buildOrLoad(weights_file, gallery_items, index_file, rebuild_index, std::move(progress_callback));
}

void RoiSearch::build(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                      const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    impl_->build(weights_file, gallery_items, index_file, std::move(progress_callback));
}

void RoiSearch::load(const fs::path &weights_file, const fs::path &index_file)
{
    impl_->load(weights_file, index_file);
}

std::vector<RoiSearchResult> RoiSearch::search(const fs::path &query_image, const RoiSearchBox &roi, int top_k)
{
    return impl_->search(query_image, roi, top_k);
}

bool RoiSearch::isReady() const noexcept
{
    return impl_ && impl_->isReady();
}

const RoiSearchConfig &RoiSearch::config() const noexcept
{
    return impl_->config();
}

const fs::path &RoiSearch::indexPath() const noexcept
{
    return impl_->indexPath();
}

std::vector<int64_t> RoiSearch::galleryIds() const
{
    return impl_->galleryIds();
}

int RoiSearch::featureDim() const noexcept
{
    return impl_->featureDim();
}

fs::path RoiSearch::defaultIndexPath(const fs::path &output_dir, const std::string &model_name,
                                     const std::string &feature_name)
{
    (void)model_name;
    (void)feature_name;
    return irt::util::resolveOutputFilePath({}, output_dir, ".faiss");
}

} // namespace irt::features
