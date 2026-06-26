/**
 * @file ImageSearch.cpp
 * @brief ``ImageSearch`` 公共 API 的实现与图库/索引路径辅助函数。
 */

#include "priv/ImageSearchImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/util/FileManifest.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {

namespace {

/**
 * @brief 将字符串转换为小写。
 * @param value 原始字符串。
 * @return 小写字符串。
 */
std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

} // namespace

ImageSearch::ImageSearch(ImageSearchConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

ImageSearch::~ImageSearch() = default;

ImageSearch::ImageSearch(ImageSearch &&other) noexcept = default;

ImageSearch &ImageSearch::operator=(ImageSearch &&other) noexcept = default;

void ImageSearch::buildOrLoad(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file,
                              bool rebuild_index, ImageSearchBuildProgressCallback progress_callback)
{
    impl_->buildOrLoad(weights_file, gallery_dir, index_file, rebuild_index, std::move(progress_callback));
}

void ImageSearch::build(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file,
                        ImageSearchBuildProgressCallback progress_callback)
{
    impl_->build(weights_file, gallery_dir, index_file, std::move(progress_callback));
}

void ImageSearch::build(const fs::path &weights_file, const std::vector<fs::path> &gallery_images,
                        const fs::path &index_file, ImageSearchBuildProgressCallback progress_callback)
{
    impl_->build(weights_file, gallery_images, index_file, std::move(progress_callback));
}

void ImageSearch::load(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file)
{
    impl_->load(weights_file, gallery_dir, index_file);
}

std::vector<ImageSearchResult> ImageSearch::search(const fs::path &query_image, int top_k)
{
    return impl_->search(query_image, top_k);
}

bool ImageSearch::isReady() const noexcept
{
    return impl_ && impl_->isReady();
}

const ImageSearchConfig &ImageSearch::config() const noexcept
{
    return impl_->config();
}

const fs::path &ImageSearch::indexPath() const noexcept
{
    return impl_->indexPath();
}

std::vector<fs::path> ImageSearch::galleryImages() const
{
    return impl_->galleryImages();
}

int ImageSearch::featureDim() const noexcept
{
    return impl_->featureDim();
}

bool ImageSearch::isImageFile(const fs::path &path)
{
    if (!path.has_extension())
    {
        return false;
    }

    const std::string ext = toLower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

std::vector<fs::path> ImageSearch::collectGalleryImages(const fs::path &gallery_dir)
{
    if (!fs::exists(gallery_dir) || !fs::is_directory(gallery_dir))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Gallery directory does not exist: %s",
                             gallery_dir.string().c_str());
    }

    // 递归扫描后统一转为绝对路径，保证同一图库在不同工作目录下生成稳定的路径映射。
    std::vector<fs::path> images;
    for (const auto &entry : fs::recursive_directory_iterator(gallery_dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        if (isImageFile(entry.path()))
        {
            images.push_back(fs::absolute(entry.path()));
        }
    }

    std::sort(images.begin(), images.end());
    if (images.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "No images found under gallery directory: %s",
                             gallery_dir.string().c_str());
    }
    return images;
}

fs::path ImageSearch::defaultIndexPath(const fs::path &gallery_dir, const std::string &model_name,
                                       const std::string &feature_name)
{
    (void)model_name;
    (void)feature_name;
    return irt::util::resolveOutputFilePath({}, gallery_dir, ".faiss");
}

} // namespace irt::features
