/**
 * @file ImageSearch.cpp
 * @brief ``ImageSearch`` 公共 API 的实现与图库/索引路径辅助函数。
 */

#include "priv/ImageSearchImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
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

/**
 * @brief 将字符串规范化为安全的文件名片段。
 *
 * 非字母数字字符替换为下划线，用于生成默认索引文件名。
 *
 * @param value 原始字符串（如模型名、特征名）。
 * @return 可用于文件名的 stem 字符串。
 */
std::string sanitizeFileStem(std::string_view value)
{
    std::string stem;
    stem.reserve(value.size());
    for (unsigned char ch : value)
    {
        if (std::isalnum(ch))
        {
            stem.push_back(static_cast<char>(ch));
        }
        else
        {
            stem.push_back('_');
        }
    }
    return stem;
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
                              bool rebuild_index)
{
    impl_->buildOrLoad(weights_file, gallery_dir, index_file, rebuild_index);
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
    return gallery_dir / (sanitizeFileStem(model_name) + "_" + sanitizeFileStem(feature_name) + ".faiss");
}

} // namespace irt::features
