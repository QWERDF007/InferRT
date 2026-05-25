#pragma once

#include <inferrt/features/ImageSearch.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace faiss {
struct Index;
} // namespace faiss

namespace irt::features {

namespace priv {
class ImageSearchFeatureExtractor;
} // namespace priv

class ImageSearch::Impl
{
public:
    Impl(std::string model_name, std::string feature_name);
    ~Impl();

    Impl(const Impl &)            = delete;
    Impl &operator=(const Impl &) = delete;

    void buildOrLoad(const std::filesystem::path &weights_file, const std::filesystem::path &gallery_dir,
                     const std::filesystem::path &index_file, bool rebuild_index);

    std::vector<ImageSearchResult> search(const std::filesystem::path &query_image, int top_k);

    bool                               isReady() const noexcept;
    const std::string                 &modelName() const noexcept;
    const std::string                 &featureName() const noexcept;
    const std::filesystem::path       &indexPath() const noexcept;
    std::vector<std::filesystem::path> galleryImages() const;
    int                                featureDim() const noexcept;

private:
    void ensureExtractor();

    std::string                                        model_name_;
    std::string                                        feature_name_;
    std::filesystem::path                              weights_file_;
    std::filesystem::path                              gallery_dir_;
    std::filesystem::path                              index_path_;
    std::vector<std::filesystem::path>                 gallery_images_;
    std::unique_ptr<faiss::Index>                      index_;
    std::unique_ptr<priv::ImageSearchFeatureExtractor> extractor_;
    int                                                feature_dim_{0};
};

} // namespace irt::features
