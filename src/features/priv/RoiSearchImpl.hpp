#pragma once

/**
 * @file RoiSearchImpl.hpp
 * @brief ``RoiSearch::Impl`` PIMPL 声明。
 */

#include <inferrt/features/RoiSearch.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace faiss {
struct Index;
struct IndexFlat;

namespace gpu {
class StandardGpuResources;
} // namespace gpu
} // namespace faiss

namespace irt::features {

namespace priv {
class RoiFeatureExtractor;
struct FaissIndexBundle;
} // namespace priv

/**
 * @brief ``RoiSearch`` 的私有实现。
 */
class RoiSearch::Impl
{
public:
    explicit Impl(RoiSearchConfig config);
    ~Impl();

    Impl(const Impl &)            = delete;
    Impl &operator=(const Impl &) = delete;

    void buildOrLoad(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                     const std::filesystem::path &index_file, bool rebuild_index,
                     RoiSearchBuildProgressCallback progress_callback);

    void build(const std::filesystem::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
               const std::filesystem::path &index_file, RoiSearchBuildProgressCallback progress_callback);

    void load(const std::filesystem::path &weights_file, const std::filesystem::path &index_file);

    std::vector<RoiSearchResult> search(const std::filesystem::path &query_image, const RoiSearchBox &roi, int top_k);

    std::vector<RoiSearchResult> search(const RoiSearchItem &query, int top_k);

    bool                         isReady() const noexcept;
    const RoiSearchConfig       &config() const noexcept;
    const std::filesystem::path &indexPath() const noexcept;
    std::vector<int64_t>         galleryIds() const;
    int                          featureDim() const noexcept;

    RoiFeatureMatrixView featureView() const;
    RoiFeatureWorkStats featureWorkStats() const noexcept;
    std::vector<RoiSearchResult> searchByRoiId(int64_t roi_id, int top_k);
    std::vector<RoiSearchResult> repeatSearch(int top_k);

private:
    void releaseState();
    std::vector<RoiSearchResult> searchVector(const float* query, int top_k);
    std::vector<float> last_query_feature_;
    void buildWithItems(const std::filesystem::path &weights_file, std::vector<RoiSearchItem> gallery_items,
                        const std::filesystem::path &index_file, RoiSearchBuildProgressCallback progress_callback);
    void installIndex(const std::filesystem::path &weights_file, const std::filesystem::path &index_file,
                      priv::FaissIndexBundle bundle, std::vector<int64_t> gallery_ids, int feature_dim,
                      std::unique_ptr<priv::RoiFeatureExtractor> extractor);
    void ensureExtractor();
    const faiss::IndexFlat *getHostFlatIndex() const noexcept;

    RoiSearchConfig config_{}; ///< ROI 检索配置。

    std::filesystem::path weights_file_; ///< 模型文件路径。
    std::filesystem::path index_path_;   ///< Faiss 索引路径。

    std::vector<int64_t> gallery_ids_; ///< 与 Faiss 向量 ID 对应的 ROI ID。

    std::unique_ptr<faiss::IndexFlat>                 host_flat_index_;     ///< CPU 宿主端精确向量索引（供 featureView 及 CPU 零拷贝聚类）。
    std::unique_ptr<faiss::gpu::StandardGpuResources> faiss_gpu_resources_; ///< GPU Faiss 资源。
    std::unique_ptr<faiss::Index>                     index_;               ///< Faiss 索引。
    std::unique_ptr<priv::RoiFeatureExtractor>        extractor_;           ///< ROI 特征抽取器。

    int feature_dim_{0}; ///< ROI 特征向量维度。
};

} // namespace irt::features
