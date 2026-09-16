#pragma once

/**
 * @file DinoIndexStore.hpp
 * @brief 索引根目录中的紧凑数组写入与流式读取。
 */

#include "DinoDescriptors.hpp"
#include "DinoIndexStoreTypes.hpp"
#include "DinoStorageFormat.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace irt::features::priv {

/**
 * @brief 索引持久化契约（Manifest 强类型定义）。
 *
 * 集中管理写入 index.yaml 的 manifest 元数据以及建库、预检查和 search 共享的唯一兼容性校验逻辑。
 */
struct DinoIndexContract
{
    std::string         schema_version{"1.0"};
    std::string         feature_version{"v4_hadamard_sparse_fine"};
    std::string         model_name{"dinov3_vits16"};
    std::string         weights_id{"default"};
    int                 encoder_edge{512};
    int                 patch_size{16};
    int                 token_dimension{384};
    std::vector<int>    gallery_tile_edges{512, 1024, 2048};
    double              view_overlap{0.25};
    std::vector<double> region_window_ratios{1.0, 0.5};
    double              region_window_stride_ratio{0.5};
    double              region_min_valid_fraction{0.5};
    int                 coarse_dimension{96};
    int                 local_representatives{64};
    bool                merge_enabled{true};
    double              merge_epsilon{0.10};
    int                 max_leaf_side_patches{4};
    bool                quantize_int8{true};
    std::string         preprocess_description{};

    bool operator==(const DinoIndexContract &) const = default;

    static DinoIndexContract fromConfig(const DinoRegionSearchConfig &config);
    YAML::Node toYamlNode() const;
    static DinoIndexContract fromYamlNode(const YAML::Node &node);

    /**
     * @brief 校验与传入请求配置的兼容性。
     * @return 兼容时返回空字符串，不兼容时返回描述具体字段差异的错误说明。
     */
    std::string checkCompatibility(const DinoRegionSearchConfig &config) const;
};

/** @brief 检查索引根目录是否需要重新构建。 */
bool dinoIndexNeedsRebuild(const std::filesystem::path &index_root, const DinoRegionSearchConfig &config);

/** @brief 直接写入索引根目录，最后写出读取所需的 index.yaml。 */
class DinoIndexWriter
{
public:
    DinoIndexWriter(const std::filesystem::path &index_root, const DinoIndexContract &contract);

    ~DinoIndexWriter();

    DinoIndexWriter(const DinoIndexWriter &)            = delete;
    DinoIndexWriter &operator=(const DinoIndexWriter &) = delete;

    /** @brief 登记一张图库图像。 */
    void addImage(const DinoImageIdentity &record);

    /** @brief 开始登记一个视图，返回索引内的视图下标。 */
    int addView(int image_index, const DinoViewPlan &plan);

    /** @brief 追加一条区域描述。 */
    void addRegion(int view_id, const DinoRegionDescriptor &meta, const std::vector<float> &vector);

    /** @brief 追加一条局部描述。 */
    void addLocal(int view_id, const DinoLocalLeaf &meta, const std::vector<float> &vector);

    /** @brief 返回已完成视图数。 */
    size_t viewCount() const noexcept
    {
        return views_.size();
    }

    size_t regionCount() const noexcept
    {
        return region_count_;
    }

    size_t localCount() const noexcept
    {
        return local_count_;
    }

    /** @brief 已登记的图像身份记录。 */
    const std::vector<DinoImageIdentity> &imageIdentities() const noexcept
    {
        return images_;
    }

    size_t imageViewCount(size_t index) const;
    size_t imageRegionCount(size_t index) const;
    size_t imageLocalCount(size_t index) const;
    uint64_t imagePatchCount(size_t index) const;

    uint64_t regionBytes() const noexcept
    {
        return region_bytes_;
    }

    uint64_t localBytes() const noexcept
    {
        return local_bytes_;
    }

    float maxMergeRadius() const noexcept
    {
        return max_merge_radius_;
    }

    double meanMergeRadius() const noexcept
    {
        return radius_samples_ > 0U ? mean_merge_radius_ / static_cast<double>(radius_samples_) : 0.0;
    }

    float maxQuantizationDelta() const noexcept
    {
        return max_quantization_delta_;
    }

    double meanQuantizationDelta() const noexcept
    {
        return quantization_samples_ > 0U ? quantization_delta_sum_ / static_cast<double>(quantization_samples_) : 0.0;
    }

    uint64_t originalPatchCount() const noexcept
    {
        return original_patch_count_;
    }

    const DinoIndexContract &contract() const noexcept
    {
        return contract_;
    }

    /** @brief 完成数组与 index.yaml，返回描述统计。 */
    DinoBuildReport finish();

    /** @brief 停止写入并移除未完成索引的读取入口。 */
    void abort();

    const std::filesystem::path &indexPath() const noexcept
    {
        return index_root_;
    }

private:
    void closeWritersIfOpen();

    std::filesystem::path        index_root_{};
    DinoIndexContract            contract_{};
    size_t                       descriptor_dim_{0};
    bool                         quantize_{true};

    std::unique_ptr<DinoNpyWriter> region_meta_writer_{};
    std::unique_ptr<DinoNpyWriter> local_meta_writer_{};
    std::unique_ptr<std::ofstream> region_vector_writer_{};
    std::unique_ptr<std::ofstream> local_vector_writer_{};
    std::unique_ptr<std::ofstream> region_scale_writer_{};
    std::unique_ptr<std::ofstream> local_scale_writer_{};

    std::vector<DinoIndexView>   views_{};
    std::vector<int64_t>         region_offsets_{0};
    std::vector<int64_t>         local_offsets_{0};
    std::vector<DinoImageIdentity> images_{};
    std::vector<size_t>          image_view_counts_{};
    std::vector<size_t>          image_region_counts_{};
    std::vector<size_t>          image_local_counts_{};
    std::vector<uint64_t>        image_patch_counts_{};

    size_t   region_count_{0};
    size_t   local_count_{0};
    uint64_t original_patch_count_{0};
    uint64_t region_bytes_{0};
    uint64_t local_bytes_{0};
    float    max_merge_radius_{0.0F};
    double   mean_merge_radius_{0.0};
    uint64_t radius_samples_{0};
    float    max_quantization_delta_{0.0F};
    double   quantization_delta_sum_{0.0};
    uint64_t quantization_samples_{0};
    bool     finished_{false};
};

/** @brief 读取索引根目录；描述数组按请求区间流式读取。 */
class DinoIndexReader
{
public:
    explicit DinoIndexReader(const std::filesystem::path &index_root);

    const DinoIndexContract &contract() const noexcept
    {
        return contract_;
    }

    /**
     * @brief 校验索引契约与当前配置的兼容性。
     * @throws irt::Exception 当存在硬参数或版本不一致时抛出 NOT_READY。
     */
    void validateContract(const DinoRegionSearchConfig &config) const;

    const std::vector<DinoImageIdentity> &images() const noexcept
    {
        return images_;
    }

    const std::vector<DinoIndexView> &views() const noexcept
    {
        return views_;
    }

    size_t descriptorDim() const noexcept
    {
        return descriptor_dim_;
    }

    bool quantized() const noexcept
    {
        return quantized_;
    }

    size_t regionCount() const noexcept
    {
        return region_count_;
    }

    size_t localCount() const noexcept
    {
        return local_count_;
    }

    uint64_t indexBytes() const noexcept
    {
        return index_bytes_;
    }

    /** @brief 视图内区域描述的连续区间。 */
    DinoDescriptorRange regionRange(size_t view_id) const;

    /** @brief 视图内局部描述的连续区间。 */
    DinoDescriptorRange localRange(size_t view_id) const;

    /** @brief 读取并解码指定区间的区域描述向量（维度为 ``descriptorDim()``）。 */
    void readRegionVectors(size_t begin, size_t count, std::vector<float> &out) const;

    /** @brief 读取并解码指定区间的局部描述向量。 */
    void readLocalVectors(size_t begin, size_t count, std::vector<float> &out) const;

    /** @brief 读取一段紧凑区域描述；返回的指针在下一次读取前有效。 */
    DinoCompactBlock readRegionCompact(size_t begin, size_t count, DinoCompactScratch &scratch) const;

    /** @brief 读取一段紧凑局部描述；返回的指针在下一次读取前有效。 */
    DinoCompactBlock readLocalCompact(size_t begin, size_t count, DinoCompactScratch &scratch) const;

    /** @brief 读取区域描述元数据。 */
    std::vector<DinoRegionDescriptor> readRegionMeta(size_t begin, size_t count) const;

    /** @brief 读取局部描述元数据。 */
    std::vector<DinoLocalLeaf> readLocalMeta(size_t begin, size_t count) const;

    /** @brief 读取单条区域描述元数据。 */
    DinoRegionDescriptor regionMetaAt(size_t index) const;

    /** @brief 读取单条局部描述元数据。 */
    DinoLocalLeaf localMetaAt(size_t index) const;

    /** @brief 定位一条区域描述所属的视图；索引越界时抛出。 */
    int viewOfRegionDescriptor(size_t index) const;

    /** @brief 定位一条局部描述所属的视图；索引越界时抛出。 */
    int viewOfLocalDescriptor(size_t index) const;

    /** @brief 定位图像下标；不存在时返回 -1。 */
    int imageIndexById(int64_t image_id) const;

    /** @brief 返回索引根目录。 */
    const std::filesystem::path &indexPath() const noexcept
    {
        return index_root_;
    }

private:
    void loadViews();

    /** @brief 紧凑读取的共享实现：向量与标量归约都经由此处，避免解码逻辑分叉。 */
    DinoCompactBlock readCompact(std::ifstream &vectors, std::ifstream &scales, size_t total_count, size_t begin,
                                 size_t count, const char *label, DinoCompactScratch &scratch) const;

    std::filesystem::path        index_root_{};
    DinoIndexContract            contract_{};
    std::vector<DinoImageIdentity> images_{};
    std::vector<DinoIndexView>   views_{};
    std::vector<int64_t>         offsets_{};
    size_t                       descriptor_dim_{0};
    size_t                       region_count_{0};
    size_t                       local_count_{0};
    uint64_t                     index_bytes_{0};
    bool                         quantized_{true};

    std::unique_ptr<DinoNpyReader> region_meta_{};
    std::unique_ptr<DinoNpyReader> local_meta_{};
    mutable std::unique_ptr<std::ifstream> region_vectors_{};
    mutable std::unique_ptr<std::ifstream> local_vectors_{};
    mutable std::unique_ptr<std::ifstream> region_scales_{};
    mutable std::unique_ptr<std::ifstream> local_scales_{};
};

} // namespace irt::features::priv
