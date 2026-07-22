#pragma once

/**
 * @file RoiFeatureExtractor.hpp
 * @brief ROI 特征图抽取、ROIAlign 与归一化的共用实现。
 */

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/features/RoiFeature.hpp>
#include <inferrt/ops/RoIAlign.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace irt::features {

namespace {

struct PatchTokenGrid
{
    int height{0};
    int width{0};
};

inline ImageSearchConfig featureSearchConfig(const RoiFeatureConfig &config)
{
    const ImageSearchConfig &base = config;
    return base;
}

inline bool isValidRoi(const RoiFeatureBox &roi) noexcept
{
    return std::isfinite(roi.x1) && std::isfinite(roi.y1) && std::isfinite(roi.x2) && std::isfinite(roi.y2)
        && roi.x2 > roi.x1 && roi.y2 > roi.y1;
}

inline void validateRoi(const RoiFeatureBox &roi)
{
    if (!isValidRoi(roi))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI must be finite and satisfy x2 > x1, y2 > y1");
    }
}

inline bool isPatchTokenFeature(const RoiFeatureConfig &config)
{
    return config.feature_name == "x_norm_patchtokens";
}

inline PatchTokenGrid inferPatchTokenGrid(int64_t token_count, int input_height, int input_width)
{
    if (token_count <= 0 || token_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid patch token count for RoiFeatureExtractor");
    }

    const auto tokens       = static_cast<int>(token_count);
    const auto target_ratio = input_width > 0 && input_height > 0
                                ? static_cast<double>(input_height) / static_cast<double>(input_width)
                                : 1.0;

    PatchTokenGrid best{1, tokens};
    double         best_score = std::numeric_limits<double>::infinity();
    auto           evaluate   = [&](int height, int width)
    {
        const auto ratio = static_cast<double>(height) / static_cast<double>(width);
        const auto score = std::abs(std::log(ratio / target_ratio));
        if (score < best_score)
        {
            best       = PatchTokenGrid{height, width};
            best_score = score;
        }
    };

    for (int factor = 1; factor <= tokens / factor; ++factor)
    {
        if (tokens % factor != 0)
        {
            continue;
        }
        const int other = tokens / factor;
        evaluate(factor, other);
        evaluate(other, factor);
    }
    return best;
}

} // namespace

namespace priv {

/**
 * @brief OpenCV PCA 投影器，负责按通道维训练和批量投影。
 */
class RoiPcaProjector
{
public:
    /**
     * @brief 使用特征图空间位置上的通道向量训练 PCA。
     * @param features 扁平化 ``sample_count x input_dim`` 通道特征。
     * @param sample_count PCA 训练样本数量。
     * @param input_dim 原始特征图通道数。
     * @param output_dim PCA 输出通道数。
     */
    void train(const std::vector<float> &features, size_t sample_count, int input_dim, int output_dim)
    {
        if (sample_count == 0 || input_dim <= 0 || output_dim <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA requires non-empty features");
        }
        if (sample_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA feature count is too large");
        }
        if (output_dim > input_dim || output_dim > static_cast<int>(sample_count))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI PCA dim must not exceed feature channel count or sample count");
        }
        if (features.size() != sample_count * static_cast<size_t>(input_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA training feature size mismatch");
        }

        cv::Mat input(static_cast<int>(sample_count), input_dim, CV_32F, const_cast<float *>(features.data()));
        pca_        = cv::PCA(input, cv::Mat(), cv::PCA::DATA_AS_ROW, output_dim);
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        normalizeLoadedMats();
        validateReady();
    }

    /**
     * @brief 批量投影 ``count x inputDim()`` 通道特征。
     */
    std::vector<float> project(const float *features, size_t count) const
    {
        validateReady();
        if (count == 0)
        {
            return {};
        }
        if (features == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA input feature pointer is null");
        }
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA projection batch is too large");
        }

        cv::Mat input(static_cast<int>(count), input_dim_, CV_32F, const_cast<float *>(features));
        cv::Mat output;
        pca_.project(input, output);
        if (output.rows != static_cast<int>(count) || output.cols != output_dim_ || output.type() != CV_32F)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA projection shape mismatch");
        }

        std::vector<float> projected(count * static_cast<size_t>(output_dim_));
        if (output.isContinuous())
        {
            std::memcpy(projected.data(), output.ptr<float>(), projected.size() * sizeof(float));
        }
        else
        {
            for (size_t row = 0; row < count; ++row)
            {
                std::memcpy(projected.data() + row * static_cast<size_t>(output_dim_),
                            output.ptr<float>(static_cast<int>(row)), static_cast<size_t>(output_dim_) * sizeof(float));
            }
        }
        return projected;
    }

    /** @brief 获取 PCA 输入通道数。 */
    int inputDim() const noexcept
    {
        return input_dim_;
    }

    /** @brief 获取 PCA 输出通道数。 */
    int outputDim() const noexcept
    {
        return output_dim_;
    }

private:
    /**
     * @brief 统一加载矩阵的类型和形状。
     */
    void normalizeLoadedMats()
    {
        if (!pca_.mean.empty())
        {
            pca_.mean = pca_.mean.reshape(1, 1);
            if (pca_.mean.type() != CV_32F)
            {
                pca_.mean.convertTo(pca_.mean, CV_32F);
            }
        }
        if (!pca_.eigenvectors.empty() && pca_.eigenvectors.type() != CV_32F)
        {
            pca_.eigenvectors.convertTo(pca_.eigenvectors, CV_32F);
        }
        if (!pca_.eigenvalues.empty() && pca_.eigenvalues.type() != CV_32F)
        {
            pca_.eigenvalues.convertTo(pca_.eigenvalues, CV_32F);
        }
    }

    /**
     * @brief 校验 PCA 参数是否完整。
     */
    void validateReady() const
    {
        if (input_dim_ <= 0 || output_dim_ <= 0 || pca_.mean.total() != static_cast<size_t>(input_dim_)
            || pca_.eigenvectors.rows != output_dim_ || pca_.eigenvectors.cols != input_dim_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI PCA parameters");
        }
    }

    cv::PCA pca_{};         ///< OpenCV PCA 参数。
    int     input_dim_{0};  ///< PCA 输入通道数。
    int     output_dim_{0}; ///< PCA 输出通道数。
};

/**
 * @brief ROI 特征抽取器：模型特征图 + ROIAlign + 展平归一化。
 */
class RoiFeatureExtractor
{
    /**
     * @brief ROI 特征张量布局。
     */
    enum class RoiFeatureLayout
    {
        SpatialNchw, ///< 标准 ``B x C x H x W`` 空间特征图。
        PatchTokens, ///< DINO ``B x tokens x dim`` patch token。
    };

    /**
     * @brief ROIAlign 可直接消费的 NCHW 特征图视图。
     */
    struct PreparedFeatureMap
    {
        std::vector<float> owned_data;    ///< patch token 转置或 PCA 投影后持有的 NCHW 数据。
        const float       *data{nullptr}; ///< 指向 NCHW 特征图数据。
        nvinfer1::Dims     dims{};        ///< NCHW 形状。

        PreparedFeatureMap() = default;

        PreparedFeatureMap(const PreparedFeatureMap &other)
            : owned_data(other.owned_data)
            , data(owned_data.empty() ? other.data : owned_data.data())
            , dims(other.dims)
        {
        }

        PreparedFeatureMap &operator=(const PreparedFeatureMap &other)
        {
            if (this != &other)
            {
                owned_data = other.owned_data;
                data       = owned_data.empty() ? other.data : owned_data.data();
                dims       = other.dims;
            }
            return *this;
        }

        PreparedFeatureMap(PreparedFeatureMap &&other) noexcept
            : owned_data(std::move(other.owned_data))
            , data(owned_data.empty() ? other.data : owned_data.data())
            , dims(other.dims)
        {
            other.data = nullptr;
        }

        PreparedFeatureMap &operator=(PreparedFeatureMap &&other) noexcept
        {
            if (this != &other)
            {
                owned_data = std::move(other.owned_data);
                data       = owned_data.empty() ? other.data : owned_data.data();
                dims       = other.dims;
                other.data = nullptr;
            }
            return *this;
        }
    };

public:
    RoiFeatureExtractor(const RoiFeatureConfig &config, const std::filesystem::path &weights_file)
        : config_(config)
        , image_extractor_(config.model_name, config.feature_name, weights_file, featureSearchConfig(config))
    {
        const auto dims = image_extractor_.featureTensorShape();
        if (dims.nbDims == 4 && dims.d[0] > 0 && dims.d[1] > 0 && dims.d[2] > 0 && dims.d[3] > 0)
        {
            layout_           = RoiFeatureLayout::SpatialNchw;
            feature_channels_ = static_cast<int>(dims.d[1]);
        }
        else if (dims.nbDims == 3 && dims.d[0] > 0 && dims.d[1] > 0 && dims.d[2] > 0 && isPatchTokenFeature(config_))
        {
            layout_     = RoiFeatureLayout::PatchTokens;
            patch_grid_ = inferPatchTokenGrid(dims.d[1], image_extractor_.inputHeight(), image_extractor_.inputWidth());
            feature_channels_ = static_cast<int>(dims.d[2]);
        }
        else
        {
            throw irt::Exception(
                irt::Status::ERROR_INVALID_ARGUMENT,
                "RoiFeature tensor must be NCHW map or DINO x_norm_patchtokens with shape BxTokensxDim");
        }

        if (config_.use_pca && (config_.pca_dim <= 0 || config_.pca_dim > feature_channels_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI PCA dim must be positive and not exceed feature channel count");
        }

        feature_dim_ = roiFeatureDim(feature_channels_);
    }

    /**
     * @brief 获取 ROI 特征向量维度。
     */
    int featureDim() const noexcept
    {
        return config_.use_pca ? roiFeatureDim(config_.pca_dim) : feature_dim_;
    }

    /**
     * @brief 提取单个 ROI 的归一化特征。
     */
    std::vector<float> extract(const RoiFeatureItem &item)
    {
        return extractItems(std::vector<RoiFeatureItem>{item});
    }

    /**
     * @brief 获取模型支持的最大特征提取 batch。
     */
    size_t maxBatchSize() const noexcept
    {
        return image_extractor_.maxBatchSize();
    }

    /**
     * @brief 批量提取 ROI 特征。
     */
    std::vector<float> extractBatch(const std::vector<RoiFeatureItem> &items, size_t begin, size_t count)
    {
        if (begin > items.size() || count > items.size() - begin)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch range is invalid");
        }

        std::vector<RoiFeatureItem> selected;
        selected.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            selected.push_back(items[begin + i]);
        }
        return extractItems(selected);
    }

    /**
     * @brief 对一组 ROI 去重图像路径后执行一次模型前向和批量 ROIAlign。
     *
     * 同一张图的多个标注只生成一份空间特征图；ROIAlign 可以在同一份特征图上消费任意数量
     * 的 ROI，从而避免按 ROI 重复执行模型前向。
     */
    std::vector<float> extractItems(const std::vector<RoiFeatureItem> &items)
    {
        if (items.empty())
        {
            return {};
        }

        std::vector<std::filesystem::path> paths;
        std::vector<RoiFeatureBox> rois;
        std::vector<size_t>       image_indices;
        std::map<std::filesystem::path, size_t> path_to_index;
        paths.reserve(items.size());
        rois.reserve(items.size());
        image_indices.reserve(items.size());
        for (const auto &item : items)
        {
            validateRoi(item.roi);
            const auto [it, inserted] = path_to_index.emplace(item.image_path, paths.size());
            if (inserted)
            {
                paths.push_back(item.image_path);
            }
            rois.push_back(item.roi);
            image_indices.push_back(it->second);
        }
        if (paths.size() > maxBatchSize())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI feature batch contains more unique images than model max batch size");
        }

        const auto tensor      = image_extractor_.extractFeatureTensorBatch(paths, 0, paths.size());
        const auto feature_map = prepareFeatureMap(tensor);
        std::vector<float> features;
        if (!config_.use_pca)
        {
            features = roiAlignAndFlatten(feature_map, tensor.original_sizes, rois, image_indices);
        }
        else
        {
            features.resize(items.size() * static_cast<size_t>(featureDim()));
            std::vector<std::vector<size_t>> roi_indices_by_image(paths.size());
            for (size_t roi_index = 0; roi_index < image_indices.size(); ++roi_index)
            {
                roi_indices_by_image[image_indices[roi_index]].push_back(roi_index);
            }
            for (size_t image_index = 0; image_index < paths.size(); ++image_index)
            {
                auto single_map = singleFeatureMap(feature_map, image_index);
                auto reduced    = applyLocalPcaToFeatureMap(single_map);
                std::vector<RoiFeatureBox> image_rois;
                std::vector<size_t>       local_indices;
                image_rois.reserve(roi_indices_by_image[image_index].size());
                local_indices.resize(roi_indices_by_image[image_index].size(), 0);
                for (const auto roi_index : roi_indices_by_image[image_index])
                {
                    image_rois.push_back(rois[roi_index]);
                }
                const auto image_features
                    = roiAlignAndFlatten(reduced, {tensor.original_sizes[image_index]}, image_rois, local_indices);
                const auto dim = static_cast<size_t>(featureDim());
                for (size_t local_index = 0; local_index < roi_indices_by_image[image_index].size(); ++local_index)
                {
                    const auto output_index = roi_indices_by_image[image_index][local_index];
                    std::copy_n(image_features.data() + local_index * dim, dim, features.data() + output_index * dim);
                }
            }
        }

        const auto dim = static_cast<size_t>(featureDim());
        for (size_t i = 0; i < rois.size(); ++i)
        {
            normalizeFeature(features.data() + i * dim, dim, config_.norm);
        }
        return features;
    }

    /**
     * @brief 按图像分组提取完整 ROI 特征矩阵。
     *
     * 同一图像的所有 ROI 共享一次模型特征图；不同图像按模型 batch 打包。输出始终保持
     * 输入 ROI 顺序，适合直接送入 HDBSCAN。
     */
    std::vector<float> extractAll(const std::vector<RoiFeatureItem> &items,
                                  const std::function<void(size_t, size_t, size_t, size_t)> &progress = {})
    {
        if (items.empty())
        {
            return {};
        }

        const auto dim = static_cast<size_t>(featureDim());
        std::vector<float> all_features(items.size() * dim);
        struct ImageGroup
        {
            std::filesystem::path image_path;
            std::vector<size_t> roi_indices;
        };

        std::vector<ImageGroup> groups;
        std::map<std::filesystem::path, size_t> group_by_path;
        groups.reserve(items.size());
        for (size_t index = 0; index < items.size(); ++index)
        {
            const auto [it, inserted] = group_by_path.emplace(items[index].image_path, groups.size());
            if (inserted)
            {
                groups.push_back(ImageGroup{items[index].image_path, {}});
            }
            groups[it->second].roi_indices.push_back(index);
        }

        std::vector<size_t> packed_roi_indices;
        size_t              packed_image_count{0};
        size_t              processed_count{0};
        size_t              batch_index{0};
        auto process_packed = [&]
        {
            if (packed_roi_indices.empty())
            {
                return;
            }

            std::vector<RoiFeatureItem> batch_items;
            batch_items.reserve(packed_roi_indices.size());
            for (const auto roi_index : packed_roi_indices)
            {
                batch_items.push_back(items[roi_index]);
            }
            const auto features = extractItems(batch_items);
            if (features.size() != batch_items.size() * dim)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch size mismatch");
            }
            for (size_t local_index = 0; local_index < packed_roi_indices.size(); ++local_index)
            {
                const auto output_index = packed_roi_indices[local_index];
                std::copy_n(features.data() + local_index * dim, dim, all_features.data() + output_index * dim);
            }

            processed_count += packed_roi_indices.size();
            if (progress)
            {
                progress(batch_index++, packed_roi_indices.front(), packed_roi_indices.size(), processed_count);
            }
            packed_roi_indices.clear();
            packed_image_count = 0;
        };

        for (const auto &group : groups)
        {
            if (!packed_roi_indices.empty() && packed_image_count >= maxBatchSize())
            {
                process_packed();
            }
            packed_roi_indices.insert(packed_roi_indices.end(), group.roi_indices.begin(), group.roi_indices.end());
            ++packed_image_count;
        }
        process_packed();
        return all_features;
    }

private:
    /**
     * @brief 根据通道数计算 ROIAlign 展平后的维度。
     */
    int roiFeatureDim(int channels) const noexcept
    {
        return channels * config_.pooled_height * config_.pooled_width;
    }

    /**
     * @brief 校验 NCHW 特征图并返回元素数量。
     */
    static size_t validateNchwTensor(const FeatureTensorBatch &tensor, const nvinfer1::Dims &dims)
    {
        if (dims.nbDims != 4 || dims.d[0] <= 0 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature requires NCHW feature tensor");
        }
        const auto expected = static_cast<size_t>(dims.d[0]) * static_cast<size_t>(dims.d[1])
                            * static_cast<size_t>(dims.d[2]) * static_cast<size_t>(dims.d[3]);
        if (tensor.data.size() != expected)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature tensor size mismatch");
        }
        return expected;
    }

    /**
     * @brief 校验已准备的 NCHW 特征图并返回元素数量。
     */
    static size_t validatePreparedFeatureMap(const PreparedFeatureMap &feature_map)
    {
        const auto &dims = feature_map.dims;
        if (dims.nbDims != 4 || dims.d[0] <= 0 || dims.d[1] <= 0 || dims.d[2] <= 0 || dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature requires NCHW feature map");
        }
        if (feature_map.data == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature feature map data is null");
        }
        return static_cast<size_t>(dims.d[0]) * static_cast<size_t>(dims.d[1]) * static_cast<size_t>(dims.d[2])
             * static_cast<size_t>(dims.d[3]);
    }

    /**
     * @brief 将 DINO patch token 重排为 ``B x C x patch_h x patch_w``。
     */
    PreparedFeatureMap preparePatchTokenMap(const FeatureTensorBatch &tensor) const
    {
        if (tensor.dims.nbDims != 3 || tensor.dims.d[0] <= 0 || tensor.dims.d[1] <= 0 || tensor.dims.d[2] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "RoiFeature patch token tensor must have shape BxTokensxDim");
        }

        const int batch    = static_cast<int>(tensor.dims.d[0]);
        const int tokens   = static_cast<int>(tensor.dims.d[1]);
        const int channels = static_cast<int>(tensor.dims.d[2]);
        if (tokens != patch_grid_.height * patch_grid_.width || channels != feature_channels_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature patch token grid mismatch");
        }

        const auto expected = static_cast<size_t>(batch) * static_cast<size_t>(tokens) * static_cast<size_t>(channels);
        if (tensor.data.size() != expected)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiFeature patch token tensor size mismatch");
        }

        PreparedFeatureMap prepared;
        prepared.dims.nbDims = 4;
        prepared.dims.d[0]   = batch;
        prepared.dims.d[1]   = channels;
        prepared.dims.d[2]   = patch_grid_.height;
        prepared.dims.d[3]   = patch_grid_.width;
        prepared.owned_data.resize(expected);

        for (int b = 0; b < batch; ++b)
        {
            for (int y = 0; y < patch_grid_.height; ++y)
            {
                for (int x = 0; x < patch_grid_.width; ++x)
                {
                    const int token = y * patch_grid_.width + x;
                    for (int c = 0; c < channels; ++c)
                    {
                        const auto src
                            = (static_cast<size_t>(b) * static_cast<size_t>(tokens) + static_cast<size_t>(token))
                                * static_cast<size_t>(channels)
                            + static_cast<size_t>(c);
                        const auto dst
                            = ((static_cast<size_t>(b) * static_cast<size_t>(channels) + static_cast<size_t>(c))
                                   * static_cast<size_t>(patch_grid_.height)
                               + static_cast<size_t>(y))
                                * static_cast<size_t>(patch_grid_.width)
                            + static_cast<size_t>(x);
                        prepared.owned_data[dst] = tensor.data[src];
                    }
                }
            }
        }

        prepared.data = prepared.owned_data.data();
        return prepared;
    }

    /**
     * @brief 准备 ROIAlign 输入特征图。
     */
    PreparedFeatureMap prepareFeatureMap(const FeatureTensorBatch &tensor) const
    {
        if (layout_ == RoiFeatureLayout::PatchTokens)
        {
            return preparePatchTokenMap(tensor);
        }

        validateNchwTensor(tensor, tensor.dims);
        PreparedFeatureMap prepared;
        prepared.data = tensor.data.data();
        prepared.dims = tensor.dims;
        return prepared;
    }

    /**
     * @brief 将 NCHW 特征图转换为 ``N*H*W x C`` 的通道向量矩阵。
     */
    std::vector<float> featureMapRows(const PreparedFeatureMap &feature_map) const
    {
        validatePreparedFeatureMap(feature_map);
        const int batch    = static_cast<int>(feature_map.dims.d[0]);
        const int channels = static_cast<int>(feature_map.dims.d[1]);
        const int height   = static_cast<int>(feature_map.dims.d[2]);
        const int width    = static_cast<int>(feature_map.dims.d[3]);

        const auto row_count = static_cast<size_t>(batch) * static_cast<size_t>(height) * static_cast<size_t>(width);
        std::vector<float> rows(row_count * static_cast<size_t>(channels));
        for (int b = 0; b < batch; ++b)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const auto row = (static_cast<size_t>(b) * static_cast<size_t>(height) + static_cast<size_t>(y))
                                       * static_cast<size_t>(width)
                                   + static_cast<size_t>(x);
                    for (int c = 0; c < channels; ++c)
                    {
                        const auto src
                            = ((static_cast<size_t>(b) * static_cast<size_t>(channels) + static_cast<size_t>(c))
                                   * static_cast<size_t>(height)
                               + static_cast<size_t>(y))
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(x);
                        rows[row * static_cast<size_t>(channels) + static_cast<size_t>(c)] = feature_map.data[src];
                    }
                }
            }
        }
        return rows;
    }

    /**
     * @brief 返回一个批次样本的 NCHW 视图。
     */
    PreparedFeatureMap singleFeatureMap(const PreparedFeatureMap &feature_map, size_t index) const
    {
        validatePreparedFeatureMap(feature_map);
        if (index >= static_cast<size_t>(feature_map.dims.d[0]))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature map batch index is invalid");
        }

        const auto sample_size = static_cast<size_t>(feature_map.dims.d[1])
                                * static_cast<size_t>(feature_map.dims.d[2])
                                * static_cast<size_t>(feature_map.dims.d[3]);
        PreparedFeatureMap single;
        single.data      = feature_map.data + index * sample_size;
        single.dims      = feature_map.dims;
        single.dims.d[0] = 1;
        return single;
    }

    /**
     * @brief 对当前特征图通道维训练本地 PCA，并还原为 ROIAlign 可消费的 NCHW 特征图。
     */
    PreparedFeatureMap applyLocalPcaToFeatureMap(const PreparedFeatureMap &feature_map) const
    {
        validatePreparedFeatureMap(feature_map);
        const int  batch     = static_cast<int>(feature_map.dims.d[0]);
        const int  channels  = static_cast<int>(feature_map.dims.d[1]);
        const int  height    = static_cast<int>(feature_map.dims.d[2]);
        const int  width     = static_cast<int>(feature_map.dims.d[3]);
        const auto row_count = static_cast<size_t>(batch) * static_cast<size_t>(height) * static_cast<size_t>(width);
        if (config_.pca_dim <= 0 || config_.pca_dim > channels || static_cast<size_t>(config_.pca_dim) > row_count)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ROI PCA dim must not exceed feature channel count or local feature-map sample count");
        }

        auto            rows = featureMapRows(feature_map);
        RoiPcaProjector projector;
        projector.train(rows, row_count, channels, config_.pca_dim);
        const auto projected  = projector.project(rows.data(), row_count);
        const int  out_ch     = config_.pca_dim;
        const auto out_stride = static_cast<size_t>(out_ch);

        PreparedFeatureMap reduced;
        reduced.dims      = feature_map.dims;
        reduced.dims.d[1] = out_ch;
        reduced.owned_data.resize(static_cast<size_t>(batch) * static_cast<size_t>(out_ch) * static_cast<size_t>(height)
                                  * static_cast<size_t>(width));
        for (int b = 0; b < batch; ++b)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const auto row = (static_cast<size_t>(b) * static_cast<size_t>(height) + static_cast<size_t>(y))
                                       * static_cast<size_t>(width)
                                   + static_cast<size_t>(x);
                    for (int c = 0; c < out_ch; ++c)
                    {
                        const auto dst
                            = ((static_cast<size_t>(b) * static_cast<size_t>(out_ch) + static_cast<size_t>(c))
                                   * static_cast<size_t>(height)
                               + static_cast<size_t>(y))
                                * static_cast<size_t>(width)
                            + static_cast<size_t>(x);
                        reduced.owned_data[dst] = projected[row * out_stride + static_cast<size_t>(c)];
                    }
                }
            }
        }
        reduced.data = reduced.owned_data.data();
        return reduced;
    }

    /**
     * @brief 将原图 ROI 映射到特征图坐标并批量执行 ROIAlign。
     */
    std::vector<float> roiAlignAndFlatten(const PreparedFeatureMap &feature_map,
                                           const std::vector<ImageSize> &image_sizes,
                                           const std::vector<RoiFeatureBox> &rois,
                                           const std::vector<size_t> &image_indices = {}) const
    {
        validatePreparedFeatureMap(feature_map);
        if (image_sizes.size() != static_cast<size_t>(feature_map.dims.d[0])
            || (!image_indices.empty() && image_indices.size() != rois.size()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI batch shape mismatch");
        }
        if (rois.empty())
        {
            return {};
        }

        const auto feature_height = static_cast<float>(feature_map.dims.d[2]);
        const auto feature_width  = static_cast<float>(feature_map.dims.d[3]);
        std::vector<float> mapped_rois(rois.size() * 5);
        for (size_t i = 0; i < rois.size(); ++i)
        {
            validateRoi(rois[i]);
            const auto offset = i * 5;
            const auto image_index = image_indices.empty() ? i : image_indices[i];
            if (image_index >= image_sizes.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI image index is out of range");
            }
            const auto &mapped_image_size = image_sizes[image_index];
            if (mapped_image_size.width <= 0 || mapped_image_size.height <= 0)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid source image size for ROI mapping");
            }
            mapped_rois[offset]     = static_cast<float>(image_index);
            mapped_rois[offset + 1] = rois[i].x1 * feature_width / static_cast<float>(mapped_image_size.width);
            mapped_rois[offset + 2] = rois[i].y1 * feature_height / static_cast<float>(mapped_image_size.height);
            mapped_rois[offset + 3] = rois[i].x2 * feature_width / static_cast<float>(mapped_image_size.width);
            mapped_rois[offset + 4] = rois[i].y2 * feature_height / static_cast<float>(mapped_image_size.height);
        }

        const int64_t input_shape[4]{
            feature_map.dims.d[0],
            feature_map.dims.d[1],
            feature_map.dims.d[2],
            feature_map.dims.d[3],
        };
        const irt::ops::RoIAlign roi_align({config_.pooled_height, config_.pooled_width}, 1.0f, config_.sampling_ratio,
                                           config_.aligned);
        auto feature = roi_align.forward(feature_map.data, input_shape, mapped_rois.data(),
                                         static_cast<int64_t>(rois.size()));
        const auto expected_size = rois.size() * static_cast<size_t>(roiFeatureDim(static_cast<int>(feature_map.dims.d[1])));
        if (feature.size() != expected_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ROI feature size");
        }
        return feature;
    }

    RoiFeatureConfig       config_{};                              ///< ROI 检索配置。
    ImageFeatureExtractor image_extractor_;                       ///< 共用图像特征图抽取器。
    RoiFeatureLayout      layout_{RoiFeatureLayout::SpatialNchw}; ///< 特征张量布局。
    PatchTokenGrid        patch_grid_{};                          ///< patch token 推断出的空间网格。
    int                   feature_channels_{0};                   ///< ROIAlign 输入通道数。
    int                   feature_dim_{0};                        ///< ROI 特征向量维度。
};

} // namespace priv


} // namespace irt::features
