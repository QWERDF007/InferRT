/**
 * @file RoiSearchImpl.cpp
 * @brief ROI 搜索匹配 PIMPL 实现。
 */

#include "RoiSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/ops/RoIAlign.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#pragma warning(pop)

#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {
namespace {

/**
 * @brief 将 ROI 配置切片为图像检索共用配置。
 */
ImageSearchConfig featureSearchConfig(const RoiSearchConfig &config)
{
    const ImageSearchConfig &base = config;
    return base;
}

/**
 * @brief 判断 ROI 坐标是否合法。
 */
bool isValidRoi(const RoiSearchBox &roi) noexcept
{
    return std::isfinite(roi.x1) && std::isfinite(roi.y1) && std::isfinite(roi.x2) && std::isfinite(roi.y2)
        && roi.x2 > roi.x1 && roi.y2 > roi.y1;
}

/**
 * @brief 校验 ROI 坐标，失败时抛出异常。
 */
void validateRoi(const RoiSearchBox &roi)
{
    if (!isValidRoi(roi))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI must be finite and satisfy x2 > x1, y2 > y1");
    }
}

/**
 * @brief 校验并规范化单个 ROI 条目。
 */
RoiSearchItem normalizeItem(const RoiSearchItem &item)
{
    if (item.image_path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI image path must not be empty");
    }

    std::error_code ec;
    const bool      exists          = fs::exists(item.image_path, ec);
    const bool      is_regular_file = exists && fs::is_regular_file(item.image_path, ec);
    if (!is_regular_file)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI image path does not exist: %s",
                             item.image_path.string().c_str());
    }
    if (!ImageSearch::isImageFile(item.image_path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ROI image file: %s",
                             item.image_path.string().c_str());
    }
    validateRoi(item.roi);
    return RoiSearchItem{fs::absolute(item.image_path), item.roi};
}

/**
 * @brief 校验并规范化 ROI 特征库条目。
 */
std::vector<RoiSearchItem> normalizeItems(const std::vector<RoiSearchItem> &items)
{
    if (items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI item list must not be empty");
    }

    std::vector<RoiSearchItem> normalized;
    normalized.reserve(items.size());
    for (const auto &item : items)
    {
        normalized.push_back(normalizeItem(item));
    }
    return normalized;
}

/**
 * @brief 由索引路径推导 ROI 映射文件路径。
 */
fs::path roiMappingPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".rois.txt";
}

/**
 * @brief 获取写入元数据的有效 PCA 维度。
 */
int effectivePcaDim(const RoiSearchConfig &config) noexcept
{
    return config.use_pca ? config.pca_dim : 0;
}

const char *roiPcaModeName(const RoiSearchConfig &config) noexcept
{
    return config.use_pca ? "local_image" : "none";
}

/**
 * @brief DINO patch token 网格尺寸。
 */
struct PatchTokenGrid
{
    int height{0}; ///< patch 网格高度。
    int width{0};  ///< patch 网格宽度。
};

/**
 * @brief 判断当前特征是否为可恢复空间网格的 patch token。
 */
bool isPatchTokenFeature(const RoiSearchConfig &config)
{
    return config.feature_name == "x_norm_patchtokens";
}

/**
 * @brief 根据 token 数量和模型输入宽高推断 patch token 网格。
 *
 * DINOv2/DINOv3 的 ``x_norm_patchtokens`` 形状为 ``B x tokens x dim``。这里不依赖模型私有的 patch
 * size，而是在所有可整除因子中选择最接近模型输入宽高比的 ``H x W``。
 */
PatchTokenGrid inferPatchTokenGrid(int64_t token_count, int input_height, int input_width)
{
    if (token_count <= 0 || token_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid patch token count for RoiSearch");
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

/**
 * @brief 保存 Faiss ID 到 ROI 条目的映射。
 */
void saveRoiMapping(const fs::path &mapping_path, const std::vector<RoiSearchItem> &items)
{
    std::ofstream output(mapping_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI mapping file: %s",
                             mapping_path.string().c_str());
    }

    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto &item : items)
    {
        output << item.image_path.generic_string() << '\t' << item.roi.x1 << '\t' << item.roi.y1 << '\t' << item.roi.x2
               << '\t' << item.roi.y2 << '\n';
    }
}

/**
 * @brief 从映射文件加载 ROI 条目列表。
 */
std::vector<RoiSearchItem> loadRoiMapping(const fs::path &mapping_path)
{
    std::ifstream input(mapping_path);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI mapping file: %s",
                             mapping_path.string().c_str());
    }

    std::vector<RoiSearchItem> items;
    std::string                line;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> fields;
        size_t                   begin = 0;
        while (begin <= line.size())
        {
            const auto end = line.find('\t', begin);
            fields.push_back(line.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos)
            {
                break;
            }
            begin = end + 1;
        }
        if (fields.size() != 5)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid ROI mapping line: %s", line.c_str());
        }

        RoiSearchItem item;
        item.image_path = fields[0];
        item.roi.x1     = std::stof(fields[1]);
        item.roi.y1     = std::stof(fields[2]);
        item.roi.x2     = std::stof(fields[3]);
        item.roi.y2     = std::stof(fields[4]);
        validateRoi(item.roi);
        items.push_back(std::move(item));
    }
    return items;
}

/**
 * @brief 判断两个 ROI 框是否近似相等。
 */
bool roiNearlyEquals(const RoiSearchBox &lhs, const RoiSearchBox &rhs) noexcept
{
    constexpr float eps = 1.0e-5f;
    return std::abs(lhs.x1 - rhs.x1) <= eps && std::abs(lhs.y1 - rhs.y1) <= eps && std::abs(lhs.x2 - rhs.x2) <= eps
        && std::abs(lhs.y2 - rhs.y2) <= eps;
}

/**
 * @brief 判断 ROI 条目列表是否与映射文件内容一致。
 */
bool itemsMatch(const std::vector<RoiSearchItem> &lhs, const std::vector<RoiSearchItem> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i].image_path != rhs[i].image_path || !roiNearlyEquals(lhs[i].roi, rhs[i].roi))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief 序列化布尔值。
 */
const char *boolName(bool value) noexcept
{
    return value ? "true" : "false";
}

/**
 * @brief 保存 ROI 搜索索引元数据。
 */
void saveRoiMetadata(const fs::path &metadata_path, const RoiSearchConfig &config)
{
    std::ofstream output(metadata_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open ROI metadata file: %s",
                             metadata_path.string().c_str());
    }

    output << "kind=roi_search\n";
    output << "model=" << config.model_name << "\n";
    output << "feature=" << config.feature_name << "\n";
    output << "model_backend=" << irt::model::modelBackendName(config.model_backend) << "\n";
    output << "model_device=" << irt::model::modelDeviceName(config.model_device) << "\n";
    output << "preprocess_backend=" << priv::preprocessBackendName(config.preprocess_backend) << "\n";
    output << "norm=" << priv::featureNormName(config.norm) << "\n";
    output << "faiss_backend=" << priv::faissBackendName(config.faiss_backend) << "\n";
    output << "index_storage=" << priv::indexStorageName(config.index_storage) << "\n";
    output << "model_batch_size=" << config.model_batch_size << "\n";
    output << "index_kind=" << priv::indexKindName(config) << "\n";
    output << "roi_pooled_height=" << config.pooled_height << "\n";
    output << "roi_pooled_width=" << config.pooled_width << "\n";
    output << "roi_sampling_ratio=" << config.sampling_ratio << "\n";
    output << "roi_aligned=" << boolName(config.aligned) << "\n";
    output << "roi_use_pca=" << boolName(config.use_pca) << "\n";
    output << "roi_pca_mode=" << roiPcaModeName(config) << "\n";
    output << "roi_pca_dim=" << effectivePcaDim(config) << "\n";
}

/**
 * @brief 加载 key=value 元数据。
 */
std::unordered_map<std::string, std::string> loadMetadata(const fs::path &metadata_path)
{
    std::ifstream input(metadata_path);
    if (!input)
    {
        return {};
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string                                  line;
    while (std::getline(input, line))
    {
        const auto separator = line.find('=');
        if (separator == std::string::npos)
        {
            continue;
        }
        metadata.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return metadata;
}

/**
 * @brief 判断元数据字段是否与期望值一致。
 */
bool metadataEquals(const std::unordered_map<std::string, std::string> &metadata, const std::string &key,
                    const std::string &expected)
{
    const auto it = metadata.find(key);
    return it != metadata.end() && it->second == expected;
}

/**
 * @brief 判断磁盘上的 ROI 索引是否匹配当前配置。
 */
bool existingRoiIndexMatchesConfig(const fs::path &index_path, const RoiSearchConfig &config)
{
    const auto metadata = loadMetadata(priv::metadataPathFromIndex(index_path));
    if (metadata.empty())
    {
        return false;
    }

    return metadataEquals(metadata, "kind", "roi_search") && metadataEquals(metadata, "model", config.model_name)
        && metadataEquals(metadata, "feature", config.feature_name)
        && metadataEquals(metadata, "model_backend", irt::model::modelBackendName(config.model_backend))
        && metadataEquals(metadata, "model_device", irt::model::modelDeviceName(config.model_device))
        && metadataEquals(metadata, "preprocess_backend", priv::preprocessBackendName(config.preprocess_backend))
        && metadataEquals(metadata, "norm", priv::featureNormName(config.norm))
        && metadataEquals(metadata, "faiss_backend", priv::faissBackendName(config.faiss_backend))
        && metadataEquals(metadata, "index_storage", priv::indexStorageName(config.index_storage))
        && metadataEquals(metadata, "index_kind", priv::indexKindName(config))
        && metadataEquals(metadata, "roi_pooled_height", std::to_string(config.pooled_height))
        && metadataEquals(metadata, "roi_pooled_width", std::to_string(config.pooled_width))
        && metadataEquals(metadata, "roi_sampling_ratio", std::to_string(config.sampling_ratio))
        && metadataEquals(metadata, "roi_aligned", boolName(config.aligned))
        && metadataEquals(metadata, "roi_use_pca", boolName(config.use_pca))
        && metadataEquals(metadata, "roi_pca_mode", roiPcaModeName(config))
        && metadataEquals(metadata, "roi_pca_dim", std::to_string(effectivePcaDim(config)));
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
    };

public:
    RoiFeatureExtractor(const RoiSearchConfig &config, const fs::path &weights_file)
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
                "RoiSearch feature tensor must be NCHW map or DINO x_norm_patchtokens with shape BxTokensxDim");
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
     * @brief 获取原始特征图通道数。
     */
    int featureChannels() const noexcept
    {
        return feature_channels_;
    }

    /**
     * @brief 提取单个 ROI 的归一化特征。
     */
    std::vector<float> extract(const RoiSearchItem &item)
    {
        auto feature = extractRaw(item);
        normalizeFeature(feature, config_.norm);
        return feature;
    }

    /**
     * @brief 提取单个 ROI 的未归一化特征。
     */
    std::vector<float> extractRaw(const RoiSearchItem &item)
    {
        validateRoi(item.roi);
        const std::vector<fs::path> paths{item.image_path};
        const auto                  tensor = image_extractor_.extractFeatureTensorBatch(paths, 0, 1);
        return roiAlignAndFlatten(tensor, item.roi, 0);
    }

    /**
     * @brief 批量提取 ROI 特征。
     */
    std::vector<float> extractBatch(const std::vector<RoiSearchItem> &items, size_t begin, size_t count)
    {
        const int dim      = featureDim();
        auto      features = extractRawBatch(items, begin, count);
        for (size_t i = 0; i < count; ++i)
        {
            normalizeFeature(features.data() + i * static_cast<size_t>(dim), static_cast<size_t>(dim), config_.norm);
        }
        return features;
    }

    /**
     * @brief 批量提取未归一化 ROI 特征。
     */
    std::vector<float> extractRawBatch(const std::vector<RoiSearchItem> &items, size_t begin, size_t count)
    {
        if (begin > items.size() || count > items.size() - begin)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch range is invalid");
        }

        const int          dim = featureDim();
        std::vector<float> features;
        features.reserve(count * static_cast<size_t>(dim));
        for (size_t i = 0; i < count; ++i)
        {
            auto feature = extractRaw(items[begin + i]);
            features.insert(features.end(), feature.begin(), feature.end());
        }
        return features;
    }

    /**
     * @brief 按任意下标批量提取 ROI 特征。
     */
    std::vector<float> extractBatch(const std::vector<RoiSearchItem> &items, const std::vector<size_t> &indices)
    {
        const int dim      = featureDim();
        auto      features = extractRawBatch(items, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            normalizeFeature(features.data() + i * static_cast<size_t>(dim), static_cast<size_t>(dim), config_.norm);
        }
        return features;
    }

    /**
     * @brief 按任意下标批量提取未归一化 ROI 特征。
     */
    std::vector<float> extractRawBatch(const std::vector<RoiSearchItem> &items, const std::vector<size_t> &indices)
    {
        const int          dim = featureDim();
        std::vector<float> features;
        features.reserve(indices.size() * static_cast<size_t>(dim));
        for (const auto index : indices)
        {
            if (index >= items.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI feature batch index is invalid");
            }
            auto feature = extractRaw(items[index]);
            features.insert(features.end(), feature.begin(), feature.end());
        }
        return features;
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
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch requires NCHW feature tensor");
        }
        const auto expected = static_cast<size_t>(dims.d[0]) * static_cast<size_t>(dims.d[1])
                            * static_cast<size_t>(dims.d[2]) * static_cast<size_t>(dims.d[3]);
        if (tensor.data.size() != expected)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch feature tensor size mismatch");
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
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch requires NCHW feature map");
        }
        if (feature_map.data == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch feature map data is null");
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
                                 "RoiSearch patch token tensor must have shape BxTokensxDim");
        }

        const int batch    = static_cast<int>(tensor.dims.d[0]);
        const int tokens   = static_cast<int>(tensor.dims.d[1]);
        const int channels = static_cast<int>(tensor.dims.d[2]);
        if (tokens != patch_grid_.height * patch_grid_.width || channels != feature_channels_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch patch token grid mismatch");
        }

        const auto expected = static_cast<size_t>(batch) * static_cast<size_t>(tokens) * static_cast<size_t>(channels);
        if (tensor.data.size() != expected)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "RoiSearch patch token tensor size mismatch");
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
     * @brief 对当前特征图通道维训练本地 PCA，并还原为 ROIAlign 可消费的 NCHW 特征图。
     */
    PreparedFeatureMap applyLocalPcaToFeatureMap(PreparedFeatureMap feature_map) const
    {
        validatePreparedFeatureMap(feature_map);
        const int batch    = static_cast<int>(feature_map.dims.d[0]);
        const int channels = static_cast<int>(feature_map.dims.d[1]);
        const int height   = static_cast<int>(feature_map.dims.d[2]);
        const int width    = static_cast<int>(feature_map.dims.d[3]);
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
     * @brief 将原图 ROI 映射到特征图坐标并执行 ROIAlign。
     */
    std::vector<float> roiAlignAndFlatten(const FeatureTensorBatch &tensor, const RoiSearchBox &roi,
                                          size_t batch_index) const
    {
        auto feature_map = prepareFeatureMap(tensor);
        if (config_.use_pca)
        {
            feature_map = applyLocalPcaToFeatureMap(std::move(feature_map));
        }

        if (batch_index >= tensor.original_sizes.size() || batch_index >= static_cast<size_t>(feature_map.dims.d[0]))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI batch index is invalid");
        }

        const auto image_size = tensor.original_sizes[batch_index];
        if (image_size.width <= 0 || image_size.height <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid source image size for ROI mapping");
        }

        const auto feature_height = static_cast<float>(feature_map.dims.d[2]);
        const auto feature_width  = static_cast<float>(feature_map.dims.d[3]);
        const auto scale_x        = feature_width / static_cast<float>(image_size.width);
        const auto scale_y        = feature_height / static_cast<float>(image_size.height);

        const std::array<float, 5> mapped_roi{
            static_cast<float>(batch_index), roi.x1 * scale_x, roi.y1 * scale_y, roi.x2 * scale_x, roi.y2 * scale_y,
        };
        const int64_t input_shape[4]{
            feature_map.dims.d[0],
            feature_map.dims.d[1],
            feature_map.dims.d[2],
            feature_map.dims.d[3],
        };

        const irt::ops::RoIAlign roi_align({config_.pooled_height, config_.pooled_width}, 1.0f, config_.sampling_ratio,
                                           config_.aligned);
        auto                     feature = roi_align.forward(feature_map.data, input_shape, mapped_roi.data(), 1);
        if (feature.size() != static_cast<size_t>(roiFeatureDim(static_cast<int>(feature_map.dims.d[1]))))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ROI feature size");
        }
        return feature;
    }

    RoiSearchConfig       config_{};                              ///< ROI 检索配置。
    ImageFeatureExtractor image_extractor_;                       ///< 共用图像特征图抽取器。
    RoiFeatureLayout      layout_{RoiFeatureLayout::SpatialNchw}; ///< 特征张量布局。
    PatchTokenGrid        patch_grid_{};                          ///< patch token 推断出的空间网格。
    int                   feature_channels_{0};                   ///< ROIAlign 输入通道数。
    int                   feature_dim_{0};                        ///< ROI 特征向量维度。
};

} // namespace priv

namespace {

} // namespace

RoiSearch::Impl::Impl(RoiSearchConfig config)
    : config_(std::move(config))
{
    if (config_.faiss_backend == ImageSearchFaissBackend::GPU)
    {
        config_.index_storage = ImageSearchIndexStorage::RAM;
    }
    if (!irt::model::isSupportedModel(config_.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", config_.model_name.c_str());
    }
    if (config_.feature_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "feature name must not be empty");
    }
    if (config_.pooled_height <= 0 || config_.pooled_width <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI pooled size must be positive");
    }
    if (config_.sampling_ratio < -1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI sampling ratio must be -1 or non-negative");
    }
    if (config_.pca_dim < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be non-negative");
    }
    if (config_.use_pca && config_.pca_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI PCA dim must be positive when PCA is enabled");
    }
    priv::validateFeatureSearchConfig(config_, "RoiSearch");
}

RoiSearch::Impl::~Impl() = default;

void RoiSearch::Impl::buildOrLoad(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                                  const fs::path &index_file, bool rebuild_index,
                                  RoiSearchBuildProgressCallback progress_callback)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());

    if (!rebuild_index && fs::exists(index_file) && fs::exists(roiMappingPathFromIndex(index_file))
        && existingRoiIndexMatchesConfig(index_file, config_))
    {
        const auto mapped_items = loadRoiMapping(roiMappingPathFromIndex(index_file));
        if (itemsMatch(normalized_items, mapped_items))
        {
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
            load(weights_file, index_file);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0,
                                      gallery_items_.size(), gallery_items_.size());
            return;
        }
    }

    buildWithItems(weights_file, std::move(normalized_items), index_file, std::move(progress_callback));
}

void RoiSearch::Impl::build(const fs::path &weights_file, const std::vector<RoiSearchItem> &gallery_items,
                            const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());
    buildWithItems(weights_file, std::move(normalized_items), index_file, std::move(progress_callback));
}

void RoiSearch::Impl::load(const fs::path &weights_file, const fs::path &index_file)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index_file must not be empty");
    }
    if (!fs::exists(index_file))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI index file does not exist: %s",
                             index_file.string().c_str());
    }
    if (!fs::exists(roiMappingPathFromIndex(index_file)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI mapping file does not exist: %s",
                             roiMappingPathFromIndex(index_file).string().c_str());
    }
    if (!existingRoiIndexMatchesConfig(index_file, config_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index metadata does not match current config: %s",
                             priv::metadataPathFromIndex(index_file).string().c_str());
    }

    auto items = loadRoiMapping(roiMappingPathFromIndex(index_file));
    auto faiss = priv::loadConfiguredFaissIndex(index_file, config_);
    if (config_.use_pca)
    {
        const auto expected_dim = static_cast<faiss::idx_t>(config_.pca_dim * config_.pooled_height * config_.pooled_width);
        if (!faiss.index || faiss.index->d != expected_dim)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI local PCA dim does not match Faiss index dim");
        }
    }
    if (static_cast<faiss::idx_t>(items.size()) != faiss.index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ROI index size (%lld) does not match mapping size (%zu)",
                             static_cast<long long>(faiss.index->ntotal), items.size());
    }

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    index_path_          = index_file;
    faiss_gpu_resources_ = std::move(faiss.gpu_resources);
    index_               = std::move(faiss.index);
    gallery_items_       = std::move(items);
    extractor_.reset();
    feature_dim_ = static_cast<int>(index_->d);
}

void RoiSearch::Impl::buildWithItems(const fs::path &weights_file, std::vector<RoiSearchItem> gallery_items,
                                     const fs::path &index_file, RoiSearchBuildProgressCallback progress_callback)
{
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 0, 1);
    auto extractor = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 1, 1);

    if (!index_file.parent_path().empty())
    {
        fs::create_directories(index_file.parent_path());
    }

    const int final_feature_dim = extractor->featureDim();
    auto      faiss             = priv::buildConfiguredFaissIndex(
        gallery_items.size(), final_feature_dim, index_file, config_,
        [&](size_t index) { return extractor->extract(gallery_items[index]); }, [&](size_t begin, size_t count)
        { return extractor->extractBatch(gallery_items, begin, count); }, [&](const std::vector<size_t> &indices)
        { return extractor->extractBatch(gallery_items, indices); }, progress_callback);
    saveRoiMapping(roiMappingPathFromIndex(index_file), gallery_items);

    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    index_path_          = index_file;
    faiss_gpu_resources_ = std::move(faiss.gpu_resources);
    index_               = std::move(faiss.index);
    gallery_items_       = std::move(gallery_items);
    feature_dim_         = final_feature_dim;
    extractor_           = std::move(extractor);

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 0, 1);
    saveRoiMetadata(priv::metadataPathFromIndex(index_path_), config_);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 1, 1);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0, gallery_items_.size(),
                              gallery_items_.size());
}

std::vector<RoiSearchResult> RoiSearch::Impl::search(const fs::path &query_image, const RoiSearchBox &roi, int top_k)
{
    if (!index_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "RoiSearch index is not ready");
    }
    if (top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }
    if (index_->ntotal <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "RoiSearch index is empty");
    }

    const auto query_item = normalizeItem(RoiSearchItem{query_image, roi});
    ensureExtractor();
    const auto query_feature = extractor_->extract(query_item);
    if (query_feature.size() != static_cast<size_t>(index_->d))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ROI query feature dimension mismatch");
    }

    const int                 result_count = std::min(top_k, static_cast<int>(index_->ntotal));
    std::vector<faiss::idx_t> indices(result_count);
    std::vector<float>        distances(result_count);
    index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());

    std::vector<RoiSearchResult> results;
    results.reserve(static_cast<size_t>(result_count));
    for (int i = 0; i < result_count; ++i)
    {
        if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_items_.size())
        {
            continue;
        }
        const auto item_index = static_cast<size_t>(indices[i]);
        results.push_back(
            {distances[i], gallery_items_[item_index].image_path, gallery_items_[item_index].roi, item_index});
    }
    return results;
}

bool RoiSearch::Impl::isReady() const noexcept
{
    return index_ != nullptr;
}

const RoiSearchConfig &RoiSearch::Impl::config() const noexcept
{
    return config_;
}

const fs::path &RoiSearch::Impl::indexPath() const noexcept
{
    return index_path_;
}

std::vector<RoiSearchItem> RoiSearch::Impl::galleryItems() const
{
    return gallery_items_;
}

int RoiSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void RoiSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        extractor_ = std::make_unique<priv::RoiFeatureExtractor>(config_, weights_file_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
