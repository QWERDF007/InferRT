#include <inferrt/features/ImageSearch.hpp>

#include <cuda_runtime_api.h>
#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {

namespace {

using DeviceBuffer = irt::model::DeviceBuffer;
using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

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
 * @brief 对特征向量执行 L2 归一化。
 * @param values 待归一化的特征向量。
 */
void l2Normalize(std::vector<float> &values)
{
    const float sum_sq = std::inner_product(values.begin(), values.end(), values.begin(), 0.0f);
    if (sum_sq <= 0.0f)
    {
        return;
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (float &value : values)
    {
        value *= inv_norm;
    }
}

/**
 * @brief 将模型名或特征名转换为适合文件名使用的安全字符串。
 * @param value 原始名称。
 * @return 处理后的文件名 stem。
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

/**
 * @brief 根据索引文件路径生成配套的图片路径映射文件路径。
 * @param index_path Faiss 索引文件路径。
 * @return 路径映射文件路径。
 */
fs::path mappingPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".paths.txt";
}

/**
 * @brief 根据索引文件路径生成配套的元数据文件路径。
 * @param index_path Faiss 索引文件路径。
 * @return 元数据文件路径。
 */
fs::path metadataPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".meta.txt";
}

/**
 * @brief 保存向量 id 到图片路径的映射文件。
 * @param mapping_path 输出映射文件路径。
 * @param image_paths 图片路径列表。
 */
void savePathMapping(const fs::path &mapping_path, const std::vector<fs::path> &image_paths)
{
    std::ofstream output(mapping_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open mapping file: %s",
                             mapping_path.string().c_str());
    }

    for (const auto &image_path : image_paths)
    {
        output << image_path.generic_string() << "\n";
    }
}

/**
 * @brief 加载向量 id 到图片路径的映射文件。
 * @param mapping_path 映射文件路径。
 * @return 图片路径列表。
 */
std::vector<fs::path> loadPathMapping(const fs::path &mapping_path)
{
    std::ifstream input(mapping_path);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open mapping file: %s",
                             mapping_path.string().c_str());
    }

    std::vector<fs::path> image_paths;
    std::string           line;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            image_paths.emplace_back(line);
        }
    }
    return image_paths;
}

/**
 * @brief 保存索引构建元数据。
 * @param metadata_path 元数据文件路径。
 * @param gallery_dir 图库目录。
 * @param model_name 模型名。
 * @param feature_name 特征张量名。
 */
void saveMetadata(const fs::path &metadata_path, const fs::path &gallery_dir, const std::string &model_name,
                  const std::string &feature_name)
{
    std::ofstream output(metadata_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open metadata file: %s",
                             metadata_path.string().c_str());
    }

    output << "model=" << model_name << "\n";
    output << "feature=" << feature_name << "\n";
    output << "gallery_dir=" << fs::absolute(gallery_dir).generic_string() << "\n";
}

/**
 * @brief 基于分类、ViT 或 DINO 模型中间特征提取向量的辅助类。
 */
class FeatureExtractor
{
public:
    /**
     * @brief 构造特征提取器并加载 feature-only 模型。
     * @param model_name 模型名。
     * @param feature_name 特征张量名。
     * @param weights_file 权重文件路径。
     */
    FeatureExtractor(std::string model_name, std::string feature_name, const fs::path &weights_file)
        : model_name_(std::move(model_name))
        , feature_name_(std::move(feature_name))
    {
        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames({feature_name_});
        config->setOutputTensorNames({feature_name_});
        config->setFeatureOnly(true);

        model_ = irt::model::CreateModel(model_name_, std::move(config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 model_name_.c_str());
        }

        model_->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        model_->buildOrLoad(weights_file.string());

        output_name_ = model_->modelConfig().outputTensorNames().front();
        output_dims_ = model_->tensorShape(output_name_);
        output_type_ = model_->tensorDataType(output_name_);
        if (output_type_ != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected float feature tensor for %s, got unsupported data type",
                                 output_name_.c_str());
        }

        const auto &input_tensor_names = model_->modelConfig().inputTensorNames();
        if (input_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch model must expose at least one input tensor");
        }

        const auto input_shape = model_->tensorShape(input_tensor_names.front());
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != 3 || input_shape.d[2] <= 0
            || input_shape.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch expects input shape 1x3xHxW, got %s",
                                 dimsToCsv(input_shape).c_str());
        }

        input_height_ = static_cast<int>(input_shape.d[2]);
        input_width_  = static_cast<int>(input_shape.d[3]);
        feature_dim_  = elementCount(output_dims_);
        device_input_.resize(elementCount(input_shape), nvinfer1::DataType::kFLOAT);
        device_output_.resize(feature_dim_, nvinfer1::DataType::kFLOAT);
    }

    /**
     * @brief 析构时优先释放模型，再释放 CUDA buffer。
     */
    ~FeatureExtractor()
    {
        // TensorRT context/engine 先释放，再释放 CUDA buffer。
        model_.reset();
    }

    /**
     * @brief 返回单张图片对应的特征维度。
     * @return 特征维度。
     */
    int featureDim() const noexcept
    {
        return static_cast<int>(feature_dim_);
    }

    /**
     * @brief 提取单张图片的归一化特征向量。
     * @param image_path 输入图片路径。
     * @return L2 归一化后的特征。
     */
    std::vector<float> extract(const fs::path &image_path)
    {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto          preprocessed = irt::model::ImageNetUtil::preprocess(image, cv::Size(input_width_, input_height_));
        const auto          input_data   = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
        std::vector<float>  feature(feature_dim_);
        std::vector<void *> buffers{device_input_.data(), device_output_.data()};
        const auto          stream = model_->resolveExecutionStream();

        checkCuda(cudaMemcpyAsync(device_input_.data(), input_data.data(), input_data.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");
        model_->forwardFeatures(buffers, stream, true);
        checkCuda(cudaMemcpyAsync(feature.data(), device_output_.data(), feature.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(D2H feature)");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature extraction)");
        l2Normalize(feature);
        return feature;
    }

private:
    std::string                         model_name_;
    std::string                         feature_name_;
    std::unique_ptr<irt::model::IModel> model_;
    std::string                         output_name_;
    nvinfer1::Dims                      output_dims_{};
    nvinfer1::DataType                  output_type_{};
    int                                 input_height_{224};
    int                                 input_width_{224};
    size_t                              feature_dim_{0};
    DeviceBuffer                        device_input_;
    DeviceBuffer                        device_output_;
};

/**
 * @brief 基于图库图片构建 Faiss 内积索引并持久化到磁盘。
 * @param gallery_images 图库图片列表。
 * @param extractor 特征提取器。
 * @param index_path 索引输出路径。
 * @return 构建完成的 Faiss 索引。
 */
std::unique_ptr<faiss::Index> buildIndex(const std::vector<fs::path> &gallery_images, FeatureExtractor &extractor,
                                         const fs::path &index_path)
{
    auto index = std::make_unique<faiss::IndexFlatIP>(extractor.featureDim());

    for (const auto &image_path : gallery_images)
    {
        const auto feature = extractor.extract(image_path);
        index->add(1, feature.data());
    }

    faiss::write_index(index.get(), index_path.string().c_str());
    savePathMapping(mappingPathFromIndex(index_path), gallery_images);
    return index;
}

/**
 * @brief 从磁盘加载 Faiss 索引及其图片路径映射。
 * @param index_path Faiss 索引文件路径。
 * @return 索引对象与图片路径列表。
 */
std::pair<std::unique_ptr<faiss::Index>, std::vector<fs::path>> loadIndex(const fs::path &index_path)
{
    auto index = std::unique_ptr<faiss::Index>(faiss::read_index(index_path.string().c_str()));
    if (!index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load Faiss index: %s",
                             index_path.string().c_str());
    }

    auto image_paths = loadPathMapping(mappingPathFromIndex(index_path));
    if (static_cast<faiss::idx_t>(image_paths.size()) != index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index size (%lld) does not match path mapping size (%zu)",
                             static_cast<long long>(index->ntotal), image_paths.size());
    }

    return {std::move(index), std::move(image_paths)};
}

} // namespace

class ImageSearch::Impl
{
public:
    /**
     * @brief 构造图像检索私有实现。
     * @param model_name 模型名称。
     * @param feature_name 特征名称。
     */
    Impl(std::string model_name, std::string feature_name)
        : model_name_(std::move(model_name))
        , feature_name_(std::move(feature_name))
    {
        if (!irt::model::isSupportedModel(model_name_))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", model_name_.c_str());
        }
        if (feature_name_.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "feature name must not be empty");
        }
    }

    /**
     * @brief 构建或加载图库索引。
     */
    void buildOrLoad(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file,
                     bool rebuild_index)
    {
        weights_file_ = weights_file;
        gallery_dir_  = gallery_dir;
        index_path_   = index_file.empty() ? ImageSearch::defaultIndexPath(gallery_dir_, model_name_, feature_name_)
                                           : index_file;

        if (!rebuild_index && fs::exists(index_path_) && fs::exists(mappingPathFromIndex(index_path_)))
        {
            auto loaded     = loadIndex(index_path_);
            index_          = std::move(loaded.first);
            gallery_images_ = std::move(loaded.second);
            feature_dim_    = static_cast<int>(index_->d);
            return;
        }

        auto extractor = std::make_unique<FeatureExtractor>(model_name_, feature_name_, weights_file_);
        gallery_images_ = ImageSearch::collectGalleryImages(gallery_dir_);
        if (!index_path_.parent_path().empty())
        {
            fs::create_directories(index_path_.parent_path());
        }
        index_       = buildIndex(gallery_images_, *extractor, index_path_);
        feature_dim_ = extractor->featureDim();
        extractor_   = std::move(extractor);
        saveMetadata(metadataPathFromIndex(index_path_), gallery_dir_, model_name_, feature_name_);
    }

    /**
     * @brief 搜索查询图片。
     */
    std::vector<ImageSearchResult> search(const fs::path &query_image, int top_k)
    {
        if (!index_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is not ready");
        }
        if (top_k <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
        }
        if (index_->ntotal <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is empty");
        }

        ensureExtractor();
        const auto query_feature = extractor_->extract(query_image);

        const int                 result_count = std::min(top_k, static_cast<int>(index_->ntotal));
        std::vector<faiss::idx_t> indices(result_count);
        std::vector<float>        distances(result_count);
        index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());

        std::vector<ImageSearchResult> results;
        results.reserve(static_cast<size_t>(result_count));
        for (int i = 0; i < result_count; ++i)
        {
            if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_images_.size())
            {
                continue;
            }
            results.push_back({distances[i], gallery_images_[static_cast<size_t>(indices[i])]});
        }
        return results;
    }

    /**
     * @brief 确保查询阶段特征提取器已经构造。
     */
    void ensureExtractor()
    {
        if (!extractor_)
        {
            extractor_   = std::make_unique<FeatureExtractor>(model_name_, feature_name_, weights_file_);
            feature_dim_ = extractor_->featureDim();
        }
    }

    std::string                       model_name_;
    std::string                       feature_name_;
    fs::path                          weights_file_;
    fs::path                          gallery_dir_;
    fs::path                          index_path_;
    std::vector<fs::path>             gallery_images_;
    std::unique_ptr<faiss::Index>     index_;
    std::unique_ptr<FeatureExtractor> extractor_;
    int                               feature_dim_{0};
};

ImageSearch::ImageSearch(std::string model_name, std::string feature_name)
    : impl_(std::make_unique<Impl>(std::move(model_name), std::move(feature_name)))
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
    return impl_ && impl_->index_ != nullptr;
}

const std::string &ImageSearch::modelName() const noexcept
{
    return impl_->model_name_;
}

const std::string &ImageSearch::featureName() const noexcept
{
    return impl_->feature_name_;
}

const fs::path &ImageSearch::indexPath() const noexcept
{
    return impl_->index_path_;
}

std::vector<fs::path> ImageSearch::galleryImages() const
{
    return impl_->gallery_images_;
}

int ImageSearch::featureDim() const noexcept
{
    return impl_->feature_dim_;
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
