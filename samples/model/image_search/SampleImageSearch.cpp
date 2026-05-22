#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char *kDefaultModelName   = "resnet18";
constexpr const char *kDefaultFeatureName = "layer4";
constexpr int         kDefaultTopK        = 5;

/**
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

/**
 * @brief 图像检索 sample 的命令行参数集合。
 */
struct Arguments
{
    std::string model_name{kDefaultModelName};
    std::string feature_name{kDefaultFeatureName};
    fs::path    weights_file;
    fs::path    gallery_dir;
    fs::path    query_image;
    fs::path    index_file;
    int         top_k{kDefaultTopK};
    bool        rebuild_index{false};
};

/**
 * @brief 简单的 CUDA 设备内存 RAII 封装。
 */
class CudaBuffer
{
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t num_bytes)
    {
        allocate(num_bytes);
    }

    ~CudaBuffer()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer &)            = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr_(other.ptr_)
        , num_bytes_(other.num_bytes_)
    {
        other.ptr_       = nullptr;
        other.num_bytes_ = 0;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                cudaFree(ptr_);
            }
            ptr_             = other.ptr_;
            num_bytes_       = other.num_bytes_;
            other.ptr_       = nullptr;
            other.num_bytes_ = 0;
        }
        return *this;
    }

    void allocate(size_t num_bytes)
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_       = nullptr;
            num_bytes_ = 0;
        }

        if (num_bytes == 0)
        {
            return;
        }

        checkCuda(cudaMalloc(&ptr_, num_bytes), "cudaMalloc");
        num_bytes_ = num_bytes;
    }

    void *get() const noexcept
    {
        return ptr_;
    }

private:
    static void checkCuda(cudaError_t status, const char *op)
    {
        if (status != cudaSuccess)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
        }
    }

    void  *ptr_{nullptr};
    size_t num_bytes_{0};
};

/**
 * @brief 将文件扩展名转换为小写，便于无大小写差异比较。
 * @param value 原始字符串。
 * @return 小写结果。
 */
std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

/**
 * @brief 判断路径是否为支持的图片文件。
 * @param path 待判断路径。
 * @return 是图片返回 true，否则返回 false。
 */
bool isImageFile(const fs::path &path)
{
    if (!path.has_extension())
    {
        return false;
    }

    const std::string ext = toLower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

/**
 * @brief 包装 CUDA 调用，失败时抛出带上下文的异常。
 * @param status CUDA 返回状态。
 * @param op 当前操作名。
 */
void checkCuda(cudaError_t status, const char *op)
{
    if (status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
    }
}

/**
 * @brief 计算张量元素总数，并校验维度均为正数。
 * @param dims TensorRT 维度对象。
 * @return 元素总数。
 */
size_t elementCount(const nvinfer1::Dims &dims)
{
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor shape contains non-positive dimension: %d", dims.d[i]);
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
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
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Build and query a Faiss image retrieval index from InferRT features");
    options.add_options()("weights-file,w", "Weights file (.wts)", cxxopts::value<std::string>())(
        "gallery-dir,g", "Gallery image directory", cxxopts::value<std::string>())("query-image,q", "Query image path",
                                                                                   cxxopts::value<std::string>())(
        "model", "Built-in model name", cxxopts::value<std::string>()->default_value(kDefaultModelName))(
        "feature", "Feature tensor name", cxxopts::value<std::string>()->default_value(kDefaultFeatureName))(
        "topk", "Top-k nearest results", cxxopts::value<int>()->default_value(std::to_string(kDefaultTopK)))(
        "index", "Faiss index path", cxxopts::value<std::string>()->default_value(""))(
        "rebuild-index", "Force rebuild of the Faiss index")("h,help", "Show help");
    return options;
}

/**
 * @brief 解析并校验命令行参数。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 解析后的参数。
 */
Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default model: " << kDefaultModelName << std::endl;
        std::cout << "Default feature tensor: " << kDefaultFeatureName << std::endl;
        std::cout << "Default top-k: " << kDefaultTopK << std::endl;
        std::cout << "If --index is omitted, the sample uses <gallery_dir>/<model>_<feature>.faiss" << std::endl;
        std::cout << "Supported models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            std::cout << ' ' << model_name;
        }
        std::cout << std::endl;
        throw HelpRequested{};
    }

    if (!result.count("weights-file") || !result.count("gallery-dir") || !result.count("query-image"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--weights-file, --gallery-dir and --query-image are required");
    }

    Arguments args;
    args.model_name    = result["model"].as<std::string>();
    args.feature_name  = result["feature"].as<std::string>();
    args.weights_file  = result["weights-file"].as<std::string>();
    args.gallery_dir   = result["gallery-dir"].as<std::string>();
    args.query_image   = result["query-image"].as<std::string>();
    args.index_file    = result["index"].as<std::string>();
    args.top_k         = result["topk"].as<int>();
    args.rebuild_index = result.count("rebuild-index") > 0;

    if (args.top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }

    if (!irt::model::isSupportedModel(args.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", args.model_name.c_str());
    }

    return args;
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
 * @brief 生成默认 Faiss 索引文件路径。
 * @param gallery_dir 图库目录。
 * @param model_name 模型名。
 * @param feature_name 特征张量名。
 * @return 默认索引文件路径。
 */
fs::path defaultIndexPath(const fs::path &gallery_dir, const std::string &model_name, const std::string &feature_name)
{
    return gallery_dir / (sanitizeFileStem(model_name) + "_" + sanitizeFileStem(feature_name) + ".faiss");
}

/**
 * @brief 递归收集图库目录下的所有图片。
 * @param gallery_dir 图库根目录。
 * @return 排序后的图片路径列表。
 */
std::vector<fs::path> collectGalleryImages(const fs::path &gallery_dir)
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
 * @brief 基于分类模型中间特征提取向量的辅助类。
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
        config->setFeatureOnly(true);

        model_ = irt::model::CreateModel(model_name_, std::move(config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 model_name_.c_str());
        }

        model_->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        model_->buildOrLoad(weights_file.string());

        output_name_ = model_->modelConfig().featureTensorNames().front();
        output_dims_ = model_->tensorShape(output_name_);
        output_type_ = model_->tensorDataType(output_name_);
        if (output_type_ != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected float feature tensor for %s, got unsupported data type",
                                 output_name_.c_str());
        }

        feature_dim_ = elementCount(output_dims_);
        device_input_.allocate(sizeof(float) * 3 * 224 * 224);
        device_output_.allocate(sizeof(float) * feature_dim_);
    }

    /**
     * @brief 析构时优先释放模型，再释放 CUDA buffer。
     */
    ~FeatureExtractor()
    {
        // Ensure TensorRT context/engine are torn down before CUDA buffers they reference.
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

        const auto          preprocessed = irt::model::ImageNetUtil::preprocess(image);
        const auto          input_data   = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
        std::vector<float>  feature(feature_dim_);
        std::vector<void *> buffers{device_input_.get(), device_output_.get()};
        const auto          stream = model_->resolveExecutionStream();

        checkCuda(cudaMemcpyAsync(device_input_.get(), input_data.data(), input_data.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");
        model_->forwardFeatures(buffers, stream, true);
        checkCuda(cudaMemcpyAsync(feature.data(), device_output_.get(), feature.size() * sizeof(float),
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
    size_t                              feature_dim_{0};
    CudaBuffer                          device_input_;
    CudaBuffer                          device_output_;
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

    std::cout << "Building Faiss index from " << gallery_images.size() << " gallery images..." << std::endl;
    for (size_t i = 0; i < gallery_images.size(); ++i)
    {
        const auto feature = extractor.extract(gallery_images[i]);
        index->add(1, feature.data());
        std::cout << "[" << (i + 1) << "/" << gallery_images.size() << "] indexed "
                  << gallery_images[i].filename().string() << std::endl;
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

/**
 * @brief 构建或加载图像检索索引，并返回查询图片的 Top-K 相似结果。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        const auto     args          = parseArguments(argc, argv);
        const fs::path index_path    = args.index_file.empty()
                                         ? defaultIndexPath(args.gallery_dir, args.model_name, args.feature_name)
                                         : args.index_file;
        const fs::path mapping_path  = mappingPathFromIndex(index_path);
        const fs::path metadata_path = metadataPathFromIndex(index_path);

        std::unique_ptr<faiss::Index> index;
        std::vector<fs::path>         gallery_images;

        if (!args.rebuild_index && fs::exists(index_path) && fs::exists(mapping_path))
        {
            std::cout << "Loading existing Faiss index: " << fs::absolute(index_path).string() << std::endl;
            auto loaded    = loadIndex(index_path);
            index          = std::move(loaded.first);
            gallery_images = std::move(loaded.second);
        }
        else
        {
            FeatureExtractor extractor(args.model_name, args.feature_name, args.weights_file);
            gallery_images = collectGalleryImages(args.gallery_dir);
            if (!index_path.parent_path().empty())
            {
                fs::create_directories(index_path.parent_path());
            }
            index = buildIndex(gallery_images, extractor, index_path);
            saveMetadata(metadata_path, args.gallery_dir, args.model_name, args.feature_name);
            std::cout << "Saved Faiss index to: " << fs::absolute(index_path).string() << std::endl;
        }

        FeatureExtractor extractor(args.model_name, args.feature_name, args.weights_file);
        const auto       query_feature = extractor.extract(args.query_image);

        const int                 top_k = std::min(args.top_k, static_cast<int>(index->ntotal));
        std::vector<faiss::idx_t> indices(top_k);
        std::vector<float>        distances(top_k);
        index->search(1, query_feature.data(), top_k, distances.data(), indices.data());

        std::cout << "Query image: " << fs::absolute(args.query_image).string() << std::endl;
        std::cout << "Model: " << args.model_name << ", feature tensor: " << args.feature_name << std::endl;
        std::cout << "Top " << top_k << " similar images:" << std::endl;

        for (int i = 0; i < top_k; ++i)
        {
            if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_images.size())
            {
                continue;
            }

            std::cout << (i + 1) << ". score=" << distances[i]
                      << " image=" << gallery_images[static_cast<size_t>(indices[i])].string() << std::endl;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const HelpRequested &)
    {
        return 0;
    }
}
