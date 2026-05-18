#include <cuda_runtime_api.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cmath>
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

struct Arguments
{
    std::string model_name{ kDefaultModelName };
    std::string feature_name{ kDefaultFeatureName };
    fs::path weights_file;
    fs::path gallery_dir;
    fs::path query_image;
    fs::path index_file;
    int      top_k{ kDefaultTopK };
    bool     rebuild_index{ false };
};

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

    void  *ptr_{ nullptr };
    size_t num_bytes_{ 0 };
};

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isImageFile(const fs::path &path)
{
    if (!path.has_extension())
    {
        return false;
    }

    const std::string ext = toLower(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

void checkCuda(cudaError_t status, const char *op)
{
    if (status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
    }
}

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

cv::Mat preprocess(const cv::Mat &image)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    cv::Mat resized;
    cv::resize(normalized, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);
    return resized;
}

std::vector<float> makeInputTensor(const cv::Mat &preprocessed)
{
    std::vector<float>   input_data(1 * 3 * 224 * 224);
    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(input_data.data() + c * 224 * 224, channels[c].data, 224 * 224 * sizeof(float));
    }
    return input_data;
}

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

fs::path mappingPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".paths.txt";
}

fs::path metadataPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".meta.txt";
}

void printUsage(const char *program_name)
{
    std::cerr << "Usage: " << program_name
              << " <weights_file.wts> <gallery_dir> <query_image>"
              << " [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--rebuild-index]"
              << std::endl;
    std::cerr << "Default model: " << kDefaultModelName << std::endl;
    std::cerr << "Default feature tensor: " << kDefaultFeatureName << std::endl;
    std::cerr << "Default top-k: " << kDefaultTopK << std::endl;
    std::cerr << "If --index is omitted, the sample uses <gallery_dir>/<model>_<feature>.faiss" << std::endl;
    std::cerr << "Supported models:";
    for (const auto &model_name : irt::model::getRegisteredModelNames())
    {
        std::cerr << ' ' << model_name;
    }
    std::cerr << std::endl;
}

Arguments parseArguments(int argc, char *argv[])
{
    if (argc < 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Expected at least 3 positional arguments");
    }

    Arguments args;
    args.weights_file = argv[1];
    args.gallery_dir  = argv[2];
    args.query_image  = argv[3];

    for (int i = 4; i < argc; ++i)
    {
        const std::string option = argv[i];
        if (option == "--rebuild-index")
        {
            args.rebuild_index = true;
            continue;
        }

        if (option == "--model")
        {
            if (i + 1 >= argc)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--model requires a value");
            }
            args.model_name = argv[++i];
            continue;
        }

        if (option == "--feature")
        {
            if (i + 1 >= argc)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--feature requires a value");
            }
            args.feature_name = argv[++i];
            continue;
        }

        if (option == "--topk")
        {
            if (i + 1 >= argc)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--topk requires a value");
            }
            args.top_k = std::stoi(argv[++i]);
            continue;
        }

        if (option == "--index")
        {
            if (i + 1 >= argc)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--index requires a value");
            }
            args.index_file = argv[++i];
            continue;
        }

        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown argument: %s", option.c_str());
    }

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

fs::path defaultIndexPath(const fs::path &gallery_dir, const std::string &model_name, const std::string &feature_name)
{
    return gallery_dir / (sanitizeFileStem(model_name) + "_" + sanitizeFileStem(feature_name) + ".faiss");
}

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

class FeatureExtractor
{
public:
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

    ~FeatureExtractor()
    {
        // Ensure TensorRT context/engine are torn down before CUDA buffers they reference.
        model_.reset();
    }

    int featureDim() const noexcept
    {
        return static_cast<int>(feature_dim_);
    }

    std::vector<float> extract(const fs::path &image_path)
    {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto preprocessed = preprocess(image);
        const auto input_data   = makeInputTensor(preprocessed);
        std::vector<float> feature(feature_dim_);
        std::vector<void *> buffers{device_input_.get(), device_output_.get()};

        checkCuda(cudaMemcpy(device_input_.get(), input_data.data(), input_data.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(H2D input)");
        model_->forwardFeatures(buffers);
        checkCuda(cudaMemcpy(feature.data(), device_output_.get(), feature.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(D2H feature)");
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

int main(int argc, char *argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);
        const fs::path index_path = args.index_file.empty()
                                        ? defaultIndexPath(args.gallery_dir, args.model_name, args.feature_name)
                                        : args.index_file;
        const fs::path mapping_path = mappingPathFromIndex(index_path);
        const fs::path metadata_path = metadataPathFromIndex(index_path);

        std::unique_ptr<faiss::Index> index;
        std::vector<fs::path>         gallery_images;

        if (!args.rebuild_index && fs::exists(index_path) && fs::exists(mapping_path))
        {
            std::cout << "Loading existing Faiss index: " << fs::absolute(index_path).string() << std::endl;
            auto loaded = loadIndex(index_path);
            index = std::move(loaded.first);
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
        const auto query_feature = extractor.extract(args.query_image);

        const int top_k = std::min(args.top_k, static_cast<int>(index->ntotal));
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
        printUsage(argv[0]);
        return -1;
    }
}
