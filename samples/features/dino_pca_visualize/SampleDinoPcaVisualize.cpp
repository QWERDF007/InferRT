#include <SampleSupport.hpp>

#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock        = irt::util::TimingClock;
using DeviceBuffer = irt::model::DeviceBuffer;
using irt::model::checkCuda;
using irt::model::dataTypeToString;
using irt::model::dimsToCsv;
using irt::model::elementCount;
using irt::model::elementSize;
using irt::samples::HelpRequested;
using irt::util::elapsedMs;
using irt::util::printTimingStats;
using irt::util::sanitizeFileStem;
using irt::util::summarizeTimings;
using irt::util::TimingStats;

constexpr const char *kDefaultModel   = "dinov2_vits14";
constexpr const char *kDefaultFeature = "x_norm_patchtokens";
const fs::path        kDefaultOutputDir{"dino_pca_visualize_cpp"};

struct Arguments
{
    std::string             model_name{kDefaultModel};
    fs::path                weights_file;
    std::string             feature_name{kDefaultFeature};
    fs::path                image_path;
    fs::path                output_dir;
    float                   threshold{0.5F};
    irt::model::ModelRuntime runtime{};
    int                     warmup{0};
    int                     repeat{1};
};

struct IterationTiming
{
    double h2d_ms{0.0};
    double inference_ms{0.0};
    double d2h_ms{0.0};

    double totalMs() const noexcept
    {
        return h2d_ms + inference_ms + d2h_ms;
    }
};

struct OutputTensor
{
    std::string                         name;
    irt::Shape                          dims{};
    irt::TensorDataType                 data_type{irt::TensorDataType::F32};
    size_t                              element_count{0};
    size_t                              num_bytes{0};
    DeviceBuffer                        device;
    std::vector<char>                   host_bytes;
    std::vector<std::max_align_t>       graph_storage;
};

struct PatchFeatureMatrix
{
    cv::Mat features;
    int     grid_h{0};
    int     grid_w{0};
    int     channels{0};
};

cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Visualize DINO patch tokens with OpenCV PCA");
    options.add_options()("model,m", "Built-in DINO model name",
                          cxxopts::value<std::string>()->default_value(kDefaultModel))(
        "weights-file,w", "Weights/model file. Empty uses assets/models/<family>/<model>.wts",
        cxxopts::value<std::string>()->default_value(""))(
        "feature,f", "Feature tensor name", cxxopts::value<std::string>()->default_value(kDefaultFeature))(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "output-dir,o", "Output directory", cxxopts::value<std::string>()->default_value(""))(
        "threshold,t", "Background mask threshold after 1D PCA normalization",
        cxxopts::value<float>()->default_value("0.5"))(
        "runtime", "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
        cxxopts::value<std::string>()->default_value("tensorrt:0"));
    irt::samples::addTimingOptions(options, "Timed feature forward iterations");
    options.add_options()("h,help", "Show help");
    return options;
}

Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << irt::model::ImageNetUtil::kDefaultImagePath.generic_string() << '\n';
        std::cout << "Default output dir: " << kDefaultOutputDir.generic_string() << '\n';
        std::cout << "The sample expects a patch-token feature such as x_norm_patchtokens." << std::endl;
        throw HelpRequested{};
    }

    Arguments args;
    args.model_name   = result["model"].as<std::string>();
    args.weights_file = result["weights-file"].as<std::string>();
    args.feature_name = result["feature"].as<std::string>();
    args.image_path   = result["image-path"].as<std::string>();
    args.output_dir   = result["output-dir"].as<std::string>();
    args.threshold    = result["threshold"].as<float>();
    args.runtime      = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    const auto timing = irt::samples::parseTimingOptions(result);
    args.warmup       = timing.warmup;
    args.repeat       = timing.repeat;

    if (irt::util::trim(args.model_name).empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--model must not be empty");
    }
    if (irt::util::trim(args.feature_name).empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--feature must not be empty");
    }
    if (args.threshold < 0.0F || args.threshold > 1.0F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--threshold must be in [0, 1]");
    }
    return args;
}

fs::path resolveImagePath(const fs::path &project_root, const fs::path &configured)
{
    if (configured.empty())
    {
        return project_root / irt::model::ImageNetUtil::kDefaultImagePath;
    }
    return irt::samples::resolvePath(project_root, configured);
}

fs::path resolveWeightsPath(const fs::path &project_root, const Arguments &args)
{
    if (!args.weights_file.empty())
    {
        return irt::samples::resolvePath(project_root, args.weights_file);
    }
    if (args.runtime.backend() != irt::model::ModelRuntime::Backend::TensorRT)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--weights-file is required for non-TensorRT backends");
    }
    std::string family = args.model_name.find("dinov3") != std::string::npos ? "dinov3" : "dinov2";
    return project_root / "assets" / "models" / family / (args.model_name + ".wts");
}

fs::path resolveOutputDir(const fs::path &configured)
{
    if (configured.empty())
    {
        return fs::current_path() / kDefaultOutputDir;
    }
    return configured.is_absolute() ? configured : fs::current_path() / configured;
}

std::vector<float> preprocessImage(const cv::Mat &image, const irt::Shape &input_dims)
{
    if (input_dims.rank() != 4 || input_dims[0] != 1 || input_dims[1] != 3 || input_dims[2] <= 0
        || input_dims[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "DINO PCA sample expects input shape 1x3xHxW, got %s", dimsToCsv(input_dims).c_str());
    }

    const cv::Mat preprocessed = irt::model::ImageNetUtil::preprocess(
        image, cv::Size(static_cast<int>(input_dims[3]), static_cast<int>(input_dims[2])));
    return irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
}

std::pair<int, int> inferPatchGrid(size_t token_count, int input_h, int input_w)
{
    if (token_count == 0 || token_count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid patch token count: %zu", token_count);
    }

    const auto square = static_cast<int>(std::llround(std::sqrt(static_cast<double>(token_count))));
    if (static_cast<size_t>(square) * static_cast<size_t>(square) == token_count)
    {
        return {square, square};
    }

    const double target_aspect
        = input_h > 0 && input_w > 0 ? static_cast<double>(input_w) / static_cast<double>(input_h) : 1.0;
    int    best_h     = 0;
    int    best_w     = 0;
    double best_error = std::numeric_limits<double>::infinity();
    for (size_t h = 1; h * h <= token_count; ++h)
    {
        if (token_count % h != 0)
        {
            continue;
        }
        const size_t w = token_count / h;
        const double aspect_a = static_cast<double>(w) / static_cast<double>(h);
        const double error_a  = std::abs(aspect_a - target_aspect);
        if (error_a < best_error)
        {
            best_error = error_a;
            best_h     = static_cast<int>(h);
            best_w     = static_cast<int>(w);
        }

        const double aspect_b = static_cast<double>(h) / static_cast<double>(w);
        const double error_b  = std::abs(aspect_b - target_aspect);
        if (error_b < best_error)
        {
            best_error = error_b;
            best_h     = static_cast<int>(w);
            best_w     = static_cast<int>(h);
        }
    }

    if (best_h <= 0 || best_w <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Failed to infer patch grid from token count: %zu", token_count);
    }
    return {best_h, best_w};
}

PatchFeatureMatrix makePatchFeatureMatrix(const OutputTensor &tensor, const irt::Shape &input_dims)
{
    if (tensor.data_type != irt::TensorDataType::F32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "PCA visualization expects float32 feature tensor, got %s",
                             irt::model::dataTypeToString(tensor.data_type).c_str());
    }

    const auto *data = reinterpret_cast<const float *>(tensor.host_bytes.data());
    if (data == nullptr || tensor.host_bytes.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature tensor is empty");
    }

    PatchFeatureMatrix matrix;
    const int input_h = input_dims.rank() == 4 ? static_cast<int>(input_dims[2]) : 0;
    const int input_w = input_dims.rank() == 4 ? static_cast<int>(input_dims[3]) : 0;

    if (tensor.dims.rank() == 3)
    {
        if (tensor.dims[0] != 1 || tensor.dims[1] <= 0 || tensor.dims[2] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected feature shape [1, tokens, channels], got %s",
                                 dimsToCsv(tensor.dims).c_str());
        }
        const int tokens   = static_cast<int>(tensor.dims[1]);
        const int channels = static_cast<int>(tensor.dims[2]);
        auto [grid_h, grid_w] = inferPatchGrid(static_cast<size_t>(tokens), input_h, input_w);
        matrix.features       = cv::Mat(tokens, channels, CV_32F, const_cast<float *>(data)).clone();
        matrix.grid_h         = grid_h;
        matrix.grid_w         = grid_w;
        matrix.channels       = channels;
        return matrix;
    }

    if (tensor.dims.rank() == 2)
    {
        if (tensor.dims[0] <= 0 || tensor.dims[1] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected feature shape [tokens, channels], got %s",
                                 dimsToCsv(tensor.dims).c_str());
        }
        const int tokens   = static_cast<int>(tensor.dims[0]);
        const int channels = static_cast<int>(tensor.dims[1]);
        auto [grid_h, grid_w] = inferPatchGrid(static_cast<size_t>(tokens), input_h, input_w);
        matrix.features       = cv::Mat(tokens, channels, CV_32F, const_cast<float *>(data)).clone();
        matrix.grid_h         = grid_h;
        matrix.grid_w         = grid_w;
        matrix.channels       = channels;
        return matrix;
    }

    if (tensor.dims.rank() == 4)
    {
        if (tensor.dims[0] != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Only single-image feature maps are supported, got %s",
                                 dimsToCsv(tensor.dims).c_str());
        }

        const int d1 = static_cast<int>(tensor.dims[1]);
        const int d2 = static_cast<int>(tensor.dims[2]);
        const int d3 = static_cast<int>(tensor.dims[3]);
        if (d1 <= 0 || d2 <= 0 || d3 <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid 4D feature shape: %s",
                                 dimsToCsv(tensor.dims).c_str());
        }

        const bool nchw = d1 >= d2 && d1 >= d3;
        const int  channels = nchw ? d1 : d3;
        const int  grid_h   = nchw ? d2 : d1;
        const int  grid_w   = nchw ? d3 : d2;
        matrix.features     = cv::Mat(grid_h * grid_w, channels, CV_32F);
        for (int y = 0; y < grid_h; ++y)
        {
            for (int x = 0; x < grid_w; ++x)
            {
                auto *row = matrix.features.ptr<float>(y * grid_w + x);
                for (int c = 0; c < channels; ++c)
                {
                    const size_t src = nchw
                        ? (static_cast<size_t>(c) * static_cast<size_t>(grid_h) + static_cast<size_t>(y))
                              * static_cast<size_t>(grid_w)
                              + static_cast<size_t>(x)
                        : (static_cast<size_t>(y) * static_cast<size_t>(grid_w) + static_cast<size_t>(x))
                              * static_cast<size_t>(channels)
                              + static_cast<size_t>(c);
                    row[c] = data[src];
                }
            }
        }
        matrix.grid_h   = grid_h;
        matrix.grid_w   = grid_w;
        matrix.channels = channels;
        return matrix;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "Unsupported patch-token feature shape %s. Use x_norm_patchtokens.",
                         dimsToCsv(tensor.dims).c_str());
}

cv::Mat projectPca(const cv::Mat &features, int components)
{
    if (features.type() != CV_32F || features.channels() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "PCA input must be a CV_32F matrix");
    }
    if (features.rows < components || features.cols < components)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "PCA requires rows and channels >= %d, got rows=%d channels=%d", components,
                             features.rows, features.cols);
    }

    const cv::PCA pca(features, cv::Mat(), cv::PCA::DATA_AS_ROW, components);
    cv::Mat       projected;
    pca.project(features, projected);
    if (projected.type() != CV_32F)
    {
        projected.convertTo(projected, CV_32F);
    }
    return projected;
}

std::vector<unsigned char> makeBackgroundMask(const cv::Mat &features, float threshold)
{
    const cv::Mat projected = projectPca(features, 1);
    double        min_value = 0.0;
    double        max_value = 0.0;
    cv::minMaxLoc(projected, &min_value, &max_value);

    std::vector<unsigned char> mask(static_cast<size_t>(features.rows), 0);
    const double               range = max_value - min_value;
    if (std::abs(range) < 1e-12)
    {
        return mask;
    }

    for (int row = 0; row < projected.rows; ++row)
    {
        const double normalized = (static_cast<double>(projected.at<float>(row, 0)) - min_value) / range;
        mask[static_cast<size_t>(row)] = normalized > static_cast<double>(threshold) ? 255U : 0U;
    }
    return mask;
}

cv::Mat zeroBackgroundFeatures(const cv::Mat &features, const std::vector<unsigned char> &background_mask)
{
    if (background_mask.size() != static_cast<size_t>(features.rows))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Background mask size does not match feature rows");
    }

    cv::Mat zeroed = features.clone();
    for (int row = 0; row < zeroed.rows; ++row)
    {
        if (background_mask[static_cast<size_t>(row)] != 0U)
        {
            zeroed.row(row).setTo(0);
        }
    }
    return zeroed;
}

unsigned char scaleProjectedValue(float value, double min_value, double max_value)
{
    const double range = max_value - min_value;
    if (std::abs(range) < 1e-12 || !std::isfinite(static_cast<double>(value)))
    {
        return 0U;
    }
    const double normalized = (static_cast<double>(value) - min_value) / range;
    const double scaled     = std::clamp(normalized, 0.0, 1.0) * 255.0;
    return static_cast<unsigned char>(std::lround(scaled));
}

cv::Mat projectedToGridImage(const cv::Mat                         &projected,
                             int                                    grid_h,
                             int                                    grid_w,
                             const std::vector<unsigned char>       *black_mask)
{
    if (projected.type() != CV_32F || projected.cols != 3 || projected.rows != grid_h * grid_w)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Projected PCA shape must be [grid_h * grid_w, 3], got %dx%d for grid %dx%d",
                             projected.rows, projected.cols, grid_h, grid_w);
    }
    if (black_mask != nullptr && black_mask->size() != static_cast<size_t>(projected.rows))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Black mask size does not match PCA rows");
    }

    std::array<double, 3> min_values{};
    std::array<double, 3> max_values{};
    for (int c = 0; c < 3; ++c)
    {
        cv::minMaxLoc(projected.col(c), &min_values[static_cast<size_t>(c)], &max_values[static_cast<size_t>(c)]);
    }

    cv::Mat grid(grid_h, grid_w, CV_8UC3);
    for (int row = 0; row < projected.rows; ++row)
    {
        cv::Vec3b pixel{0, 0, 0};
        if (black_mask == nullptr || (*black_mask)[static_cast<size_t>(row)] == 0U)
        {
            const auto *src = projected.ptr<float>(row);
            pixel[0] = scaleProjectedValue(src[0], min_values[0], max_values[0]);
            pixel[1] = scaleProjectedValue(src[1], min_values[1], max_values[1]);
            pixel[2] = scaleProjectedValue(src[2], min_values[2], max_values[2]);
        }
        grid.at<cv::Vec3b>(row / grid_w, row % grid_w) = pixel;
    }
    return grid;
}

cv::Mat maskToGridImage(const std::vector<unsigned char> &mask, int grid_h, int grid_w)
{
    if (mask.size() != static_cast<size_t>(grid_h * grid_w))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Mask size does not match patch grid");
    }

    cv::Mat image(grid_h, grid_w, CV_8UC1);
    for (int row = 0; row < grid_h * grid_w; ++row)
    {
        image.at<unsigned char>(row / grid_w, row % grid_w) = mask[static_cast<size_t>(row)];
    }
    return image;
}

cv::Mat resizeToOriginal(const cv::Mat &grid, cv::Size original_size)
{
    cv::Mat resized;
    cv::resize(grid, resized, original_size, 0.0, 0.0, cv::INTER_NEAREST);
    return resized;
}

void saveImage(const fs::path &path, const cv::Mat &image)
{
    if (!cv::imwrite(path.string(), image))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write image: %s",
                             path.string().c_str());
    }
}

void writeManifest(const fs::path            &path,
                   const Arguments          &args,
                   const fs::path           &weights_path,
                   const fs::path           &image_path,
                   const irt::Shape          &input_dims,
                   const OutputTensor       &feature,
                   const PatchFeatureMatrix &matrix,
                   const fs::path           &direct_path,
                   const fs::path           &background_path,
                   const fs::path           &mask_path)
{
    std::ofstream manifest(path);
    if (!manifest)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file: %s",
                             path.string().c_str());
    }

    manifest << "version=1\n";
    manifest << "sample=dino_pca_visualize\n";
    manifest << "runtime=" << args.runtime.toString() << "\n";
    manifest << "model_name=" << args.model_name << "\n";
    manifest << "weights_file=" << fs::absolute(weights_path).generic_string() << "\n";
    manifest << "image_path=" << fs::absolute(image_path).generic_string() << "\n";
    manifest << "feature_tensor_name=" << feature.name << "\n";
    manifest << "feature_tensor_dims=" << dimsToCsv(feature.dims) << "\n";
    manifest << "input_dims=" << dimsToCsv(input_dims) << "\n";
    manifest << "pca_rows=" << matrix.features.rows << "\n";
    manifest << "pca_channels=" << matrix.channels << "\n";
    manifest << "patch_grid=" << matrix.grid_h << "x" << matrix.grid_w << "\n";
    manifest << "threshold=" << args.threshold << "\n";
    manifest << "direct_pca3=" << direct_path.filename().generic_string() << "\n";
    manifest << "background_zeroed_pca3=" << background_path.filename().generic_string() << "\n";
    manifest << "background_mask=" << mask_path.filename().generic_string() << "\n";
}

size_t findFeatureOutputIndex(const std::vector<OutputTensor> &outputs, const std::string &feature_name)
{
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i].name == feature_name)
        {
            return i;
        }
    }
    if (outputs.size() == 1)
    {
        return 0;
    }

    std::string names;
    for (const auto &output : outputs)
    {
        if (!names.empty())
        {
            names += ", ";
        }
        names += output.name;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature output %s was not found. Outputs: %s",
                         feature_name.c_str(), names.c_str());
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const auto total_start = Clock::now();
        const Arguments args  = parseArguments(argc, argv);

        if (!irt::model::isSupportedModel(args.model_name))
        {
            std::cerr << "Unsupported model: " << args.model_name << std::endl;
            return -1;
        }

        const fs::path project_root
            = irt::util::findProjectRoot(argv[0], {irt::model::ImageNetUtil::kDefaultImagePath}, __FILE__);
        const fs::path image_path   = resolveImagePath(project_root, args.image_path);
        const fs::path weights_path = resolveWeightsPath(project_root, args);
        const fs::path output_dir   = resolveOutputDir(args.output_dir);

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames({args.feature_name});
        config->setOutputTensorNames({args.feature_name});
        config->setFeatureOnly(true);
        config->setRuntime(args.runtime);

        const std::string runtime_model_name
            = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT ? args.model_name : "onnx";
        auto model = irt::model::CreateModel(runtime_model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 runtime_model_name.c_str());
        }

        model->setLogLevel(irt::model::LogLevel::Info);
        std::cout << "Building or loading feature-only model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(weights_path.string());
        const auto build_end = Clock::now();

        const auto image_load_start = Clock::now();
        cv::Mat    image            = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }
        const auto image_load_end = Clock::now();

        const auto input_tensor_names = model->ioTensorNames(irt::TensorIOMode::Input);
        if (input_tensor_names.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "DINO PCA sample expects exactly one input tensor, got %zu",
                                 input_tensor_names.size());
        }

        const std::string &input_name = input_tensor_names.front();
        irt::Shape          input_dims = model->tensorShape(input_name);
        if (input_dims.rank() != 4 || input_dims[1] != 3 || input_dims[2] <= 0 || input_dims[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "DINO PCA sample expects input shape Nx3xHxW, got %s",
                                 dimsToCsv(input_dims).c_str());
        }
        input_dims[0] = 1;
        model->setTensorShape(input_name, input_dims);

        const auto preprocess_start = Clock::now();
        const auto input_data       = preprocessImage(image, input_dims);
        const auto preprocess_end   = Clock::now();

        const bool uses_tensorrt = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
        const cudaStream_t stream = uses_tensorrt ? reinterpret_cast<cudaStream_t>(model->resolveExecutionStream()) : nullptr;

        DeviceBuffer d_input;
        if (uses_tensorrt)
        {
            d_input = DeviceBuffer(input_data.size(), irt::TensorDataType::F32);
        }

        const auto output_names = model->ioTensorNames(irt::TensorIOMode::Output);
        if (output_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model has no feature outputs");
        }

        std::vector<OutputTensor> outputs;
        outputs.reserve(output_names.size());
        for (const auto &output_name : output_names)
        {
            OutputTensor tensor;
            tensor.name          = output_name;
            tensor.dims          = model->tensorShape(output_name);
            tensor.data_type     = model->tensorDataType(output_name);
            tensor.element_count = elementCount(tensor.dims);
            tensor.num_bytes     = tensor.element_count * elementSize(tensor.data_type);
            tensor.host_bytes.resize(tensor.num_bytes);
            if (uses_tensorrt)
            {
                tensor.device = DeviceBuffer(tensor.element_count, tensor.data_type);
            }
            else
            {
                const size_t aligned_words
                    = (tensor.num_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
                tensor.graph_storage.resize(aligned_words);
            }
            outputs.push_back(std::move(tensor));
        }

        std::vector<irt::BufferView> buffers;
        buffers.reserve(1 + outputs.size());
        buffers.push_back(uses_tensorrt
                              ? irt::BufferView::fromBytes(d_input.data(), d_input.sizeBytes(), irt::MemoryKind::DEVICE,
                                                           input_name)
                              : irt::BufferView::fromBytes(const_cast<float *>(input_data.data()),
                                                           input_data.size() * sizeof(float), irt::MemoryKind::HOST,
                                                           input_name));
        for (auto &output : outputs)
        {
            buffers.push_back(uses_tensorrt
                              ? irt::BufferView::fromBytes(output.device.data(), output.device.sizeBytes(),
                                                               irt::MemoryKind::DEVICE, output.name)
                              : irt::BufferView::fromBytes(output.graph_storage.data(), output.num_bytes,
                                                               irt::MemoryKind::HOST, output.name));
        }

        auto run_feature_once = [&]() -> IterationTiming
        {
            IterationTiming timing;

            const auto h2d_start = Clock::now();
            if (uses_tensorrt)
            {
                checkCuda(cudaMemcpyAsync(d_input.data(), input_data.data(), input_data.size() * sizeof(float),
                                          cudaMemcpyHostToDevice, stream),
                          "cudaMemcpyAsync(H2D input)");
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(H2D input)");
            }
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->forwardFeatures(buffers, reinterpret_cast<std::uintptr_t>(stream), true);
            if (uses_tensorrt)
            {
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature forward)");
            }
            const auto inference_end = Clock::now();
            timing.inference_ms      = elapsedMs(inference_start, inference_end);

            const auto d2h_start = Clock::now();
            if (uses_tensorrt)
            {
                for (auto &output : outputs)
                {
                    checkCuda(cudaMemcpyAsync(output.host_bytes.data(), output.device.data(), output.num_bytes,
                                              cudaMemcpyDeviceToHost, stream),
                              "cudaMemcpyAsync(D2H feature)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H feature)");
            }
            else
            {
                for (auto &output : outputs)
                {
                    std::memcpy(output.host_bytes.data(), output.graph_storage.data(), output.num_bytes);
                }
            }
            const auto d2h_end = Clock::now();
            timing.d2h_ms      = elapsedMs(d2h_start, d2h_end);
            return timing;
        };

        std::cout << "Running feature forward warmup=" << args.warmup << ", repeat=" << args.repeat << "..."
                  << std::endl;
        for (int i = 0; i < args.warmup; ++i)
        {
            (void)run_feature_once();
        }

        std::vector<double> h2d_times_ms;
        std::vector<double> inference_times_ms;
        std::vector<double> d2h_times_ms;
        std::vector<double> end_to_end_times_ms;
        h2d_times_ms.reserve(static_cast<size_t>(args.repeat));
        inference_times_ms.reserve(static_cast<size_t>(args.repeat));
        d2h_times_ms.reserve(static_cast<size_t>(args.repeat));
        end_to_end_times_ms.reserve(static_cast<size_t>(args.repeat));

        const auto infer_loop_start = Clock::now();
        for (int i = 0; i < args.repeat; ++i)
        {
            const auto timing = run_feature_once();
            h2d_times_ms.push_back(timing.h2d_ms);
            inference_times_ms.push_back(timing.inference_ms);
            d2h_times_ms.push_back(timing.d2h_ms);
            end_to_end_times_ms.push_back(timing.totalMs());
        }
        const auto infer_loop_end = Clock::now();

        const auto matrix_start = Clock::now();
        const size_t feature_index = findFeatureOutputIndex(outputs, args.feature_name);
        const auto  &feature       = outputs[feature_index];
        auto         matrix        = makePatchFeatureMatrix(feature, input_dims);
        const auto   matrix_end    = Clock::now();

        const auto direct_pca_start = Clock::now();
        const auto direct_projected = projectPca(matrix.features, 3);
        const auto direct_grid      = projectedToGridImage(direct_projected, matrix.grid_h, matrix.grid_w, nullptr);
        const auto direct_pca_end   = Clock::now();

        const auto mask_start      = Clock::now();
        const auto background_mask = makeBackgroundMask(matrix.features, args.threshold);
        const auto mask_end        = Clock::now();

        const auto background_pca_start = Clock::now();
        const auto zeroed_features      = zeroBackgroundFeatures(matrix.features, background_mask);
        const auto background_projected = projectPca(zeroed_features, 3);
        const auto background_grid
            = projectedToGridImage(background_projected, matrix.grid_h, matrix.grid_w, &background_mask);
        const auto background_pca_end = Clock::now();

        const auto save_start = Clock::now();
        fs::create_directories(output_dir);
        const std::string image_stem   = sanitizeFileStem(image_path.stem().string(), "image");
        const fs::path    direct_path  = output_dir / (image_stem + "_pca3.png");
        const fs::path    bg_path      = output_dir / (image_stem + "_background_zeroed_pca3.png");
        const fs::path    mask_path    = output_dir / (image_stem + "_background_mask.png");
        const fs::path    manifest_path = output_dir / "manifest.txt";

        saveImage(direct_path, resizeToOriginal(direct_grid, image.size()));
        saveImage(bg_path, resizeToOriginal(background_grid, image.size()));
        saveImage(mask_path, resizeToOriginal(maskToGridImage(background_mask, matrix.grid_h, matrix.grid_w),
                                              image.size()));
        writeManifest(manifest_path, args, weights_path, image_path, input_dims, feature, matrix, direct_path, bg_path,
                      mask_path);
        const auto save_end = Clock::now();

        const auto total_end = Clock::now();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Saved outputs to: " << fs::absolute(output_dir).string() << '\n';
        std::cout << "  direct_pca3: " << direct_path.filename().string() << '\n';
        std::cout << "  background_zeroed_pca3: " << bg_path.filename().string() << '\n';
        std::cout << "  background_mask: " << mask_path.filename().string() << '\n';
        std::cout << "Feature: " << feature.name << " dims=[" << dimsToCsv(feature.dims) << "], PCA matrix="
                  << matrix.features.rows << "x" << matrix.channels << ", patch_grid=" << matrix.grid_h << "x"
                  << matrix.grid_w << '\n';
        std::cout << "Runtime: " << args.runtime.toString() << '\n';
        std::cout << "Timing:\n";
        std::cout << "  build_or_load: " << elapsedMs(build_start, build_end) << " ms\n";
        std::cout << "  image_load: " << elapsedMs(image_load_start, image_load_end) << " ms\n";
        std::cout << "  preprocess: " << elapsedMs(preprocess_start, preprocess_end) << " ms\n";
        printTimingStats(std::cout, "h2d", summarizeTimings(h2d_times_ms), "\n  ");
        printTimingStats(std::cout, "inference", summarizeTimings(inference_times_ms), "\n  ");
        printTimingStats(std::cout, "d2h", summarizeTimings(d2h_times_ms), "\n  ");
        printTimingStats(std::cout, "end_to_end", summarizeTimings(end_to_end_times_ms), "\n  ");
        std::cout << '\n';
        std::cout << "  timed_loop_wall: " << elapsedMs(infer_loop_start, infer_loop_end) << " ms\n";
        std::cout << "  feature_matrix: " << elapsedMs(matrix_start, matrix_end) << " ms\n";
        std::cout << "  direct_pca3: " << elapsedMs(direct_pca_start, direct_pca_end) << " ms\n";
        std::cout << "  background_mask_pca1: " << elapsedMs(mask_start, mask_end) << " ms\n";
        std::cout << "  background_zeroed_pca3: " << elapsedMs(background_pca_start, background_pca_end) << " ms\n";
        std::cout << "  save_outputs: " << elapsedMs(save_start, save_end) << " ms\n";
        std::cout << "  total: " << elapsedMs(total_start, total_end) << " ms\n";
        return 0;
    }
    catch (const HelpRequested &)
    {
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
