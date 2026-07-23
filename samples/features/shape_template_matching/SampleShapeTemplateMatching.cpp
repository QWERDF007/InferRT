/**
 * @file SampleShapeTemplateMatching.cpp
 * @brief 形状模板匹配两阶段示例程序。
 */

#include <SampleSupport.hpp>
#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ShapeTemplateMatcher.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using irt::samples::HelpRequested;

/** @brief 示例运行阶段。 */
enum class Stage
{
    Train,
    Match,
};

/** @brief 训练阶段命令行参数。 */
struct TrainingArguments
{
    fs::path              template_image; ///< 模板输入图像；可以是大图。
    fs::path              template_mask;  ///< 可选目标掩膜。
    std::vector<cv::Rect> template_rois;  ///< 可重复指定的大图裁剪区域；为空时训练整图。
    fs::path              save_templates; ///< 训练后写入的标准 YAML 文件。
    float                                     angle_begin{0.0f};
    float                                     angle_end{90.0f};
    float                                     angle_step{15.0f};
    float                                     scale_begin{1.0f};
    float                                     scale_end{1.0f};
    float                                     scale_step{1.0f};
    irt::features::ShapeTemplateMatcherConfig config;
};

/** @brief 匹配阶段命令行参数。 */
struct MatchingArguments
{
    fs::path                load_templates; ///< 训练阶段生成的标准 YAML 文件。
    fs::path                source_image;   ///< 待搜索大图。
    fs::path                search_mask;    ///< 可选搜索区域掩膜。
    fs::path                output_image{"shape_template_matching_result.png"};
    float                   threshold{-1.0f}; ///< 负数时使用模板文件中保存的阈值。
    int                     template_stride{1}; ///< 近似模式：每隔多少个模板变体扫描一次。
    int                     scan_step{0};       ///< 近似模式：0 使用模板文件配置，正数覆盖空间扫描步长。
    int                     max_parallelism{0}; ///< 匹配线程数；0 使用模板配置或自动选择。
};

/** @brief 计时控制参数。 */
struct TimingArguments
{
    int warmup{3};  ///< 不计入统计的预热次数。
    int repeat{10}; ///< 计入统计的重复次数。
};

/** @brief 示例程序命令行参数集合。 */
struct Arguments
{
    Stage             stage{Stage::Train};
    irt::features::ShapeTemplateMatcherVersion version{irt::features::ShapeTemplateMatcherVersion::V1};
    TrainingArguments training;
    MatchingArguments matching;
    TimingArguments   timing;
};

/** @brief 解析独立实现版本名称。 */
irt::features::ShapeTemplateMatcherVersion parseMatcherVersion(const std::string &text)
{
    if (text == "v0")
    {
        return irt::features::ShapeTemplateMatcherVersion::V0;
    }
    if (text == "v1")
    {
        return irt::features::ShapeTemplateMatcherVersion::V1;
    }
    if (text == "v2")
    {
        return irt::features::ShapeTemplateMatcherVersion::V2;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--version must be v0, v1, or v2");
}

/**
 * @brief 从磁盘读取图像并在失败时抛出 InferRT 异常。
 */
cv::Mat loadImage(const fs::path &path, int flags, const char *name)
{
    if (path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s path must not be empty", name);
    }
    cv::Mat image = cv::imread(path.string(), flags);
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load %s: %s", name, path.string().c_str());
    }
    return image;
}

/**
 * @brief 解析 ``x,y,width,height`` 格式的模板裁剪区域。
 */
cv::Rect parseRoi(const std::string &text)
{
    std::istringstream stream(text);
    cv::Rect           roi;
    char               comma1 = 0;
    char               comma2 = 0;
    char               comma3 = 0;
    std::string        trailing;
    if (!(stream >> roi.x >> comma1 >> roi.y >> comma2 >> roi.width >> comma3 >> roi.height) || comma1 != ','
        || comma2 != ',' || comma3 != ',' || (stream >> trailing))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "template-roi must use the format x,y,width,height");
    }
    return roi;
}

/**
 * @brief 校验裁剪区域是否完全位于输入图像内。
 */
void validateRoi(const cv::Rect &roi, const cv::Size &image_size)
{
    if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 || roi.x >= image_size.width
        || roi.y >= image_size.height || roi.width > image_size.width - roi.x || roi.height > image_size.height - roi.y)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "template-roi must be fully inside the template image");
    }
}

/** @brief 为输出文件创建父目录。 */
void createParentDirectory(const fs::path &path)
{
    if (!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
}

/**
 * @brief 输出以毫秒为单位的重复执行耗时统计。
 */
void printTimingStats(const char *operation, int warmup, const std::vector<double> &samples_ms)
{
    if (samples_ms.empty())
    {
        return;
    }

    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const double total  = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    const double mean   = total / static_cast<double>(samples_ms.size());
    const double median = sorted.size() % 2 == 0 ? (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) * 0.5
                                                 : sorted[sorted.size() / 2];
    const double squared_error      = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0,
                                                      [mean](double sum, double value)
                                                      {
                                                     const double delta = value - mean;
                                                     return sum + delta * delta;
                                                 });
    const double standard_deviation = std::sqrt(squared_error / static_cast<double>(samples_ms.size()));

    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << "timing (" << operation << "): warmup=" << warmup
           << ", repeats=" << samples_ms.size() << ", total=" << total << " ms, avg=" << mean
           << " ms, median=" << median << " ms, min=" << sorted.front() << " ms, max=" << sorted.back()
           << " ms, stddev=" << standard_deviation << " ms";
    std::cout << output.str() << std::endl;
}

/**
 * @brief 构造按运行阶段分组的命令行选项定义。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Two-stage shape-based template matching sample");

    options.add_options("Stage")("mode", "Stage to run: train or match (required)",
                                 cxxopts::value<std::string>()->default_value(""))("h,help", "Show help");

    options.add_options("Implementation")(
        "version", "Matcher implementation: v0 (scalar), v1 (AVX2), or v2 (AVX512F/BW)",
        cxxopts::value<std::string>()->default_value("v1"));

    options.add_options("Timing")("warmup", "Warmup iterations for the selected stage; excluded from timing",
                                  cxxopts::value<int>()->default_value("3"))(
        "repeat", "Timed repetitions for the selected stage", cxxopts::value<int>()->default_value("10"));

    options.add_options("Training")("template", "Training image; it may be a cropped template or a larger image",
                                    cxxopts::value<std::string>()->default_value(""))(
        "template-roi", "Repeatable crop rectangle in the training image: x,y,width,height",
        cxxopts::value<std::string>()->default_value(""))(
        "template-mask", "Optional object mask; cropped-template size is allowed only with one template-roi",
        cxxopts::value<std::string>()->default_value(""))("save-templates",
                                                          "Output standard YAML path for the trained templates",
                                                          cxxopts::value<std::string>()->default_value(""))(
        "angle-begin", "First training angle in degrees", cxxopts::value<float>()->default_value("0"))(
        "angle-end", "Last training angle in degrees", cxxopts::value<float>()->default_value("90"))(
        "angle-step", "Training angle step in degrees", cxxopts::value<float>()->default_value("15"))(
        "scale-begin", "First training scale", cxxopts::value<float>()->default_value("1"))(
        "scale-end", "Last training scale", cxxopts::value<float>()->default_value("1"))(
        "scale-step", "Training scale step", cxxopts::value<float>()->default_value("1"))(
        "features", "Maximum feature points per template", cxxopts::value<int>()->default_value("96"))(
        "weak-threshold", "Weak gradient magnitude threshold", cxxopts::value<float>()->default_value("10"))(
        "strong-threshold", "Strong template candidate magnitude threshold", cxxopts::value<float>()->default_value("20"))(
        "max-label-difference", "Circular orientation tolerance in bins [0,4]",
        cxxopts::value<int>()->default_value("1"))(
        "min-feature-distance", "Minimum template feature spacing; 0 selects an area-based default",
        cxxopts::value<float>()->default_value("0"))(
        "template-scan-step", "Exact spatial scan step stored in the template file",
        cxxopts::value<int>()->default_value("1"))(
        "train-parallelism", "v1/v2 training worker count; 0 chooses automatically, v0 always stays original serial",
        cxxopts::value<int>()->default_value("0"))(
        "gaussian-gradient", "v1 approximate preprocessing: apply 5x5 GaussianBlur before Sobel (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "orientation-histogram", "v1 approximate preprocessing: apply 3x3 orientation majority filter (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "edge-nms", "v1 approximate preprocessing: suppress non-maximum edge responses (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "edge-connectivity", "v1 approximate preprocessing: keep weak edges connected to strong seeds (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "polarity-invariant", "v1 approximate preprocessing: fold opposite gradient polarities (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "spatial-spread", "v1 approximate preprocessing: spread labels to nearby weak pixels (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "reuse-base-features", "v1 approximate training: reuse base features for angle/scale variants (default off)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "max-results", "Matching result limit saved in the template file; 0 means unlimited", cxxopts::value<int>()->default_value("0"))(
        "nms", "Matching NMS IoU threshold saved in the template file; negative disables NMS",
        cxxopts::value<float>()->default_value("0.3"));

    options.add_options("Matching")("load-templates", "Input standard YAML template file",
                                    cxxopts::value<std::string>()->default_value(""))(
        "source", "Source image to search", cxxopts::value<std::string>()->default_value(""))(
        "search-mask", "Optional source-image search mask", cxxopts::value<std::string>()->default_value(""))(
        "threshold,t", "Match threshold in [0, 100]; negative uses the value saved in the template file",
        cxxopts::value<float>()->default_value("-1"))(
        "template-stride", "Approximate mode: search every Nth template variant; 1 scans all variants exactly",
        cxxopts::value<int>()->default_value("1"))(
        "scan-step", "Approximate mode: override spatial scan step; 0 uses the template-file setting",
        cxxopts::value<int>()->default_value("0"))(
        "parallelism", "Match worker count; 0 uses the template setting or automatic selection",
        cxxopts::value<int>()->default_value("0"))(
        "output,o", "Output visualization image",
        cxxopts::value<std::string>()->default_value("shape_template_matching_result.png"));
    return options;
}

/**
 * @brief 解析并校验示例程序命令行参数。
 */
Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        throw HelpRequested{};
    }

    Arguments args;
    args.version       = parseMatcherVersion(result["version"].as<std::string>());
    args.timing.warmup = result["warmup"].as<int>();
    args.timing.repeat = result["repeat"].as<int>();
    if (args.timing.warmup < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--warmup must be non-negative");
    }
    if (args.timing.repeat <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--repeat must be positive");
    }

    const auto mode = result["mode"].as<std::string>();
    if (mode == "train")
    {
        args.stage                           = Stage::Train;
        auto &training                       = args.training;
        training.template_image              = result["template"].as<std::string>();
        training.template_mask               = result["template-mask"].as<std::string>();
        training.save_templates              = result["save-templates"].as<std::string>();
        training.angle_begin                 = result["angle-begin"].as<float>();
        training.angle_end                   = result["angle-end"].as<float>();
        training.angle_step                  = result["angle-step"].as<float>();
        training.scale_begin                 = result["scale-begin"].as<float>();
        training.scale_end                   = result["scale-end"].as<float>();
        training.scale_step                  = result["scale-step"].as<float>();
        training.config.num_features         = result["features"].as<int>();
        training.config.min_features         = std::min(8, std::max(1, training.config.num_features));
        training.config.weak_threshold       = result["weak-threshold"].as<float>();
        training.config.strong_threshold     = result["strong-threshold"].as<float>();
        training.config.match_threshold      = 85.0f;
        training.config.max_results          = result["max-results"].as<int>();
        training.config.nms_threshold        = result["nms"].as<float>();
        training.config.max_label_difference = result["max-label-difference"].as<int>();
        training.config.min_feature_distance = result["min-feature-distance"].as<float>();
        training.config.scan_step            = result["template-scan-step"].as<int>();
        training.config.max_training_parallelism = result["train-parallelism"].as<int>();
        training.config.use_gaussian_gradient = result["gaussian-gradient"].as<bool>();
        training.config.use_orientation_histogram = result["orientation-histogram"].as<bool>();
        training.config.use_edge_nms = result["edge-nms"].as<bool>();
        training.config.use_edge_connectivity = result["edge-connectivity"].as<bool>();
        training.config.use_polarity_invariant = result["polarity-invariant"].as<bool>();
        training.config.use_spatial_spread = result["spatial-spread"].as<bool>();
        training.config.reuse_base_features_for_variants = result["reuse-base-features"].as<bool>();

        if ((training.config.use_gaussian_gradient || training.config.use_orientation_histogram
             || training.config.use_edge_nms || training.config.use_edge_connectivity
             || training.config.use_polarity_invariant || training.config.use_spatial_spread
             || training.config.reuse_base_features_for_variants)
            && args.version != irt::features::ShapeTemplateMatcherVersion::V1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "v1 approximate preprocessing/variant-reuse options require --version v1");
        }

        // cxxopts 的 vector 值以逗号分隔，而 ROI 本身也使用逗号。保留字符串选项，
        // 再从原始参数顺序中收集每次出现的 --template-roi，避免将一个 ROI 拆成四项。
        for (const auto &argument : result.arguments())
        {
            if (argument.key() == "template-roi")
            {
                training.template_rois.push_back(parseRoi(argument.value()));
            }
        }
        if (training.template_image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--template is required in train mode");
        }
        if (training.save_templates.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--save-templates is required in train mode");
        }
        if (training.config.max_training_parallelism < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "--train-parallelism must be non-negative");
        }
    }
    else if (mode == "match")
    {
        args.stage              = Stage::Match;
        auto &matching          = args.matching;
        matching.load_templates = result["load-templates"].as<std::string>();
        matching.source_image   = result["source"].as<std::string>();
        matching.search_mask    = result["search-mask"].as<std::string>();
        matching.threshold      = result["threshold"].as<float>();
        matching.output_image   = result["output"].as<std::string>();
        matching.template_stride = result["template-stride"].as<int>();
        matching.scan_step       = result["scan-step"].as<int>();
        matching.max_parallelism = result["parallelism"].as<int>();
        if (matching.load_templates.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--load-templates is required in match mode");
        }
        if (matching.source_image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--source is required in match mode");
        }
        if (!std::isfinite(matching.threshold) || matching.threshold > 100.0f)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "--threshold must be finite and no greater than 100");
        }
        if (matching.template_stride <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--template-stride must be positive");
        }
        if (matching.scan_step < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--scan-step must be non-negative");
        }
        if (matching.max_parallelism < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--parallelism must be non-negative");
        }
    }
    else
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--mode must be either train or match");
    }
    return args;
}

/**
 * @brief 将匹配框和分数绘制到输出图像。
 */
void drawMatches(cv::Mat &image, const std::vector<irt::features::ShapeTemplateMatch> &matches)
{
    if (image.channels() == 1)
    {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }
    for (const auto &match : matches)
    {
        const cv::Rect box(match.x, match.y, match.width, match.height);
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        const std::string label = "t=" + std::to_string(match.template_id) + " "
                                + std::to_string(static_cast<int>(std::round(match.similarity))) + "% a="
                                + std::to_string(static_cast<int>(std::round(match.angle_degrees)));
        cv::putText(image, label, cv::Point(match.x, std::max(12, match.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
}

/** @brief 执行模板训练、统计核心训练耗时并保存模板文件。 */
void runTraining(const TrainingArguments &args, const TimingArguments &timing,
                 irt::features::ShapeTemplateMatcherVersion version)
{
    struct TemplateRoiInput
    {
        cv::Rect roi;
        cv::Mat  image;
        cv::Mat  mask;
    };

    const cv::Mat full_image = loadImage(args.template_image, cv::IMREAD_UNCHANGED, "template image");
    std::vector<cv::Rect> rois = args.template_rois;
    if (rois.empty())
    {
        rois.emplace_back(0, 0, full_image.cols, full_image.rows);
    }
    for (const auto &roi : rois)
    {
        validateRoi(roi, full_image.size());
    }

    cv::Mat mask;
    if (!args.template_mask.empty())
    {
        mask = loadImage(args.template_mask, cv::IMREAD_GRAYSCALE, "template mask");
        if (mask.size() != full_image.size() && (rois.size() != 1 || mask.size() != rois.front().size()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "template mask must match the full training image; a cropped mask is allowed only "
                                 "when exactly one template-roi is provided");
        }
    }

    std::vector<TemplateRoiInput> templates;
    templates.reserve(rois.size());
    for (const auto &roi : rois)
    {
        TemplateRoiInput input;
        input.roi   = roi;
        input.image = full_image(roi).clone();
        if (!mask.empty())
        {
            input.mask = mask.size() == full_image.size() ? mask(roi).clone() : mask;
        }
        templates.push_back(std::move(input));
    }

    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(
        args.angle_begin, args.angle_end, args.angle_step, args.scale_begin, args.scale_end, args.scale_step);

    std::vector<irt::features::ShapeTemplateTrainingInput> training_inputs;
    training_inputs.reserve(templates.size());
    for (const auto &input : templates)
    {
        training_inputs.push_back(irt::features::ShapeTemplateTrainingInput{input.image, input.mask});
    }
    const bool use_batch_training = training_inputs.size() > 1;

    const auto addAllTemplateVariants = [&training_inputs, &variants, use_batch_training](
                                            irt::features::IShapeTemplateMatcher &matcher)
    {
        if (!use_batch_training)
        {
            const auto &input = training_inputs.front();
            return matcher.addTemplateVariants(input.image, input.object_mask, variants);
        }

        const auto ids_by_input = matcher.addTemplateVariantsBatch(training_inputs, variants);
        std::vector<int> ids;
        ids.reserve(training_inputs.size() * variants.size());
        for (const auto &input_ids : ids_by_input)
        {
            ids.insert(ids.end(), input_ids.begin(), input_ids.end());
        }
        return ids;
    };

    for (int iteration = 0; iteration < timing.warmup; ++iteration)
    {
        auto warmup_matcher = irt::features::createShapeTemplateMatcher(version, args.config);
        (void)addAllTemplateVariants(*warmup_matcher);
    }

    std::unique_ptr<irt::features::IShapeTemplateMatcher> trained_matcher;
    std::vector<int>                                    template_ids;
    std::vector<double>                                 samples_ms;
    samples_ms.reserve(static_cast<size_t>(timing.repeat));
    for (int iteration = 0; iteration < timing.repeat; ++iteration)
    {
        auto       matcher = irt::features::createShapeTemplateMatcher(version, args.config);
        const auto start   = std::chrono::steady_clock::now();
        auto ids = addAllTemplateVariants(*matcher);
        const auto stop = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        if (iteration == timing.repeat - 1)
        {
            trained_matcher = std::move(matcher);
            template_ids    = std::move(ids);
        }
    }

    createParentDirectory(args.save_templates);
    trained_matcher->save(args.save_templates);

    std::cout << "stage: train" << std::endl;
    std::cout << "version: " << irt::features::shapeTemplateMatcherVersionName(version) << std::endl;
    std::cout << "template input: " << fs::absolute(args.template_image).string() << std::endl;
    std::cout << "template rois: " << templates.size() << std::endl;
    for (size_t index = 0; index < templates.size(); ++index)
    {
        const auto &roi = templates[index].roi;
        std::cout << "  " << index << ": (" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height
                  << ")" << std::endl;
    }
    std::cout << "variants per roi: " << variants.size() << std::endl;
    if (version == irt::features::ShapeTemplateMatcherVersion::V0)
    {
        std::cout << "training parallelism: 1 (v0 original serial path)" << std::endl;
        std::cout << "training scheduler: per-roi original serial path" << std::endl;
    }
    else if (args.config.max_training_parallelism == 0)
    {
        std::cout << "training parallelism: auto (v1/v2 optimized path)" << std::endl;
        std::cout << "training scheduler: "
                  << (use_batch_training ? "global roi x variant task queue" : "single-input variant task queue")
                  << std::endl;
    }
    else
    {
        std::cout << "training parallelism: " << args.config.max_training_parallelism
                  << " (v1/v2 optimized path)" << std::endl;
        std::cout << "training scheduler: "
                  << (use_batch_training ? "global roi x variant task queue" : "single-input variant task queue")
                  << std::endl;
    }
    std::cout << "templates: " << template_ids.size() << std::endl;
    std::cout << "gaussian gradient: " << (args.config.use_gaussian_gradient ? "on" : "off") << std::endl;
    std::cout << "orientation histogram: " << (args.config.use_orientation_histogram ? "on" : "off") << std::endl;
    std::cout << "edge NMS: " << (args.config.use_edge_nms ? "on" : "off") << std::endl;
    std::cout << "edge connectivity: " << (args.config.use_edge_connectivity ? "on" : "off") << std::endl;
    std::cout << "polarity invariant: " << (args.config.use_polarity_invariant ? "on" : "off") << std::endl;
    std::cout << "spatial spread: " << (args.config.use_spatial_spread ? "on" : "off") << std::endl;
    std::cout << "reuse base features: " << (args.config.reuse_base_features_for_variants ? "on" : "off")
              << std::endl;
    std::cout << "template scan step: " << args.config.scan_step << std::endl;
    std::cout << "template file: " << fs::absolute(args.save_templates).string() << std::endl;
    printTimingStats(use_batch_training ? "train/addTemplateVariantsBatch" : "train/addTemplateVariants",
                     timing.warmup, samples_ms);
}

/** @brief 加载模板文件，统计匹配耗时并保存可视化结果。 */
void runMatching(const MatchingArguments &args, const TimingArguments &timing,
                 irt::features::ShapeTemplateMatcherVersion version)
{
    auto matcher = irt::features::createShapeTemplateMatcher(version);
    matcher->load(args.load_templates);

    cv::Mat source = loadImage(args.source_image, cv::IMREAD_UNCHANGED, "source image");
    cv::Mat search_mask;
    if (!args.search_mask.empty())
    {
        search_mask = loadImage(args.search_mask, cv::IMREAD_GRAYSCALE, "search mask");
    }

    irt::features::ShapeTemplateMatchOptions match_options;
    match_options.template_stride = args.template_stride;
    match_options.scan_step       = args.scan_step;
    match_options.max_parallelism = args.max_parallelism;

    for (int iteration = 0; iteration < timing.warmup; ++iteration)
    {
        (void)matcher->match(source, args.threshold, search_mask, match_options);
    }

    std::vector<irt::features::ShapeTemplateMatch> matches;
    std::vector<double>                            samples_ms;
    samples_ms.reserve(static_cast<size_t>(timing.repeat));
    for (int iteration = 0; iteration < timing.repeat; ++iteration)
    {
        const auto start           = std::chrono::steady_clock::now();
        auto current_matches = matcher->match(source, args.threshold, search_mask, match_options);
        const auto stop            = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        if (iteration == timing.repeat - 1)
        {
            matches = std::move(current_matches);
        }
    }

    drawMatches(source, matches);
    createParentDirectory(args.output_image);
    if (!cv::imwrite(args.output_image.string(), source))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                             args.output_image.string().c_str());
    }

    std::cout << "stage: match" << std::endl;
    std::cout << "version: " << irt::features::shapeTemplateMatcherVersionName(version) << std::endl;
    std::cout << "templates: " << matcher->numTemplates() << std::endl;
    std::cout << "gaussian gradient: " << (matcher->config().use_gaussian_gradient ? "on" : "off") << std::endl;
    std::cout << "orientation histogram: " << (matcher->config().use_orientation_histogram ? "on" : "off") << std::endl;
    std::cout << "edge NMS: " << (matcher->config().use_edge_nms ? "on" : "off") << std::endl;
    std::cout << "edge connectivity: " << (matcher->config().use_edge_connectivity ? "on" : "off") << std::endl;
    std::cout << "polarity invariant: " << (matcher->config().use_polarity_invariant ? "on" : "off") << std::endl;
    std::cout << "spatial spread: " << (matcher->config().use_spatial_spread ? "on" : "off") << std::endl;
    std::cout << "reuse base features: " << (matcher->config().reuse_base_features_for_variants ? "on" : "off")
              << std::endl;
    std::cout << "template stride: " << match_options.template_stride << std::endl;
    std::cout << "scan step: " << (match_options.scan_step > 0 ? match_options.scan_step : matcher->config().scan_step)
              << (match_options.scan_step > 0 ? " (override)" : " (template config)") << std::endl;
    std::cout << "parallelism: " << (match_options.max_parallelism > 0 ? match_options.max_parallelism
                                                                        : matcher->config().max_parallelism)
              << (match_options.max_parallelism > 0 ? " (override; 0=auto)" : " (template config; 0=auto)")
              << std::endl;
    std::cout << "matches: " << matches.size() << std::endl;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        const auto &match = matches[i];
        std::cout << (i + 1) << ". template=" << match.template_id << " score=" << match.similarity << " box=("
                  << match.x << "," << match.y << "," << match.width << "," << match.height << ") angle="
                  << match.angle_degrees << " scale=" << match.scale << std::endl;
    }
    std::cout << "output: " << fs::absolute(args.output_image).string() << std::endl;
    printTimingStats("match/IShapeTemplateMatcher::match", timing.warmup, samples_ms);
}

} // namespace

/**
 * @brief 根据 ``--mode`` 执行模板训练或模板匹配。
 */
int main(int argc, char *argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);
        if (args.stage == Stage::Train)
        {
            runTraining(args.training, args.timing, args.version);
        }
        else
        {
            runMatching(args.matching, args.timing, args.version);
        }
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
