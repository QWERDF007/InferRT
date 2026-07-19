/**
 * @file SampleShapeTemplateMatching.cpp
 * @brief 形状模板匹配示例程序。
 */

#include <SampleSupport.hpp>

#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using irt::samples::HelpRequested;

/**
 * @brief 示例程序命令行参数集合。
 */
struct Arguments
{
    fs::path template_image; ///< 模板图像路径；为空时使用合成模板。
    fs::path source_image;   ///< 待匹配源图路径；为空时使用合成源图。
    fs::path mask_image;     ///< 可选模板掩膜路径。
    fs::path load_templates; ///< 已训练模板文件路径。
    fs::path save_templates; ///< 训练后保存模板文件路径。
    fs::path output_image{"shape_template_matching_result.png"}; ///< 可视化输出路径。
    std::string class_id{"part"};                                ///< 模板类别 ID。
    float threshold{85.0f};                                      ///< 匹配阈值。
    float angle_begin{0.0f};                                     ///< 起始训练角度。
    float angle_end{90.0f};                                      ///< 结束训练角度。
    float angle_step{15.0f};                                     ///< 训练角度步长。
    float scale_begin{1.0f};                                     ///< 起始训练尺度。
    float scale_end{1.0f};                                       ///< 结束训练尺度。
    float scale_step{1.0f};                                      ///< 训练尺度步长。
    int   max_results{20};                                       ///< 最大输出匹配数量。
    irt::features::ShapeTemplateMatcherConfig config;            ///< 模板匹配器配置。
};

/**
 * @brief 生成默认 L 形模板，用于无参数运行示例。
 */
cv::Mat makeLShape(int size = 64)
{
    cv::Mat image(size, size, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Rect(12, 12, 38, 9), cv::Scalar(255), cv::FILLED);
    cv::rectangle(image, cv::Rect(12, 12, 9, 38), cv::Scalar(255), cv::FILLED);
    return image;
}

/**
 * @brief 生成包含一个旋转目标的默认合成源图。
 */
cv::Mat makeSyntheticScene(const cv::Mat &object)
{
    const auto rotated = irt::features::ShapeTemplateMatcher::transform(
        object, irt::features::ShapeTemplateVariant{90.0f, 1.0f});
    cv::Mat scene(160, 180, CV_8UC1, cv::Scalar(0));
    rotated.copyTo(scene(cv::Rect(72, 54, rotated.cols, rotated.rows)));
    return scene;
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
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load %s: %s", name,
                             path.string().c_str());
    }
    return image;
}

/**
 * @brief 构造命令行选项定义。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Shape-based template matching sample");
    options.add_options()("template", "Template image. If omitted, a synthetic template is used.",
                          cxxopts::value<std::string>()->default_value(""))(
        "source", "Source image. If omitted, a synthetic source is used.",
        cxxopts::value<std::string>()->default_value(""))(
        "mask", "Optional template mask image", cxxopts::value<std::string>()->default_value(""))(
        "load-templates", "Load trained templates from YAML/XML instead of training",
        cxxopts::value<std::string>()->default_value(""))(
        "save-templates", "Save trained templates to YAML/XML",
        cxxopts::value<std::string>()->default_value(""))(
        "output,o", "Output visualization image",
        cxxopts::value<std::string>()->default_value("shape_template_matching_result.png"))(
        "class-id,c", "Template class id", cxxopts::value<std::string>()->default_value("part"))(
        "threshold,t", "Match threshold in [0, 100]", cxxopts::value<float>()->default_value("85"))(
        "angle-begin", "First training angle in degrees", cxxopts::value<float>()->default_value("0"))(
        "angle-end", "Last training angle in degrees", cxxopts::value<float>()->default_value("90"))(
        "angle-step", "Training angle step in degrees", cxxopts::value<float>()->default_value("15"))(
        "scale-begin", "First training scale", cxxopts::value<float>()->default_value("1"))(
        "scale-end", "Last training scale", cxxopts::value<float>()->default_value("1"))(
        "scale-step", "Training scale step", cxxopts::value<float>()->default_value("1"))(
        "features", "Maximum feature points per template", cxxopts::value<int>()->default_value("96"))(
        "max-results", "Maximum displayed matches", cxxopts::value<int>()->default_value("20"))(
        "nms", "Per-class NMS IoU threshold; negative disables NMS",
        cxxopts::value<float>()->default_value("0.3"))("h,help", "Show help");
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
    args.template_image              = result["template"].as<std::string>();
    args.source_image                = result["source"].as<std::string>();
    args.mask_image                  = result["mask"].as<std::string>();
    args.load_templates              = result["load-templates"].as<std::string>();
    args.save_templates              = result["save-templates"].as<std::string>();
    args.output_image                = result["output"].as<std::string>();
    args.class_id                    = result["class-id"].as<std::string>();
    args.threshold                   = result["threshold"].as<float>();
    args.angle_begin                 = result["angle-begin"].as<float>();
    args.angle_end                   = result["angle-end"].as<float>();
    args.angle_step                  = result["angle-step"].as<float>();
    args.scale_begin                 = result["scale-begin"].as<float>();
    args.scale_end                   = result["scale-end"].as<float>();
    args.scale_step                  = result["scale-step"].as<float>();
    args.max_results                 = result["max-results"].as<int>();
    args.config.num_features         = result["features"].as<int>();
    args.config.min_features         = std::min(8, std::max(1, args.config.num_features));
    args.config.weak_threshold       = 10.0f;
    args.config.strong_threshold     = 20.0f;
    args.config.match_threshold      = args.threshold;
    args.config.max_results          = args.max_results;
    args.config.nms_threshold        = result["nms"].as<float>();
    args.config.max_label_difference = 1;
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
        cv::rectangle(image, box, cv::Scalar(0, 220, 255), 2);
        const std::string label = match.class_id + " "
                                + std::to_string(static_cast<int>(std::round(match.similarity))) + "% a="
                                + std::to_string(static_cast<int>(std::round(match.angle_degrees)));
        cv::putText(image, label, cv::Point(match.x, std::max(12, match.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
    }
}

} // namespace

/**
 * @brief 训练或加载形状模板，在源图中匹配并保存可视化结果。
 */
int main(int argc, char *argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);

        cv::Mat templ;
        cv::Mat source;
        cv::Mat mask;
        if (args.template_image.empty() && args.source_image.empty())
        {
            templ  = makeLShape();
            source = makeSyntheticScene(templ);
        }
        else
        {
            templ  = args.template_image.empty() ? makeLShape() : loadImage(args.template_image, cv::IMREAD_UNCHANGED,
                                                                            "template image");
            source = loadImage(args.source_image, cv::IMREAD_UNCHANGED, "source image");
            if (!args.mask_image.empty())
            {
                mask = loadImage(args.mask_image, cv::IMREAD_GRAYSCALE, "template mask");
            }
        }

        irt::features::ShapeTemplateMatcher matcher(args.config);
        if (!args.load_templates.empty())
        {
            matcher.load(args.load_templates);
        }
        else
        {
            const auto variants = irt::features::ShapeTemplateMatcher::makeAngleScaleVariants(
                args.angle_begin, args.angle_end, args.angle_step, args.scale_begin, args.scale_end, args.scale_step);
            matcher.addTemplateVariants(templ, args.class_id, mask, variants);
            if (!args.save_templates.empty())
            {
                matcher.save(args.save_templates);
            }
        }

        auto matches = matcher.match(source, args.threshold, {args.class_id});
        drawMatches(source, matches);
        if (!args.output_image.parent_path().empty())
        {
            fs::create_directories(args.output_image.parent_path());
        }
        if (!cv::imwrite(args.output_image.string(), source))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                                 args.output_image.string().c_str());
        }

        std::cout << "templates: " << matcher.numTemplates(args.class_id) << std::endl;
        std::cout << "matches: " << matches.size() << std::endl;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const auto &match = matches[i];
            std::cout << (i + 1) << ". class=" << match.class_id << " template=" << match.template_id
                      << " score=" << match.similarity << " box=(" << match.x << "," << match.y << ","
                      << match.width << "," << match.height << ") angle=" << match.angle_degrees
                      << " scale=" << match.scale << std::endl;
        }
        std::cout << "output: " << fs::absolute(args.output_image).string() << std::endl;
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
