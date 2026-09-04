#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>


namespace irt::model {
namespace {

int cvInterpolation(const irt::Interpolation interpolation)
{
    switch (interpolation)
    {
    case irt::Interpolation::Nearest:
        return cv::INTER_NEAREST;
    case irt::Interpolation::Linear:
        return cv::INTER_LINEAR;
    case irt::Interpolation::Cubic:
        return cv::INTER_CUBIC;
    case irt::Interpolation::Area:
        return cv::INTER_AREA;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported preprocessing interpolation");
}

int colorConversionCode(const irt::ColorFormat source, const irt::ColorFormat destination)
{
    if (source == destination)
    {
        return -1;
    }
    using irt::ColorFormat;
    switch (source)
    {
    case ColorFormat::BGR:
        switch (destination)
        {
        case ColorFormat::RGB:  return cv::COLOR_BGR2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGR2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_BGR2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_BGR2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGB:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGB2BGR;
        case ColorFormat::GRAY: return cv::COLOR_RGB2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGB2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_RGB2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::GRAY:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_GRAY2BGR;
        case ColorFormat::RGB:  return cv::COLOR_GRAY2RGB;
        case ColorFormat::BGRA: return cv::COLOR_GRAY2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_GRAY2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::BGRA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_BGRA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_BGRA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGRA2GRAY;
        case ColorFormat::RGBA: return cv::COLOR_BGRA2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGBA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGBA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_RGBA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_RGBA2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGBA2BGRA;
        default:                break;
        }
        break;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "Unsupported preprocessing color conversion (%d -> %d)", static_cast<int>(source),
                         static_cast<int>(destination));
}

} // namespace

size_t elementCount(const irt::Shape &shape)
{
    return shape.elementCount();
}

size_t elementSize(irt::TensorDataType data_type)
{
    const auto size = irt::dataTypeSize(data_type);
    if (size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported tensor data type");
    }
    return size;
}

size_t dataTypeSize(irt::TensorDataType data_type)
{
    return elementSize(data_type);
}

std::string dataTypeToString(irt::TensorDataType data_type)
{
    return std::string(irt::dataTypeToString(data_type));
}

std::string dimsToCsv(const irt::Shape &shape)
{
    std::string result;
    for (size_t index = 0; index < shape.rank(); ++index)
    {
        if (index > 0)
        {
            result += ",";
        }
        result += std::to_string(shape[index]);
    }
    return result;
}

std::string dimsToString(const irt::Shape &shape)
{
    return "[" + dimsToCsv(shape) + "]";
}

const std::filesystem::path ImageNetUtil::kDefaultImagePath = "assets/pics/dog.jpg";
const std::filesystem::path ImageNetUtil::kDefaultLabelPath = "assets/imagenet1000_clsidx_to_labels.txt";

/**
 * @brief 读取 ImageNet 标签文件。
 * @param label_file 标签文件路径。
 * @return 长度为 1000 的标签数组。
 */
std::vector<std::string> readImagenetLabels(const std::string &label_file)
{
    std::vector<std::string> labels(1000);
    std::ifstream            file(label_file);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open weights file: %s", label_file.c_str());
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t colon_pos = line.find(": ");
        if (colon_pos != std::string::npos)
        {
            int         idx   = std::stoi(line.substr(0, colon_pos));
            std::string label = line.substr(colon_pos + 2);
            if (label.size() >= 2 && label.front() == '\'' && label.back() == ',')
            {
                label = label.substr(1, label.size() - 3);
            }
            if (idx >= 0 && idx < 1000)
            {
                labels[idx] = label;
            }
        }
    }

    return labels;
}

ImageNetUtil::PreprocessResult ImageNetUtil::preprocessWithGeometry(const cv::Mat &image,
                                                                     const irt::PreprocessSpec &spec)
{
    spec.validate();
    if (image.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (image.depth() != CV_8U)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Preprocessing expects an 8-bit image, got depth=%d",
                             image.depth());
    }
    const int expected_source_channels = irt::colorChannels(spec.src_color);
    if (image.channels() != expected_source_channels
        || (spec.source_channels != 0 && image.channels() != spec.source_channels))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Preprocessing source expects %d channels, got %d", expected_source_channels,
                             image.channels());
    }

    cv::Mat converted;
    const int conversion = colorConversionCode(spec.src_color, spec.dst_color);
    if (conversion < 0)
    {
        converted = image;
    }
    else
    {
        cv::cvtColor(image, converted, conversion);
    }

    PreprocessResult result;
    result.geometry = irt::resolvePreprocessGeometry(spec, image.cols, image.rows);

    cv::Mat resized;
    cv::Rect content_rect;
    const cv::Size target(spec.input_width, spec.input_height);
    const int      interpolation = cvInterpolation(spec.interpolation);
    switch (spec.padding_mode)
    {
    case irt::PaddingMode::DirectResize:
        cv::resize(converted, resized, target, 0.0, 0.0, interpolation);
        break;
    case irt::PaddingMode::Letterbox:
    {
        cv::resize(converted, resized,
                   cv::Size(result.geometry.resized_width, result.geometry.resized_height), 0.0, 0.0, interpolation);
        content_rect = cv::Rect(result.geometry.pad_left, result.geometry.pad_top, resized.cols, resized.rows);
        if (!spec.pad_after_normalize)
        {
            cv::Mat canvas(target.height, target.width, resized.type(),
                           cv::Scalar(spec.pad_value, spec.pad_value, spec.pad_value, spec.pad_value));
            resized.copyTo(canvas(content_rect));
            resized = std::move(canvas);
        }
        break;
    }
    case irt::PaddingMode::CenterCrop:
    {
        cv::resize(converted, resized,
                   cv::Size(result.geometry.resized_width, result.geometry.resized_height), 0.0, 0.0, interpolation);
        resized = resized(cv::Rect(result.geometry.crop_left, result.geometry.crop_top, target.width, target.height))
                      .clone();
        break;
    }
    }

    cv::Mat normalized;
    resized.convertTo(normalized, CV_MAKETYPE(CV_32F, spec.input_channels), spec.scale);
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    for (size_t channel = 0; channel < channels.size(); ++channel)
    {
        channels[channel].convertTo(channels[channel], CV_32F, 1.0, -spec.mean[channel]);
        channels[channel] /= spec.stddev[channel];
    }
    cv::merge(channels, normalized);
    if (spec.pad_after_normalize)
    {
        result.image = cv::Mat::zeros(target.height, target.width, normalized.type());
        normalized.copyTo(result.image(content_rect));
    }
    else
    {
        result.image = std::move(normalized);
    }
    return result;
}

cv::Mat ImageNetUtil::preprocess(const cv::Mat &image, const irt::PreprocessSpec &spec)
{
    return preprocessWithGeometry(image, spec).image;
}

cv::Mat ImageNetUtil::preprocess(const cv::Mat &bgr_image, cv::Size target_size)
{
    if (target_size.width <= 0 || target_size.height <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid target size: %dx%d", target_size.width,
                             target_size.height);
    }

    irt::PreprocessSpec spec;
    spec.input_width  = target_size.width;
    spec.input_height = target_size.height;
    spec.input_channels = 3;
    spec.source_channels = 3;
    return preprocess(bgr_image, spec);
}

std::vector<float> ImageNetUtil::imageToTensorCHW(const cv::Mat &image)
{
    if (image.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (image.depth() != CV_32F || image.channels() <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Expected a float image, got type=%d", image.type());
    }

    const size_t plane_size = irt::checkedSizeMul(static_cast<size_t>(image.rows), static_cast<size_t>(image.cols),
                                                  "Image plane");
    const size_t tensor_size = irt::checkedSizeMul(static_cast<size_t>(image.channels()), plane_size,
                                                   "Image tensor");
    std::vector<float>   tensor(tensor_size);
    std::vector<cv::Mat> channels(static_cast<size_t>(image.channels()));
    cv::split(image, channels);
    const size_t channel_bytes = irt::checkedSizeMul(plane_size, sizeof(float), "Image channel");
    for (size_t c = 0; c < channels.size(); ++c)
    {
        std::memcpy(tensor.data() + irt::checkedSizeMul(c, plane_size, "Image channel offset"), channels[c].data,
                    channel_bytes);
    }
    return tensor;
}

} // namespace irt::model
