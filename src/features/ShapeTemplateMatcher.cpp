/**
 * @file ShapeTemplateMatcher.cpp
 * @brief ``ShapeTemplateMatcher`` 公共 API 转发实现。
 */

#include "priv/ShapeTemplateMatcherImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features {

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : ShapeTemplateMatcher(std::move(config), true)
{
}

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcherConfig config, bool use_avx2)
    : impl_(std::make_unique<Impl>(std::move(config), use_avx2))
{
}

ShapeTemplateMatcher::~ShapeTemplateMatcher() = default;

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcher &&other) noexcept = default;

ShapeTemplateMatcher &ShapeTemplateMatcher::operator=(ShapeTemplateMatcher &&other) noexcept = default;

int ShapeTemplateMatcher::addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
                                      ShapeTemplateVariant variant)
{
    return impl_->addTemplate(image, class_id, object_mask, variant);
}

int ShapeTemplateMatcher::addTemplateFile(const fs::path &image_file, const std::string &class_id,
                                          const fs::path &mask_file, ShapeTemplateVariant variant)
{
    return impl_->addTemplateFile(image_file, class_id, mask_file, variant);
}

std::vector<int> ShapeTemplateMatcher::addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                                           const cv::Mat &object_mask,
                                                           const std::vector<ShapeTemplateVariant> &variants)
{
    return impl_->addTemplateVariants(image, class_id, object_mask, variants);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::match(const cv::Mat &image, float threshold,
                                                            const std::vector<std::string> &class_ids,
                                                            const cv::Mat &search_mask) const
{
    return impl_->match(image, threshold, class_ids, search_mask);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::matchFile(const fs::path &image_file, float threshold,
                                                                const std::vector<std::string> &class_ids,
                                                                const fs::path &mask_file) const
{
    return impl_->matchFile(image_file, threshold, class_ids, mask_file);
}

void ShapeTemplateMatcher::clear()
{
    impl_->clear();
}

bool ShapeTemplateMatcher::empty() const noexcept
{
    return impl_ == nullptr || impl_->empty();
}

int ShapeTemplateMatcher::numClasses() const noexcept
{
    return impl_ ? impl_->numClasses() : 0;
}

int ShapeTemplateMatcher::numTemplates() const noexcept
{
    return impl_ ? impl_->numTemplates() : 0;
}

int ShapeTemplateMatcher::numTemplates(const std::string &class_id) const noexcept
{
    return impl_ ? impl_->numTemplates(class_id) : 0;
}

std::vector<std::string> ShapeTemplateMatcher::classIds() const
{
    return impl_->classIds();
}

const ShapeTemplateInfo &ShapeTemplateMatcher::getTemplate(const std::string &class_id, int template_id) const
{
    return impl_->getTemplate(class_id, template_id);
}

const ShapeTemplateMatcherConfig &ShapeTemplateMatcher::config() const noexcept
{
    return impl_->config();
}

void ShapeTemplateMatcher::save(const fs::path &template_file) const
{
    impl_->save(template_file);
}

void ShapeTemplateMatcher::load(const fs::path &template_file)
{
    impl_->load(template_file);
}

std::vector<ShapeTemplateVariant> ShapeTemplateMatcher::makeAngleScaleVariants(float angle_begin_degrees,
                                                                               float angle_end_degrees,
                                                                               float angle_step_degrees,
                                                                               float scale_begin, float scale_end,
                                                                               float scale_step)
{
    if (!std::isfinite(angle_begin_degrees) || !std::isfinite(angle_end_degrees)
        || !std::isfinite(angle_step_degrees) || angle_step_degrees <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Angle range must be finite and angle_step_degrees must be positive");
    }
    if (!std::isfinite(scale_begin) || !std::isfinite(scale_end) || !std::isfinite(scale_step)
        || scale_step <= 0.0f || scale_begin <= 0.0f || scale_end <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Scale range must be finite, positive, and scale_step must be positive");
    }
    if (angle_end_degrees + 1.0e-6f < angle_begin_degrees)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Angle range end must be >= begin");
    }
    if (scale_end + 1.0e-6f < scale_begin)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Scale range end must be >= begin");
    }

    std::vector<ShapeTemplateVariant> variants;
    constexpr float eps = 1.0e-6f;
    for (float scale = scale_begin; scale <= scale_end + eps; scale += scale_step)
    {
        for (float angle = angle_begin_degrees; angle <= angle_end_degrees + eps; angle += angle_step_degrees)
        {
            variants.push_back(ShapeTemplateVariant{angle, scale});
        }
    }
    return variants;
}

cv::Mat ShapeTemplateMatcher::transform(const cv::Mat &image, ShapeTemplateVariant variant,
                                        const cv::Scalar &border_value)
{
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "image must not be empty");
    }
    if (!std::isfinite(variant.angle_degrees) || !std::isfinite(variant.scale) || variant.scale <= 0.0f)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Shape template variant is invalid");
    }

    const cv::Point2f center(static_cast<float>(image.cols) * 0.5f, static_cast<float>(image.rows) * 0.5f);
    const cv::Mat     matrix = cv::getRotationMatrix2D(center, variant.angle_degrees, variant.scale);

    cv::Mat output;
    cv::warpAffine(image, output, matrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
    return output;
}

} // namespace irt::features
