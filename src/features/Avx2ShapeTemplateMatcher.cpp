/**
 * @file Avx2ShapeTemplateMatcher.cpp
 * @brief v1 AVX2 形状模板匹配器实现。
 */

#include "priv/ShapeTemplateMatcherEngine.hpp"

#include <inferrt/features/ShapeTemplateMatcher.hpp>
#include <inferrt/features/v1/ShapeTemplateMatcher.hpp>

#include <memory>
#include <utility>

namespace irt::features::v1 {

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : impl_(std::make_unique<detail::ShapeTemplateMatcherEngine>(
          std::move(config), detail::createShapeTemplateMatcherKernel(detail::ShapeTemplateMatcherBackend::Avx2)))
{
}

ShapeTemplateMatcher::~ShapeTemplateMatcher() = default;
ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept = default;
ShapeTemplateMatcher &ShapeTemplateMatcher::operator=(ShapeTemplateMatcher &&) noexcept = default;

int ShapeTemplateMatcher::addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
                                      ShapeTemplateVariant variant)
{
    return impl_->addTemplate(image, class_id, object_mask, variant);
}

int ShapeTemplateMatcher::addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                                          const std::filesystem::path &mask_file, ShapeTemplateVariant variant)
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

std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::matchFile(const std::filesystem::path &image_file,
                                                                float threshold,
                                                                const std::vector<std::string> &class_ids,
                                                                const std::filesystem::path &mask_file) const
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
    return impl_ == nullptr ? 0 : impl_->numClasses();
}

int ShapeTemplateMatcher::numTemplates() const noexcept
{
    return impl_ == nullptr ? 0 : impl_->numTemplates();
}

int ShapeTemplateMatcher::numTemplates(const std::string &class_id) const noexcept
{
    return impl_ == nullptr ? 0 : impl_->numTemplates(class_id);
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

void ShapeTemplateMatcher::save(const std::filesystem::path &template_file) const
{
    impl_->save(template_file);
}

void ShapeTemplateMatcher::load(const std::filesystem::path &template_file)
{
    impl_->load(template_file);
}

std::vector<ShapeTemplateVariant>
ShapeTemplateMatcher::makeAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees,
                                             float angle_step_degrees, float scale_begin, float scale_end,
                                             float scale_step)
{
    return makeShapeTemplateAngleScaleVariants(angle_begin_degrees, angle_end_degrees, angle_step_degrees,
                                               scale_begin, scale_end, scale_step);
}

cv::Mat ShapeTemplateMatcher::transform(const cv::Mat &image, ShapeTemplateVariant variant,
                                        const cv::Scalar &border_value)
{
    return transformShapeTemplateImage(image, variant, border_value);
}

} // namespace irt::features::v1
