/**
 * @file ShapeTemplateMatcherBase.cpp
 * @brief 形状模板匹配内部适配器转发流程的实现。
 */

#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include "priv/ShapeTemplateMatcherAdapter.hpp"

#include <utility>

namespace irt::features::priv {

ShapeTemplateMatcherAdapter::ShapeTemplateMatcherAdapter(std::unique_ptr<IShapeTemplateMatcher> implementation)
    : implementation_(std::move(implementation))
{
}

ShapeTemplateMatcherAdapter::~ShapeTemplateMatcherAdapter() = default;
ShapeTemplateMatcherAdapter::ShapeTemplateMatcherAdapter(ShapeTemplateMatcherAdapter &&) noexcept = default;
ShapeTemplateMatcherAdapter &ShapeTemplateMatcherAdapter::operator=(ShapeTemplateMatcherAdapter &&) noexcept = default;

int ShapeTemplateMatcherAdapter::addTemplate(const cv::Mat &image, const cv::Mat &object_mask,
                                             ShapeTemplateVariant variant)
{
    return implementation_->addTemplate(image, object_mask, variant);
}

int ShapeTemplateMatcherAdapter::addTemplateFile(const std::filesystem::path &image_file,
                                                 const std::filesystem::path &mask_file, ShapeTemplateVariant variant)
{
    return implementation_->addTemplateFile(image_file, mask_file, variant);
}

std::vector<int> ShapeTemplateMatcherAdapter::addTemplateVariants(
    const cv::Mat &image, const cv::Mat &object_mask, const std::vector<ShapeTemplateVariant> &variants)
{
    return implementation_->addTemplateVariants(image, object_mask, variants);
}

std::vector<std::vector<int>> ShapeTemplateMatcherAdapter::addTemplateVariantsBatch(
    const std::vector<ShapeTemplateTrainingInput> &inputs,
    const std::vector<ShapeTemplateVariant> &variants)
{
    return implementation_->addTemplateVariantsBatch(inputs, variants);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcherAdapter::match(
    const cv::Mat &image, float threshold, const cv::Mat &search_mask, ShapeTemplateMatchOptions options) const
{
    return implementation_->match(image, threshold, search_mask, options);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcherAdapter::matchFile(
    const std::filesystem::path &image_file, float threshold, const std::filesystem::path &mask_file,
    ShapeTemplateMatchOptions options) const
{
    return implementation_->matchFile(image_file, threshold, mask_file, options);
}

void ShapeTemplateMatcherAdapter::clear()
{
    implementation_->clear();
}

bool ShapeTemplateMatcherAdapter::empty() const noexcept
{
    return implementation_ == nullptr || implementation_->empty();
}

int ShapeTemplateMatcherAdapter::numTemplates() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->numTemplates();
}

const ShapeTemplateInfo &ShapeTemplateMatcherAdapter::getTemplate(int template_id) const
{
    return implementation_->getTemplate(template_id);
}

const ShapeTemplateMatcherConfig &ShapeTemplateMatcherAdapter::config() const noexcept
{
    return implementation_->config();
}

void ShapeTemplateMatcherAdapter::save(const std::filesystem::path &template_file) const
{
    implementation_->save(template_file);
}

void ShapeTemplateMatcherAdapter::load(const std::filesystem::path &template_file)
{
    implementation_->load(template_file);
}

std::vector<ShapeTemplateVariant>
ShapeTemplateMatcherAdapter::makeAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees,
                                                     float angle_step_degrees, float scale_begin, float scale_end,
                                                     float scale_step)
{
    return ::irt::features::makeShapeTemplateAngleScaleVariants(angle_begin_degrees, angle_end_degrees,
                                                                 angle_step_degrees, scale_begin, scale_end,
                                                                 scale_step);
}

cv::Mat ShapeTemplateMatcherAdapter::transform(const cv::Mat &image, ShapeTemplateVariant variant,
                                                const cv::Scalar &border_value)
{
    return ::irt::features::transformShapeTemplateImage(image, variant, border_value);
}

} // namespace irt::features::priv
