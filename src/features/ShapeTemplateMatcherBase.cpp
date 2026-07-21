/**
 * @file ShapeTemplateMatcherBase.cpp
 * @brief 形状模板匹配版本共用转发流程的实现。
 */

#include <inferrt/features/ShapeTemplateMatcher.hpp>
#include <inferrt/features/ShapeTemplateMatcherBase.hpp>

#include <utility>

namespace irt::features::detail {

ShapeTemplateMatcherBase::ShapeTemplateMatcherBase(std::unique_ptr<IShapeTemplateMatcher> implementation)
    : implementation_(std::move(implementation))
{
}

ShapeTemplateMatcherBase::~ShapeTemplateMatcherBase() = default;
ShapeTemplateMatcherBase::ShapeTemplateMatcherBase(ShapeTemplateMatcherBase &&) noexcept = default;
ShapeTemplateMatcherBase &ShapeTemplateMatcherBase::operator=(ShapeTemplateMatcherBase &&) noexcept = default;

int ShapeTemplateMatcherBase::addTemplate(const cv::Mat &image, const std::string &class_id,
                                          const cv::Mat &object_mask, ShapeTemplateVariant variant)
{
    return implementation_->addTemplate(image, class_id, object_mask, variant);
}

int ShapeTemplateMatcherBase::addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                                              const std::filesystem::path &mask_file, ShapeTemplateVariant variant)
{
    return implementation_->addTemplateFile(image_file, class_id, mask_file, variant);
}

std::vector<int> ShapeTemplateMatcherBase::addTemplateVariants(
    const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
    const std::vector<ShapeTemplateVariant> &variants)
{
    return implementation_->addTemplateVariants(image, class_id, object_mask, variants);
}

std::vector<std::vector<int>> ShapeTemplateMatcherBase::addTemplateVariantsBatch(
    const std::vector<ShapeTemplateTrainingInput> &inputs,
    const std::vector<ShapeTemplateVariant> &variants)
{
    return implementation_->addTemplateVariantsBatch(inputs, variants);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcherBase::match(
    const cv::Mat &image, float threshold, const std::vector<std::string> &class_ids,
    const cv::Mat &search_mask, ShapeTemplateMatchOptions options) const
{
    return implementation_->match(image, threshold, class_ids, search_mask, options);
}

std::vector<ShapeTemplateMatch> ShapeTemplateMatcherBase::matchFile(
    const std::filesystem::path &image_file, float threshold, const std::vector<std::string> &class_ids,
    const std::filesystem::path &mask_file, ShapeTemplateMatchOptions options) const
{
    return implementation_->matchFile(image_file, threshold, class_ids, mask_file, options);
}

void ShapeTemplateMatcherBase::clear()
{
    implementation_->clear();
}

bool ShapeTemplateMatcherBase::empty() const noexcept
{
    return implementation_ == nullptr || implementation_->empty();
}

int ShapeTemplateMatcherBase::numClasses() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->numClasses();
}

int ShapeTemplateMatcherBase::numTemplates() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->numTemplates();
}

int ShapeTemplateMatcherBase::numTemplates(const std::string &class_id) const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->numTemplates(class_id);
}

std::vector<std::string> ShapeTemplateMatcherBase::classIds() const
{
    return implementation_->classIds();
}

const ShapeTemplateInfo &ShapeTemplateMatcherBase::getTemplate(const std::string &class_id, int template_id) const
{
    return implementation_->getTemplate(class_id, template_id);
}

const ShapeTemplateMatcherConfig &ShapeTemplateMatcherBase::config() const noexcept
{
    return implementation_->config();
}

void ShapeTemplateMatcherBase::save(const std::filesystem::path &template_file) const
{
    implementation_->save(template_file);
}

void ShapeTemplateMatcherBase::load(const std::filesystem::path &template_file)
{
    implementation_->load(template_file);
}

std::vector<ShapeTemplateVariant>
ShapeTemplateMatcherBase::makeAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees,
                                                  float angle_step_degrees, float scale_begin, float scale_end,
                                                  float scale_step)
{
    return ::irt::features::makeShapeTemplateAngleScaleVariants(angle_begin_degrees, angle_end_degrees,
                                                                 angle_step_degrees, scale_begin, scale_end,
                                                                 scale_step);
}

cv::Mat ShapeTemplateMatcherBase::transform(const cv::Mat &image, ShapeTemplateVariant variant,
                                             const cv::Scalar &border_value)
{
    return ::irt::features::transformShapeTemplateImage(image, variant, border_value);
}

} // namespace irt::features::detail
