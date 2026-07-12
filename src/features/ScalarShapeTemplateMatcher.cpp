#include <inferrt/features/scalar/ShapeTemplateMatcher.hpp>

#include <utility>

namespace irt::features::scalar {

ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcherConfig config)
    : impl_(new irt::features::ShapeTemplateMatcher(std::move(config), false))
{
}

ShapeTemplateMatcher::~ShapeTemplateMatcher() = default;
ShapeTemplateMatcher::ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept = default;
ShapeTemplateMatcher &ShapeTemplateMatcher::operator=(ShapeTemplateMatcher &&) noexcept = default;

int ShapeTemplateMatcher::addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &mask,
                                      ShapeTemplateVariant variant)
{
    return impl_->addTemplate(image, class_id, mask, variant);
}
int ShapeTemplateMatcher::addTemplateFile(const std::filesystem::path &image, const std::string &class_id,
                                          const std::filesystem::path &mask, ShapeTemplateVariant variant)
{
    return impl_->addTemplateFile(image, class_id, mask, variant);
}
std::vector<int> ShapeTemplateMatcher::addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                                            const cv::Mat &mask, const std::vector<ShapeTemplateVariant> &variants)
{
    return impl_->addTemplateVariants(image, class_id, mask, variants);
}
std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::match(const cv::Mat &image, float threshold,
                                                             const std::vector<std::string> &class_ids, const cv::Mat &mask) const
{
    return impl_->match(image, threshold, class_ids, mask);
}
std::vector<ShapeTemplateMatch> ShapeTemplateMatcher::matchFile(const std::filesystem::path &image, float threshold,
                                                                 const std::vector<std::string> &class_ids, const std::filesystem::path &mask) const
{
    return impl_->matchFile(image, threshold, class_ids, mask);
}
void ShapeTemplateMatcher::clear() { impl_->clear(); }
bool ShapeTemplateMatcher::empty() const noexcept { return impl_->empty(); }
int ShapeTemplateMatcher::numClasses() const noexcept { return impl_->numClasses(); }
int ShapeTemplateMatcher::numTemplates() const noexcept { return impl_->numTemplates(); }
int ShapeTemplateMatcher::numTemplates(const std::string &class_id) const noexcept { return impl_->numTemplates(class_id); }
std::vector<std::string> ShapeTemplateMatcher::classIds() const { return impl_->classIds(); }
const ShapeTemplateInfo &ShapeTemplateMatcher::getTemplate(const std::string &class_id, int template_id) const { return impl_->getTemplate(class_id, template_id); }
const ShapeTemplateMatcherConfig &ShapeTemplateMatcher::config() const noexcept { return impl_->config(); }
void ShapeTemplateMatcher::save(const std::filesystem::path &path) const { impl_->save(path); }
void ShapeTemplateMatcher::load(const std::filesystem::path &path) { impl_->load(path); }
std::vector<ShapeTemplateVariant> ShapeTemplateMatcher::makeAngleScaleVariants(float begin, float end, float step, float scale_begin, float scale_end, float scale_step)
{
    return irt::features::ShapeTemplateMatcher::makeAngleScaleVariants(begin, end, step, scale_begin, scale_end, scale_step);
}
cv::Mat ShapeTemplateMatcher::transform(const cv::Mat &image, ShapeTemplateVariant variant, const cv::Scalar &border)
{
    return irt::features::ShapeTemplateMatcher::transform(image, variant, border);
}

} // namespace irt::features::scalar