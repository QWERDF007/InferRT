#pragma once

/** @file scalar/ShapeTemplateMatcher.hpp
 * @brief 不使用 AVX2 intrinsic 的形状模板匹配器接口。 */
#include <inferrt/features/ShapeTemplateMatcher.hpp>

namespace irt::features::scalar {

/** @brief 与 AVX2 版本接口、模板文件格式和结果类型兼容的标量实现。 */
class INFERRT_FEATURES_API ShapeTemplateMatcher
{
public:
    explicit ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~ShapeTemplateMatcher();
    ShapeTemplateMatcher(const ShapeTemplateMatcher &) = delete;
    ShapeTemplateMatcher &operator=(const ShapeTemplateMatcher &) = delete;
    ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept;
    ShapeTemplateMatcher &operator=(ShapeTemplateMatcher &&) noexcept;
    int addTemplate(const cv::Mat &, const std::string &, const cv::Mat & = cv::Mat(), ShapeTemplateVariant = {});
    int addTemplateFile(const std::filesystem::path &, const std::string &, const std::filesystem::path & = {}, ShapeTemplateVariant = {});
    std::vector<int> addTemplateVariants(const cv::Mat &, const std::string &, const cv::Mat &, const std::vector<ShapeTemplateVariant> &);
    std::vector<ShapeTemplateMatch> match(const cv::Mat &, float = -1.0f, const std::vector<std::string> & = {}, const cv::Mat & = cv::Mat()) const;
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &, float = -1.0f, const std::vector<std::string> & = {}, const std::filesystem::path & = {}) const;
    void clear(); bool empty() const noexcept; int numClasses() const noexcept; int numTemplates() const noexcept;
    int numTemplates(const std::string &) const noexcept; std::vector<std::string> classIds() const;
    const ShapeTemplateInfo &getTemplate(const std::string &, int) const; const ShapeTemplateMatcherConfig &config() const noexcept;
    void save(const std::filesystem::path &) const; void load(const std::filesystem::path &);
    static std::vector<ShapeTemplateVariant> makeAngleScaleVariants(float, float, float, float = 1.0f, float = 1.0f, float = 1.0f);
    static cv::Mat transform(const cv::Mat &, ShapeTemplateVariant, const cv::Scalar & = cv::Scalar());
private:
    std::unique_ptr<irt::features::ShapeTemplateMatcher> impl_;
};
} // namespace irt::features::scalar