#pragma once

/**
 * @file v0/ShapeTemplateMatcher.hpp
 * @brief 原始标量形状模板匹配器。
 */

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features::detail {
class ShapeTemplateMatcherEngine;
}

namespace irt::features::v0 {

/** @brief 原始标量实现；不使用 SIMD intrinsic。 */
class INFERRT_FEATURES_API ShapeTemplateMatcher final : public IShapeTemplateMatcher
{
public:
    explicit ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});
    ~ShapeTemplateMatcher() override;
    ShapeTemplateMatcher(const ShapeTemplateMatcher &)            = delete;
    ShapeTemplateMatcher &operator=(const ShapeTemplateMatcher &) = delete;
    ShapeTemplateMatcher(ShapeTemplateMatcher &&) noexcept;
    ShapeTemplateMatcher &operator=(ShapeTemplateMatcher &&) noexcept;

    int addTemplate(const cv::Mat &, const std::string &, const cv::Mat & = cv::Mat(),
                    ShapeTemplateVariant = {}) override;
    int addTemplateFile(const std::filesystem::path &, const std::string &,
                        const std::filesystem::path & = {}, ShapeTemplateVariant = {}) override;
    std::vector<int> addTemplateVariants(const cv::Mat &, const std::string &, const cv::Mat &,
                                         const std::vector<ShapeTemplateVariant> &) override;
    std::vector<ShapeTemplateMatch> match(const cv::Mat &, float = -1.0f,
                                          const std::vector<std::string> & = {},
                                          const cv::Mat & = cv::Mat(),
                                          ShapeTemplateMatchOptions = {}) const override;
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &, float = -1.0f,
                                              const std::vector<std::string> & = {},
                                              const std::filesystem::path & = {},
                                              ShapeTemplateMatchOptions = {}) const override;
    void clear() override;
    bool empty() const noexcept override;
    int numClasses() const noexcept override;
    int numTemplates() const noexcept override;
    int numTemplates(const std::string &) const noexcept override;
    std::vector<std::string> classIds() const override;
    const ShapeTemplateInfo &getTemplate(const std::string &, int) const override;
    const ShapeTemplateMatcherConfig &config() const noexcept override;
    void save(const std::filesystem::path &) const override;
    void load(const std::filesystem::path &) override;

    static std::vector<ShapeTemplateVariant> makeAngleScaleVariants(float, float, float,
                                                                     float = 1.0f, float = 1.0f,
                                                                     float = 1.0f);
    static cv::Mat transform(const cv::Mat &, ShapeTemplateVariant, const cv::Scalar & = cv::Scalar());

private:
    std::unique_ptr<detail::ShapeTemplateMatcherEngine> impl_;
};

} // namespace irt::features::v0
