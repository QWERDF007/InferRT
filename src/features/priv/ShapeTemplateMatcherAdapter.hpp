#pragma once

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features::priv {

/**
 * Shared private adapter for the scalar and SIMD implementations.
 *
 * The installed interface is IShapeTemplateMatcher. This adapter keeps the
 * forwarding and ownership policy in one implementation without making the
 * algorithm or ISA classes part of the installed interface.
 */
class ShapeTemplateMatcherAdapter : public IShapeTemplateMatcher
{
public:
    ~ShapeTemplateMatcherAdapter() override;
    ShapeTemplateMatcherAdapter(const ShapeTemplateMatcherAdapter &)            = delete;
    ShapeTemplateMatcherAdapter &operator=(const ShapeTemplateMatcherAdapter &) = delete;
    ShapeTemplateMatcherAdapter(ShapeTemplateMatcherAdapter &&) noexcept;
    ShapeTemplateMatcherAdapter &operator=(ShapeTemplateMatcherAdapter &&) noexcept;

    int addTemplate(const cv::Mat &, const cv::Mat & = cv::Mat(), ShapeTemplateVariant = {}) final;
    int addTemplateFile(const std::filesystem::path &, const std::filesystem::path & = {},
                        ShapeTemplateVariant = {}) final;
    std::vector<int> addTemplateVariants(const cv::Mat &, const cv::Mat &,
                                         const std::vector<ShapeTemplateVariant> &) final;
    std::vector<std::vector<int>>
    addTemplateVariantsBatch(const std::vector<ShapeTemplateTrainingInput> &, const std::vector<ShapeTemplateVariant> &)
        final;
    std::vector<ShapeTemplateMatch> match(const cv::Mat &, float = -1.0f, const cv::Mat & = cv::Mat(),
                                          ShapeTemplateMatchOptions = {}) const final;
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &, float = -1.0f,
                                              const std::filesystem::path & = {},
                                              ShapeTemplateMatchOptions = {}) const final;
    void clear() final;
    bool empty() const noexcept final;
    int numTemplates() const noexcept final;
    const ShapeTemplateInfo &getTemplate(int) const final;
    const ShapeTemplateMatcherConfig &config() const noexcept final;
    void save(const std::filesystem::path &) const final;
    void load(const std::filesystem::path &) final;

    static std::vector<ShapeTemplateVariant> makeAngleScaleVariants(float, float, float, float = 1.0f,
                                                                     float = 1.0f, float = 1.0f);
    static cv::Mat transform(const cv::Mat &, ShapeTemplateVariant, const cv::Scalar & = cv::Scalar());

protected:
    explicit ShapeTemplateMatcherAdapter(std::unique_ptr<IShapeTemplateMatcher> implementation);

private:
    std::unique_ptr<IShapeTemplateMatcher> implementation_;
};

} // namespace irt::features::priv
