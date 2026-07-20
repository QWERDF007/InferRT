#pragma once

/**
 * @file IShapeTemplateMatcher.hpp
 * @brief 可扩展的形状模板匹配版本公共接口。
 */

#include <inferrt/features/ShapeTemplateMatcherTypes.hpp>

namespace irt::features {

/**
 * @brief 形状模板匹配器的版本无关接口。
 *
 * v0、v1 及未来的 AVX512 版本均实现该接口；模板文件格式和结果数据结构在版本间保持兼容。
 */
class INFERRT_FEATURES_API IShapeTemplateMatcher
{
public:
    virtual ~IShapeTemplateMatcher() = default;

    IShapeTemplateMatcher(const IShapeTemplateMatcher &)                   = delete;
    IShapeTemplateMatcher &operator=(const IShapeTemplateMatcher &)        = delete;
    IShapeTemplateMatcher(IShapeTemplateMatcher &&) noexcept               = default;
    IShapeTemplateMatcher &operator=(IShapeTemplateMatcher &&) noexcept = default;

    virtual int addTemplate(const cv::Mat &image, const std::string &class_id,
                            const cv::Mat &object_mask = cv::Mat(), ShapeTemplateVariant variant = {}) = 0;
    virtual int addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                                const std::filesystem::path &mask_file = {}, ShapeTemplateVariant variant = {}) = 0;
    virtual std::vector<int> addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                                 const cv::Mat &object_mask,
                                                 const std::vector<ShapeTemplateVariant> &variants) = 0;
    virtual std::vector<ShapeTemplateMatch> match(const cv::Mat &image, float threshold = -1.0f,
                                                  const std::vector<std::string> &class_ids = {},
                                                  const cv::Mat &search_mask = cv::Mat()) const = 0;
    virtual std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &image_file,
                                                      float threshold = -1.0f,
                                                      const std::vector<std::string> &class_ids = {},
                                                      const std::filesystem::path &mask_file = {}) const = 0;
    virtual void clear() = 0;
    virtual bool empty() const noexcept = 0;
    virtual int numClasses() const noexcept = 0;
    virtual int numTemplates() const noexcept = 0;
    virtual int numTemplates(const std::string &class_id) const noexcept = 0;
    virtual std::vector<std::string> classIds() const = 0;
    virtual const ShapeTemplateInfo &getTemplate(const std::string &class_id, int template_id) const = 0;
    virtual const ShapeTemplateMatcherConfig &config() const noexcept = 0;
    virtual void save(const std::filesystem::path &template_file) const = 0;
    virtual void load(const std::filesystem::path &template_file) = 0;

protected:
    IShapeTemplateMatcher() = default;
};

} // namespace irt::features
