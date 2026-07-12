#pragma once

/**
 * @file ShapeTemplateMatcherImpl.hpp
 * @brief ``ShapeTemplateMatcher`` 私有实现声明。
 */

#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include <map>

namespace irt::features {

/**
 * @brief ``ShapeTemplateMatcher`` 的私有实现。
 */
class ShapeTemplateMatcher::Impl
{
public:
    /**
     * @brief 构造私有实现并校验配置。
     * @param config 模板匹配器配置。
     */
    explicit Impl(ShapeTemplateMatcherConfig config, bool use_avx2 = true);

    /** @brief 析构私有实现。 */
    ~Impl() = default;

    Impl(const Impl &)            = delete;
    Impl &operator=(const Impl &) = delete;

    /** @brief 从内存图像训练单个模板。 */
    int addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
                    ShapeTemplateVariant variant);

    /** @brief 从图像文件训练单个模板。 */
    int addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                        const std::filesystem::path &mask_file, ShapeTemplateVariant variant);

    /** @brief 基于旋转/缩放参数批量生成模板。 */
    std::vector<int> addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                         const cv::Mat &object_mask,
                                         const std::vector<ShapeTemplateVariant> &variants);

    /** @brief 在内存图像中匹配已训练模板。 */
    std::vector<ShapeTemplateMatch> match(const cv::Mat &image, float threshold,
                                          const std::vector<std::string> &class_ids,
                                          const cv::Mat                  &search_mask) const;

    /** @brief 在图像文件中匹配已训练模板。 */
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &image_file, float threshold,
                                              const std::vector<std::string> &class_ids,
                                              const std::filesystem::path    &mask_file) const;

    /** @brief 清空模板库。 */
    void clear();

    /** @brief 判断模板库是否为空。 */
    bool empty() const noexcept;

    /** @brief 获取类别数量。 */
    int numClasses() const noexcept;

    /** @brief 获取模板总数。 */
    int numTemplates() const noexcept;

    /** @brief 获取指定类别的模板数量。 */
    int numTemplates(const std::string &class_id) const noexcept;

    /** @brief 获取类别 ID 列表。 */
    std::vector<std::string> classIds() const;

    /** @brief 获取指定模板信息。 */
    const ShapeTemplateInfo &getTemplate(const std::string &class_id, int template_id) const;

    /** @brief 获取当前配置。 */
    const ShapeTemplateMatcherConfig &config() const noexcept;

    /** @brief 保存模板库。 */
    void save(const std::filesystem::path &template_file) const;

    /** @brief 加载模板库。 */
    void load(const std::filesystem::path &template_file);

private:
    using TemplateMap = std::map<std::string, std::vector<ShapeTemplateInfo>>; ///< 类别到模板列表的映射。

    ShapeTemplateMatcherConfig config_{};   ///< 当前匹配器配置。
    TemplateMap                templates_; ///< 已训练模板库。
    bool                       use_avx2_{true}; ///< 是否使用 AVX2 热点内核。
};

} // namespace irt::features
