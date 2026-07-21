#pragma once

/**
 * @file ShapeTemplateMatcherBase.hpp
 * @brief 形状模板匹配版本共用的转发基类。
 */

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <memory>

namespace irt::features::detail {

/**
 * @brief 为 v0、v1 与 v2 公开匹配器提供完全一致的接口流程。
 *
 * @details 该类只负责把公共 API 转发给版本私有的具体实现，不包含标量或 SIMD 算法，
 * 也不选择、包含或调用任一版本的代码。各版本仅在构造时注入自己的实现，因此训练、
 * 持久化、匹配、NMS 和结果访问的调用流程在架构层面保持一致。
 */
class INFERRT_FEATURES_API ShapeTemplateMatcherBase : public IShapeTemplateMatcher
{
public:
    ~ShapeTemplateMatcherBase() override;
    ShapeTemplateMatcherBase(const ShapeTemplateMatcherBase &)            = delete;
    ShapeTemplateMatcherBase &operator=(const ShapeTemplateMatcherBase &) = delete;
    ShapeTemplateMatcherBase(ShapeTemplateMatcherBase &&) noexcept;
    ShapeTemplateMatcherBase &operator=(ShapeTemplateMatcherBase &&) noexcept;

    /** @brief 向当前版本实现添加一个模板。 */
    int addTemplate(const cv::Mat &, const cv::Mat & = cv::Mat(), ShapeTemplateVariant = {}) final;
    /** @brief 从图像文件添加一个模板。 */
    int addTemplateFile(const std::filesystem::path &, const std::filesystem::path & = {},
                        ShapeTemplateVariant = {}) final;
    /** @brief 按给定角度和尺度变体批量训练模板。 */
    std::vector<int> addTemplateVariants(const cv::Mat &, const cv::Mat &,
                                         const std::vector<ShapeTemplateVariant> &) final;
    /** @brief 对多个训练输入统一调度同一组角度和尺度变体。 */
    std::vector<std::vector<int>>
    addTemplateVariantsBatch(const std::vector<ShapeTemplateTrainingInput> &,
                             const std::vector<ShapeTemplateVariant> &) final;
    /** @brief 在内存图像中执行模板匹配。 */
    std::vector<ShapeTemplateMatch> match(const cv::Mat &, float = -1.0f, const cv::Mat & = cv::Mat(),
                                          ShapeTemplateMatchOptions = {}) const final;
    /** @brief 从图像文件执行模板匹配。 */
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &, float = -1.0f,
                                              const std::filesystem::path & = {},
                                              ShapeTemplateMatchOptions = {}) const final;
    /** @brief 清空当前模板库。 */
    void clear() final;
    /** @brief 判断当前模板库是否为空。 */
    bool empty() const noexcept final;
    /** @brief 获取全部模板数量。 */
    int numTemplates() const noexcept final;
    /** @brief 获取指定模板的元数据和特征。 */
    const ShapeTemplateInfo &getTemplate(int) const final;
    /** @brief 获取当前配置。 */
    const ShapeTemplateMatcherConfig &config() const noexcept final;
    /** @brief 保存紧凑 v3 模板文件。 */
    void save(const std::filesystem::path &) const final;
    /** @brief 加载紧凑 v3 模板文件。 */
    void load(const std::filesystem::path &) final;

    /** @brief 生成闭区间角度/尺度模板变体。 */
    static std::vector<ShapeTemplateVariant> makeAngleScaleVariants(float, float, float,
                                                                     float = 1.0f, float = 1.0f,
                                                                     float = 1.0f);
    /** @brief 按训练相同规则对图像做中心旋转/缩放。 */
    static cv::Mat transform(const cv::Mat &, ShapeTemplateVariant,
                             const cv::Scalar & = cv::Scalar());

protected:
    /** @brief 接管版本私有实现；仅供 v0/v1/v2 公开匹配器构造函数调用。 */
    explicit ShapeTemplateMatcherBase(std::unique_ptr<IShapeTemplateMatcher> implementation);

private:
    std::unique_ptr<IShapeTemplateMatcher> implementation_; ///< 版本私有的实际算法实现。
};

} // namespace irt::features::detail
