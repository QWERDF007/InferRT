#pragma once

/**
 * @file ShapeTemplateMatcherEngine.hpp
 * @brief 形状模板匹配的版本无关引擎与可替换 SIMD 内核。
 */

#include <inferrt/features/IShapeTemplateMatcher.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace irt::features::detail {

inline constexpr int           kShapeTemplateOrientationBins = 8;
inline constexpr unsigned char kShapeTemplateInvalidLabel     = 255;

/** @brief 源图梯度幅值、角度和量化标签。 */
struct ShapeTemplateQuantizedGradient
{
    cv::Mat labels;
    cv::Mat magnitude;
    cv::Mat angle;
};

/** @brief 模板训练阶段的候选特征点。 */
struct ShapeTemplateCandidate
{
    int   x{0};
    int   y{0};
    int   label{0};
    float angle_degrees{0.0f};
    float score{0.0f};
};

using ShapeTemplateResponseTable = std::array<std::array<unsigned char, kShapeTemplateOrientationBins>,
                                              kShapeTemplateOrientationBins>;

/** @brief 模板匹配使用的源图响应数据，以及单特征分数的整数分母。 */
struct ShapeTemplateResponseMaps
{
    std::array<cv::Mat, kShapeTemplateOrientationBins> maps; ///< 按模板方向索引的 ``CV_8U`` 响应图。
    std::array<float, kShapeTemplateOrientationBins> average_responses{}; ///< 各模板方向响应均值，用于 v1/v2 剪枝排序。
    cv::Mat quantized_labels;                                ///< 源图量化方向；v1/v2 可直接 SIMD 查表，减少响应图切换。
    ShapeTemplateResponseTable response_table{};             ///< 方向标签到响应分子的稳定查找表。
    int denominator_per_feature{1};                          ///< 单个特征点满分对应的分母。
};

/** @brief 内核完成滑窗评分后返回的候选位置。 */
struct ShapeTemplateScoredPosition
{
    int   x{0};
    int   y{0};
    float score{0.0f};
};

/**
 * @brief SIMD 相关热点的策略接口。
 *
 * @details 引擎负责模板管理、序列化、NMS 和滑窗调度；不同版本只注入各自的热点内核。
 * 后续新增 AVX512 时无需修改公共流程。
 */
class ShapeTemplateMatcherKernel
{
public:
    virtual ~ShapeTemplateMatcherKernel() = default;

    /** @brief 当前版本是否需要引擎物化 8 张响应图。 */
    virtual bool needsMaterializedResponseMaps() const noexcept = 0;
    /** @brief 当前版本是否使用 SIMD 专属的并行模板训练路径。 */
    virtual bool usesOptimizedTemplateTraining() const noexcept = 0;
    virtual void fillQuantizedLabels(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                                     float threshold, cv::Mat &labels) const = 0;
    virtual std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const = 0;
    virtual void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                                 const ShapeTemplateResponseTable &table) const = 0;

    /**
     * @brief 扫描一个模板并返回达到阈值的位置。
     *
     * v0 保持参考 ``shape_based_matching`` 的标量逐位置累加语义；v1/v2 可将同一语义替换为
     * 批量 SIMD 累加与早停。引擎仍统一负责模板调度、类别信息、排序和 NMS。
     * @param full_search_mask ``true`` 表示搜索掩膜没有零值；SIMD 版本可据此跳过候选掩膜判断。
    */
    virtual std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size, int scan_step,
                 float threshold) const = 0;
};

/** @brief 供各 kernel 共享的稳定方向量化和候选排序规则。 */
int  quantizeShapeTemplateAngle(float angle_degrees) noexcept;
void sortShapeTemplateCandidates(std::vector<ShapeTemplateCandidate> &candidates);

/**
 * @brief 与版本无关的模板流程核心。
 *
 * @details 具体 v0/v1/v2 实现只在构造时传入不同内核；本类统一执行训练、持久化、匹配和 NMS，
 * 从而保证各版本的流程和结果数据结构一致。
 */
class ShapeTemplateMatcherEngine : public IShapeTemplateMatcher
{
public:
    ShapeTemplateMatcherEngine(ShapeTemplateMatcherConfig config,
                               std::unique_ptr<ShapeTemplateMatcherKernel> kernel);
    ~ShapeTemplateMatcherEngine() override;

    ShapeTemplateMatcherEngine(const ShapeTemplateMatcherEngine &)            = delete;
    ShapeTemplateMatcherEngine &operator=(const ShapeTemplateMatcherEngine &) = delete;
    ShapeTemplateMatcherEngine(ShapeTemplateMatcherEngine &&) noexcept;
    ShapeTemplateMatcherEngine &operator=(ShapeTemplateMatcherEngine &&) noexcept;

    /** @brief 添加单个模板。 */
    int addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
                    ShapeTemplateVariant variant) override;
    /** @brief 从文件添加单个模板。 */
    int addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                        const std::filesystem::path &mask_file, ShapeTemplateVariant variant) override;
    /** @brief 批量训练角度/尺度模板变体。 */
    std::vector<int> addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                         const cv::Mat &object_mask,
                                         const std::vector<ShapeTemplateVariant> &variants) override;
    /** @brief 在内存图像中执行匹配。 */
    std::vector<ShapeTemplateMatch> match(const cv::Mat &image, float threshold,
                                          const std::vector<std::string> &class_ids,
                                          const cv::Mat &search_mask,
                                          ShapeTemplateMatchOptions options) const override;
    /** @brief 从文件执行匹配。 */
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &image_file, float threshold,
                                              const std::vector<std::string> &class_ids,
                                              const std::filesystem::path &mask_file,
                                              ShapeTemplateMatchOptions options) const override;
    /** @brief 清空模板库。 */
    void clear() override;
    /** @brief 判断模板库是否为空。 */
    bool empty() const noexcept override;
    /** @brief 获取类别数量。 */
    int numClasses() const noexcept override;
    /** @brief 获取全部模板数量。 */
    int numTemplates() const noexcept override;
    /** @brief 获取指定类别的模板数量。 */
    int numTemplates(const std::string &class_id) const noexcept override;
    /** @brief 获取所有类别 ID。 */
    std::vector<std::string> classIds() const override;
    /** @brief 获取指定模板信息。 */
    const ShapeTemplateInfo &getTemplate(const std::string &class_id, int template_id) const override;
    /** @brief 获取配置。 */
    const ShapeTemplateMatcherConfig &config() const noexcept override;
    /** @brief 保存紧凑 v2 模板。 */
    void save(const std::filesystem::path &template_file) const override;
    /** @brief 加载紧凑 v2 模板。 */
    void load(const std::filesystem::path &template_file) override;

private:
    using TemplateMap = std::map<std::string, std::vector<ShapeTemplateInfo>>;

    ShapeTemplateMatcherConfig                 config_{};
    TemplateMap                                templates_;
    std::unique_ptr<ShapeTemplateMatcherKernel> kernel_;
};

std::vector<ShapeTemplateVariant>
makeShapeTemplateAngleScaleVariants(float angle_begin_degrees, float angle_end_degrees, float angle_step_degrees,
                                    float scale_begin, float scale_end, float scale_step);
cv::Mat transformShapeTemplateImage(const cv::Mat &image, ShapeTemplateVariant variant,
                                    const cv::Scalar &border_value);

} // namespace irt::features::detail
