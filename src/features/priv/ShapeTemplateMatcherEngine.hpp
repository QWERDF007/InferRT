#pragma once

/**
 * @file ShapeTemplateMatcherEngine.hpp
 * @brief 形状模板匹配的版本无关引擎与可替换 SIMD 内核。
 */

#include <inferrt/features/ShapeTemplateMatcherTypes.hpp>

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
    std::array<float, kShapeTemplateOrientationBins> average_responses{}; ///< 各模板方向响应均值，用于 v1 剪枝排序。
    cv::Mat quantized_labels;                                ///< 源图量化方向；v1 可直接 SIMD 查表，减少响应图切换。
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

/** @brief 内部热点实现类型；新增 AVX512 时仅需添加一个枚举值和对应 kernel。 */
enum class ShapeTemplateMatcherBackend
{
    Scalar,
    Avx2,
};

/**
 * @brief SIMD 相关热点的策略接口。
 *
 * 引擎负责模板管理、序列化、NMS 和滑窗调度；不同版本仅替换这些热点。未来 AVX512
 * 实现只需新增一个 kernel、一个 ``ShapeTemplateMatcherBackend`` 枚举值和一个版本包装器，
 * 无需修改训练、匹配、持久化或 NMS 流程。
 */
class ShapeTemplateMatcherKernel
{
public:
    virtual ~ShapeTemplateMatcherKernel() = default;

    /** @brief 当前版本是否需要引擎物化 8 张响应图。 */
    virtual bool needsMaterializedResponseMaps() const noexcept = 0;
    virtual void fillQuantizedLabels(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                                     float threshold, cv::Mat &labels) const = 0;
    virtual std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const = 0;
    virtual void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                                 const ShapeTemplateResponseTable &table) const = 0;

    /**
     * @brief 扫描一个模板并返回达到阈值的位置。
     *
     * v0 保持参考 ``shape_based_matching`` 的标量逐位置累加语义；v1 可将同一语义替换为
     * 批量 SIMD 累加与早停。引擎仍统一负责模板调度、类别信息、排序和 NMS，因此未来 v2/AVX512
     * 只需实现该热点接口。
    */
    virtual std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, cv::Size image_size, int scan_step, float threshold) const = 0;
};

/** @brief 根据内部后端创建热点内核。 */
std::unique_ptr<ShapeTemplateMatcherKernel>
createShapeTemplateMatcherKernel(ShapeTemplateMatcherBackend backend);

/** @brief 由标量 kernel 翻译单元提供。 */
std::unique_ptr<ShapeTemplateMatcherKernel> createScalarShapeTemplateMatcherKernel();

/** @brief 由 AVX2 kernel 翻译单元提供。 */
std::unique_ptr<ShapeTemplateMatcherKernel> createAvx2ShapeTemplateMatcherKernel();

/** @brief 供各 kernel 共享的稳定方向量化和候选排序规则。 */
int  quantizeShapeTemplateAngle(float angle_degrees) noexcept;
void sortShapeTemplateCandidates(std::vector<ShapeTemplateCandidate> &candidates);

/**
 * @brief 与实现版本无关的模板管理与匹配引擎。
 */
class ShapeTemplateMatcherEngine
{
public:
    ShapeTemplateMatcherEngine(ShapeTemplateMatcherConfig config,
                               std::unique_ptr<ShapeTemplateMatcherKernel> kernel);
    ~ShapeTemplateMatcherEngine();

    ShapeTemplateMatcherEngine(const ShapeTemplateMatcherEngine &)            = delete;
    ShapeTemplateMatcherEngine &operator=(const ShapeTemplateMatcherEngine &) = delete;
    ShapeTemplateMatcherEngine(ShapeTemplateMatcherEngine &&) noexcept;
    ShapeTemplateMatcherEngine &operator=(ShapeTemplateMatcherEngine &&) noexcept;

    int addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask,
                    ShapeTemplateVariant variant);
    int addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                        const std::filesystem::path &mask_file, ShapeTemplateVariant variant);
    std::vector<int> addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                         const cv::Mat &object_mask,
                                         const std::vector<ShapeTemplateVariant> &variants);
    std::vector<ShapeTemplateMatch> match(const cv::Mat &image, float threshold,
                                          const std::vector<std::string> &class_ids,
                                          const cv::Mat &search_mask,
                                          ShapeTemplateMatchOptions options) const;
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &image_file, float threshold,
                                              const std::vector<std::string> &class_ids,
                                              const std::filesystem::path &mask_file,
                                              ShapeTemplateMatchOptions options) const;
    void clear();
    bool empty() const noexcept;
    int numClasses() const noexcept;
    int numTemplates() const noexcept;
    int numTemplates(const std::string &class_id) const noexcept;
    std::vector<std::string> classIds() const;
    const ShapeTemplateInfo &getTemplate(const std::string &class_id, int template_id) const;
    const ShapeTemplateMatcherConfig &config() const noexcept;
    void save(const std::filesystem::path &template_file) const;
    void load(const std::filesystem::path &template_file);

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
