#pragma once

/**
 * @file ShapeTemplateMatcher.hpp
 * @brief 基于梯度方向的形状模板匹配公共 API。
 */

#include <inferrt/features/Export.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features {

namespace scalar {
class ShapeTemplateMatcher;
}

inline constexpr int   kDefaultShapeTemplateNumFeatures     = 128;   ///< 默认每个模板最多保留的特征点数。
inline constexpr int   kDefaultShapeTemplateMinFeatures     = 4;     ///< 默认每个模板至少需要的有效特征点数。
inline constexpr float kDefaultShapeTemplateWeakThreshold   = 30.0f; ///< 默认源图梯度弱阈值。
inline constexpr float kDefaultShapeTemplateStrongThreshold = 60.0f; ///< 默认模板候选特征强阈值。
inline constexpr float kDefaultShapeTemplateMatchThreshold  = 80.0f; ///< 默认匹配分数阈值。
inline constexpr float kDefaultShapeTemplateNmsThreshold    = 0.3f;  ///< 默认同类别 NMS IoU 阈值。

/**
 * @brief 形状模板匹配器配置。
 */
struct ShapeTemplateMatcherConfig
{
    int   num_features{kDefaultShapeTemplateNumFeatures};        ///< 每个模板最多保留的梯度方向特征点数。
    int   min_features{kDefaultShapeTemplateMinFeatures};        ///< 模板训练成功所需的最少有效特征点数。
    float weak_threshold{kDefaultShapeTemplateWeakThreshold};     ///< 源图量化梯度方向时使用的幅值阈值。
    float strong_threshold{kDefaultShapeTemplateStrongThreshold}; ///< 模板候选特征点的梯度幅值阈值。
    int   max_label_difference{1};                               ///< 环形方向 bin 容差，合法范围为 ``[0, 4]``。
    float match_threshold{kDefaultShapeTemplateMatchThreshold};   ///< 默认匹配分数阈值，合法范围为 ``[0, 100]``。
    float nms_threshold{kDefaultShapeTemplateNmsThreshold};       ///< 同类别 NMS 的 IoU 阈值；小于 0 时关闭 NMS。
    int   max_results{0};                                        ///< 最多返回的匹配数量；0 表示不限制。
    int   scan_step{1};                                          ///< 滑窗扫描步长，单位为像素。
    int   max_parallelism{0};                                    ///< 模板扫描工作线程数；0 表示自动，1 表示串行。
    float min_feature_distance{0.0f};                            ///< 贪心选点最小间距；0 表示按模板面积自动估计。
};

/**
 * @brief 生成旋转/缩放模板变体时使用的元数据。
 */
struct ShapeTemplateVariant
{
    float angle_degrees{0.0f}; ///< 训练图像逆时针旋转角度，单位为度。
    float scale{1.0f};         ///< 训练图像缩放倍率。
};

/**
 * @brief 模板中的一个量化梯度方向特征点。
 */
struct ShapeTemplateFeature
{
    int   x{0};                ///< 相对模板左上角的 x 坐标。
    int   y{0};                ///< 相对模板左上角的 y 坐标。
    int   label{0};            ///< 量化方向标签，范围为 ``[0, 7]``。
    float angle_degrees{0.0f}; ///< 原始梯度方向角，单位为度。
};

/**
 * @brief 已训练模板的元数据和选中的特征点集合。
 */
struct ShapeTemplateInfo
{
    std::string class_id;            ///< 调用方提供的类别 ID。
    int         template_id{-1};     ///< 当前类别内的模板 ID。
    int         width{0};            ///< 裁剪后的模板特征包围盒宽度。
    int         height{0};           ///< 裁剪后的模板特征包围盒高度。
    int         tl_x{0};             ///< 特征包围盒在训练图像中的左上角 x 坐标。
    int         tl_y{0};             ///< 特征包围盒在训练图像中的左上角 y 坐标。
    float       angle_degrees{0.0f}; ///< 模板变体对应的旋转角度元数据。
    float       scale{1.0f};         ///< 模板变体对应的缩放倍率元数据。

    std::vector<ShapeTemplateFeature> features; ///< 模板保留的稀疏梯度方向特征点。
};

/**
 * @brief 一个形状模板匹配结果。
 */
struct ShapeTemplateMatch
{
    int         x{0};                ///< 源图坐标系下匹配框左上角 x 坐标。
    int         y{0};                ///< 源图坐标系下匹配框左上角 y 坐标。
    int         width{0};            ///< 匹配模板宽度。
    int         height{0};           ///< 匹配模板高度。
    float       similarity{0.0f};    ///< 梯度方向一致性分数，范围为 ``[0, 100]``。
    std::string class_id;            ///< 命中的类别 ID。
    int         template_id{-1};     ///< 命中的类别内模板 ID。
    float       angle_degrees{0.0f}; ///< 命中模板的旋转角度元数据。
    float       scale{1.0f};         ///< 命中模板的缩放倍率元数据。
};

/**
 * @brief 使用量化图像梯度的形状模板匹配器。
 *
 * 实现参考 ``shape_based_matching``/LINEMOD 的核心思路：训练阶段把目标形状表示为稀疏梯度方向特征点，
 * 匹配阶段在源图中滑窗统计特征点平移后的方向一致性，并按分数和同类别 NMS 返回结果。
 */
class INFERRT_FEATURES_API ShapeTemplateMatcher
{
public:
    /**
     * @brief 构造形状模板匹配器。
     * @param config 匹配器配置。
     */
    explicit ShapeTemplateMatcher(ShapeTemplateMatcherConfig config = {});

    /** @brief 析构形状模板匹配器。 */
    ~ShapeTemplateMatcher();

    ShapeTemplateMatcher(const ShapeTemplateMatcher &)            = delete;
    ShapeTemplateMatcher &operator=(const ShapeTemplateMatcher &) = delete;

    /** @brief 移动构造形状模板匹配器。 */
    ShapeTemplateMatcher(ShapeTemplateMatcher &&other) noexcept;

    /** @brief 移动赋值形状模板匹配器。 */
    ShapeTemplateMatcher &operator=(ShapeTemplateMatcher &&other) noexcept;

    /**
     * @brief 从内存图像和可选目标掩膜训练一个模板。
     * @param image 训练图像，支持灰度、BGR 或 BGRA。
     * @param class_id 调用方提供的类别 ID。
     * @param object_mask 可选目标掩膜，非零区域参与训练。
     * @param variant 模板变体的旋转/缩放元数据。
     * @return 当前类别内的模板 ID。
     */
    int addTemplate(const cv::Mat &image, const std::string &class_id, const cv::Mat &object_mask = cv::Mat(),
                    ShapeTemplateVariant variant = {});

    /**
     * @brief 从图像文件和可选掩膜文件训练一个模板。
     * @param image_file 训练图像文件路径。
     * @param class_id 调用方提供的类别 ID。
     * @param mask_file 可选目标掩膜文件路径。
     * @param variant 模板变体的旋转/缩放元数据。
     * @return 当前类别内的模板 ID。
     */
    int addTemplateFile(const std::filesystem::path &image_file, const std::string &class_id,
                        const std::filesystem::path &mask_file = {}, ShapeTemplateVariant variant = {});

    /**
     * @brief 基于一张训练图生成并添加多组旋转/缩放模板。
     * @param image 原始训练图像。
     * @param class_id 调用方提供的类别 ID。
     * @param object_mask 可选目标掩膜，非零区域参与训练。
     * @param variants 需要训练的旋转/缩放变体列表。
     * @return 成功添加的模板 ID 列表，顺序与 ``variants`` 一致。
     */
    std::vector<int> addTemplateVariants(const cv::Mat &image, const std::string &class_id,
                                         const cv::Mat &object_mask,
                                         const std::vector<ShapeTemplateVariant> &variants);

    /**
     * @brief 在源图中匹配已训练模板。
     * @param image 待搜索源图，支持灰度、BGR 或 BGRA。
     * @param threshold 匹配阈值；小于 0 时使用配置中的 ``match_threshold``。
     * @param class_ids 可选类别过滤列表；为空时匹配所有类别。
     * @param search_mask 可选搜索掩膜，非零区域参与搜索。
     * @return 按相似度从高到低排序并经过 NMS/数量限制后的匹配结果。
     */
    std::vector<ShapeTemplateMatch> match(const cv::Mat &image, float threshold = -1.0f,
                                          const std::vector<std::string> &class_ids = {},
                                          const cv::Mat                  &search_mask = cv::Mat()) const;

    /**
     * @brief 从源图文件中匹配已训练模板。
     * @param image_file 待搜索源图文件路径。
     * @param threshold 匹配阈值；小于 0 时使用配置中的 ``match_threshold``。
     * @param class_ids 可选类别过滤列表；为空时匹配所有类别。
     * @param mask_file 可选搜索掩膜文件路径。
     * @return 按相似度从高到低排序并经过 NMS/数量限制后的匹配结果。
     */
    std::vector<ShapeTemplateMatch> matchFile(const std::filesystem::path &image_file, float threshold = -1.0f,
                                              const std::vector<std::string> &class_ids = {},
                                              const std::filesystem::path    &mask_file = {}) const;

    /** @brief 清空所有已训练模板。 */
    void clear();

    /** @brief 判断当前是否没有任何模板。 */
    bool empty() const noexcept;

    /** @brief 获取已训练类别数量。 */
    int numClasses() const noexcept;

    /** @brief 获取所有类别下的模板总数。 */
    int numTemplates() const noexcept;

    /** @brief 获取指定类别下的模板数量。 */
    int numTemplates(const std::string &class_id) const noexcept;

    /** @brief 获取所有类别 ID，按字符串升序返回。 */
    std::vector<std::string> classIds() const;

    /**
     * @brief 获取指定类别和模板 ID 对应的模板信息。
     * @throws irt::Exception 当类别或模板 ID 不存在时抛出。
     */
    const ShapeTemplateInfo &getTemplate(const std::string &class_id, int template_id) const;

    /** @brief 获取当前匹配器配置。 */
    const ShapeTemplateMatcherConfig &config() const noexcept;

    /**
     * @brief 将已训练模板和配置保存为 OpenCV YAML/XML 文件。
     * @param template_file 输出模板文件路径。
     */
    void save(const std::filesystem::path &template_file) const;

    /**
     * @brief 从 OpenCV YAML/XML 文件加载模板和配置。
     * @param template_file 输入模板文件路径。
     */
    void load(const std::filesystem::path &template_file);

    /**
     * @brief 根据闭区间角度/尺度范围生成模板变体列表。
     * @param angle_begin_degrees 起始角度，单位为度。
     * @param angle_end_degrees 结束角度，单位为度，包含端点。
     * @param angle_step_degrees 角度步长，必须为正数。
     * @param scale_begin 起始缩放倍率。
     * @param scale_end 结束缩放倍率，包含端点。
     * @param scale_step 缩放步长，必须为正数。
     * @return 先按 scale、再按 angle 展开的模板变体列表。
     */
    static std::vector<ShapeTemplateVariant> makeAngleScaleVariants(float angle_begin_degrees,
                                                                    float angle_end_degrees,
                                                                    float angle_step_degrees, float scale_begin = 1.0f,
                                                                    float scale_end = 1.0f,
                                                                    float scale_step = 1.0f);

    /**
     * @brief 对图像应用与模板变体训练一致的中心旋转/缩放变换。
     * @param image 输入图像。
     * @param variant 旋转/缩放参数。
     * @param border_value 仿射变换的边界填充值。
     * @return 变换后的图像，尺寸与输入图像一致。
     */
    static cv::Mat transform(const cv::Mat &image, ShapeTemplateVariant variant,
                             const cv::Scalar &border_value = cv::Scalar());

private:
    ShapeTemplateMatcher(ShapeTemplateMatcherConfig config, bool use_avx2);
    friend class scalar::ShapeTemplateMatcher;

    class Impl;
    std::unique_ptr<Impl> impl_; ///< 私有实现，隐藏 OpenCV 训练、匹配和序列化细节。
};

} // namespace irt::features
