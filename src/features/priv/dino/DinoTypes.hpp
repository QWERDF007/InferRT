#pragma once

/**
 * @file DinoTypes.hpp
 * @brief DINO 区域检索的几何、视图、描述子与候选基础类型。
 *
 * 本文件是区域检索内部坐标约定的唯一来源：矩形为半开区间 ``[x0, y0, x1, y1)``，
 * 坐标归属由结构体名称显式表达（``canonical`` / ``input`` / ``patch``）。
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace irt::features::priv {

/** @brief 半开区间矩形 ``[x0, y0, x1, y1)``。 */
struct DinoRect
{
    double x0{0.0};
    double y0{0.0};
    double x1{0.0};
    double y1{0.0};
    double width() const noexcept { return x1 - x0; }
    double height() const noexcept { return y1 - y0; }
    double area() const noexcept { const auto w=width(); const auto h=height(); return w>0.0&&h>0.0?w*h:0.0; }
    bool empty() const noexcept { return !(width()>0.0)||!(height()>0.0); }
    bool contains(double x,double y) const noexcept { return x>=x0&&x<x1&&y>=y0&&y<y1; }
    static DinoRect fromCorners(double ax,double ay,double bx,double by) noexcept
    { return DinoRect{ax<bx?ax:bx, ay<by?ay:by, ax<bx?bx:ax, ay<by?by:ay}; }
    static DinoRect intersect(const DinoRect&a,const DinoRect&b) noexcept
    { DinoRect r{a.x0>b.x0?a.x0:b.x0,a.y0>b.y0?a.y0:b.y0,a.x1<b.x1?a.x1:b.x1,a.y1<b.y1?a.y1:b.y1}; if(r.x1<r.x0)r.x1=r.x0; if(r.y1<r.y0)r.y1=r.y0; return r; }
    static DinoRect unite(const DinoRect&a,const DinoRect&b) noexcept
    { return DinoRect{a.x0<b.x0?a.x0:b.x0,a.y0<b.y0?a.y0:b.y0,a.x1>b.x1?a.x1:b.x1,a.y1>b.y1?a.y1:b.y1}; }
};

/** @brief 二维点。 */
struct DinoPoint
{
    double x{0.0};
    double y{0.0};
};

/**
 * @brief 二维仿射变换，按行主序保存 ``[a b c; d e f; 0 0 1]``。
 *
 * 变换链一律由矩阵组合还原坐标，不对坐标做逐步整数取整累加。
 */
struct DinoAffine
{
    double m[6]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};

    static DinoAffine identity() noexcept
    {
        return DinoAffine{};
    }

    static DinoAffine scaleTranslate(const double sx, const double sy, const double tx, const double ty) noexcept
    {
        return DinoAffine{sx, 0.0, tx, 0.0, sy, ty};
    }

    /** @brief 组合变换：先应用 ``other``，再应用当前变换。 */
    DinoAffine compose(const DinoAffine &other) const noexcept
    {
        DinoAffine result;
        result.m[0] = m[0] * other.m[0] + m[1] * other.m[3];
        result.m[1] = m[0] * other.m[1] + m[1] * other.m[4];
        result.m[2] = m[0] * other.m[2] + m[1] * other.m[5] + m[2];
        result.m[3] = m[3] * other.m[0] + m[4] * other.m[3];
        result.m[4] = m[3] * other.m[1] + m[4] * other.m[4];
        result.m[5] = m[3] * other.m[2] + m[4] * other.m[5] + m[5];
        return result;
    }

    /** @brief 求逆；行列式接近零时返回单位矩阵。 */
    DinoAffine inverse() const noexcept
    {
        const double det = m[0] * m[4] - m[1] * m[3];
        if (det > -1e-12 && det < 1e-12)
        {
            return DinoAffine{};
        }
        const double inv_det = 1.0 / det;
        DinoAffine result;
        result.m[0] = m[4] * inv_det;
        result.m[1] = -m[1] * inv_det;
        result.m[3] = -m[3] * inv_det;
        result.m[4] = m[0] * inv_det;
        result.m[2] = -(result.m[0] * m[2] + result.m[1] * m[5]);
        result.m[5] = -(result.m[3] * m[2] + result.m[4] * m[5]);
        return result;
    }

    DinoPoint apply(const double x, const double y) const noexcept
    {
        return DinoPoint{m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]};
    }

    double applyX(const double x, const double y) const noexcept
    {
        return m[0] * x + m[1] * y + m[2];
    }

    double applyY(const double x, const double y) const noexcept
    {
        return m[3] * x + m[4] * y + m[5];
    }

    /** @brief 变换矩形；仅对轴对齐、无旋转的仿射保证仍是矩形。 */
    DinoRect apply(const DinoRect &rect) const noexcept
    {
        const auto tl = apply(rect.x0, rect.y0);
        const auto br = apply(rect.x1, rect.y1);
        return DinoRect::fromCorners(tl.x, tl.y, br.x, br.y);
    }

    /** @brief 轴向缩放因子，用于换算采样间隔。 */
    double scaleX() const noexcept
    {
        return m[0];
    }

    double scaleY() const noexcept
    {
        return m[4];
    }
};

/** @brief canonical 图像上的查询区域，bbox 与 polygon 互斥。 */
struct DinoRoi
{
    bool                   is_polygon{false};
    DinoRect               bbox{};
    std::vector<DinoPoint> polygon{};

    /** @brief 返回包围盒；polygon 模式由顶点派生。 */
    DinoRect boundingBox() const noexcept
    {
        if (!is_polygon)
        {
            return bbox;
        }
        if (polygon.empty())
        {
            return DinoRect{};
        }
        DinoRect result{polygon.front().x, polygon.front().y, polygon.front().x, polygon.front().y};
        for (const auto &point : polygon)
        {
            if (point.x < result.x0)
            {
                result.x0 = point.x;
            }
            if (point.y < result.y0)
            {
                result.y0 = point.y;
            }
            if (point.x > result.x1)
            {
                result.x1 = point.x;
            }
            if (point.y > result.y1)
            {
                result.y1 = point.y;
            }
        }
        return result;
    }
};

/**
 * @brief 一张图库图像的身份与几何记录。
 *
 * 图像身份由规范化源路径确定；文件大小与修改时间用于缓存失效。
 */
struct DinoImageIdentity
{
    std::string image_id{};         ///< 规范化源路径 UTF-8。
    std::string source_path{};      ///< 规范化绝对路径。
    int         width{0};           ///< canonical 宽度。
    int         height{0};          ///< canonical 高度。
    int64_t     file_size{0};
    int64_t     mtime_ns{0};
    bool        exif_orientation_applied{false};
    std::string decode_pipeline{};  ///< 解码库与方向处理标识，进入签名。
    std::string failure_reason{};   ///< 非空表示该文件未能写入索引。
};

/**
 * @brief 一个视图的几何规划结果。
 *
 * ``source_rect`` 是 canonical 空间中映射到模型输入光栅的区域（可能超出原图边界）；
 * ``valid_input_rect`` 是输入光栅中真正承载有效原图像素的矩形，padding 部分不计入
 * 池化权重，也不参与局部描述。
 */
struct DinoViewPlan
{
    DinoRect   source_rect{};      ///< canonical 空间源区域，可超出原图。
    DinoRect   valid_input_rect{}; ///< 输入光栅中的有效像素矩形。
    DinoAffine canonical_to_input{}; ///< canonical px -> 模型输入 px。
    DinoAffine input_to_canonical{}; ///< 模型输入 px -> canonical px。
    int        input_width{0};     ///< 模型输入宽度。
    int        input_height{0};    ///< 模型输入高度。
    int        patch_size{0};      ///< 骨干 patch 边长。
    int        grid_height{0};     ///< patch 网格高度。
    int        grid_width{0};      ///< patch 网格宽度。
    bool       is_full_view{false}; ///< 是否为整图视图。
    int        scale_index{-1};     ///< 所属源边长尺度下标；整图视图为 -1。

    /** @brief 输入光栅 1 像素对应的 canonical 像素数。 */
    double canonicalPerInputPxX() const noexcept
    {
        return canonical_to_input.scaleX() > 0.0 ? 1.0 / canonical_to_input.scaleX() : 0.0;
    }

    double canonicalPerInputPxY() const noexcept
    {
        return canonical_to_input.scaleY() > 0.0 ? 1.0 / canonical_to_input.scaleY() : 0.0;
    }

    /** @brief 单个 patch 在该轴上覆盖的 canonical 像素数。 */
    double sourcePxPerPatchX() const noexcept
    {
        return static_cast<double>(patch_size) * canonicalPerInputPxX();
    }

    double sourcePxPerPatchY() const noexcept
    {
        return static_cast<double>(patch_size) * canonicalPerInputPxY();
    }

    /** @brief 第 ``row``/``col`` 个 patch 在 canonical 空间覆盖的矩形。 */
    DinoRect patchRect(const int row, const int col) const noexcept
    {
        const double x0 = static_cast<double>(col) * patch_size;
        const double y0 = static_cast<double>(row) * patch_size;
        return input_to_canonical.apply(
            DinoRect{x0, y0, x0 + static_cast<double>(patch_size), y0 + static_cast<double>(patch_size)});
    }

    int patchCount() const noexcept
    {
        return grid_height * grid_width;
    }
};

/** @brief 一个视图的模型输入光栅与 patch 网格。 */
struct DinoFeatureGrid
{
    DinoViewPlan       plan{};      ///< 视图几何。
    int                channels{0}; ///< 特征维度 D。
    std::vector<float> tokens{};    ///< ``grid_height * grid_width * channels``，逐 patch L2 归一化。
    std::vector<float> valid_area{}; ///< 逐 patch 有效面积占比 ``[0, 1]``。

    const float *token(const int row, const int col) const noexcept
    {
        return tokens.data()
             + (static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width) + static_cast<size_t>(col))
                   * static_cast<size_t>(channels);
    }

    bool patchValid(const int row, const int col) const noexcept
    {
        return valid_area[static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width)
                          + static_cast<size_t>(col)]
             > 0.0f;
    }
};

/** @brief 区域整体描述：窗口池化向量与其 canonical 覆盖范围。 */
struct DinoRegionDescriptor
{
    int      view_id{0};
    int      grid_row{0};
    int      grid_col{0};
    int      grid_height{0};
    int      grid_width{0};
    DinoRect source_rect{};  ///< 窗口在 canonical 空间的覆盖范围。
    float    valid_fraction{0.0f};
};

/** @brief 局部描述叶节点：代表特征与空间覆盖。 */
struct DinoLocalLeaf
{
    int      view_id{0};
    int      grid_row{0};       ///< 叶节点覆盖矩形起点行。
    int      grid_col{0};       ///< 叶节点覆盖矩形起点列。
    int      grid_height{0};    ///< 叶节点覆盖矩形高度（patch 数）。
    int      grid_width{0};     ///< 叶节点覆盖矩形宽度（patch 数）。
    int      rep_row{0};        ///< 代表 patch 行。
    int      rep_col{0};        ///< 代表 patch 列。
    int      member_count{0};   ///< 有效成员 patch 数。
    float    radius{0.0f};      ///< 原始 FP32 归一化特征上的最大替换距离。
};

/** @brief 未压缩基线条目：逐有效 patch 的独立局部描述。 */
struct DinoLocalPatch
{
    int   view_id{0};
    int   grid_row{0};
    int   grid_col{0};
    float valid_area{0.0f};
};

/** @brief 粗选候选。 */
struct DinoCandidate
{
    std::string image_id{};
    int        view_id{0};
    int        query_view_id{0};
    bool       from_region{false};
    bool       from_local{false};
    float      region_score{-std::numeric_limits<float>::infinity()};
    float      local_score{-std::numeric_limits<float>::infinity()};
    float      score{0.0f};
    DinoRect   source_bbox{};  ///< canonical 空间的候选区域。
};

/** @brief 最终结果框。 */
struct DinoMatchResult
{
    std::string image_id{};
    std::string source_path{};
    DinoRect    bbox{};       ///< canonical 空间结果框。
    float       score{0.0f};
    float       template_similarity{0.0f};
    float       query_coverage{0.0f};
    float       spatial_consistency{0.0f};
    bool        from_region_channel{false};
    bool        from_local_channel{false};
    int         coarse_view_id{-1};
};

/** @brief 一条 INT8 编码后的描述向量。 */
struct DinoQuantizedVector
{
    std::vector<int8_t> codes{};  ///< INT8 码字。
    float               scale{0.0f};    ///< ``max(|r|)/127``。
    float               inv_norm{0.0f}; ///< ``1 / ||scale * codes||2``。
};

} // namespace irt::features::priv
