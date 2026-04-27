#include "OpResizeImpl.hpp"

#include "saturate.cuh"
#include "type.cuh"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpResize.h>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda::priv {

// OpenCV的定点算术常量
static const int INTER_RESIZE_COEF_BITS    = 11;
static const int INTER_RESIZE_COEF_SCALE   = 1 << INTER_RESIZE_COEF_BITS;                       // 2048
static const int SHIFT                     = INTER_RESIZE_COEF_BITS * 2;                        // 22
static const int DELTA                     = 1 << (SHIFT - 1);                                  // 2097152
static const int INTER_RESIZE_COEF_SCALE_2 = INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE; // 4194304

/**
 * @brief Resize 区域坐标和插值系数计算 (bilinear/bicubic/lanczos)
 * 
 * @tparam CT2 插值小数部分计算类型 (如 float2, double2)
 * @param[in] dst_coord 目标像素中心坐标
 * @param[in] scale 缩放比例
 * @param[out] s 插值起始坐标
 * @param[out] f 插值小数部分
 */
template<typename CT2>
__device__ __forceinline__ void cal_interpolation(const int2 dst_coord, const double2 scale, int2 &s, CT2 &f)
{
    CT2 src_coord;
    src_coord.x = (dst_coord.x + 0.5) * scale.x - 0.5;
    src_coord.y = (dst_coord.y + 0.5) * scale.y - 0.5;

    if constexpr (std::is_same_v<CT2, double2>)
    {
        s.x = __double2int_rd(src_coord.x);
        s.y = __double2int_rd(src_coord.y);
    }
    else if constexpr (std::is_same_v<CT2, float2>)
    {
        s.x = __float2int_rd(src_coord.x);
        s.y = __float2int_rd(src_coord.y);
    }
    else
    {
        s.x = __double2int_rd(src_coord.x);
        s.y = __double2int_rd(src_coord.y);
    }

    f.x = src_coord.x - s.x;
    f.y = src_coord.y - s.y;
}

/**
 * @brief Resize Linear 区域坐标和插值系数计算
 * 
 * @tparam CT2 插值小数部分计算类型 (如 float2, double2)
 * @param[in] ssize 源图像尺寸
 * @param[in] dst_coord 目标像素中心坐标
 * @param[in] scale 缩放比例
 * @param[out] s 插值起始坐标
 * @param[out] f 插值小数部分
 * @param[out] s1 插值结束坐标
 */
template<typename CT2>
__device__ __forceinline__ void cal_bilinear_interpolation(const int2 ssize, const int2 dst_coord, const double2 scale,
                                                           int2 &s, CT2 &f, int2 &s1)
{
    cal_interpolation<CT2>(dst_coord, scale, s, f);

    if (s.x < 0)
    {
        s.x = 0;
        f.x = 0.0;
    }
    else if (s.x >= ssize.x - 1)
    {
        s.x = ssize.x - 1;
        f.x = 0.0;
    }

    if (s.y < 0)
    {
        s.y = 0;
        f.y = 0.0;
    }
    else if (s.y >= ssize.y - 1)
    {
        s.y = ssize.y - 1;
        f.y = 0.0;
    }

    s1.x = min(s.x + 1, ssize.x - 1);
    s1.y = min(s.y + 1, ssize.y - 1);
}

/**
 * @brief 双线性插值图像缩放
 * 
 * @tparam T 图像数据类型 (如 float, uint8_t)
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale X轴缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 * @note +0.5: 像素通常认为是一个 小方格，而不是一个点
 *      (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
 *      -0.5: 保证源图和目标图的像素中心对齐，避免锯齿
 */
template<typename T, typename CT, int CH>
__global__ void resize_bilinear_kernel(const T *src, T *dst, const double2 scale, const int2 ssize, const int sstride,
                                       const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;
    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // sx, sy
    int2 s1; // sx+1, sy+1

    using CT2 = make_vector2_t<CT>;
    CT2 f;

    cal_bilinear_interpolation<CT2>(ssize, dst_coord, scale, s, f, s1);

    using CT4 = make_vector4_t<CT>;
    CT4 w;

    // 双线性插值的四个权重: 左上、右上、左下、右下
    w.x = (1 - f.x) * (1 - f.y);
    w.y = f.x * (1 - f.y);
    w.z = (1 - f.x) * f.y;
    w.w = f.x * f.y;

    T *v1 = const_cast<T *>(src) + s.y * sstride + s.x * CH;
    T *v2 = const_cast<T *>(src) + s.y * sstride + s1.x * CH;
    T *v3 = const_cast<T *>(src) + s1.y * sstride + s.x * CH;
    T *v4 = const_cast<T *>(src) + s1.y * sstride + s1.x * CH;

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        CT v              = w.x * v1[i] + w.y * v2[i] + w.z * v3[i] + w.w * v4[i];
        dst[dst_base + i] = saturate_cast<T>(v);
    }
}

/**
 * @brief 双线性插值图像缩放, 定点计算版本, 对齐 OpenCV
 * 
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale X轴缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename CT, int CH>
__global__ void u8_resize_bilinear_kernel(const uint8_t *src, uint8_t *dst, const double2 scale, const int2 ssize,
                                          const int sstride, const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // sx, sy
    int2 s1; // sx+1, sy+1

    using CT2 = make_vector2_t<CT>;
    CT2 f;

    cal_bilinear_interpolation<CT2>(ssize, dst_coord, scale, s, f, s1);

    // 定点插值系数: 将浮点小数部分量化为 11 位定点整数 (INTER_RESIZE_COEF_SCALE = 2048)
    short2 alpha;
    alpha.x = saturate_cast<short>((1.0f - f.x) * (float)INTER_RESIZE_COEF_SCALE); // 左权重
    alpha.y = INTER_RESIZE_COEF_SCALE - alpha.x;                                   // 右权重

    short2 beta;
    beta.x = saturate_cast<short>((1.0f - f.y) * (float)INTER_RESIZE_COEF_SCALE); // 上权重
    beta.y = INTER_RESIZE_COEF_SCALE - beta.x;                                     // 下权重

    // 获取源数据指针
    uint8_t *row0 = const_cast<uint8_t *>(src) + s.y * sstride;
    uint8_t *row1 = const_cast<uint8_t *>(src) + s1.y * sstride;

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        // 水平插值（模拟HResizeLinear的行为）
        // WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
        int2 hval;
        hval.x = row0[s.x * CH + i] * alpha.x + row0[s1.x * CH + i] * alpha.y;
        hval.y = row1[s.x * CH + i] * alpha.x + row1[s1.x * CH + i] * alpha.y;

        // 垂直插值 - 模拟VResizeLinear<uchar>的公式
        // dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
        int2 term;
        term.x     = (beta.x * (hval.x >> 4)) >> 16; // 第一项
        term.y     = (beta.y * (hval.y >> 4)) >> 16; // 第二项
        int result = (term.x + term.y + 2) >> 2;     // 加法、加2、右移2位

        // 使用与OpenCV完全相同的转换方式：直接uchar()转换
        dst[dst_base + i] = (uint8_t)result;
    }
}

/**
 * @brief 最近邻插值图像缩放
 * 
 * @tparam T 图像数据类型 (如 float, uint8_t)
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale X轴缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename T, typename CT, int CH>
__global__ void resize_nearest_kernel(const T *src, T *dst, const double2 scale, const int2 ssize, const int sstride,
                                      const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    using CT2 = make_vector2_t<CT>;
    CT2 src_coord;

    src_coord.x = dst_coord.x * scale.x;
    src_coord.y = dst_coord.y * scale.y;

    int2 s;
    if constexpr (std::is_same_v<CT2, double2>)
    {
        s.x = __double2int_rd(src_coord.x);
        s.y = __double2int_rd(src_coord.y);
    }
    else if constexpr (std::is_same_v<CT2, float2>)
    {
        s.x = __float2int_rd(src_coord.x);
        s.y = __float2int_rd(src_coord.y);
    }
    else
    {
        s.x = __double2int_rd(src_coord.x);
        s.y = __double2int_rd(src_coord.y);
    }

    s.x = max(0, min(s.x, ssize.x - 1));
    s.y = max(0, min(s.y, ssize.y - 1));

    const int src_base = s.y * sstride + s.x * CH;
    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;
#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        dst[dst_base + i] = src[src_base + i];
    }
}


/**
 * @brief 三次插值系数计算 (Keys cubic, A=-0.75, 与 OpenCV 一致)
 * 
 * @tparam CT 浮点计算类型 (float 或 double)
 * @param[in] x 小数部分 (0 <= x < 1)
 * @param[out] coeffs 4 个插值权重系数
 */
template<typename CT>
__device__ __forceinline__ void interpolateCubic(CT x, CT coeffs[4])
{
    const CT A = static_cast<CT>(-0.75); // Keys cubic 参数, OpenCV 默认值

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A; // x-1 位置的权重
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;                          // x   位置的权重
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;        // x+1 位置的权重
    coeffs[3] = static_cast<CT>(1) - coeffs[0] - coeffs[1] - coeffs[2];         // x+2 位置的权重 (由归一化条件推导)
}


/**
 * @brief 双三次插值图像缩放 (浮点版本)
 * 
 * @tparam T 图像数据类型 (如 float, uint8_t)
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale 缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度 (以元素为单位)
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度 (以元素为单位)
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename T, typename CT, int CH>
__global__ void resize_bicubic_kernel(const T *src, T *dst, const double2 scale, const int2 ssize, const int sstride,
                                      const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // 源图像中对应的整数坐标
    using CT2 = make_vector2_t<CT>;
    CT2 f;   // 小数部分, 用于插值权重计算
    cal_interpolation<CT2>(dst_coord, scale, s, f);

    CT alpha[4], beta[4]; // x/y 方向的 4 个三次插值系数
    interpolateCubic<CT>(f.x, alpha);
    interpolateCubic<CT>(f.y, beta);

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) // 遍历 y 方向 4 个采样点
        {
            int syj     = max(0, min(s.y + j - 1, ssize.y - 1)); // 边界裁剪
            CT  row_sum = 0;

#pragma unroll
            for (int i = 0; i < 4; ++i) // 遍历 x 方向 4 个采样点
            {
                int sxi = max(0, min(s.x + i - 1, ssize.x - 1)); // 边界裁剪
                row_sum += alpha[i] * src[syj * sstride + sxi * CH + ch];
            }

            sum += beta[j] * row_sum; // y 方向加权
        }

        dst[dst_base + ch] = saturate_cast<T>(sum); // 饱和截断到目标类型范围
    }
}

/**
 * @brief 双三次插值图像缩放, 定点计算版本 (uint8_t 专用)
 * 
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale 缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度 (以元素为单位)
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度 (以元素为单位)
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename CT, int CH>
__global__ void u8_resize_bicubic_kernel(const uint8_t *src, uint8_t *dst, const double2 scale, const int2 ssize,
                                         const int sstride, const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // 源图像中对应的整数坐标
    using CT2 = make_vector2_t<CT>;
    CT2 f;   // 小数部分
    cal_interpolation<CT2>(dst_coord, scale, s, f);

    CT alpha[4], beta[4]; // x/y 方向的 4 个三次插值系数 (浮点)
    interpolateCubic<CT>(f.x, alpha);
    interpolateCubic<CT>(f.y, beta);

    // 将浮点插值系数量化为定点整数, 对齐 OpenCV 的定点计算方式
    short ialpha[4], ibeta[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        ialpha[k] = saturate_cast<short>(alpha[k] * INTER_RESIZE_COEF_SCALE);
        ibeta[k]  = saturate_cast<short>(beta[k] * INTER_RESIZE_COEF_SCALE);
    }

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        int sum = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j) // 遍历 y 方向 4 个采样点
        {
            int syj     = max(0, min(s.y + j - 1, ssize.y - 1)); // 边界裁剪
            int row_sum = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) // 遍历 x 方向 4 个采样点
            {
                int sxi = max(0, min(s.x + i - 1, ssize.x - 1)); // 边界裁剪
                row_sum += ialpha[i] * src[syj * sstride + sxi * CH + ch];
            }
            sum += ibeta[j] * row_sum; // y 方向加权
        }

        // 定点结果还原: 加 DELTA 偏移后右移 SHIFT 位, 等效于四舍五入
        dst[dst_base + ch] = saturate_cast<uint8_t>((sum + DELTA) >> SHIFT);
    }
}


/**
 * @brief Lanczos4 插值的预计算三角函数系数表 (常量内存)
 * 
 * 存储 sin(y0) 和 cos(y0) 在 8 个采样点处的线性组合系数,
 * 避免在每个线程中重复计算三角函数, 对齐 OpenCV 的实现
 * cs[i] = {sin((i-3)*PI/4) 的系数, cos((i-3)*PI/4) 的系数}
 */
__constant__ double cs[8][2] = {
    {                                  1,                                   0},  // i=0: sin(0)
    {-0.70710678118654752440084436210485, -0.70710678118654752440084436210485},  // i=1: sin(-PI/4)
    {                                  0,                                   1},  // i=2: sin(-PI/2)
    { 0.70710678118654752440084436210485, -0.70710678118654752440084436210485},  // i=3: sin(-3PI/4)
    {                                 -1,                                   0},  // i=4: sin(-PI)
    { 0.70710678118654752440084436210485,  0.70710678118654752440084436210485},  // i=5: sin(-5PI/4)
    {                                  0,                                  -1},  // i=6: sin(-3PI/2)
    {-0.70710678118654752440084436210485,  0.70710678118654752440084436210485}   // i=7: sin(-7PI/4)
};

/**
 * @brief Lanczos4 插值系数计算 (a=4, 8 个采样点, 与 OpenCV 一致)
 * 
 * Lanczos4 核函数: L(x) = sinc(x) * sinc(x/4), |x| < 4; 0, |x| >= 4
 * 利用预计算的三角函数系数表加速, 并对 sinc(0) 的奇点做特殊处理
 * 
 * @tparam CT 浮点计算类型 (float 或 double)
 * @param[in] x 小数部分 (0 <= x < 1)
 * @param[out] coeffs 8 个插值权重系数 (归一化后)
 */
template<typename CT>
__device__ __forceinline__ void interpolateLanczos4(CT x, CT *coeffs)
{
    CT     sum = 0;
    double y0  = -(x + 3) * CV_PI * 0.25; // 初始角度, 对应采样中心偏移
    float  s0, c0;
    __sincosf(y0, &s0, &c0); // 同时计算 sin 和 cos, 比分开调用更快

    for (int i = 0; i < 8; i++)
    {
        CT y0_ = (x + 3 - i); // 当前采样点相对于中心的偏移
        if (fabs(y0_) >= 1e-6f)
        {
            // 非零点: 利用预计算系数表计算 sinc 值
            double y  = -y0_ * CV_PI * 0.25;
            coeffs[i] = (CT)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        }
        else
        {
            // special handling for 'x' values:
            // - ~0.0: 0 0 0 1 0 0 0 0
            // - ~1.0: 0 0 0 0 1 0 0 0
            // sinc(0) 奇点特殊处理: 该采样点权重设为极大值, 归一化后即为 1
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }

    // 归一化, 使权重之和为 1
    sum = 1.0 / sum;
    for (int i = 0; i < 8; i++) coeffs[i] *= sum;
}

/**
 * @brief Lanczos4 插值图像缩放 (浮点版本)
 * 
 * @tparam T 图像数据类型 (如 float)
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale 缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度 (以元素为单位)
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度 (以元素为单位)
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename T, typename CT, int CH>
__global__ void resize_lanczos_kernel(const T *src, T *dst, const double2 scale, const int2 ssize, const int sstride,
                                      const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // 源图像中对应的整数坐标
    using CT2 = make_vector2_t<CT>;
    CT2 f;   // 小数部分, 用于插值权重计算
    cal_interpolation<CT2>(dst_coord, scale, s, f);

    // 计算 Lanczos4 插值权重 (8 个采样点)
    CT alpha[8], beta[8];
    interpolateLanczos4<CT>(f.x, alpha);
    interpolateLanczos4<CT>(f.y, beta);

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0;

#pragma unroll
        for (int j = 0; j < 8; ++j) // 遍历 y 方向 8 个采样点
        {
            int syj     = max(0, min(s.y + j - 3, ssize.y - 1)); // Lanczos4 中心在第 4 个位置 (索引 3)
            CT  row_sum = 0;

#pragma unroll
            for (int i = 0; i < 8; ++i) // 遍历 x 方向 8 个采样点
            {
                int sxi = max(0, min(s.x + i - 3, ssize.x - 1)); // Lanczos4 中心在第 4 个位置 (索引 3)
                row_sum += alpha[i] * src[syj * sstride + sxi * CH + ch];
            }

            sum += beta[j] * row_sum; // y 方向加权
        }

        dst[dst_base + ch] = saturate_cast<T>(sum);
    }
}

/**
 * @brief Lanczos4 插值图像缩放, 定点计算版本 (uint8_t 专用)
 * 
 * @tparam CT 插值小数部分计算类型 (如 float, double)
 * @tparam CH 通道数 (1,3,4)
 * @param[in] src 源图像数据指针
 * @param[out] dst 目标图像数据指针
 * @param[in] scale 缩放比例 (x: src_w / dst_w, y: src_h / dst_h)
 * @param[in] ssize 源图像尺寸
 * @param[in] sstride 源图像行宽度 (以元素为单位)
 * @param[in] dsize 目标图像尺寸
 * @param[in] dstride 目标图像行宽度 (以元素为单位)
 * @param[in] dst_N 目标图像总像素数 (dst_h * dst_w)
 */
template<typename CT, int CH>
__global__ void u8_resize_lanczos_kernel(const uint8_t *src, uint8_t *dst, const double2 scale, const int2 ssize,
                                         const int sstride, const int2 dsize, const int dstride, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    int2 s;  // 源图像中对应的整数坐标
    using CT2 = make_vector2_t<CT>;
    CT2 f;   // 小数部分
    cal_interpolation<CT2>(dst_coord, scale, s, f);

    CT alpha[8], beta[8]; // x/y 方向的 8 个 Lanczos4 插值系数 (浮点)
    interpolateLanczos4<CT>(f.x, alpha);
    interpolateLanczos4<CT>(f.y, beta);

    // 将浮点插值系数量化为定点整数, 对齐 OpenCV 的定点计算方式
    short ialpha[8], ibeta[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        ialpha[k] = saturate_cast<short>(alpha[k] * INTER_RESIZE_COEF_SCALE);
        ibeta[k]  = saturate_cast<short>(beta[k] * INTER_RESIZE_COEF_SCALE);
    }

    const int dst_base = dst_coord.y * dstride + dst_coord.x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        int sum = 0;

#pragma unroll
        for (int j = 0; j < 8; ++j) // 遍历 y 方向 8 个采样点
        {
            int syj     = max(0, min(s.y + j - 3, ssize.y - 1)); // Lanczos4 中心在第 4 个位置 (索引 3)
            int row_sum = 0;

#pragma unroll
            for (int i = 0; i < 8; ++i) // 遍历 x 方向 8 个采样点
            {
                int sxi = max(0, min(s.x + i - 3, ssize.x - 1)); // Lanczos4 中心在第 4 个位置 (索引 3)
                row_sum += ialpha[i] * src[syj * sstride + sxi * CH + ch];
            }

            sum += ibeta[j] * row_sum; // y 方向加权
        }

        // 定点结果还原: 加 DELTA 偏移后右移 SHIFT 位, 等效于四舍五入
        dst[dst_base + ch] = saturate_cast<uint8_t>((sum + DELTA) >> SHIFT);
    }
}

/**
 * @brief 启动双线性插值 resize kernel 的统一入口
 * 
 * 根据 T 类型自动选择: uint8_t 使用定点版本, 其他类型使用浮点版本
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 * @tparam C 通道数 (编译期常量)
 */
template<typename T, typename CT, int C>
inline void launch_bilinear_kernel(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                                   const int2 &dsize, const int dstride, const int dst_N, const int grid_size,
                                   const int block_size, cudaStream_t stream)
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        u8_resize_bilinear_kernel<CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
    else
    {
        resize_bilinear_kernel<T, CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
}

/**
 * @brief 启动双三次插值 resize kernel 的统一入口
 * 
 * 根据 T 类型自动选择: uint8_t 使用定点版本, 其他类型使用浮点版本
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 * @tparam C 通道数 (编译期常量)
 */
template<typename T, typename CT, int C>
inline void launch_bicubic_kernel(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                                  const int2 &dsize, const int dstride, const int dst_N, const int grid_size,
                                  const int block_size, cudaStream_t stream)
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        u8_resize_bicubic_kernel<CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
    else
    {
        resize_bicubic_kernel<T, CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
}

/**
 * @brief 启动 Lanczos4 插值 resize kernel 的统一入口
 * 
 * 根据 T 类型自动选择: uint8_t 使用定点版本, 其他类型使用浮点版本
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 * @tparam C 通道数 (编译期常量)
 */
template<typename T, typename CT, int C>
inline void launch_lanczos_kernel(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                                  const int2 &dsize, const int dstride, const int dst_N, const int grid_size,
                                  const int block_size, cudaStream_t stream)
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        u8_resize_lanczos_kernel<CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
    else
    {
        resize_lanczos_kernel<T, CT, C>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
    }
}

/**
 * @brief 双线性插值 resize 入口, 根据通道数分发到对应的 kernel 模板实例
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 */
template<typename T, typename CT>
void resize_bilinear(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                     const int2 &dsize, const int dstride, const int CH, const int dst_N, const int grid_size,
                     const int block_size, cudaStream_t stream)
{
    switch (CH)
    {
    case 1:
        launch_bilinear_kernel<T, CT, 1>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                         block_size, stream);
        break;
    case 3:
        launch_bilinear_kernel<T, CT, 3>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                         block_size, stream);
        break;
    case 4:
        launch_bilinear_kernel<T, CT, 4>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                         block_size, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1/3/4");
    }
}

/**
 * @brief 双三次插值 resize 入口, 根据通道数分发到对应的 kernel 模板实例
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 */
template<typename T, typename CT>
void resize_bicubic(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                    const int2 &dsize, const int dstride, const int CH, const int dst_N, const int grid_size,
                    const int block_size, cudaStream_t stream)
{
    switch (CH)
    {
    case 1:
        launch_bicubic_kernel<T, CT, 1>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    case 3:
        launch_bicubic_kernel<T, CT, 3>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    case 4:
        launch_bicubic_kernel<T, CT, 4>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1/3/4");
    }
}

/**
 * @brief 最近邻插值 resize 入口, 根据通道数分发到对应的 kernel 模板实例
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 */
template<typename T, typename CT>
void resize_nearest(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                    const int2 &dsize, const int dstride, const int CH, const int dst_N, const int grid_size,
                    const int block_size, cudaStream_t stream)
{
    switch (CH)
    {
    case 1:
        resize_nearest_kernel<T, CT, 1>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        break;
    case 3:
        resize_nearest_kernel<T, CT, 3>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        break;
    case 4:
        resize_nearest_kernel<T, CT, 4>
            <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1/3/4");
    }
}

/**
 * @brief Lanczos4 插值 resize 入口, 根据通道数分发到对应的 kernel 模板实例
 * 
 * @tparam T 图像数据类型
 * @tparam CT 插值计算浮点类型
 */
template<typename T, typename CT>
void resize_lanczos(const T *d_src, T *d_dst, const double2 &scale, const int2 &ssize, const int sstride,
                    const int2 &dsize, const int dstride, const int CH, const int dst_N, const int grid_size,
                    const int block_size, cudaStream_t stream)
{
    switch (CH)
    {
    case 1:
        launch_lanczos_kernel<T, CT, 1>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    case 3:
        launch_lanczos_kernel<T, CT, 3>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    case 4:
        launch_lanczos_kernel<T, CT, 4>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N, grid_size,
                                        block_size, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1/3/4");
    }
}

/**
 * @brief Resize 操作的主入口, 根据插值方法调用对应的 resize 实现
 * 
 * @tparam T 图像数据类型 (uint8_t 或 float)
 * @param[in] d_src 源图像设备内存指针
 * @param[out] d_dst 目标图像设备内存指针
 * @param[in] ssize 源图像尺寸 (width, height)
 * @param[in] sstride 源图像行宽度 (以元素为单位, 含 padding)
 * @param[in] dsize 目标图像尺寸 (width, height)
 * @param[in] dstride 目标图像行宽度 (以元素为单位, 含 padding)
 * @param[in] CH 通道数 (1/3/4)
 * @param[in] interpolation 插值方法
 * @param[in] stream CUDA 流
 */
template<typename T>
void ResizeImpl<T>::RunResize(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                              const int dstride, const int CH, const int interpolation, cudaStream_t stream)
{
    // 计算缩放比例和网格参数
    // 注意此处不直接用 ssize / dsize, 因为可能由于精度问题导致和 OpenCV 存在精度差异, 特别是对于 LINE_NEAREST
    double2 scale;
    scale.x = 1.0 / (static_cast<double>(dsize.x) / ssize.x);
    scale.y = 1.0 / (static_cast<double>(dsize.y) / ssize.y);

    const int dst_N      = dsize.x * dsize.y;  // 目标图像总像素数
    const int block_size = 256;                  // 每个 CUDA 线程块的线程数
    const int grid_size  = (dst_N + block_size - 1) / block_size; // 向上取整计算所需线程块数

    // 根据插值方法选择对应的 resize 实现
    switch (interpolation)
    {
    case cv::INTER_LINEAR:
    {
        resize_bilinear<T, float>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, CH, dst_N, grid_size, block_size,
                                  stream);
        break;
    }
    case cv::INTER_CUBIC:
    {
        resize_bicubic<T, float>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, CH, dst_N, grid_size, block_size,
                                 stream);
        break;
    }
    case cv::INTER_NEAREST:
    {
        resize_nearest<T, double>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, CH, dst_N, grid_size, block_size,
                                  stream);
        break;
    }
    case cv::INTER_LANCZOS4:
    {
        resize_lanczos<T, float>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, CH, dst_N, grid_size, block_size,
                                 stream);
        break;
    }
    default:
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Interpolation method not implemented");
    }
    }
}

// 显式模板实例化: uint8_t 和 float 两种图像数据类型
template void ResizeImpl<uint8_t>::RunResize(const uint8_t *, uint8_t *, const int2, const int, const int2, const int,
                                             const int, const int, cudaStream_t);
template void ResizeImpl<float>::RunResize(const float *, float *, const int2, const int, const int2, const int,
                                           const int, const int, cudaStream_t);

} // namespace irt::cvcuda::priv
