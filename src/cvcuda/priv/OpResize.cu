#include "OpResize.cuh"
#include "saturate.cuh"
#include "type.cuh"

#include <inferrt/cvcuda/OpResize.h>

namespace irt::cvcuda {

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
    else if constexpr (std::is_same_v<CT2, float>)
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

    short2 alpha;
    alpha.x = saturate_cast<short>((1.0f - f.x) * (float)INTER_RESIZE_COEF_SCALE);
    alpha.y = INTER_RESIZE_COEF_SCALE - alpha.x;

    short2 beta;
    beta.x = saturate_cast<short>((1.0f - f.y) * (float)INTER_RESIZE_COEF_SCALE);
    beta.y = INTER_RESIZE_COEF_SCALE - beta.x;

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

template<typename T, typename CT>
void resize_bilinear(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                     const int CH, cudaStream_t stream)
{
    double2 scale;
    scale.x = static_cast<double>(ssize.x) / dsize.x;
    scale.y = static_cast<double>(ssize.y) / dsize.y;

    const int dst_N      = dsize.x * dsize.y;
    const int block_size = 256;
    const int grid_size  = (dst_N + block_size - 1) / block_size;

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        if (CH == 1)
        {
            u8_resize_bilinear_kernel<CT, 1>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
        else if (CH == 3)
        {
            u8_resize_bilinear_kernel<CT, 3>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
        else if (CH == 4)
        {
            u8_resize_bilinear_kernel<CT, 4>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
    }
    else
    {
        if (CH == 1)
        {
            resize_bilinear_kernel<T, CT, 1>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
        else if (CH == 3)
        {
            resize_bilinear_kernel<T, CT, 3>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
        else if (CH == 4)
        {
            resize_bilinear_kernel<T, CT, 4>
                <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dstride, dst_N);
        }
    }
}

template INFERRT_CVCUDA_API void resize_bilinear<uint8_t, float>(const uint8_t *, uint8_t *, const int2, const int,
                                                                 const int2, const int, const int, cudaStream_t);
template INFERRT_CVCUDA_API void resize_bilinear<float, float>(const float *, float *, const int2, const int,
                                                               const int2, const int, const int, cudaStream_t);
template INFERRT_CVCUDA_API void resize_bilinear<float, double>(const float *, float *, const int2, const int,
                                                                const int2, const int, const int, cudaStream_t);

} // namespace irt::cvcuda