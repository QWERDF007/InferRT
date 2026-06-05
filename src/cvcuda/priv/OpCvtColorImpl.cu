#include "OpCvtColorImpl.hpp"

#include "saturate.cuh"

#include <inferrt/core/Exception.hpp>
#include <inferrt/util/CheckError.hpp>
#include <opencv2/imgproc.hpp>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace irt::cvcuda::priv {

// 棰滆壊閫氶亾杈呭姪缁撴瀯浣?- 鍙傝€?OpenCV 瀹炵幇
template<typename _Tp>
struct ColorChannel
{
    static __device__ __forceinline__ _Tp max()
    {
        if constexpr (std::is_same_v<_Tp, uint8_t>)
        {
            return 255;
        }
        else if constexpr (std::is_same_v<_Tp, int>)
        {
            return INT_MAX;
        }
        else if constexpr (std::is_same_v<_Tp, float>)
        {
            return 1.f;
        }
        else
        {
            return 1.0;
        }
    }

    static __device__ __forceinline__ _Tp half()
    {
        if constexpr (std::is_same_v<_Tp, float>)
        {
            return 0.5f;
        }
        else
        {
            return (_Tp)(max() / 2 + 1);
        }
    }
};

template<typename T>
__global__ void gray2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        dst[base + i] = src[tid];
    }
}

template<typename T>
__global__ void gray2bgra_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 4;

#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        dst[base + i] = src[tid];
    }
    dst[base + 3] = ColorChannel<T>::max();
}

template<typename T>
__global__ void bgr2rgb_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    dst[base]     = src[base + 2];
    dst[base + 1] = src[base + 1];
    dst[base + 2] = src[base];
}

template<typename T>
__global__ void bgr2rgba_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = tid * 3;
    const int dst_base = tid * 4;

    dst[dst_base]     = src[src_base + 2];
    dst[dst_base + 1] = src[src_base + 1];
    dst[dst_base + 2] = src[src_base];
    dst[dst_base + 3] = ColorChannel<T>::max();
}

template<typename T>
__global__ void bgr2bgra_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = tid * 3;
    const int dst_base = tid * 4;

    dst[dst_base]     = src[src_base];
    dst[dst_base + 1] = src[src_base + 1];
    dst[dst_base + 2] = src[src_base + 2];
    dst[dst_base + 3] = ColorChannel<T>::max();
}

#define CV_DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))

static constexpr float B2YF = 0.114f;
static constexpr float G2YF = 0.587f;
static constexpr float R2YF = 0.299f;

static constexpr int gray_shift = 15;

static constexpr int RY15 = 9798;  // == R2YF*32768 + 0.5
static constexpr int GY15 = 19235; // == G2YF*32768 + 0.5
static constexpr int BY15 = 3735;  // == B2YF*32768 + 0.5

template<typename T>
__global__ void bgr2gray_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        dst[tid] = src[src_base] * B2YF + src[src_base + 1] * G2YF + src[src_base + 2] * R2YF;
    }
    else
    {
        dst[tid] = CV_DESCALE(src[src_base] * BY15 + src[src_base + 1] * GY15 + src[src_base + 2] * RY15, gray_shift);
    }
}

template<typename T>
__global__ void rgb2gray_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        dst[tid] = src[src_base] * R2YF + src[src_base + 1] * G2YF + src[src_base + 2] * B2YF;
    }
    else
    {
        dst[tid] = CV_DESCALE(src[src_base] * RY15 + src[src_base + 1] * GY15 + src[src_base + 2] * BY15, gray_shift);
    }
}

static constexpr int R2Y = 4899; // == R2YF*16384
static constexpr int G2Y = 9617; // == G2YF*16384
static constexpr int B2Y = 1868; // == B2YF*16384

static constexpr int yuv_shift = 14;

//to YCbCr
static constexpr float YCBF = 0.564f; // == 1/2/(1-B2YF)
static constexpr float YCRF = 0.713f; // == 1/2/(1-R2YF)
static constexpr int   YCBI = 9241;   // == YCBF*16384
static constexpr int   YCRI = 11682;  // == YCRF*16384

//to YUV
static constexpr float B2UF = 0.492f;
static constexpr float R2VF = 0.877f;
static constexpr int   B2UI = 8061;  // == B2UF*16384
static constexpr int   R2VI = 14369; // == R2VF*16384

//from YUV
static constexpr float U2BF = 2.032f;
static constexpr float U2GF = -0.395f;
static constexpr float V2GF = -0.581f;
static constexpr float V2RF = 1.140f;
static constexpr int   U2BI = 33292;
static constexpr int   U2GI = -6472;
static constexpr int   V2GI = -9519;
static constexpr int   V2RI = 18678;

//from YCrCb
static constexpr float CB2BF = 1.773f;
static constexpr float CB2GF = -0.344f;
static constexpr float CR2GF = -0.714f;
static constexpr float CR2RF = 1.403f;
static constexpr int   CB2BI = 29049;
static constexpr int   CB2GI = -5636;
static constexpr int   CR2GI = -11698;
static constexpr int   CR2RI = 22987;

template<typename T>
__global__ void bgr2YCrCb_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * B2YF + src[base + 1] * G2YF + src[base + 2] * R2YF;
        float Cr = (src[base + 2] - Y) * YCRF + delta;
        float Cb = (src[base] - Y) * YCBF + delta;

        dst[base]     = Y;
        dst[base + 1] = Cr;
        dst[base + 2] = Cb;
    }
    else
    {
        const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * B2Y + src[base + 1] * G2Y + src[base + 2] * R2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base + 2] - Y) * YCRI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base] - Y) * YCBI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 1] = saturate_cast<T>(Cr);
        dst[base + 2] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void YCrCb2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T Y  = src[base];
    T Cr = src[base + 1];
    T Cb = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float b = Y + (Cb - delta) * CB2BF;
        float g = Y + (Cb - delta) * CB2GF + (Cr - delta) * CR2GF;
        float r = Y + (Cr - delta) * CR2RF;

        dst[base]     = b;
        dst[base + 1] = g;
        dst[base + 2] = r;
    }
    else
    {
        const uint8_t delta = ColorChannel<T>::half();

        int b = Y + CV_DESCALE((Cb - delta) * CB2BI, yuv_shift);
        int g = Y + CV_DESCALE((Cb - delta) * CB2GI + (Cr - delta) * CR2GI, yuv_shift);
        int r = Y + CV_DESCALE((Cr - delta) * CR2RI, yuv_shift);

        dst[base]     = saturate_cast<T>(b);
        dst[base + 1] = saturate_cast<T>(g);
        dst[base + 2] = saturate_cast<T>(r);
    }
}

template<typename T>
__global__ void bgr2yuv_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * B2YF + src[base + 1] * G2YF + src[base + 2] * R2YF;
        float Cr = (src[base + 2] - Y) * R2VF + delta;
        float Cb = (src[base] - Y) * B2UF + delta;

        dst[base]     = Y;
        dst[base + 2] = Cr;
        dst[base + 1] = Cb;
    }
    else
    {
        const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * B2Y + src[base + 1] * G2Y + src[base + 2] * R2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base + 2] - Y) * R2VI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base] - Y) * B2UI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Cr);
        dst[base + 1] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void yuv2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T Y  = src[base];
    T Cr = src[base + 2];
    T Cb = src[base + 1];

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float b = Y + (Cb - delta) * U2BF;
        float g = Y + (Cb - delta) * U2GF + (Cr - delta) * V2GF;
        float r = Y + (Cr - delta) * V2RF;

        dst[base]     = b;
        dst[base + 1] = g;
        dst[base + 2] = r;
    }
    else
    {
        const uint8_t delta = ColorChannel<T>::half();

        int b = Y + CV_DESCALE((Cb - delta) * U2BI, yuv_shift);
        int g = Y + CV_DESCALE((Cb - delta) * U2GI + (Cr - delta) * V2GI, yuv_shift);
        int r = Y + CV_DESCALE((Cr - delta) * V2RI, yuv_shift);

        dst[base]     = saturate_cast<T>(b);
        dst[base + 1] = saturate_cast<T>(g);
        dst[base + 2] = saturate_cast<T>(r);
    }
}

template<typename T>
__global__ void rgb2YCrCb_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * R2YF + src[base + 1] * G2YF + src[base + 2] * B2YF;
        float Cr = (src[base] - Y) * YCRF + delta;
        float Cb = (src[base + 2] - Y) * YCBF + delta;

        dst[base]     = Y;
        dst[base + 1] = Cr;
        dst[base + 2] = Cb;
    }
    else
    {
        int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * R2Y + src[base + 1] * G2Y + src[base + 2] * B2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base] - Y) * YCRI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base + 2] - Y) * YCBI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 1] = saturate_cast<T>(Cr);
        dst[base + 2] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void rgb2yuv_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * R2YF + src[base + 1] * G2YF + src[base + 2] * B2YF;
        float Cr = (src[base] - Y) * R2VF + delta;
        float Cb = (src[base + 2] - Y) * B2UF + delta;

        dst[base]     = Y;
        dst[base + 2] = Cr;
        dst[base + 1] = Cb;
    }
    else
    {
        int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * R2Y + src[base + 1] * G2Y + src[base + 2] * B2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base] - Y) * R2VI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base + 2] - Y) * B2UI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Cr);
        dst[base + 1] = saturate_cast<T>(Cb);
    }
}

static constexpr int hsv_shift = 12;

// int hrange = depth == CV_32F ? 360 : isFullRange ? 256 : 180;

__device__ __forceinline__ int sdiv_table(int v)
{
    // OpenCV: sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
    return saturate_cast<int>((255 << hsv_shift) / (1. * v));
}

__device__ __forceinline__ int hdiv_table180(int v)
{
    // OpenCV: hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
    return saturate_cast<int>((180 << hsv_shift) / (6. * v));
}

template<typename T>
__global__ void bgr2hsv_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float v    = max(b, max(g, r));
        float vmin = min(b, min(g, r));
        float diff = v - vmin;

        float s = diff / (float)(fabsf(v) + FLT_EPSILON);
        diff    = (float)(60. / (diff + FLT_EPSILON));

        float h;
        if (v == r)
            h = (g - b) * diff;
        else if (v == g)
            h = (b - r) * diff + 120.f;
        else
            h = (r - g) * diff + 240.f;
        h += h < 0.f ? 360.f : 0.f;

        // dst[base]  = h * hscale; // hscale = 360.f * (1.f / 360.f);
        dst[base]     = h;
        dst[base + 1] = s;
        dst[base + 2] = v;
    }
    else
    {
        int v    = max(b, max(g, r));
        int vmin = min(b, min(g, r));

        const T   diff = saturate_cast<T>(v - vmin);
        const int vr   = v == r ? -1 : 0;
        const int vg   = v == g ? -1 : 0;

        const int s = (diff * sdiv_table(v) + (1 << (hsv_shift - 1))) >> hsv_shift;

        int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h     = (h * hdiv_table180(diff) + (1 << (hsv_shift - 1))) >> hsv_shift;
        h += (h >> 31) & 180; // 浣跨敤浣嶈繍绠楁浛浠ｅ垎鏀細h < 0 ? 180 : 0

        dst[base]     = saturate_cast<T>(h);
        dst[base + 1] = (T)s;
        dst[base + 2] = (T)v;
    }
}

__constant__ int sector_data[6][4] = {
    {1, 3, 0},
    {1, 0, 2},
    {3, 0, 1},
    {0, 2, 1},
    {0, 1, 3},
    {2, 1, 0}
};

__device__ void hsv2bgr_f(float h, float s, float v, float &b, float &g, float &r, const float hscale)
{
    if (s == 0)
        b = g = r = v;
    else
    {
        float tab[4];
        int   sector;
        h *= hscale;
        h      = fmodf(h, 6.f);
        sector = __float2int_rd(h);
        h -= sector;
        if ((unsigned)sector >= 6u)
        {
            sector = 0;
            h      = 0.f;
        }

        tab[0] = v;
        tab[1] = v * (1.f - s);
        tab[2] = v * (1.f - s * h);
        tab[3] = v * (1.f - s * (1.f - h));

        b = tab[sector_data[sector][0]];
        g = tab[sector_data[sector][1]];
        r = tab[sector_data[sector][2]];
    }
}

template<typename T>
__global__ void hsv2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T h = src[base];
    T s = src[base + 1];
    T v = src[base + 2];

    float b, g, r;

    if constexpr (std::is_same_v<T, float>)
    {
        const float hs = 6.f / 360; // hs = 6.f / _hrange = 6.f / 360;
        hsv2bgr_f(h, s, v, b, g, r, hs);
        dst[base]     = b;
        dst[base + 1] = g;
        dst[base + 2] = r;
    }
    else
    {
        const float hs = 6.f / 180;
        hsv2bgr_f(h, s * (1.0f / 255.0f), v * (1.0f / 255.0f), b, g, r, hs);
        dst[base]     = saturate_cast<T>(b * 255.0f);
        dst[base + 1] = saturate_cast<T>(g * 255.0f);
        dst[base + 2] = saturate_cast<T>(r * 255.0f);
    }
}

template<typename T>
__global__ void rgb2hsv_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float v    = max(b, max(g, r));
        float vmin = min(b, min(g, r));
        float diff = v - vmin;

        float s = diff / (float)(fabsf(v) + FLT_EPSILON);
        diff    = (float)(60. / (diff + FLT_EPSILON));

        float h;
        if (v == r)
            h = (g - b) * diff;
        else if (v == g)
            h = (b - r) * diff + 120.f;
        else
            h = (r - g) * diff + 240.f;
        h += h < 0.f ? 360.f : 0.f;

        dst[base]     = h;
        dst[base + 1] = s;
        dst[base + 2] = v;
    }
    else
    {
        int v    = max(b, max(g, r));
        int vmin = min(b, min(g, r));

        const T   diff = saturate_cast<T>(v - vmin);
        const int vr   = v == r ? -1 : 0;
        const int vg   = v == g ? -1 : 0;

        const int s = (diff * sdiv_table(v) + (1 << (hsv_shift - 1))) >> hsv_shift;

        int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h     = (h * hdiv_table180(diff) + (1 << (hsv_shift - 1))) >> hsv_shift;
        h += (h >> 31) & 180; // 浣跨敤浣嶈繍绠楁浛浠ｅ垎鏀細h < 0 ? 180 : 0

        dst[base]     = saturate_cast<T>(h);
        dst[base + 1] = (T)s;
        dst[base + 2] = (T)v;
    }
}

__device__ void bgr2hls_f(float b, float g, float r, float &h, float &l, float &s)
{
    float vmax = max(b, max(g, r));
    float vmin = min(b, min(g, r));
    float diff = vmax - vmin;

    l = (vmax + vmin) * 0.5f;
    if (diff > FLT_EPSILON)
    {
        s    = l < 0.5f ? diff / (vmax + vmin) : diff / (2 - vmax - vmin);
        diff = 60.f / diff;

        if (vmax == r)
            h = (g - b) * diff;
        else if (vmax == g)
            h = (b - r) * diff + 120.f;
        else
            h = (r - g) * diff + 240.f;

        if (h < 0.f)
            h += 360.f;
    }
}

template<typename T>
__global__ void bgr2hls_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    float h = 0.f, l = 0.f, s = 0.f;

    if constexpr (std::is_same_v<T, float>)
    {
        bgr2hls_f(b, g, r, h, l, s);

        // dst[base]     = h * hscale; hscale = 360.f / 360.f;
        dst[base]     = h;
        dst[base + 1] = l;
        dst[base + 2] = s;
    }
    else
    {
        float b_f = b * (1.f / 255.f);
        float g_f = g * (1.f / 255.f);
        float r_f = r * (1.f / 255.f);

        bgr2hls_f(b_f, g_f, r_f, h, l, s);

        // dst[base]     = saturate_cast<T>(h * hscale); // hscale = 180 / 360.f;
        dst[base]     = saturate_cast<T>(h * 0.5f);
        dst[base + 1] = saturate_cast<T>(l * 255.f);
        dst[base + 2] = saturate_cast<T>(s * 255.f);
    }
}

__device__ void hls2bgr_f(float h, float l, float s, float &b, float &g, float &r, const float hscale)
{
    if (s == 0)
        b = g = r = l;
    else
    {
        float tab[4];
        int   sector;
        float p2 = l <= 0.5f ? l * (1 + s) : l + s - l * s;
        float p1 = 2 * l - p2;

        h *= hscale;
        // We need both loops to clamp (e.g. for h == -1e-40).
        while (h < 0) h += 6;
        while (h >= 6) h -= 6;

        sector = __float2int_rd(h);
        h -= sector;

        tab[0] = p2;
        tab[1] = p1;
        tab[2] = p1 + (p2 - p1) * (1 - h);
        tab[3] = p1 + (p2 - p1) * h;

        b = tab[sector_data[sector][0]];
        g = tab[sector_data[sector][1]];
        r = tab[sector_data[sector][2]];
    }
}

template<typename T>
__global__ void hls2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T h = src[base];
    T l = src[base + 1];
    T s = src[base + 2];

    float b, g, r;

    if constexpr (std::is_same_v<T, float>)
    {
        const float hs = 6.f / 360;
        hls2bgr_f(h, l, s, b, g, r, hs);
        dst[base]     = b;
        dst[base + 1] = g;
        dst[base + 2] = r;
    }
    else
    {
        const float hs = 6.f / 180;
        hls2bgr_f(h, l * (1.f / 255.f), s * (1.f / 255.f), b, g, r, hs);

        dst[base]     = saturate_cast<T>(b * 255.f);
        dst[base + 1] = saturate_cast<T>(g * 255.f);
        dst[base + 2] = saturate_cast<T>(r * 255.f);
    }
}

template<typename T>
__global__ void rgb2hls_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

    float h = 0.f, l = 0.f, s = 0.f;

    if constexpr (std::is_same_v<T, float>)
    {
        bgr2hls_f(b, g, r, h, l, s);

        // dst[base]     = h * hscale; hscale = 360.f / 360.f;
        dst[base]     = h;
        dst[base + 1] = l;
        dst[base + 2] = s;
    }
    else
    {
        float b_f = b * (1.f / 255.f);
        float g_f = g * (1.f / 255.f);
        float r_f = r * (1.f / 255.f);

        bgr2hls_f(b_f, g_f, r_f, h, l, s);

        // dst[base]     = saturate_cast<T>(h * hscale); // hscale = 180 / 360.f;
        dst[base]     = saturate_cast<T>(h * 0.5f);
        dst[base + 1] = saturate_cast<T>(l * 255.f);
        dst[base + 2] = saturate_cast<T>(s * 255.f);
    }
}

static constexpr int xyz_shift = 12;

__constant__ int   sRGB2XYZ_D65_i[9] = {1689, 1465, 739, 871, 2929, 296, 79, 488, 3892};
__constant__ float sRGB2XYZ_D65_f[9]
    = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};

template<typename T>
__global__ void bgr2xyz_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float X = saturate_cast<T>(b * sRGB2XYZ_D65_f[2] + g * sRGB2XYZ_D65_f[1] + r * sRGB2XYZ_D65_f[0]);
        float Y = saturate_cast<T>(b * sRGB2XYZ_D65_f[5] + g * sRGB2XYZ_D65_f[4] + r * sRGB2XYZ_D65_f[3]);
        float Z = saturate_cast<T>(b * sRGB2XYZ_D65_f[8] + g * sRGB2XYZ_D65_f[7] + r * sRGB2XYZ_D65_f[6]);

        dst[base]     = X;
        dst[base + 1] = Y;
        dst[base + 2] = Z;
    }
    else
    {
        int X = CV_DESCALE(b * sRGB2XYZ_D65_i[2] + g * sRGB2XYZ_D65_i[1] + r * sRGB2XYZ_D65_i[0], xyz_shift);
        int Y = CV_DESCALE(b * sRGB2XYZ_D65_i[5] + g * sRGB2XYZ_D65_i[4] + r * sRGB2XYZ_D65_i[3], xyz_shift);
        int Z = CV_DESCALE(b * sRGB2XYZ_D65_i[8] + g * sRGB2XYZ_D65_i[7] + r * sRGB2XYZ_D65_i[6], xyz_shift);

        dst[base]     = saturate_cast<T>(X);
        dst[base + 1] = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Z);
    }
}

__constant__ int   XYZ2sRGB_D65_i[9] = {13273, -6296, -2042, -3970, 7684, 170, 228, -836, 4331};
__constant__ float XYZ2sRGB_D65_f[9]
    = {3.240479, -1.53715, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311};

template<typename T>
__global__ void xyz2bgr_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T X = src[base];
    T Y = src[base + 1];
    T Z = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float B = saturate_cast<T>(X * XYZ2sRGB_D65_f[6] + Y * XYZ2sRGB_D65_f[7] + Z * XYZ2sRGB_D65_f[8]);
        float G = saturate_cast<T>(X * XYZ2sRGB_D65_f[3] + Y * XYZ2sRGB_D65_f[4] + Z * XYZ2sRGB_D65_f[5]);
        float R = saturate_cast<T>(X * XYZ2sRGB_D65_f[0] + Y * XYZ2sRGB_D65_f[1] + Z * XYZ2sRGB_D65_f[2]);

        dst[base]     = B;
        dst[base + 1] = G;
        dst[base + 2] = R;
    }
    else
    {
        int B = CV_DESCALE(X * XYZ2sRGB_D65_i[6] + Y * XYZ2sRGB_D65_i[7] + Z * XYZ2sRGB_D65_i[8], xyz_shift);
        int G = CV_DESCALE(X * XYZ2sRGB_D65_i[3] + Y * XYZ2sRGB_D65_i[4] + Z * XYZ2sRGB_D65_i[5], xyz_shift);
        int R = CV_DESCALE(X * XYZ2sRGB_D65_i[0] + Y * XYZ2sRGB_D65_i[1] + Z * XYZ2sRGB_D65_i[2], xyz_shift);

        dst[base]     = saturate_cast<T>(B);
        dst[base + 1] = saturate_cast<T>(G);
        dst[base + 2] = saturate_cast<T>(R);
    }
}

template<typename T>
__global__ void rgb2xyz_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float X = saturate_cast<T>(b * sRGB2XYZ_D65_f[2] + g * sRGB2XYZ_D65_f[1] + r * sRGB2XYZ_D65_f[0]);
        float Y = saturate_cast<T>(b * sRGB2XYZ_D65_f[5] + g * sRGB2XYZ_D65_f[4] + r * sRGB2XYZ_D65_f[3]);
        float Z = saturate_cast<T>(b * sRGB2XYZ_D65_f[8] + g * sRGB2XYZ_D65_f[7] + r * sRGB2XYZ_D65_f[6]);

        dst[base]     = X;
        dst[base + 1] = Y;
        dst[base + 2] = Z;
    }
    else
    {
        int X = CV_DESCALE(b * sRGB2XYZ_D65_i[2] + g * sRGB2XYZ_D65_i[1] + r * sRGB2XYZ_D65_i[0], xyz_shift);
        int Y = CV_DESCALE(b * sRGB2XYZ_D65_i[5] + g * sRGB2XYZ_D65_i[4] + r * sRGB2XYZ_D65_i[3], xyz_shift);
        int Z = CV_DESCALE(b * sRGB2XYZ_D65_i[8] + g * sRGB2XYZ_D65_i[7] + r * sRGB2XYZ_D65_i[6], xyz_shift);

        dst[base]     = saturate_cast<T>(X);
        dst[base + 1] = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Z);
    }
}

// Lab 杞崲甯搁噺
static constexpr int gamma_shift = 3;
static constexpr int lab_shift   = 12;
static constexpr int lab_shift2  = 15;

static constexpr float intScale   = 255.0f * (1 << gamma_shift);
static constexpr float cbTabScale = 1.0f / (255.0f * (1 << gamma_shift));
static constexpr float lshift2    = (float)(1 << lab_shift2);
static constexpr float lthresh    = 216.0f / 24389.0f;
static constexpr float lscale     = 841.0f / 108.0f;
static constexpr float lbias      = 16.0f / 116.0f;

// 甯搁噺: 0.008856 = (6/29)^3, 7.787 = (29/3)^3/(29*4), 16/116 鈮?0.137931
static constexpr float thresh = 0.008856f;
static constexpr float scale  = 7.787037f;
static constexpr float bias   = 16.0f / 116.0f;

__device__ __forceinline__ float applyGamma(float x)
{
    //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
    return (x > 0.04045f) ? __powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92f);
}

// sRGB gamma 鏍℃鍑芥暟 (鍔ㄦ€佽绠楃増鏈紝鏇夸唬鏌ユ壘琛?
__device__ __forceinline__ unsigned short sRGBGammaTab(int i)
{
    return (unsigned short)(__float2int_rn(intScale * applyGamma(i / 255.0f)));
}

// Lab 绔嬫柟鏍规煡鎵捐〃鐨勫姩鎬佽绠楃増鏈?
__device__ __forceinline__ unsigned short LabCbrtTab(int i)
{
    float x = cbTabScale * i;
    float result;
    if (x < lthresh)
        result = lshift2 * (lscale * x + lbias);
    else
        result = lshift2 * cbrtf(x);
    return (unsigned short)(__float2int_rn(result));
}

// D65 鐧界偣
// __constant__ float D65_inv[3] = {1.0f / 0.950456f, 1.0f / 1.0f, 1.0f / 1.088754f};
// __constant__ float sRGB2XYZ_D65_f[9] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};
// BGR 鍒?XYZ 鐨勮浆鎹㈢郴鏁?(瀹氱偣鏁拌〃绀猴紝宸插綊涓€鍖栧埌 D65 鐧界偣)
// 璁＄畻鏂瑰紡: cvRound((1 << lab_shift) * sRGB2XYZ_D65[i] / D65[i])
// BGR 椤哄簭: R, G, B
__constant__ int BGR2XYZ_coeffs[9] = {
    1777, // C0: X from R = round(4096 * 0.412453 / 0.950456)
    1541, // C1: X from G = round(4096 * 0.357580 / 0.950456)
    778,  // C2: X from B = round(4096 * 0.180423 / 0.950456)
    871,  // C3: Y from R = round(4096 * 0.212671 / 1.0)
    2929, // C4: Y from G = round(4096 * 0.715160 / 1.0)
    296,  // C5: Y from B = round(4096 * 0.072169 / 1.0)
    73,   // C6: Z from R = round(4096 * 0.019334 / 1.088754)
    448,  // C7: Z from G = round(4096 * 0.119193 / 1.088754)
    3575  // C8: Z from B = round(4096 * 0.950227 / 1.088754)
};

// sRGB2XYZ_D65[i] / D65[i]
__constant__ float BGR2XYZ_coeffs_f[9] = {
    0.433953f, // C0: X from R = 0.412453 / 0.950456
    0.376219f, // C1: X from G = 0.357580 / 0.950456
    0.189828f, // C2: X from B = 0.180423 / 0.950456
    0.212671f, // C3: Y from R = 0.212671 / 1.0
    0.715160f, // C4: Y from G = 0.715160 / 1.0
    0.072169f, // C5: Y from B = 0.072169 / 1.0
    0.017758f, // C6: Z from R = 0.019334 / 1.088754
    0.109477f, // C7: Z from G = 0.119193 / 1.088754
    0.872766f  // C8: Z from B = 0.950227 / 1.088754
};

static constexpr int lshift = 1 << lab_shift;
static constexpr int Lscale = (116 * 255 + 50) / 100;
static constexpr int Lshift = -((16 * 255 * (1 << lab_shift2) + 50) / 100);

// 涓夌嚎鎬ф彃鍊煎父閲?
static constexpr int lab_base_shift  = 14;
static constexpr int LAB_BASE        = 1 << lab_base_shift; // 16384
static constexpr int lab_lut_shift   = 5;
static constexpr int LAB_LUT_DIM     = (1 << lab_lut_shift) + 1; // 33
static constexpr int trilinear_shift = 8 - lab_lut_shift + 1;    // 4
static constexpr int TRILINEAR_BASE  = 1 << trilinear_shift;     // 16

// 鍔ㄦ€佽绠?RGB 鍒?Lab 鐨勮浆鎹紙鐢ㄤ簬涓夌嚎鎬ф彃鍊肩殑椤剁偣锛?
__device__ __forceinline__ void computeLabVertex(float R, float G, float B, short &L_out, short &a_out, short &b_out)
{
    // 1. Gamma 鏍℃
    R = applyGamma(R);
    G = applyGamma(G);
    B = applyGamma(B);

    // 2. RGB 鍒?XYZ
    float X = R * BGR2XYZ_coeffs_f[0] + G * BGR2XYZ_coeffs_f[1] + B * BGR2XYZ_coeffs_f[2];
    float Y = R * BGR2XYZ_coeffs_f[3] + G * BGR2XYZ_coeffs_f[4] + B * BGR2XYZ_coeffs_f[5];
    float Z = R * BGR2XYZ_coeffs_f[6] + G * BGR2XYZ_coeffs_f[7] + B * BGR2XYZ_coeffs_f[8];

    // 3. 璁＄畻绔嬫柟鏍癸紙浣跨敤绮剧‘鐨勫垎鏁板父閲忥級
    // lthresh = 216/24389 = (6/29)^3
    // lscale = 841/108 = (29/3)^3/(29*4)
    // lbias = 16/116
    const float thresh = 216.0f / 24389.0f;
    const float scale  = 841.0f / 108.0f;
    const float bias   = 16.0f / 116.0f;

    float FX = X > thresh ? cbrtf(X) : (scale * X + bias);
    float FY = Y > thresh ? cbrtf(Y) : (scale * Y + bias);
    float FZ = Z > thresh ? cbrtf(Z) : (scale * Z + bias);

    // 4. XYZ 鍒?Lab锛堜娇鐢ㄧ簿纭殑鍒嗘暟甯搁噺锛?
    // f9033 = 29*29*29/27 = (29/3)^3 = 903.296296...
    const float f9033 = (29.0f * 29.0f * 29.0f) / 27.0f;
    float       L     = Y > thresh ? (116.0f * FY - 16.0f) : (f9033 * Y);
    float       a     = 500.0f * (FX - FY);
    float       b     = 200.0f * (FY - FZ);

    // 5. 缂栫爜涓?short (涓?OpenCV LUT 鏍煎紡涓€鑷?
    // L: LAB_BASE*L/100, a: LAB_BASE*(a+128)/256, b: LAB_BASE*(b+128)/256
    L_out = (short)(__float2int_rn(LAB_BASE * L / 100.0f));
    a_out = (short)(__float2int_rn(LAB_BASE * (a + 128.0f) / 256.0f));
    b_out = (short)(__float2int_rn(LAB_BASE * (b + 128.0f) / 256.0f));
}

// 涓夌嚎鎬ф彃鍊硷紙鍔ㄦ€佽绠楃増鏈級
__device__ __forceinline__ void trilinearInterpolate(float R, float G, float B, int &L, int &a, int &b)
{
    // 灏?[0,1] 鐨?RGB 缂╂斁鍒?LAB_BASE
    int iR = __float2int_rn(R * LAB_BASE);
    int iG = __float2int_rn(G * LAB_BASE);
    int iB = __float2int_rn(B * LAB_BASE);

    // 闄愬埗鑼冨洿
    iR = min(max(iR, 0), LAB_BASE);
    iG = min(max(iG, 0), LAB_BASE);
    iB = min(max(iB, 0), LAB_BASE);

    // 璁＄畻 LUT 绱㈠紩锛堢珛鏂逛綋鐨勫師鐐癸級
    int tx = iR >> (lab_base_shift - lab_lut_shift);
    int ty = iG >> (lab_base_shift - lab_lut_shift);
    int tz = iB >> (lab_base_shift - lab_lut_shift);

    // 纭繚绱㈠紩鍦ㄦ湁鏁堣寖鍥村唴 [0, LAB_LUT_DIM-1]
    tx = min(max(tx, 0), LAB_LUT_DIM - 1);
    ty = min(max(ty, 0), LAB_LUT_DIM - 1);
    tz = min(max(tz, 0), LAB_LUT_DIM - 1);

    // 璁＄畻鎻掑€兼潈閲?
    const int bitMask = (1 << trilinear_shift) - 1;
    int       x       = (iR >> (lab_base_shift - 8 - 1)) & bitMask;
    int       y       = (iG >> (lab_base_shift - 8 - 1)) & bitMask;
    int       z       = (iB >> (lab_base_shift - 8 - 1)) & bitMask;

    // 璁＄畻涓夌嚎鎬ф彃鍊兼潈閲嶏紙8涓《鐐癸級
    int pp = TRILINEAR_BASE - x;
    int qq = TRILINEAR_BASE - y;
    int rr = TRILINEAR_BASE - z;

    int w[8];
    w[0] = pp * qq * rr;
    w[1] = pp * qq * z;
    w[2] = pp * y * rr;
    w[3] = pp * y * z;
    w[4] = x * qq * rr;
    w[5] = x * qq * z;
    w[6] = x * y * rr;
    w[7] = x * y * z;

    // 鍔ㄦ€佽绠?8 涓《鐐圭殑 Lab 鍊?
    short       Lab[8][3]; // [椤剁偣][L,a,b]
    const float scale = 1.0f / (LAB_LUT_DIM - 1);

    for (int i = 0; i < 8; i++)
    {
        int dx = (i >> 2) & 1;
        int dy = (i >> 1) & 1;
        int dz = i & 1;

        // 浣跨敤 min 闄愬埗绱㈠紩锛屼笌 OpenCV 鐨?fill_one 鍑芥暟涓€鑷?
        int idx_x = min(tx + dx, LAB_LUT_DIM - 1);
        int idx_y = min(ty + dy, LAB_LUT_DIM - 1);
        int idx_z = min(tz + dz, LAB_LUT_DIM - 1);

        float vR = idx_x * scale;
        float vG = idx_y * scale;
        float vB = idx_z * scale;

        computeLabVertex(vR, vG, vB, Lab[i][0], Lab[i][1], Lab[i][2]);
    }

    // 涓夌嚎鎬ф彃鍊?
    L = 0;
    a = 0;
    b = 0;
    for (int i = 0; i < 8; i++)
    {
        L += Lab[i][0] * w[i];
        a += Lab[i][1] * w[i];
        b += Lab[i][2] * w[i];
    }

    // Descale
    L = CV_DESCALE(L, trilinear_shift * 3);
    a = CV_DESCALE(a, trilinear_shift * 3);
    b = CV_DESCALE(b, trilinear_shift * 3);
}

template<typename T>
__global__ void bgr2lab_kernel(T *src, T *dst, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = tid * 3;

    T blue  = src[base];
    T green = src[base + 1];
    T red   = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        // 1. Clip 杈撳叆鍒?[0, 1]
        float R = fminf(fmaxf(red, 0.0f), 1.0f);
        float G = fminf(fmaxf(green, 0.0f), 1.0f);
        float B = fminf(fmaxf(blue, 0.0f), 1.0f);

        // 2. 涓夌嚎鎬ф彃鍊艰绠?Lab
        int iL, ia, ib;
        trilinearInterpolate(R, G, B, iL, ia, ib);

        // 3. 瑙ｇ爜涓烘诞鐐?Lab 鍊?
        float L = iL * 1.0f / LAB_BASE;
        float a = ia * 1.0f / LAB_BASE;
        float b = ib * 1.0f / LAB_BASE;

        // 4. 杞崲鍒版爣鍑?Lab 鑼冨洿
        // L: [0, 100], a: [-128, 127], b: [-128, 127]
        L = L * 100.0f;
        a = a * 256.0f - 128.0f;
        b = b * 256.0f - 128.0f;

        // 5. 瀛樺偍缁撴灉
        dst[base]     = L;
        dst[base + 1] = a;
        dst[base + 2] = b;
    }
    else
    {
        // 1. Gamma 鏍℃ (浣跨敤 sRGB gamma)
        int B = sRGBGammaTab(blue);
        int G = sRGBGammaTab(green);
        int R = sRGBGammaTab(red);

        // 2. 浠庡父閲忔暟缁勮鍙栬浆鎹㈢郴鏁?
        int C0 = BGR2XYZ_coeffs[0], C1 = BGR2XYZ_coeffs[1], C2 = BGR2XYZ_coeffs[2];
        int C3 = BGR2XYZ_coeffs[3], C4 = BGR2XYZ_coeffs[4], C5 = BGR2XYZ_coeffs[5];
        int C6 = BGR2XYZ_coeffs[6], C7 = BGR2XYZ_coeffs[7], C8 = BGR2XYZ_coeffs[8];

        // 3. RGB 鍒?XYZ 杞崲骞舵煡琛ㄨ绠楃珛鏂规牴
        int X = CV_DESCALE(R * C0 + G * C1 + B * C2, lab_shift);
        int Y = CV_DESCALE(R * C3 + G * C4 + B * C5, lab_shift);
        int Z = CV_DESCALE(R * C6 + G * C7 + B * C8, lab_shift);

        int fX = LabCbrtTab(X);
        int fY = LabCbrtTab(Y);
        int fZ = LabCbrtTab(Z);

        // 4. XYZ 鍒?Lab 杞崲

        int L = CV_DESCALE(Lscale * fY + Lshift, lab_shift2);
        int a = CV_DESCALE(500 * (fX - fY) + 4194304, lab_shift2); // 128 * 锛? << 15) = 128 * 32768 = 4194304
        int b = CV_DESCALE(200 * (fY - fZ) + 4194304, lab_shift2);

        // 5. 楗卞拰杞崲骞跺瓨鍌ㄧ粨鏋?
        dst[base]     = saturate_cast<T>(L);
        dst[base + 1] = saturate_cast<T>(a);
        dst[base + 2] = saturate_cast<T>(b);
    }
}

template<typename T>
void CvtColorImpl<T>::RunCvtColor(const T *d_src, T *d_dst, const int2 size, const int code, cudaStream_t stream)
{
    const int N = size.x * size.y;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    T *src = const_cast<T *>(d_src);

#define IRT_CVCUDA_LAUNCH_CVTCOLOR(KERNEL) KERNEL<T><<<grid_size, block_size, 0, stream>>>(src, d_dst, size.y, size.x, N)

    if (code == cv::COLOR_GRAY2BGR || code == cv::COLOR_GRAY2RGB)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(gray2bgr_kernel);
    }
    else if (code == cv::COLOR_GRAY2BGRA || code == cv::COLOR_GRAY2RGBA)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(gray2bgra_kernel);
    }
    else if (code == cv::COLOR_BGR2RGB || code == cv::COLOR_RGB2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2rgb_kernel);
    }
    else if (code == cv::COLOR_BGR2RGBA || code == cv::COLOR_RGB2BGRA)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2rgba_kernel);
    }
    else if (code == cv::COLOR_BGR2BGRA || code == cv::COLOR_RGB2RGBA)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2bgra_kernel);
    }
    else if (code == cv::COLOR_BGR2GRAY)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2gray_kernel);
    }
    else if (code == cv::COLOR_RGB2GRAY)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2gray_kernel);
    }
    else if (code == cv::COLOR_BGR2YCrCb)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2YCrCb_kernel);
    }
    else if (code == cv::COLOR_RGB2YCrCb)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2YCrCb_kernel);
    }
    else if (code == cv::COLOR_YCrCb2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(YCrCb2bgr_kernel);
    }
    else if (code == cv::COLOR_BGR2YUV)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2yuv_kernel);
    }
    else if (code == cv::COLOR_RGB2YUV)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2yuv_kernel);
    }
    else if (code == cv::COLOR_YUV2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(yuv2bgr_kernel);
    }
    else if (code == cv::COLOR_BGR2HSV)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2hsv_kernel);
    }
    else if (code == cv::COLOR_RGB2HSV)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2hsv_kernel);
    }
    else if (code == cv::COLOR_HSV2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(hsv2bgr_kernel);
    }
    else if (code == cv::COLOR_BGR2HLS)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2hls_kernel);
    }
    else if (code == cv::COLOR_RGB2HLS)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2hls_kernel);
    }
    else if (code == cv::COLOR_HLS2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(hls2bgr_kernel);
    }
    else if (code == cv::COLOR_BGR2XYZ)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2xyz_kernel);
    }
    else if (code == cv::COLOR_RGB2XYZ)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(rgb2xyz_kernel);
    }
    else if (code == cv::COLOR_XYZ2BGR)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(xyz2bgr_kernel);
    }
    else if (code == cv::COLOR_BGR2Lab)
    {
        IRT_CVCUDA_LAUNCH_CVTCOLOR(bgr2lab_kernel);
    }
    else
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "CvtColor code not implemented: %d", code);
    }

#undef IRT_CVCUDA_LAUNCH_CVTCOLOR

    IRT_CHECK_THROW(cudaPeekAtLastError(), "CvtColor kernel launch failed: code=%d size=%dx%d", code, size.x, size.y);
}

template void CvtColorImpl<uint8_t>::RunCvtColor(const uint8_t *, uint8_t *, const int2, const int, cudaStream_t);
template void CvtColorImpl<float>::RunCvtColor(const float *, float *, const int2, const int, cudaStream_t);

} // namespace irt::cvcuda::priv
