#pragma once

namespace irt::cvcuda {

// OpenCV saturate_cast
template<typename _Tp, typename _Tp2>
__device__ __forceinline__ _Tp saturate_cast(_Tp2 v);

template<typename _Tp>
__device__ __forceinline__ _Tp saturate_cast(float v)
{
    return _Tp(v);
}

template<typename _Tp>
__device__ __forceinline__ _Tp saturate_cast(double v)
{
    return _Tp(v);
}

template<>
__device__ __forceinline__ unsigned char saturate_cast(unsigned char v)
{
    return v;
}

template<>
__device__ __forceinline__ unsigned char saturate_cast(int v)
{
    return (unsigned char)((unsigned)v <= 255U ? v : v > 0 ? 255 : 0);
}

template<>
__device__ __forceinline__ unsigned char saturate_cast(float v)
{
    int iv = __float2int_rn(v);
    return saturate_cast<unsigned char>(iv);
}

template<>
__device__ __forceinline__ unsigned char saturate_cast(double v)
{
    int iv = __double2int_rn(v);
    return saturate_cast<unsigned char>(iv);
}

template<>
__device__ __forceinline__ short saturate_cast(short v)
{
    return v;
}

template<>
__device__ __forceinline__ short saturate_cast(int v)
{
    return (short)((unsigned)(v + 32768) <= 65535U ? v : v > 0 ? 32767 : -32768);
}

template<>
__device__ __forceinline__ short saturate_cast(float v)
{
    int iv = __float2int_rn(v);
    return saturate_cast<short>(iv);
}

template<>
__device__ __forceinline__ short saturate_cast(double v)
{
    int iv = __double2int_rn(v);
    return saturate_cast<short>(iv);
}

template<>
__device__ __forceinline__ int saturate_cast(int v)
{
    return v;
}

template<>
__device__ __forceinline__ int saturate_cast(float v)
{
    return __float2int_rn(v);
}

template<>
__device__ __forceinline__ int saturate_cast(double v)
{
    return __double2int_rn(v);
}

} // namespace irt::cvcuda