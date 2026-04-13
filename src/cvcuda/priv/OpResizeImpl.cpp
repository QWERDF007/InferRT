#include "OpResizeImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <opencv2/opencv.hpp>

namespace irt::cvcuda::priv {

template<typename T>
void ResizeImpl<T>::operator()(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                               const int dstride, const int CH, const int interpolation, cudaStream_t stream)
{
    // 参数检查
    if (d_src == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Source pointer is null");
    }
    if (d_dst == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Destination pointer is null");
    }
    if (ssize.x <= 0 || ssize.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid source size");
    }
    if (dsize.x <= 0 || dsize.y <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid destination size");
    }
    if (CH <= 0 || CH > 4)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid channel count (must be 1-4)");
    }

    // 调用 CUDA 实现
    RunResize(d_src, d_dst, ssize, sstride, dsize, dstride, CH, interpolation, stream);
}

// 显式实例化
template void ResizeImpl<uint8_t>::operator()(const uint8_t *, uint8_t *, const int2, const int, const int2, const int,
                                              const int, const int, cudaStream_t);
template void ResizeImpl<float>::operator()(const float *, float *, const int2, const int, const int2, const int,
                                            const int, const int, cudaStream_t);

} // namespace irt::cvcuda::priv
