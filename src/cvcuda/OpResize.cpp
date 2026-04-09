#include "priv/OpResize.cuh"

#include "priv/IOperatorImpl.hpp"
#include "priv/OpResizeImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpResize.h>
#include <inferrt/cvcuda/OpResize.hpp>

namespace irt::cvcuda {

using core::ProtectCall;

template<typename T>
int resize(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH, const int interpolation,
           cudaStream_t stream)
{
    core::IRTStatus status = ProtectCall(
        [&]
        {
            int2 _ssize;
            _ssize.x = ssize.width;
            _ssize.y = ssize.height;
            int2 _dsize;
            _dsize.x = dsize.width;
            _dsize.y = dsize.height;
            int sstride;
            sstride = ssize.width * CH;
            int dstride;
            dstride = dsize.width * CH;

            switch (interpolation)
            {
            case cv::INTER_LINEAR:
            {
                priv::resize_bilinear<T, float>(d_src, d_dst, _ssize, sstride, _dsize, dstride, CH, stream);
                break;
            }
            default:
            {
                throw core::Exception(core::Status::ERROR_NOT_IMPLEMENTED, "Interpolation method not implemented");
                break;
            }
            }
        });
    return status;
}

// 显式实例化你需要的类型组合
// Windows 模板的显式实例化也需要加上 __declspec(dllexport) 才能生成导入库（.lib 文件）
template INFERRT_CVCUDA_API int resize<uint8_t>(const uint8_t *, uint8_t *, cv::Size, cv::Size, const int, const int,
                                                cudaStream_t);
template INFERRT_CVCUDA_API int resize<float>(const float *, float *, cv::Size, cv::Size, const int, const int,
                                              cudaStream_t);

Resize::Resize()
{
    impl_.reset(new priv::ResizeImpl());
}

template<typename T>
int Resize::operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH, const int interpolation,
                       cudaStream_t stream)
{
    core::IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
                throw core::Exception(core::Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");

            int2 _ssize;
            _ssize.x = ssize.width;
            _ssize.y = ssize.height;
            int2 _dsize;
            _dsize.x = dsize.width;
            _dsize.y = dsize.height;
            int sstride;
            sstride = ssize.width * CH;
            int dstride;
            dstride = dsize.width * CH;

            impl_(d_src, d_dst, _ssize, sstride, _dsize, dstride, CH, interpolation, stream);
        });
    return status;
}

} // namespace irt::cvcuda