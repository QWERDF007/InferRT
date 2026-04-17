#include "priv/OpResizeImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpResize.h>
#include <inferrt/cvcuda/OpResize.hpp>

namespace irt::cvcuda {

using irt::ProtectCall;

template<typename T>
IRTStatus resize(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH, const int interpolation,
                 cudaStream_t stream)
{
    Resize<T> resizer;
    return resizer(d_src, d_dst, ssize, dsize, CH, interpolation, stream);
}

// 显式实例化你需要的类型组合
// Windows 模板的显式实例化也需要加上 __declspec(dllexport) 才能生成导入库（.lib 文件）
template INFERRT_CVCUDA_API IRTStatus resize<uint8_t>(const uint8_t *, uint8_t *, cv::Size, cv::Size, const int,
                                                      const int, cudaStream_t);
template INFERRT_CVCUDA_API IRTStatus resize<float>(const float *, float *, cv::Size, cv::Size, const int, const int,
                                                    cudaStream_t);

template<typename T>
Resize<T>::Resize()
{
    impl_ = new priv::ResizeImpl<T>();
}

template<typename T>
Resize<T>::~Resize()
{
    if (impl_)
    {
        delete impl_;
        impl_ = nullptr;
    }
}

template<typename T>
IRTStatus Resize<T>::operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int CH,
                                const int interpolation, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");

            int2 _ssize;
            _ssize.x = ssize.width;
            _ssize.y = ssize.height;
            int2 _dsize;
            _dsize.x    = dsize.width;
            _dsize.y    = dsize.height;
            int sstride = ssize.width * CH;
            int dstride = dsize.width * CH;

            // 动态转换到具体类型
            auto *resizeImpl = static_cast<priv::ResizeImpl<T> *>(impl_);
            (*resizeImpl)(d_src, d_dst, _ssize, sstride, _dsize, dstride, CH, interpolation, stream);
        });
    return status;
}

// 显式实例化 Resize 类
template class INFERRT_CVCUDA_API Resize<uint8_t>;
template class INFERRT_CVCUDA_API Resize<float>;

} // namespace irt::cvcuda
