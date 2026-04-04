#include "priv/OpResize.cuh"

#include <inferrt/cvcuda/OpResize.h>

namespace inferrt::cvcuda {

template<typename T, int CH>
void resize(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, const int interpolation, cudaStream_t stream)
{
    // TODO
}

// 显式实例化你需要的类型组合
template void resize<uint8_t, 1>(const uint8_t *, uint8_t *, cv::Size, cv::Size, int, cudaStream_t);
template void resize<uint8_t, 3>(const uint8_t *, uint8_t *, cv::Size, cv::Size, int, cudaStream_t);
template void resize<uint8_t, 4>(const uint8_t *, uint8_t *, cv::Size, cv::Size, int, cudaStream_t);
template void resize<float, 1>(const float *, float *, cv::Size, cv::Size, int, cudaStream_t);
template void resize<float, 3>(const float *, float *, cv::Size, cv::Size, int, cudaStream_t);
template void resize<float, 4>(const float *, float *, cv::Size, cv::Size, int, cudaStream_t);

} // namespace inferrt::cvcuda