#include <gtest/gtest.h>
#include <inferrt/cvcuda/OpResize.h>
#include <opencv2/opencv.hpp>
#include <vtest/common/ValueTests.hpp>

// 参数：src_width, src_height, channels, scale, interp
_TEST_SUITE_P(MultiParamTest, vtest::ValueList<int>{1, 2, 10, 100} * vtest::ValueList<int>{1, 2, 10, 100}
                                  * vtest::ValueList<int>{1, 3, 4}
                                  * vtest::ValueList<double>{0.1, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 2.5, 3.0}
                                  * vtest::ValueList<int>{cv::INTER_LINEAR});

TEST_P(MultiParamTest, CFunc8UTest)
{
    const int    src_w  = GetParamValue<0>();
    const int    src_h  = GetParamValue<1>();
    const int    ch     = GetParamValue<2>();
    const double scale  = GetParamValue<3>();
    const int    interp = GetParamValue<4>();

    const int dst_w = std::max(1, static_cast<int>(src_w * scale));
    const int dst_h = std::max(1, static_cast<int>(src_h * scale));

    // 创建随机源图像
    cv::Mat src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    // OpenCV 参考结果
    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, interp);

    // 分配 GPU 内存
    const size_t src_bytes = src_h * src_w * ch * sizeof(uint8_t);
    const size_t dst_bytes = dst_h * dst_w * ch * sizeof(uint8_t);

    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);

    // 上传源数据（确保连续）
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    // 执行 resize
    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, interp,
                                           nullptr);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 下载结果
    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    // 比较：u8 最大差异不超过 1
    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double maxVal = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxVal);
    EXPECT_LE(maxVal, 1.0) << "src=" << src_w << "x" << src_h << " ch=" << ch << " scale=" << scale << " dst=" << dst_w
                           << "x" << dst_h;
}
