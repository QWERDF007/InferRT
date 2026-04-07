#include <cuda_runtime.h>
#include <inferrt/cvcuda/OpResize.h>

#include <iostream>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    cv::Mat img(200, 200, CV_8UC3);
    cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); // 随机填充

    double   scale = 0.25;
    cv::Size dsize(static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale));
    cv::Mat  img_resize(dsize, img.type());

    size_t s_nbytes = img.total() * img.elemSize();
    size_t d_nbytes = img_resize.total() * img_resize.elemSize();

    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;

    cudaMalloc((void **)&d_src, s_nbytes);
    cudaMalloc((void **)&d_dst, d_nbytes);
    cudaMemcpy(d_src, img.data, s_nbytes, cudaMemcpyHostToDevice);

    irt::cvcuda::resize<uint8_t>(d_src, d_dst, img.size(), dsize, img.channels(), cv::INTER_LINEAR, nullptr);
    cudaMemcpy(img_resize.data, d_dst, d_nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat cv_img_resize;
    cv::resize(img, cv_img_resize, dsize, 0, 0, cv::INTER_LINEAR);

    cv::Mat diff;
    cv::absdiff(cv_img_resize, img_resize, diff);

    // 计算实际的差异统计
    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);
    cv::Scalar meanDiff = cv::mean(diff);

    std::cout << "max diff: " << maxVal << std::endl;
    std::cout << "mean diff per channel: " << meanDiff << std::endl;
    std::cout << "mean diff (avg): " << (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0 << std::endl;

    std::vector<cv::Mat> splits;
    cv::split(diff, splits);
    int count{0};
    int countLargeDiff = 0;
    for (int i = 0; i < 3; ++i)
    {
        count += cv::countNonZero(splits[i]);
        countLargeDiff += cv::countNonZero(splits[i] > 1);
    }
    std::cout << "diff count: " << count << std::endl;
    std::cout << "channel pixels with diff > 1: " << countLargeDiff << std::endl;

    return 0;
}