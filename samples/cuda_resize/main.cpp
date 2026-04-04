#include <inferrt/cvcuda/OpResize.h>

#include <iostream>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    float  *d_src = nullptr;
    float  *d_dst = nullptr;
    cv::Mat img(100, 200, CV_8UC3);
    inferrt::cvcuda::resize<float, 3>(d_src, d_dst, img.size(), cv::Size(200, 50), 0, nullptr);
    return 0;
}