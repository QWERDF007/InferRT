#include <cuda_runtime.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpResize.h>
#include <inferrt/cvcuda/OpResize.hpp>

#include <iostream>

// // c++20 辅助函数：根据 OpenCV 类型调用对应的模板
// template<typename Func>
// void dispatchByDepth(int depth, Func &&func)
// {
//     switch (depth)
//     {
//     case CV_8U:
//         func.template operator()<uint8_t>();
//         break;
//     case CV_8S:
//         func.template operator()<int8_t>();
//         break;
//     case CV_16U:
//         func.template operator()<uint16_t>();
//         break;
//     case CV_16S:
//         func.template operator()<int16_t>();
//         break;
//     case CV_32S:
//         func.template operator()<int32_t>();
//         break;
//     case CV_32F:
//         func.template operator()<float>();
//         break;
//     case CV_64F:
//         func.template operator()<double>();
//         break;
//     default:
//         throw std::runtime_error("Unsupported image depth: " + std::to_string(depth));
//     }
// }

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    auto func = [](const cv::Mat &img, const double scale, const int interpolation)
    {
        std::cout << "img size: " << img.size << std::endl;
        std::cout << "img type: " << img.type() << std::endl;

        cv::Size dsize(static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale));
        cv::Mat  img_resize(dsize, img.type());

        size_t s_nbytes = img.total() * img.elemSize();
        size_t d_nbytes = img_resize.total() * img_resize.elemSize();

        void *d_src = nullptr;
        void *d_dst = nullptr;

        cudaMalloc((void **)&d_src, s_nbytes);
        cudaMalloc((void **)&d_dst, d_nbytes);
        cudaMemcpy(d_src, img.data, s_nbytes, cudaMemcpyHostToDevice);

        // // 根据图像数据类型调用不同的模板
        // dispatchByDepth(img.depth(),
        //                 [&]<typename T>()
        //                 {
        //                     irt::cvcuda::resize<T>(static_cast<T *>(d_src), static_cast<T *>(d_dst), img.size(), dsize,
        //                                            img.channels(), cv::INTER_LINEAR, nullptr);
        //                 });

        int depth = img.depth();
        switch (depth)
        {
        case CV_8U:
            irt::cvcuda::resize<uint8_t>(static_cast<uint8_t *>(d_src), static_cast<uint8_t *>(d_dst), img.size(),
                                         dsize, img.channels(), interpolation, nullptr);
            break;

        case CV_32F:
        {
            // irt::cvcuda::resize<float>(static_cast<float *>(d_src), static_cast<float *>(d_dst), img.size(), dsize,
            //                            img.channels(), interpolation, nullptr);
            irt::cvcuda::Resize resizer;
            resizer.operator()<float>(static_cast<float *>(d_src), static_cast<float *>(d_dst), img.size(), dsize,
                                      img.channels(), interpolation, nullptr);
            break;
        }
        default:
            std::cerr << "Unsupported image depth: " << depth << std::endl;
            cudaFree(d_src);
            cudaFree(d_dst);
            return;
        }

        cudaMemcpy(img_resize.data, d_dst, d_nbytes, cudaMemcpyDeviceToHost);
        cudaFree(d_src);
        cudaFree(d_dst);

        cv::Mat cv_img_resize;
        cv::resize(img, cv_img_resize, dsize, 0, 0, interpolation);

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
    };

    cv::Mat img(200, 200, CV_8UC3);
    cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); // 随机填充

    try
    {
        func(img, 0.25, cv::INTER_LINEAR);
    }
    catch (...)
    {
        char msg[1024];
        irt::core::GetLastErrorMessage(msg, 1024);
        std::cout << msg << std::endl;
    }

    cv::Mat img_f32;
    img.convertTo(img_f32, CV_32F, 1 / 255.0, 0);

    try
    {
        func(img_f32, 0.25, cv::INTER_LINEAR);
    }
    catch (...)
    {
        char msg[1024];
        irt::core::GetLastErrorMessage(msg, 1024);
        std::cout << msg << std::endl;
    }

    try
    {
        std::cout << "INTER_NEAREST - 0" << std::endl;
        func(img_f32, 0.25, cv::INTER_NEAREST);

        // Check if there was an error
        char      msg[1024];
        IRTStatus status = irt::core::GetLastErrorMessage(msg, 1024);
        std::cout << __FUNCTION__ << " " << __LINE__ << " code: " << status << std::endl;
        if (status != IRT_SUCCESS)
        {
            std::cout << "Error detected: " << msg << std::endl;
        }
        else
        {
            std::cout << "INTER_NEAREST - 1" << std::endl;
        }
    }
    catch (irt::core::Exception &e)
    {
        std::cout << "Catch Exception: " << e.what() << std::endl;
        char msg[1024];
        irt::core::GetLastErrorMessage(msg, 1024);
        std::cout << msg << std::endl;
    }

    std::cout << "Done!" << std::endl;

    return 0;
}