#include <cuda_runtime_api.h>
#include <inferrt/model/IModel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// 预处理函数：对齐 ImageNet 数据集的预处理方式
cv::Mat preprocess(const cv::Mat &img)
{
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // ImageNet 归一化参数
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);

    return resized;
}

// 读取 ImageNet 标签
std::vector<std::string> read_imagenet_labels(const std::string &label_file)
{
    std::vector<std::string> labels(1000);
    std::ifstream            file(label_file);
    if (!file.is_open())
    {
        std::cerr << "Failed to open label file: " << label_file << std::endl;
        return labels;
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t colon_pos = line.find(": ");
        if (colon_pos != std::string::npos)
        {
            int         idx   = std::stoi(line.substr(0, colon_pos));
            std::string label = line.substr(colon_pos + 2);
            // 移除引号
            if (label.size() >= 2 && label.front() == '\'' && label.back() == ',')
            {
                label = label.substr(1, label.size() - 3);
            }
            if (idx >= 0 && idx < 1000)
            {
                labels[idx] = label;
            }
        }
    }

    return labels;
}

int main(int argc, char *argv[])
{
    try
    {
        // 检查命令行参数
        if (argc < 3 || argc > 4)
        {
            std::cerr << "Usage: " << argv[0] << " <weights_file.wts> <image_path> [label_file]" << std::endl;
            std::cerr << "Example: " << argv[0] << " alexnet.wts dog.jpg" << std::endl;
            std::cerr << "         " << argv[0] << " alexnet.wts dog.jpg imagenet1000_clsidx_to_labels.txt"
                      << std::endl;
            return -1;
        }

        // 从命令行参数读取文件路径
        fs::path weights_file = fs::path(argv[1]);
        fs::path img_path     = fs::absolute(argv[2]);
        fs::path label_file   = (argc == 4) ? fs::canonical(argv[3]) : fs::path();

        // 创建 AlexNet 模型
        auto model = std::unique_ptr<irt::model::IModel>(irt::model::CreateModel("alexnet"));
        if (!model)
        {
            std::cerr << "Failed to create AlexNet model" << std::endl;
            return -1;
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or Loading model..." << std::endl;
        model->buildOrLoad(weights_file.string());
        std::cout << "Model loaded successfully." << std::endl;

        // 读取并预处理图像
        std::cout << "Loading image: " << img_path.generic_string() << std::endl;
        cv::Mat img = cv::imread(img_path.generic_string());
        if (img.empty())
        {
            std::cerr << "Failed to load image: " << img_path.generic_string() << std::endl;
            return -1;
        }

        cv::Mat preprocessed = preprocess(img);

        // 转换为 NCHW 格式
        std::vector<float>   input_data(1 * 3 * 224 * 224);
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);

        for (int c = 0; c < 3; ++c)
        {
            std::memcpy(input_data.data() + c * 224 * 224, channels[c].data, 224 * 224 * sizeof(float));
        }

        // 分配 GPU 内存
        void *d_input  = nullptr;
        void *d_output = nullptr;

        size_t input_size  = 1 * 3 * 224 * 224 * sizeof(float);
        size_t output_size = 1 * 1000 * sizeof(float);

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);

        // 拷贝输入数据到 GPU
        cudaMemcpy(d_input, input_data.data(), input_size, cudaMemcpyHostToDevice);

        // 执行推理
        std::vector<void *> buffers = {d_input, d_output};
        std::cout << "Running inference..." << std::endl;
        model->infer(buffers);

        // 拷贝输出数据到 CPU
        std::vector<float> output_data(1000);
        cudaMemcpy(output_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);

        // 读取标签（如果提供了标签文件）
        std::vector<std::string> labels;
        bool                     has_labels = false;
        if (fs::exists(label_file))
        {
            labels     = read_imagenet_labels(label_file.string());
            has_labels = true;
        }

        // 找到 Top-3 结果
        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < 1000; ++i)
        {
            scores.push_back({output_data[i], i});
        }
        std::partial_sort(scores.begin(), scores.begin() + 3, scores.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        // 输出结果
        std::cout << "\nTop-3 predictions:" << std::endl;
        for (int i = 0; i < 3; ++i)
        {
            int   idx        = scores[i].second;
            float confidence = scores[i].first;
            if (has_labels)
            {
                std::cout << "top: " << (i + 1) << ", confidence: " << confidence << ", label[" << idx
                          << "]: " << labels[idx] << std::endl;
            }
            else
            {
                std::cout << "top: " << (i + 1) << ", confidence: " << confidence << ", label[" << idx << ']'
                          << std::endl;
            }
        }

        // 释放 GPU 内存
        cudaFree(d_input);
        cudaFree(d_output);

        std::cout << "\nDone!" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
