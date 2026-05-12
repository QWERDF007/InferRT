#include <cuda_runtime_api.h>
#include <inferrt/model/IModel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

cv::Mat preprocess(const cv::Mat &img)
{
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);

    return resized;
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 4 || argc > 5)
        {
            std::cerr
                << "Usage: " << argv[0]
                << " <resnet18|resnet34|resnet50|resnet101|resnet152|wide_resnet50_2|wide_resnet101_2>"
                   " <weights_file.wts> <image_path> [label_file]"
                << std::endl;
            std::cerr << "Example: " << argv[0] << " resnet18 resnet18.wts dog.jpg" << std::endl;
            std::cerr << "         " << argv[0] << " resnet50 resnet50.wts dog.jpg imagenet1000_clsidx_to_labels.txt"
                      << std::endl;
            std::cerr << "         " << argv[0]
                      << " wide_resnet50_2 wide_resnet50_2.wts dog.jpg imagenet1000_clsidx_to_labels.txt"
                      << std::endl;
            std::cerr << "         " << argv[0] << " resnet152 resnet152.wts dog.jpg imagenet1000_clsidx_to_labels.txt"
                      << std::endl;
            return -1;
        }

        const std::string model_name   = argv[1];
        fs::path          weights_file = fs::path(argv[2]);
        fs::path          img_path     = fs::absolute(argv[3]);
        fs::path          label_file   = (argc == 5) ? fs::canonical(argv[4]) : fs::path();

        auto model = irt::model::CreateModel(model_name);
        if (!model)
        {
            std::cerr << "Failed to create model: " << model_name << std::endl;
            return -1;
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or Loading model..." << std::endl;
        model->buildOrLoad(weights_file.string());
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Loading image: " << img_path.generic_string() << std::endl;
        cv::Mat img = cv::imread(img_path.generic_string());
        if (img.empty())
        {
            std::cerr << "Failed to load image: " << img_path.generic_string() << std::endl;
            return -1;
        }

        cv::Mat preprocessed = preprocess(img);

        std::vector<float>   input_data(1 * 3 * 224 * 224);
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);

        for (int c = 0; c < 3; ++c)
        {
            std::memcpy(input_data.data() + c * 224 * 224, channels[c].data, 224 * 224 * sizeof(float));
        }

        void *d_input  = nullptr;
        void *d_output = nullptr;

        size_t input_size  = 1 * 3 * 224 * 224 * sizeof(float);
        size_t output_size = 1 * 1000 * sizeof(float);

        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);

        cudaMemcpy(d_input, input_data.data(), input_size, cudaMemcpyHostToDevice);

        std::vector<void *> buffers = {d_input, d_output};
        std::cout << "Running inference..." << std::endl;
        model->infer(buffers);

        std::vector<float> output_data(1000);
        cudaMemcpy(output_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);

        std::vector<std::string> labels;
        bool                     has_labels = false;
        if (!label_file.empty() && fs::exists(label_file))
        {
            labels     = irt::model::readImagenetLabels(label_file.string());
            has_labels = true;
        }

        std::vector<std::pair<float, int>> scores;
        scores.reserve(1000);
        for (int i = 0; i < 1000; ++i)
        {
            scores.push_back({output_data[i], i});
        }
        std::partial_sort(scores.begin(), scores.begin() + 3, scores.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

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
                std::cout << "top: " << (i + 1) << ", confidence: " << confidence << ", label[" << idx << "]"
                          << std::endl;
            }
        }

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
