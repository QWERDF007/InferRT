/**
 * @file BuiltinOperators.hpp
 * @brief 可在代码 DAG 中使用的内置图像流水线算子。
 */

#pragma once

#include <inferrt/engine/Pipeline.hpp>
#include <opencv2/imgproc.hpp>

#include <array>

namespace irt::engine {

struct INFERRT_ENGINE_API CpuImageToTensorOptions
{
    std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> stddev{0.229F, 0.224F, 0.225F};
    bool                 letterbox{false};
};

/** 将 submit() 提供的命名 float32 输入复制到 host tensor。 */
struct INFERRT_ENGINE_API CpuCopyTensorOptions
{
    std::string input_name;
};

struct INFERRT_ENGINE_API ResizeOptions
{
    int interpolation{cv::INTER_LINEAR};
};

struct INFERRT_ENGINE_API CvtColorOptions
{
    int code{cv::COLOR_BGR2RGB};
};

struct INFERRT_ENGINE_API NormalizeOptions
{
    std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> stddev{0.229F, 0.224F, 0.225F};
};

/** @brief 显式注册全部内置 CPU/CUDA 图像算子。 */
INFERRT_ENGINE_API void registerBuiltinOperators(OperatorRegistry &registry);

} // namespace irt::engine
