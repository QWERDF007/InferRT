/**
 * @file BuiltinOperators.hpp
 * @brief 可在代码 DAG 中使用的内置图像流水线算子。
 */

#pragma once

#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/engine/Pipeline.hpp>

namespace irt::engine {

struct INFERRT_ENGINE_API CpuImageToTensorOptions
{
    irt::PreprocessSpec preprocess{};
};

/** 将 submit() 提供的命名 float32 输入复制到 host tensor。 */
struct INFERRT_ENGINE_API CpuCopyTensorOptions
{
    std::string input_name;
};

struct INFERRT_ENGINE_API ResizeOptions
{
    irt::PreprocessSpec preprocess{};
};

struct INFERRT_ENGINE_API CvtColorOptions
{
    irt::PreprocessSpec preprocess{};
};

struct INFERRT_ENGINE_API NormalizeOptions
{
    irt::PreprocessSpec preprocess{};
};

struct INFERRT_ENGINE_API LetterBoxOptions
{
    irt::PreprocessSpec preprocess{};
};

struct INFERRT_ENGINE_API CenterCropOptions
{
    irt::PreprocessSpec preprocess{};
};

/** @brief 显式注册全部内置 CPU/CUDA 图像算子。 */
INFERRT_ENGINE_API void registerBuiltinOperators(OperatorRegistry &registry);

} // namespace irt::engine
