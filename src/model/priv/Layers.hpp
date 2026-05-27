#pragma once

#include "Weights.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cmath>

namespace irt::model {

// ============================================================================
// 常量
// ============================================================================

constexpr float kPi = 3.14159265358979323846F;

// ============================================================================
// 张量形状辅助函数
// ============================================================================

/**
 * @brief 生成与输入张量同 rank 的标量广播维度。
 */
inline nvinfer1::Dims scalarDimsLike(const nvinfer1::ITensor &input)
{
    const auto input_dims = input.getDimensions();
    nvinfer1::Dims dims{};
    dims.nbDims = input_dims.nbDims;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = 1;
    }
    return dims;
}

// ============================================================================
// 激活函数
// ============================================================================

/**
 * @brief 添加 GeLU tanh 近似激活：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))。
 */
inline nvinfer1::ITensor *addGeluApprox(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    const auto scalar_dims = scalarDimsLike(input);
    auto *half = network->addConstant(scalar_dims, ownedScalarWeight(0.5F))->getOutput(0);
    auto *one = network->addConstant(scalar_dims, ownedScalarWeight(1.0F))->getOutput(0);
    auto *sqrt_2_div_pi = network->addConstant(scalar_dims, ownedScalarWeight(std::sqrt(2.0F / kPi)))->getOutput(0);
    auto *coeff = network->addConstant(scalar_dims, ownedScalarWeight(0.044715F))->getOutput(0);

    auto *x2 = network->addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *x3 = network->addElementWise(*x2, input, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *scaled_x3
        = network->addElementWise(*x3, *coeff, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *inner = network->addElementWise(input, *scaled_x3, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    auto *scaled
        = network->addElementWise(*inner, *sqrt_2_div_pi, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *tanh = network->addActivation(*scaled, nvinfer1::ActivationType::kTANH)->getOutput(0);
    auto *one_plus_tanh = network->addElementWise(*tanh, *one, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    auto *half_x = network->addElementWise(input, *half, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    return network->addElementWise(*half_x, *one_plus_tanh, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
}

/**
 * @brief 添加精确 GeLU 激活：0.5*x*(1+erf(x/sqrt(2)))。
 */
inline nvinfer1::ITensor *addGeluExact(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto dims = scalarDimsLike(input);
    auto *half = network->addConstant(dims, ownedScalarWeight(0.5F))->getOutput(0);
    auto *one = network->addConstant(dims, ownedScalarWeight(1.0F))->getOutput(0);
    auto *inv_sqrt_2 = network->addConstant(dims, ownedScalarWeight(0.70710678118654752440F))->getOutput(0);

    auto *scaled = network->addElementWise(input, *inv_sqrt_2, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *erf = network->addUnary(*scaled, nvinfer1::UnaryOperation::kERF)->getOutput(0);
    auto *one_plus = network->addElementWise(*erf, *one, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    auto *half_x = network->addElementWise(input, *half, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    return network->addElementWise(*half_x, *one_plus, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
}

/**
 * @brief 添加 SiLU/Swish 激活：x * sigmoid(x)。
 */
inline nvinfer1::ITensor *addSilu(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *sigmoid = network->addActivation(input, nvinfer1::ActivationType::kSIGMOID)->getOutput(0);
    return network->addElementWise(input, *sigmoid, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
}

// ============================================================================
// 归一化层
// ============================================================================

/**
 * @brief 添加最后一维上的 LayerNorm。
 */
inline nvinfer1::ITensor *addLayerNorm(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const std::string &prefix, int embed_dim,
                                        float epsilon)
{
    const auto scale_weights = requireWeight(weights_map, prefix + ".weight", "LayerNorm", embed_dim);
    const auto bias_weights = requireWeight(weights_map, prefix + ".bias", "LayerNorm", embed_dim);
    auto *scale = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, scale_weights)->getOutput(0);
    auto *bias = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, bias_weights)->getOutput(0);

    const auto dims = input.getDimensions();
    const auto axes = 1U << static_cast<uint32_t>(dims.nbDims - 1);
#if TRT_VERSION >= 11500
    auto *norm = network->addNormalizationV2(input, *scale, *bias, axes);
#else
    auto *norm = network->addNormalization(input, *scale, *bias, axes);
#endif
    norm->setEpsilon(epsilon);
    return norm->getOutput(0);
}

// ============================================================================
// 线性层
// ============================================================================

/**
 * @brief 添加面向 [N, L, C] token 张量的线性层。
 */
inline nvinfer1::ITensor *addLinear3D(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int in_features,
                                       int out_features, bool bias_required = true)
{
    auto *weight = network
                       ->addConstant(nvinfer1::Dims3{1, out_features, in_features},
                                     requireWeight(weights_map, prefix + ".weight",
                                                   "Linear", static_cast<int64_t>(out_features) * in_features))
                       ->getOutput(0);
    auto *matmul = network->addMatrixMultiply(input, nvinfer1::MatrixOperation::kNONE, *weight,
                                               nvinfer1::MatrixOperation::kTRANSPOSE);
    auto *output = matmul->getOutput(0);

    const auto bias_key = prefix + ".bias";
    if (hasWeight(weights_map, bias_key))
    {
        auto *bias = network
                         ->addConstant(nvinfer1::Dims3{1, 1, out_features},
                                       requireWeight(weights_map, bias_key, "Linear", out_features))
                         ->getOutput(0);
        output = network->addElementWise(*output, *bias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    }
    else if (bias_required)
    {
        requireWeight(weights_map, bias_key, "Linear", out_features);
    }
    return output;
}

// ============================================================================
// 注意力形状变换
// ============================================================================

/**
 * @brief 将 [N, L, D] 投影结果 reshape 为 [N, H, L, head_dim]。
 */
inline nvinfer1::ITensor *reshapeToHeads(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                          int batch, int num_tokens, int num_heads, int head_dim)
{
    auto *shuffle = network->addShuffle(input);
    shuffle->setReshapeDimensions(nvinfer1::Dims4{batch, num_tokens, num_heads, head_dim});
    shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    return shuffle->getOutput(0);
}

/**
 * @brief 合并多头注意力输出为 [B, N, C]。
 */
inline nvinfer1::ITensor *mergeHeads(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                      int batch, int num_tokens, int channels)
{
    auto *shuffle = network->addShuffle(input);
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    shuffle->setReshapeDimensions(nvinfer1::Dims3{batch, num_tokens, channels});
    return shuffle->getOutput(0);
}

/**
 * @brief 从合并的 qkv 张量中分离出 q、k、v。
 *
 * qkv 形状为 [B, N, 3*C]，沿最后一维等分切割。
 */
inline void splitQkv(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &qkv, int batch, int tokens,
                     int embed_dim, nvinfer1::ITensor *&q, nvinfer1::ITensor *&k, nvinfer1::ITensor *&v)
{
    auto dims = nvinfer1::Dims3{batch, tokens, embed_dim};
    auto stride = nvinfer1::Dims3{1, 1, 1};
    q = network->addSlice(qkv, nvinfer1::Dims3{0, 0, 0}, dims, stride)->getOutput(0);
    k = network->addSlice(qkv, nvinfer1::Dims3{0, 0, embed_dim}, dims, stride)->getOutput(0);
    v = network->addSlice(qkv, nvinfer1::Dims3{0, 0, 2 * embed_dim}, dims, stride)->getOutput(0);
}

} // namespace irt::model
