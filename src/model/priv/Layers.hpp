#pragma once

#include "Weights.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace irt::model {

// ============================================================================
// 常量
// ============================================================================

constexpr float kPi = 3.14159265358979323846F;

// ============================================================================
// 张量形状辅助函数
// ============================================================================

/**
 * @brief 构造任意 rank 的 TensorRT Dims。
 * @param values 每一维的静态长度。
 * @return TensorRT Dims。
 */
inline nvinfer1::Dims makeDimsFromValues(const std::vector<int64_t> &values)
{
    if (values.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "TensorRT dims rank is too large: %zu", values.size());
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(values.size());
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = values[static_cast<size_t>(i)];
    }
    return dims;
}

/**
 * @brief 构造任意 rank 的 TensorRT Dims。
 * @param values 每一维的静态长度。
 * @return TensorRT Dims。
 */
inline nvinfer1::Dims makeDimsFromValues(std::initializer_list<int64_t> values)
{
    return makeDimsFromValues(std::vector<int64_t>(values));
}

/**
 * @brief 构造一维 Dims，常用于 shape tensor 的切片和常量长度。
 */
inline nvinfer1::Dims makeDims1(int64_t value)
{
    return makeDimsFromValues({value});
}

/**
 * @brief 构造与输入 Dims rank 相同且每一维均为 0 的 Dims。
 */
inline nvinfer1::Dims zerosLike(const nvinfer1::Dims &dims)
{
    nvinfer1::Dims result{};
    result.nbDims = dims.nbDims;
    for (int32_t i = 0; i < result.nbDims; ++i)
    {
        result.d[i] = 0;
    }
    return result;
}

/**
 * @brief 构造与输入 Dims rank 相同且每一维均为 1 的 Dims。
 */
inline nvinfer1::Dims onesLike(const nvinfer1::Dims &dims)
{
    nvinfer1::Dims result{};
    result.nbDims = dims.nbDims;
    for (int32_t i = 0; i < result.nbDims; ++i)
    {
        result.d[i] = 1;
    }
    return result;
}

/**
 * @brief 计算静态 Dims 的元素数量。
 */
inline int64_t staticVolume(const nvinfer1::Dims &dims)
{
    int64_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "static tensor volume requires positive dims");
        }
        count *= dims.d[i];
    }
    return count;
}

/**
 * @brief 生成与输入张量同 rank 的标量广播维度。
 */
inline nvinfer1::Dims scalarDimsLike(const nvinfer1::ITensor &input)
{
    const auto     input_dims = input.getDimensions();
    nvinfer1::Dims dims{};
    dims.nbDims = input_dims.nbDims;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = 1;
    }
    return dims;
}

/**
 * @brief 取输入张量第 0 维，返回长度为 1 的 int64 shape tensor。
 * @param network TensorRT 网络定义。
 * @param input 提供动态 batch 的参考张量。
 * @return 形如 `[N]` 的一维 shape tensor。
 */
inline nvinfer1::ITensor *firstDimTensor(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shape = network->addShape(input);
    auto *slice = network->addSlice(*shape->getOutput(0), makeDims1(0), makeDims1(1), makeDims1(1));
    return slice->getOutput(0);
}

/**
 * @brief 拼出以参考张量第 0 维开头、其余维度为静态值的 shape tensor。
 * @param network TensorRT 网络定义。
 * @param like 第 0 维来源张量。
 * @param static_tail 除第 0 维之外的静态维度。
 * @return 一维 int64 shape tensor，可作为动态 slice/reshape 的 size 输入。
 */
inline nvinfer1::ITensor *shapeWithFirstDimOf(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &like,
                                              const std::vector<int64_t> &static_tail)
{
    auto *first_dim = firstDimTensor(network, like);
    if (static_tail.empty())
    {
        return first_dim;
    }

    auto                            *tail = network->addConstant(makeDims1(static_cast<int64_t>(static_tail.size())),
                                                                 ownedInt64Vector(std::vector<int64_t>(static_tail)));
    std::vector<nvinfer1::ITensor *> parts{first_dim, tail->getOutput(0)};
    auto *shape = network->addConcatenation(parts.data(), static_cast<int32_t>(parts.size()));
    shape->setAxis(0);
    return shape->getOutput(0);
}

/**
 * @brief 拼出以参考张量第 0 维开头、其余维度为静态值的 shape tensor。
 */
inline nvinfer1::ITensor *shapeWithFirstDimOf(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &like,
                                              std::initializer_list<int64_t> static_tail)
{
    return shapeWithFirstDimOf(network, like, std::vector<int64_t>(static_tail));
}

/**
 * @brief 从张量中切片，并让输出第 0 维跟随输入第 0 维。
 * @param network TensorRT 网络定义。
 * @param input 待切片张量。
 * @param start 静态起始位置，rank 必须等于 `1 + static_tail.size()`。
 * @param static_tail 除第 0 维之外的输出静态尺寸。
 * @return 动态 batch 安全的切片输出。
 */
inline nvinfer1::ITensor *slicePreserveFirstDim(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                                const nvinfer1::Dims &start, std::initializer_list<int64_t> static_tail)
{
    std::vector<int64_t> size_values{1};
    size_values.insert(size_values.end(), static_tail.begin(), static_tail.end());
    const auto size   = makeDimsFromValues(size_values);
    auto      *slice  = network->addSlice(input, start, size, onesLike(size));
    auto      *dyn_sz = shapeWithFirstDimOf(network, input, static_tail);
    slice->setInput(2, *dyn_sz);
    return slice->getOutput(0);
}

/**
 * @brief 将第 0 维为 1 的常量广播到参考张量的第 0 维。
 * @param network TensorRT 网络定义。
 * @param constant 待广播常量，要求第 0 维为 1。
 * @param like 提供目标第 0 维的参考张量。
 * @return 第 0 维与 `like` 一致、其余维度与 `constant` 一致的张量。
 *
 * TensorRT Concatenation 不会按 batch 维自动广播常量；该 helper 先构造一个
 * `[N, ...]` 的零张量，再通过 ElementWise 广播常量，供 cls token、prompt token 等场景复用。
 */
inline nvinfer1::ITensor *broadcastFirstDimLike(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &constant,
                                                nvinfer1::ITensor &like)
{
    const auto dims = constant.getDimensions();
    if (dims.nbDims <= 0 || dims.d[0] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "broadcastFirstDimLike expects a tensor with leading dimension 1");
    }

    std::vector<int64_t> static_tail;
    static_tail.reserve(static_cast<size_t>(dims.nbDims - 1));
    for (int32_t i = 1; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "broadcastFirstDimLike expects static non-batch dimensions");
        }
        static_tail.push_back(dims.d[i]);
    }

    auto *zero_seed = network->addConstant(dims, ownedFloatVector(std::vector<float>(staticVolume(dims), 0.0F)));
    auto *zero_view = network->addSlice(*zero_seed->getOutput(0), zerosLike(dims), dims, onesLike(dims));
    zero_view->setMode(nvinfer1::SampleMode::kFILL);
    auto *fill = network->addConstant(scalarDimsLike(constant), ownedScalarWeight(0.0F));
    zero_view->setInput(2, *shapeWithFirstDimOf(network, like, static_tail));
    zero_view->setInput(4, *fill->getOutput(0));
    return network->addElementWise(*zero_view->getOutput(0), constant, nvinfer1::ElementWiseOperation::kSUM)
        ->getOutput(0);
}

// ============================================================================
// 激活函数
// ============================================================================

/**
 * @brief 保留 batch 维并将其余维度展平为二维张量。
 * @param network TensorRT 网络定义。
 * @param input 待展平的输入张量，约定第 0 维为 batch。
 * @return 形状为 `[N, -1]` 的张量。
 *
 * TensorRT shuffle 中的 0 表示复制输入对应维度，因此静态 batch 与动态 batch 均可复用。
 */
inline nvinfer1::ITensor *flattenPreserveBatch(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shuffle = network->addShuffle(input);
    shuffle->setReshapeDimensions(nvinfer1::Dims2{0, -1});
    return shuffle->getOutput(0);
}

/**
 * @brief 添加精确 GeLU 激活：0.5*x*(1+erf(x/sqrt(2)))。
 */
inline nvinfer1::ITensor *addGeluExact(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto  dims       = scalarDimsLike(input);
    auto *half       = network->addConstant(dims, ownedScalarWeight(0.5F))->getOutput(0);
    auto *one        = network->addConstant(dims, ownedScalarWeight(1.0F))->getOutput(0);
    auto *inv_sqrt_2 = network->addConstant(dims, ownedScalarWeight(0.70710678118654752440F))->getOutput(0);

    auto *scaled   = network->addElementWise(input, *inv_sqrt_2, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    auto *erf      = network->addUnary(*scaled, nvinfer1::UnaryOperation::kERF)->getOutput(0);
    auto *one_plus = network->addElementWise(*erf, *one, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    auto *half_x   = network->addElementWise(input, *half, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
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
    const auto bias_weights  = requireWeight(weights_map, prefix + ".bias", "LayerNorm", embed_dim);
    auto      *scale         = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, scale_weights)->getOutput(0);
    auto      *bias          = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, bias_weights)->getOutput(0);

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
                                     requireWeight(weights_map, prefix + ".weight", "Linear",
                                                   static_cast<int64_t>(out_features) * in_features))
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
inline nvinfer1::ITensor *reshapeToHeads(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                         int num_tokens, int num_heads, int head_dim)
{
    (void)batch;
    auto *shuffle = network->addShuffle(input);
    shuffle->setReshapeDimensions(nvinfer1::Dims4{0, num_tokens, num_heads, head_dim});
    shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    return shuffle->getOutput(0);
}

/**
 * @brief 合并多头注意力输出为 [B, N, C]。
 */
inline nvinfer1::ITensor *mergeHeads(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                     int num_tokens, int channels)
{
    (void)batch;
    auto *shuffle = network->addShuffle(input);
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    shuffle->setReshapeDimensions(nvinfer1::Dims3{0, num_tokens, channels});
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
    (void)batch;
    q = slicePreserveFirstDim(network, qkv, nvinfer1::Dims3{0, 0, 0}, {tokens, embed_dim});
    k = slicePreserveFirstDim(network, qkv, nvinfer1::Dims3{0, 0, embed_dim}, {tokens, embed_dim});
    v = slicePreserveFirstDim(network, qkv, nvinfer1::Dims3{0, 0, 2 * embed_dim}, {tokens, embed_dim});
}

} // namespace irt::model
