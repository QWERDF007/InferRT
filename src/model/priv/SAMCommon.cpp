#include "SAMCommon.hpp"

namespace irt::model {

nvinfer1::Dims makeDims(std::initializer_list<int32_t> values)
{
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(values.size());
    int32_t i   = 0;
    for (const auto value : values)
    {
        dims.d[i++] = value;
    }
    return dims;
}

int64_t volume(const nvinfer1::Dims &dims)
{
    size_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        count = irt::checkedSizeMul(count, irt::checkedInt64ToSize(dims.d[i], "TensorRT dimensions"),
                                    "TensorRT volume");
    }
    return irt::checkedSizeToInt64(count, "TensorRT volume");
}

std::string resolveLinearPrefix(const WeightsMap &weights_map, std::initializer_list<std::string> candidates)
{
    for (const auto &prefix : candidates)
    {
        if (hasWeight(weights_map, prefix + ".weight"))
        {
            return prefix;
        }
    }
    return candidates.size() > 0 ? *candidates.begin() : std::string{};
}

nvinfer1::ITensor *addScalar(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &like, float value)
{
    auto *constant = requireLayer(network->addConstant(scalarDimsLike(like), ownedScalarWeight(value)),
                                  "Failed to add SAM scalar constant");
    return constant->getOutput(0);
}

nvinfer1::ITensor *addLayerNormLastDim(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto dims               = scalarDimsLike(input);
    dims.d[dims.nbDims - 1] = channels;
    auto *scale     = requireLayer(network->addConstant(dims, requireWeight(weights_map, prefix + ".weight", channels)),
                                   "Failed to add SAM LayerNorm scale");
    auto *bias      = requireLayer(network->addConstant(dims, requireWeight(weights_map, prefix + ".bias", channels)),
                                   "Failed to add SAM LayerNorm bias");
    const auto axes = 1U << static_cast<uint32_t>(input.getDimensions().nbDims - 1);
#if TRT_VERSION >= 11500
    auto *norm = requireLayer(network->addNormalizationV2(input, *scale->getOutput(0), *bias->getOutput(0), axes),
                              "Failed to add SAM LayerNorm");
#else
    auto *norm = requireLayer(network->addNormalization(input, *scale->getOutput(0), *bias->getOutput(0), axes),
                              "Failed to add SAM LayerNorm");
#endif
    norm->setEpsilon(kLayerNormEps);
    return norm->getOutput(0);
}

nvinfer1::ITensor *addLayerNorm2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *mean     = requireLayer(network->addReduce(input, nvinfer1::ReduceOperation::kAVG, 1U << 1, true),
                                  "Failed to add SAM LayerNorm2d mean");
    auto *centered = requireLayer(network->addElementWise(input, *mean->getOutput(0), E::kSUB),
                                  "Failed to add SAM LayerNorm2d center");
    auto *square   = requireLayer(network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), E::kPROD),
                                  "Failed to add SAM LayerNorm2d square");
    auto *var = requireLayer(network->addReduce(*square->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true),
                             "Failed to add SAM LayerNorm2d variance");
    auto *eps = addScalar(network, *var->getOutput(0), kLayerNormEps);
    auto *var_eps
        = requireLayer(network->addElementWise(*var->getOutput(0), *eps, E::kSUM), "Failed to add SAM LayerNorm2d eps");
    auto *std = requireLayer(network->addUnary(*var_eps->getOutput(0), U::kSQRT), "Failed to add SAM LayerNorm2d sqrt");
    auto *normalized = requireLayer(network->addElementWise(*centered->getOutput(0), *std->getOutput(0), E::kDIV),
                                    "Failed to add SAM LayerNorm2d div");

    auto *scale  = requireLayer(network->addConstant(nvinfer1::Dims4{1, channels, 1, 1},
                                                     requireWeight(weights_map, prefix + ".weight", channels)),
                                "Failed to add SAM LayerNorm2d scale");
    auto *bias   = requireLayer(network->addConstant(nvinfer1::Dims4{1, channels, 1, 1},
                                                     requireWeight(weights_map, prefix + ".bias", channels)),
                                "Failed to add SAM LayerNorm2d bias");
    auto *scaled = requireLayer(network->addElementWise(*normalized->getOutput(0), *scale->getOutput(0), E::kPROD),
                                "Failed to add SAM LayerNorm2d scale product");
    return requireLayer(network->addElementWise(*scaled->getOutput(0), *bias->getOutput(0), E::kSUM),
                        "Failed to add SAM LayerNorm2d bias add")
        ->getOutput(0);
}

nvinfer1::Weights makeFusedConvBNWeight(const WeightsMap &weights_map, const std::string &prefix, int64_t conv_count,
                                        int out_channels)
{
    const auto &conv  = requireWeight(weights_map, prefix + ".c.weight", conv_count);
    const auto &gamma = requireWeight(weights_map, prefix + ".bn.weight", out_channels);
    const auto &var   = requireWeight(weights_map, prefix + ".bn.running_var", out_channels);

    const auto *conv_values  = static_cast<const float *>(conv.values);
    const auto *gamma_values = static_cast<const float *>(gamma.values);
    const auto *var_values   = static_cast<const float *>(var.values);
    const auto  per_output   = static_cast<int64_t>(conv_count / out_channels);

    std::vector<float> fused(static_cast<size_t>(conv_count));
    for (int oc = 0; oc < out_channels; ++oc)
    {
        const float scale = gamma_values[oc] / std::sqrt(var_values[oc] + kBatchNormEps);
    const auto  base  = checkedWeightProduct({oc, per_output}, "Fused ConvBN bias offset");
        for (int64_t i = 0; i < per_output; ++i)
        {
            fused[static_cast<size_t>(base + i)] = conv_values[base + i] * scale;
        }
    }
    return ownedFloatVector(std::move(fused));
}

nvinfer1::Weights makeFusedConvBNBias(const WeightsMap &weights_map, const std::string &prefix, int out_channels)
{
    const auto &gamma = requireWeight(weights_map, prefix + ".bn.weight", out_channels);
    const auto &beta  = requireWeight(weights_map, prefix + ".bn.bias", out_channels);
    const auto &mean  = requireWeight(weights_map, prefix + ".bn.running_mean", out_channels);
    const auto &var   = requireWeight(weights_map, prefix + ".bn.running_var", out_channels);

    const auto *gamma_values = static_cast<const float *>(gamma.values);
    const auto *beta_values  = static_cast<const float *>(beta.values);
    const auto *mean_values  = static_cast<const float *>(mean.values);
    const auto *var_values   = static_cast<const float *>(var.values);

    std::vector<float> fused(static_cast<size_t>(out_channels));
    for (int oc = 0; oc < out_channels; ++oc)
    {
        const float scale              = gamma_values[oc] / std::sqrt(var_values[oc] + kBatchNormEps);
        fused[static_cast<size_t>(oc)] = beta_values[oc] - mean_values[oc] * scale;
    }
    return ownedFloatVector(std::move(fused));
}

nvinfer1::ITensor *addEdgeSAMConvBN(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    nvinfer1::ITensor &input, const std::string &prefix, int in_channels,
                                    int out_channels, int kernel_size, int stride, int padding, int groups)
{
    const auto per_group_in = in_channels / groups;
    const auto conv_count   = checkedWeightProduct({out_channels, per_group_in, kernel_size, kernel_size},
                                                   "Fused ConvBN convolution weight");
    auto      *conv
        = requireLayer(network->addConvolutionNd(input, out_channels, nvinfer1::DimsHW{kernel_size, kernel_size},
                                                 makeFusedConvBNWeight(weights_map, prefix, conv_count, out_channels),
                                                 makeFusedConvBNBias(weights_map, prefix, out_channels)),
                       "Failed to add EdgeSAM fused Conv2d_BN");
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    conv->setNbGroups(groups);
    return conv->getOutput(0);
}

nvinfer1::ITensor *addEdgeSAMConvNoBias(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const std::string &key, int in_channels,
                                        int out_channels, int kernel_size, int stride, int padding)
{
    auto *conv = requireLayer(network->addConvolutionNd(input, out_channels, nvinfer1::DimsHW{kernel_size, kernel_size},
                                                         requireWeight(weights_map, key,
                                                                       checkedWeightProduct({out_channels, in_channels,
                                                                                             kernel_size, kernel_size},
                                                                                            "Fused ConvBN weight")),
                                                         emptyWeights()),
                               "Failed to add EdgeSAM bias-free convolution");
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    return conv->getOutput(0);
}

nvinfer1::ITensor *flattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                               int width, int channels)
{
    (void)batch;
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC flatten");
    shuffle->setReshapeDimensions(nvinfer1::Dims3{0, height * width, channels});
    return shuffle->getOutput(0);
}

nvinfer1::ITensor *unflattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                                 int width, int channels)
{
    (void)batch;
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC unflatten");
    shuffle->setReshapeDimensions(nvinfer1::Dims4{0, height, width, channels});
    return shuffle->getOutput(0);
}

nvinfer1::ITensor *nchwToNhwc(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NCHW->NHWC");
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    return shuffle->getOutput(0);
}

nvinfer1::ITensor *nhwcToNchw(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC->NCHW");
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    return shuffle->getOutput(0);
}

nvinfer1::ITensor *addTokenAttention(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &q, nvinfer1::ITensor &k,
                                     nvinfer1::ITensor &v, int batch, int q_tokens, int k_tokens, int num_heads,
                                     int internal_dim)
{
    const int head_dim = internal_dim / num_heads;
    auto     *q_heads  = reshapeToHeads(network, q, batch, q_tokens, num_heads, head_dim);
    auto     *k_heads  = reshapeToHeads(network, k, batch, k_tokens, num_heads, head_dim);
    auto     *v_heads  = reshapeToHeads(network, v, batch, k_tokens, num_heads, head_dim);

    auto *qk      = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                                 "Failed to add SAM attention qk");
    auto *scale   = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto *scaled  = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD),
                                 "Failed to add SAM attention scale");
    auto *softmax = requireLayer(network->addSoftMax(*scaled->getOutput(0)), "Failed to add SAM attention softmax");
    softmax->setAxes(1U << 3);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed to add SAM attention value matmul");
    return mergeHeads(network, *attended->getOutput(0), batch, q_tokens, internal_dim);
}

} // namespace irt::model
