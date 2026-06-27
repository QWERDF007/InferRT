#include "RFDETR.hpp"

#include "Layers.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace irt::model {
namespace {

using E = nvinfer1::ElementWiseOperation;
using M = nvinfer1::MatrixOperation;

constexpr int   kDefaultImageSize       = 224;
constexpr int   kRFDETRDefaultClasses   = 90;
constexpr int   kRFDETREncoderLayers    = 12;
constexpr int   kRFDETRProjectorBlocks  = 3;
constexpr int   kRFDETRGroupDETRInfer   = 1;
constexpr int   kRFDETRMaskDownsample   = 4;
constexpr float kLayerNormEps           = 1.0e-6F;
constexpr float kProposalBaseBoxScale   = 0.05F;
constexpr float kRFDETRSinePositionTemp = 10000.0F;
constexpr float kRFDETRTwoPi            = 6.28318530717958647692F;

/**
 * @brief 输入几何信息，避免在建图阶段重复解析 shape。
 */
struct RFDETRGeometry
{
    int batch{1};        ///< 构建时配置的 opt batch；动态 batch 时 TensorRT 图内用 -1。
    int channels{3};     ///< 输入通道数。
    int height{0};       ///< 输入高度。
    int width{0};        ///< 输入宽度。
    int patch_grid_h{0}; ///< DINO patch 网格高。
    int patch_grid_w{0}; ///< DINO patch 网格宽。
    int patches{0};      ///< patch token 数。
};

/**
 * @brief 多尺度 projector 输出张量与静态空间尺寸。
 */
struct FeatureLevel
{
    nvinfer1::ITensor *tensor{nullptr}; ///< NCHW feature map。
    int                height{0};       ///< feature 高度。
    int                width{0};        ///< feature 宽度。
};

/**
 * @brief 对 TensorRT 建图 API 返回值做统一空指针检查。
 */
template<typename Layer>
Layer *requireLayer(Layer *layer, const char *message)
{
    if (layer == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "%s", message);
    }
    return layer;
}

/**
 * @brief 根据 rank 构造最后一维为 `channels`、其他维为 1 的广播权重维度。
 */
nvinfer1::Dims lastDimWeightDims(int rank, int channels)
{
    if (rank <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR last-dim weight expects a valid tensor rank");
    }
    std::vector<int64_t> values(static_cast<size_t>(rank), 1);
    values.back() = channels;
    return makeDimsFromValues(values);
}

/**
 * @brief 根据 rank 构造标量广播维度。
 */
nvinfer1::Dims scalarDims(int rank)
{
    if (rank <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR scalar expects a valid tensor rank");
    }
    return makeDimsFromValues(std::vector<int64_t>(static_cast<size_t>(rank), 1));
}

/**
 * @brief 创建与输入 rank 对齐的 float 标量常量。
 */
nvinfer1::ITensor *addScalar(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &like, float value)
{
    return requireLayer(network->addConstant(scalarDims(like.getDimensions().nbDims), ownedScalarWeight(value)),
                        "Failed to add RF-DETR scalar")
        ->getOutput(0);
}

/**
 * @brief 在最后一维上做 LayerNorm。
 */
nvinfer1::ITensor *addLayerNormLastDim(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    const auto dims = input.getDimensions();
    if (dims.nbDims <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR LayerNorm input rank is invalid for %s",
                             prefix.c_str());
    }
    auto *scale_const = requireLayer(
        network->addConstant(lastDimWeightDims(dims.nbDims, channels),
                             requireWeight(weights_map, prefix + ".weight", "RF-DETR LayerNorm", channels)),
        "Failed to add RF-DETR LayerNorm scale");
    auto *bias_const = requireLayer(
        network->addConstant(lastDimWeightDims(dims.nbDims, channels),
                             requireWeight(weights_map, prefix + ".bias", "RF-DETR LayerNorm", channels)),
        "Failed to add RF-DETR LayerNorm bias");

    const auto axes = 1U << static_cast<uint32_t>(dims.nbDims - 1);
#if TRT_VERSION >= 11500
    auto *norm
        = requireLayer(network->addNormalizationV2(input, *scale_const->getOutput(0), *bias_const->getOutput(0), axes),
                       "Failed to add RF-DETR LayerNorm");
#else
    auto *norm
        = requireLayer(network->addNormalization(input, *scale_const->getOutput(0), *bias_const->getOutput(0), axes),
                       "Failed to add RF-DETR LayerNorm");
#endif
    norm->setEpsilon(kLayerNormEps);
    return norm->getOutput(0);
}

/**
 * @brief 对 NCHW feature map 的 channel 维执行官方 LayerNorm2d。
 */
nvinfer1::ITensor *addLayerNorm2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *to_nhwc = requireLayer(network->addShuffle(input), "Failed to add RF-DETR NCHW->NHWC");
    to_nhwc->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    auto *norm    = addLayerNormLastDim(network, weights_map, *to_nhwc->getOutput(0), prefix, channels);
    auto *to_nchw = requireLayer(network->addShuffle(*norm), "Failed to add RF-DETR NHWC->NCHW");
    to_nchw->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    return to_nchw->getOutput(0);
}

/**
 * @brief 添加带自定义权重 key 的 [N,L,C] 线性层。
 */
nvinfer1::ITensor *addLinear3DFromKeys(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &weight_key,
                                       const std::string &bias_key, int in_features, int out_features,
                                       bool bias_required = true)
{
    auto *weight = requireLayer(network->addConstant(nvinfer1::Dims3{1, out_features, in_features},
                                                     requireWeight(weights_map, weight_key, "RF-DETR Linear",
                                                                   static_cast<int64_t>(out_features) * in_features)),
                                "Failed to add RF-DETR linear weight")
                       ->getOutput(0);
    auto *matmul = requireLayer(network->addMatrixMultiply(input, M::kNONE, *weight, M::kTRANSPOSE),
                                "Failed to add RF-DETR linear matmul");
    auto *output = matmul->getOutput(0);
    if (hasWeight(weights_map, bias_key))
    {
        auto *bias
            = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, out_features},
                                                requireWeight(weights_map, bias_key, "RF-DETR Linear", out_features)),
                           "Failed to add RF-DETR linear bias")
                  ->getOutput(0);
        output = requireLayer(network->addElementWise(*output, *bias, E::kSUM), "Failed to add RF-DETR linear bias add")
                     ->getOutput(0);
    }
    else if (bias_required)
    {
        requireWeight(weights_map, bias_key, "RF-DETR Linear", out_features);
    }
    return output;
}

/**
 * @brief 对任意 rank 张量的最后一维执行 Linear，用于 NHWC segmentation block。
 */
nvinfer1::ITensor *addLinearLastDim(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    nvinfer1::ITensor &input, const std::string &prefix, int in_features,
                                    int out_features, bool bias_required = true)
{
    const auto dims = input.getDimensions();
    if (dims.nbDims < 2)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR Linear expects rank >= 2 for %s", prefix.c_str());
    }

    std::vector<int64_t> weight_dims(static_cast<size_t>(dims.nbDims), 1);
    weight_dims[static_cast<size_t>(dims.nbDims - 2)] = out_features;
    weight_dims[static_cast<size_t>(dims.nbDims - 1)] = in_features;
    auto *weight = requireLayer(network->addConstant(makeDimsFromValues(weight_dims),
                                                     requireWeight(weights_map, prefix + ".weight", "RF-DETR Linear",
                                                                   static_cast<int64_t>(out_features) * in_features)),
                                "Failed to add RF-DETR linear-last-dim weight")
                       ->getOutput(0);
    auto *matmul = requireLayer(network->addMatrixMultiply(input, M::kNONE, *weight, M::kTRANSPOSE),
                                "Failed to add RF-DETR linear-last-dim matmul");
    auto *output = matmul->getOutput(0);

    const auto bias_key = prefix + ".bias";
    if (hasWeight(weights_map, bias_key))
    {
        auto bias_dims               = scalarDims(dims.nbDims);
        bias_dims.d[dims.nbDims - 1] = out_features;
        auto *bias                   = requireLayer(network->addConstant(
                                      bias_dims, requireWeight(weights_map, bias_key, "RF-DETR Linear", out_features)),
                                                    "Failed to add RF-DETR linear-last-dim bias")
                         ->getOutput(0);
        output = requireLayer(network->addElementWise(*output, *bias, E::kSUM),
                              "Failed to add RF-DETR linear-last-dim bias add")
                     ->getOutput(0);
    }
    else if (bias_required)
    {
        requireWeight(weights_map, bias_key, "RF-DETR Linear", out_features);
    }
    return output;
}

/**
 * @brief 添加 [N,L,C] MLP，层名采用官方 `*.layers.<idx>` 命名。
 */
/**
 * @brief Slice a contiguous float weight block for packed PyTorch projections.
 */
nvinfer1::Weights sliceFloatWeights(const nvinfer1::Weights &weights, int64_t offset, int64_t count, const char *label)
{
    if (weights.type != nvinfer1::DataType::kFLOAT || weights.values == nullptr || offset < 0 || count < 0
        || offset + count > weights.count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid RF-DETR packed projection weight slice for %s",
                             label);
    }

    const auto *values = static_cast<const float *>(weights.values);
    return ownedFloatVector(std::vector<float>(values + offset, values + offset + count));
}

/**
 * @brief Add a [N,L,C] linear layer from already sliced weight blocks.
 */
nvinfer1::ITensor *addLinear3DFromWeights(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                          const nvinfer1::Weights &weight, const nvinfer1::Weights *bias,
                                          int in_features, int out_features)
{
    auto *weight_const = requireLayer(network->addConstant(nvinfer1::Dims3{1, out_features, in_features}, weight),
                                      "Failed to add RF-DETR sliced linear weight")
                             ->getOutput(0);
    auto *matmul = requireLayer(network->addMatrixMultiply(input, M::kNONE, *weight_const, M::kTRANSPOSE),
                                "Failed to add RF-DETR sliced linear matmul");
    auto *output = matmul->getOutput(0);
    if (bias != nullptr)
    {
        auto *bias_const = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, out_features}, *bias),
                                        "Failed to add RF-DETR sliced linear bias")
                               ->getOutput(0);
        output = requireLayer(network->addElementWise(*output, *bias_const, E::kSUM),
                              "Failed to add RF-DETR sliced linear bias add")
                     ->getOutput(0);
    }
    return output;
}

nvinfer1::ITensor *addMLP(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                          nvinfer1::ITensor &input, const std::string &prefix, const std::vector<int> &channels,
                          bool final_relu = false)
{
    nvinfer1::ITensor *x = &input;
    for (size_t i = 0; i + 1 < channels.size(); ++i)
    {
        const auto layer_prefix = prefix + ".layers." + std::to_string(i);
        x                       = addLinear3D(network, weights_map, *x, layer_prefix, channels[i], channels[i + 1]);
        if (i + 2 < channels.size() || final_relu)
        {
            x = requireLayer(network->addActivation(*x, nvinfer1::ActivationType::kRELU),
                             "Failed to add RF-DETR MLP ReLU")
                    ->getOutput(0);
        }
    }
    return x;
}

/**
 * @brief 添加二维卷积，默认使用无 bias 的 ConvX 约定。
 */
nvinfer1::ITensor *addConv2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                             nvinfer1::ITensor &input, const std::string &prefix, int in_channels, int out_channels,
                             int kernel, int stride, int padding, int groups = 1, bool bias = false)
{
    auto *conv = requireLayer(
        network->addConvolutionNd(
            input, out_channels, nvinfer1::DimsHW{kernel, kernel},
            requireWeight(weights_map, prefix + ".weight", "RF-DETR Conv2d",
                          static_cast<int64_t>(out_channels) * (in_channels / groups) * kernel * kernel),
            bias ? requireWeight(weights_map, prefix + ".bias", "RF-DETR Conv2d", out_channels) : emptyWeights()),
        "Failed to add RF-DETR Conv2d");
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    conv->setNbGroups(groups);
    return conv->getOutput(0);
}

/**
 * @brief 添加 ConvTranspose2d，匹配 projector 的上采样分支。
 */
nvinfer1::ITensor *addDeconv2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                               nvinfer1::ITensor &input, const std::string &prefix, int in_channels, int out_channels,
                               int kernel, int stride)
{
    auto *deconv
        = requireLayer(network->addDeconvolutionNd(
                           input, out_channels, nvinfer1::DimsHW{kernel, kernel},
                           requireWeight(weights_map, prefix + ".weight", "RF-DETR ConvTranspose2d",
                                         static_cast<int64_t>(in_channels) * out_channels * kernel * kernel),
                           requireWeight(weights_map, prefix + ".bias", "RF-DETR ConvTranspose2d", out_channels)),
                       "Failed to add RF-DETR ConvTranspose2d");
    deconv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    return deconv->getOutput(0);
}

/**
 * @brief 添加 projector 中的 ConvX：Conv2d + LN/BN(此处为 LN) + activation。
 */
nvinfer1::ITensor *addConvX(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                            nvinfer1::ITensor &input, const std::string &prefix, int in_channels, int out_channels,
                            int kernel, int stride, const char *activation = "silu")
{
    auto *conv = addConv2d(network, weights_map, input, prefix + ".conv", in_channels, out_channels, kernel, stride,
                           kernel / 2);
    auto *norm = addLayerNorm2d(network, weights_map, *conv, prefix + ".bn", out_channels);
    if (std::string(activation) == "relu")
    {
        return requireLayer(network->addActivation(*norm, nvinfer1::ActivationType::kRELU),
                            "Failed to add RF-DETR ConvX ReLU")
            ->getOutput(0);
    }
    return addSilu(network, *norm);
}

/**
 * @brief 添加 C2f 内部 bottleneck。
 */
nvinfer1::ITensor *addProjectorBottleneck(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                          nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *cv1 = addConvX(network, weights_map, input, prefix + ".cv1", channels, channels, 3, 1, "silu");
    return addConvX(network, weights_map, *cv1, prefix + ".cv2", channels, channels, 3, 1, "silu");
}

/**
 * @brief 添加 projector 的 C2f 模块。
 */
nvinfer1::ITensor *addProjectorC2f(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &input, const std::string &prefix, int in_channels,
                                   int out_channels, const std::string &feature_prefix,
                                   priv::IModelImpl::NamedTensorMap &named_tensors)
{
    const int hidden = out_channels / 2;
    auto     *cv1    = addConvX(network, weights_map, input, prefix + ".cv1", in_channels, 2 * hidden, 1, 1, "silu");
    named_tensors[feature_prefix + ".cv1"] = cv1;

    auto *left                               = slicePreserveFirstDim(network, *cv1, nvinfer1::Dims4{0, 0, 0, 0},
                                                                     {hidden, cv1->getDimensions().d[2], cv1->getDimensions().d[3]});
    auto *right                              = slicePreserveFirstDim(network, *cv1, nvinfer1::Dims4{0, hidden, 0, 0},
                                                                     {hidden, cv1->getDimensions().d[2], cv1->getDimensions().d[3]});
    named_tensors[feature_prefix + ".left"]  = left;
    named_tensors[feature_prefix + ".right"] = right;

    std::vector<nvinfer1::ITensor *> branches{left, right};
    auto                            *tail = right;
    for (int i = 0; i < kRFDETRProjectorBlocks; ++i)
    {
        tail = addProjectorBottleneck(network, weights_map, *tail, prefix + ".m." + std::to_string(i), hidden);
        branches.push_back(tail);
        named_tensors[feature_prefix + ".m" + std::to_string(i)] = tail;
    }

    auto *cat = requireLayer(network->addConcatenation(branches.data(), static_cast<int32_t>(branches.size())),
                             "Failed to add RF-DETR C2f concat");
    cat->setAxis(1);
    named_tensors[feature_prefix + ".cat"] = cat->getOutput(0);
    auto *cv2                              = addConvX(network, weights_map, *cat->getOutput(0), prefix + ".cv2",
                                                      static_cast<int>(branches.size()) * hidden, out_channels, 1, 1, "silu");
    named_tensors[feature_prefix + ".cv2"] = cv2;
    return cv2;
}

/**
 * @brief 判断默认 ImageNet 输入是否需要被 RF-DETR 变体尺寸覆盖。
 */
bool usesDefaultImageNetInputShape(const IModelConfig &config)
{
    if (config.inputShapes().size() != 1)
    {
        return false;
    }
    const auto &shape = config.inputShape();
    return shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageSize && shape.d[3] == kDefaultImageSize;
}

/**
 * @brief 校验 RF-DETR 输入/输出配置并解析静态几何。
 */
RFDETRGeometry resolveGeometry(const RFDETRSpec &spec, const IModelConfig &config)
{
    if (config.inputShapes().size() != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR expects exactly one input tensor");
    }
    if (!config.featureOnly() && config.outputTensorNames().size() != (spec.segmentation ? 3U : 2U))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s expects %zu output tensor names", spec.display_name,
                             spec.segmentation ? 3U : 2U);
    }

    const auto    &shape = config.inputShape();
    RFDETRGeometry geometry{};
    geometry.batch    = static_cast<int>(shape.d[0]);
    geometry.channels = static_cast<int>(shape.d[1]);
    geometry.height   = static_cast<int>(shape.d[2]);
    geometry.width    = static_cast<int>(shape.d[3]);

    const int divisor = spec.patch_size * spec.num_windows;
    if (geometry.channels != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR requires RGB input, got C=%d", geometry.channels);
    }
    if (geometry.height <= 0 || geometry.width <= 0 || geometry.height % divisor != 0 || geometry.width % divisor != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "RF-DETR input H/W must be divisible by patch_size*num_windows=%d, got H=%d W=%d", divisor,
                             geometry.height, geometry.width);
    }
    if (spec.encoder_dim % spec.encoder_heads != 0 || spec.hidden_dim % spec.self_attention_heads != 0
        || spec.hidden_dim % spec.cross_attention_heads != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "RF-DETR attention dimensions are inconsistent");
    }

    geometry.patch_grid_h = geometry.height / spec.patch_size;
    geometry.patch_grid_w = geometry.width / spec.patch_size;
    geometry.patches      = geometry.patch_grid_h * geometry.patch_grid_w;
    return geometry;
}

/**
 * @brief 按 RF-DETR 命名空间拼接权重前缀。
 */
std::string encoderPrefix()
{
    return "backbone.0.encoder.encoder";
}

/**
 * @brief 添加 DINOv2 patch embedding 与 windowed token reshape。
 */
nvinfer1::ITensor *addDINOEmbedding(const RFDETRModel &impl, nvinfer1::INetworkDefinition *network,
                                    const WeightsMap &weights_map, const RFDETRSpec &spec,
                                    const RFDETRGeometry &geometry, priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *input            = impl.addInputTensor(network);
    named_tensors["input"] = input;

    const auto patch_prefix = encoderPrefix() + ".embeddings.patch_embeddings.projection";
    auto      *patch        = requireLayer(
        network->addConvolutionNd(
            *input, spec.encoder_dim, nvinfer1::DimsHW{spec.patch_size, spec.patch_size},
            requireWeight(
                weights_map, patch_prefix + ".weight", "RF-DETR patch embedding",
                static_cast<int64_t>(spec.encoder_dim) * geometry.channels * spec.patch_size * spec.patch_size),
            requireWeight(weights_map, patch_prefix + ".bias", "RF-DETR patch embedding", spec.encoder_dim)),
        "Failed to add RF-DETR patch embedding");
    patch->setStrideNd(nvinfer1::DimsHW{spec.patch_size, spec.patch_size});

    auto *patch_tokens = requireLayer(network->addShuffle(*patch->getOutput(0)), "Failed to flatten RF-DETR patches");
    patch_tokens->setReshapeDimensions(nvinfer1::Dims3{0, spec.encoder_dim, geometry.patches});
    patch_tokens->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    named_tensors["patch_embed"] = patch_tokens->getOutput(0);

    const auto &pos_weight
        = requireWeight(weights_map, encoderPrefix() + ".embeddings.position_embeddings", "RF-DETR position embedding",
                        static_cast<int64_t>(geometry.patches + 1) * spec.encoder_dim);
    auto *pos_table
        = requireLayer(network->addConstant(nvinfer1::Dims3{1, geometry.patches + 1, spec.encoder_dim}, pos_weight),
                       "Failed to add RF-DETR position table")
              ->getOutput(0);
    auto *cls_pos = requireLayer(network->addSlice(*pos_table, nvinfer1::Dims3{0, 0, 0},
                                                   nvinfer1::Dims3{1, 1, spec.encoder_dim}, nvinfer1::Dims3{1, 1, 1}),
                                 "Failed to slice RF-DETR cls position")
                        ->getOutput(0);
    auto *patch_pos = requireLayer(network->addSlice(*pos_table, nvinfer1::Dims3{0, 1, 0},
                                                     nvinfer1::Dims3{1, geometry.patches, spec.encoder_dim},
                                                     nvinfer1::Dims3{1, 1, 1}),
                                   "Failed to slice RF-DETR patch position")
                          ->getOutput(0);

    auto *patch_with_pos = requireLayer(network->addElementWise(*patch_tokens->getOutput(0), *patch_pos, E::kSUM),
                                        "Failed to add RF-DETR patch position")
                               ->getOutput(0);
    auto *cls_token
        = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, spec.encoder_dim},
                                            requireWeight(weights_map, encoderPrefix() + ".embeddings.cls_token",
                                                          "RF-DETR cls token", spec.encoder_dim)),
                       "Failed to add RF-DETR cls token")
              ->getOutput(0);
    auto *cls_with_pos
        = requireLayer(network->addElementWise(*cls_token, *cls_pos, E::kSUM), "Failed to add RF-DETR cls position")
              ->getOutput(0);

    if (spec.num_windows <= 1)
    {
        cls_with_pos = broadcastFirstDimLike(network, *cls_with_pos, *patch_with_pos);
        std::array<nvinfer1::ITensor *, 2> tokens{cls_with_pos, patch_with_pos};
        auto *concat = requireLayer(network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size())),
                                    "Failed to concat RF-DETR DINO tokens");
        concat->setAxis(1);
        named_tensors["tokens"] = concat->getOutput(0);
        return concat->getOutput(0);
    }

    const int h_win      = geometry.patch_grid_h / spec.num_windows;
    const int w_win      = geometry.patch_grid_w / spec.num_windows;
    auto     *pixel_view = requireLayer(network->addShuffle(*patch_with_pos), "Failed to view RF-DETR window pixels");
    pixel_view->setReshapeDimensions(
        nvinfer1::Dims4{0, geometry.patch_grid_h, geometry.patch_grid_w, spec.encoder_dim});

    auto *window_view
        = requireLayer(network->addShuffle(*pixel_view->getOutput(0)), "Failed to partition RF-DETR windows");
    window_view->setReshapeDimensions(
        makeDimsFromValues({0, spec.num_windows, h_win, spec.num_windows, w_win, spec.encoder_dim}));
    window_view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *windows = requireLayer(network->addShuffle(*window_view->getOutput(0)), "Failed to flatten RF-DETR windows");
    windows->setReshapeDimensions(nvinfer1::Dims3{-1, h_win * w_win, spec.encoder_dim});

    cls_with_pos = broadcastFirstDimLike(network, *cls_with_pos, *windows->getOutput(0));
    std::array<nvinfer1::ITensor *, 2> tokens{cls_with_pos, windows->getOutput(0)};
    auto *concat = requireLayer(network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size())),
                                "Failed to concat RF-DETR window tokens");
    concat->setAxis(1);
    named_tensors["tokens"] = concat->getOutput(0);
    return concat->getOutput(0);
}

/**
 * @brief 添加 DINOv2 self-attention。
 */
nvinfer1::ITensor *addDINOSelfAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const std::string &prefix, int tokens,
                                        const RFDETRSpec &spec)
{
    auto *q = addLinear3D(network, weights_map, input, prefix + ".attention.attention.query", spec.encoder_dim,
                          spec.encoder_dim);
    auto *k = addLinear3D(network, weights_map, input, prefix + ".attention.attention.key", spec.encoder_dim,
                          spec.encoder_dim);
    auto *v = addLinear3D(network, weights_map, input, prefix + ".attention.attention.value", spec.encoder_dim,
                          spec.encoder_dim);

    const int head_dim = spec.encoder_dim / spec.encoder_heads;
    auto     *q_heads  = reshapeToHeads(network, *q, 0, tokens, spec.encoder_heads, head_dim);
    auto     *k_heads  = reshapeToHeads(network, *k, 0, tokens, spec.encoder_heads, head_dim);
    auto     *v_heads  = reshapeToHeads(network, *v, 0, tokens, spec.encoder_heads, head_dim);
    auto     *qk       = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                                      "Failed to add RF-DETR DINO qk");
    auto     *scale    = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto     *scaled
        = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD), "Failed to scale RF-DETR DINO qk");
    auto *softmax = requireLayer(network->addSoftMax(*scaled->getOutput(0)), "Failed to add RF-DETR DINO softmax");
    softmax->setAxes(1U << 3U);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed to add RF-DETR DINO attention value");
    auto *merged   = mergeHeads(network, *attended->getOutput(0), 0, tokens, spec.encoder_dim);
    return addLinear3D(network, weights_map, *merged, prefix + ".attention.output.dense", spec.encoder_dim,
                       spec.encoder_dim);
}

/**
 * @brief 添加 DINOv2 LayerScale。
 */
nvinfer1::ITensor *addLayerScale(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, const std::string &key, int channels)
{
    auto *scale = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, channels},
                                                    requireWeight(weights_map, key, "RF-DETR layer scale", channels)),
                               "Failed to add RF-DETR layer scale")
                      ->getOutput(0);
    return requireLayer(network->addElementWise(input, *scale, E::kPROD), "Failed to apply RF-DETR layer scale")
        ->getOutput(0);
}

/**
 * @brief 添加一个 DINOv2 windowed encoder block。
 */
nvinfer1::ITensor *addDINOBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, int index, bool full_attention, const RFDETRSpec &spec,
                                const RFDETRGeometry &geometry)
{
    const auto prefix   = encoderPrefix() + ".encoder.layer." + std::to_string(index);
    auto      *shortcut = &input;
    auto      *x        = &input;
    int        tokens   = static_cast<int>(x->getDimensions().d[1]);

    if (full_attention && spec.num_windows > 1)
    {
        auto *full = requireLayer(network->addShuffle(*x), "Failed to unwindow RF-DETR DINO full attention input");
        full->setReshapeDimensions(nvinfer1::Dims3{-1, spec.num_windows * spec.num_windows * tokens, spec.encoder_dim});
        x      = full->getOutput(0);
        tokens = spec.num_windows * spec.num_windows * tokens;
    }

    auto *norm1 = addLayerNormLastDim(network, weights_map, *x, prefix + ".norm1", spec.encoder_dim);
    auto *attn  = addDINOSelfAttention(network, weights_map, *norm1, prefix, tokens, spec);

    if (full_attention && spec.num_windows > 1)
    {
        auto *rewindow = requireLayer(network->addShuffle(*attn), "Failed to rewindow RF-DETR DINO attention output");
        rewindow->setReshapeDimensions(
            nvinfer1::Dims3{-1, geometry.patches / (spec.num_windows * spec.num_windows) + 1, spec.encoder_dim});
        attn = rewindow->getOutput(0);
    }

    auto *scaled_attn = addLayerScale(network, weights_map, *attn, prefix + ".layer_scale1.lambda1", spec.encoder_dim);
    auto *attn_residual
        = requireLayer(network->addElementWise(*shortcut, *scaled_attn, E::kSUM), "Failed RF-DETR DINO attn residual")
              ->getOutput(0);
    auto *norm2 = addLayerNormLastDim(network, weights_map, *attn_residual, prefix + ".norm2", spec.encoder_dim);
    auto *mlp0 = addLinear3D(network, weights_map, *norm2, prefix + ".mlp.fc1", spec.encoder_dim, spec.encoder_dim * 4);
    auto *gelu = addGeluExact(network, *mlp0);
    auto *mlp1 = addLinear3D(network, weights_map, *gelu, prefix + ".mlp.fc2", spec.encoder_dim * 4, spec.encoder_dim);
    auto *scaled_mlp = addLayerScale(network, weights_map, *mlp1, prefix + ".layer_scale2.lambda1", spec.encoder_dim);
    return requireLayer(network->addElementWise(*attn_residual, *scaled_mlp, E::kSUM),
                        "Failed RF-DETR DINO MLP residual")
        ->getOutput(0);
}

/**
 * @brief 将 windowed DINO token 恢复为 NCHW feature map。
 */
nvinfer1::ITensor *tokensToFeatureMap(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &tokens,
                                      const RFDETRSpec &spec, const RFDETRGeometry &geometry)
{
    if (spec.num_windows <= 1)
    {
        auto *patch
            = slicePreserveFirstDim(network, tokens, nvinfer1::Dims3{0, 1, 0}, {geometry.patches, spec.encoder_dim});
        auto *view = requireLayer(network->addShuffle(*patch), "Failed to view RF-DETR DINO feature");
        view->setReshapeDimensions(nvinfer1::Dims4{0, geometry.patch_grid_h, geometry.patch_grid_w, spec.encoder_dim});
        view->setSecondTranspose(nvinfer1::Permutation{0, 3, 1, 2});
        return view->getOutput(0);
    }

    const int h_win = geometry.patch_grid_h / spec.num_windows;
    const int w_win = geometry.patch_grid_w / spec.num_windows;
    auto *patch = slicePreserveFirstDim(network, tokens, nvinfer1::Dims3{0, 1, 0}, {h_win * w_win, spec.encoder_dim});
    auto *patch_view = requireLayer(network->addShuffle(*patch), "Failed to view RF-DETR window feature");
    patch_view->setReshapeDimensions(nvinfer1::Dims4{0, h_win, w_win, spec.encoder_dim});

    auto *merge_view = requireLayer(network->addShuffle(*patch_view->getOutput(0)), "Failed to merge RF-DETR windows");
    merge_view->setReshapeDimensions(
        makeDimsFromValues({-1, spec.num_windows, spec.num_windows, h_win, w_win, spec.encoder_dim}));
    merge_view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *nhwc = requireLayer(network->addShuffle(*merge_view->getOutput(0)), "Failed to restore RF-DETR feature map");
    nhwc->setReshapeDimensions(nvinfer1::Dims4{-1, geometry.patch_grid_h, geometry.patch_grid_w, spec.encoder_dim});
    nhwc->setSecondTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    return nhwc->getOutput(0);
}

/**
 * @brief 添加完整 DINOv2 windowed backbone，并返回官方 out_feature_indexes 对应的 feature maps。
 */
std::vector<nvinfer1::ITensor *> addDINOBackbone(const RFDETRModel &impl, nvinfer1::INetworkDefinition *network,
                                                 const WeightsMap &weights_map, const RFDETRSpec &spec,
                                                 const RFDETRGeometry             &geometry,
                                                 priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *x = addDINOEmbedding(impl, network, weights_map, spec, geometry, named_tensors);

    std::vector<nvinfer1::ITensor *> features;
    features.reserve(spec.out_feature_indexes.size());

    for (int i = 0; i < kRFDETREncoderLayers; ++i)
    {
        const int  stage_index = i + 1;
        const bool need_feature
            = std::find(spec.out_feature_indexes.begin(), spec.out_feature_indexes.end(), stage_index)
           != spec.out_feature_indexes.end();
        const bool full_attention = std::find(spec.out_feature_indexes.begin(), spec.out_feature_indexes.end(), i)
                                 != spec.out_feature_indexes.end();
        x = addDINOBlock(network, weights_map, *x, i, full_attention, spec, geometry);
        named_tensors["backbone.block" + std::to_string(i)] = x;

        if (need_feature)
        {
            auto *feature = tokensToFeatureMap(network, *x, spec, geometry);
            auto *norm
                = addLayerNorm2d(network, weights_map, *feature, encoderPrefix() + ".layernorm", spec.encoder_dim);
            features.push_back(norm);
        }
    }

    return features;
}

/**
 * @brief 将 `P3/P4/P5` 转换为 projector 缩放因子。
 */
float projectorScaleFactor(const std::string &scale)
{
    if (scale == "P3")
    {
        return 2.0F;
    }
    if (scale == "P4")
    {
        return 1.0F;
    }
    if (scale == "P5")
    {
        return 0.5F;
    }
    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported RF-DETR projector scale: %s", scale.c_str());
}

/**
 * @brief 添加 MultiScaleProjector 的一个输出层级。
 */
FeatureLevel addProjectorLevel(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                               const std::vector<nvinfer1::ITensor *> &features, int level_index,
                               const RFDETRSpec &spec, const RFDETRGeometry &geometry,
                               priv::IModelImpl::NamedTensorMap &named_tensors)
{
    const float scale = projectorScaleFactor(spec.projector_scales[static_cast<size_t>(level_index)]);

    std::vector<nvinfer1::ITensor *> sampled;
    sampled.reserve(features.size());
    int output_h = geometry.patch_grid_h;
    int output_w = geometry.patch_grid_w;

    for (size_t i = 0; i < features.size(); ++i)
    {
        const auto prefix
            = "backbone.0.projector.stages_sampling." + std::to_string(level_index) + "." + std::to_string(i);
        if (scale == 2.0F)
        {
            auto *up0  = addDeconv2d(network, weights_map, *features[i], prefix + ".0", spec.encoder_dim,
                                     spec.encoder_dim / 2, 2, 2);
            auto *ln0  = addLayerNorm2d(network, weights_map, *up0, prefix + ".1", spec.encoder_dim / 2);
            auto *gelu = addGeluExact(network, *ln0);
            auto *up1  = addDeconv2d(network, weights_map, *gelu, prefix + ".3", spec.encoder_dim / 2,
                                     spec.encoder_dim / 4, 2, 2);
            sampled.push_back(up1);
            output_h = geometry.patch_grid_h * 2;
            output_w = geometry.patch_grid_w * 2;
        }
        else if (scale == 1.0F)
        {
            sampled.push_back(features[i]);
        }
        else if (scale == 0.5F)
        {
            sampled.push_back(addConvX(network, weights_map, *features[i], prefix + ".0", spec.encoder_dim,
                                       spec.encoder_dim, 3, 2, "relu"));
            output_h = geometry.patch_grid_h / 2;
            output_w = geometry.patch_grid_w / 2;
        }
    }

    auto *cat = requireLayer(network->addConcatenation(sampled.data(), static_cast<int32_t>(sampled.size())),
                             "Failed to concat RF-DETR projector features");
    cat->setAxis(1);
    const auto feature_prefix              = "projector.level" + std::to_string(level_index);
    named_tensors[feature_prefix + ".cat"] = cat->getOutput(0);

    int c2f_in_channels = 0;
    if (scale == 2.0F)
    {
        c2f_in_channels = static_cast<int>(features.size()) * (spec.encoder_dim / 4);
    }
    else
    {
        c2f_in_channels = static_cast<int>(features.size()) * spec.encoder_dim;
    }

    const auto stage_prefix = "backbone.0.projector.stages." + std::to_string(level_index);
    auto      *c2f  = addProjectorC2f(network, weights_map, *cat->getOutput(0), stage_prefix + ".0", c2f_in_channels,
                                      spec.hidden_dim, feature_prefix + ".c2f", named_tensors);
    auto      *norm = addLayerNorm2d(network, weights_map, *c2f, stage_prefix + ".1", spec.hidden_dim);
    named_tensors[feature_prefix + ".norm"] = norm;
    return {norm, output_h, output_w};
}

/**
 * @brief 添加 RF-DETR projector，返回 decoder 使用的多尺度 feature levels。
 */
std::vector<FeatureLevel> addProjector(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       const std::vector<nvinfer1::ITensor *> &features, const RFDETRSpec &spec,
                                       const RFDETRGeometry &geometry, priv::IModelImpl::NamedTensorMap &named_tensors)
{
    std::vector<FeatureLevel> levels;
    levels.reserve(spec.projector_scales.size());
    for (size_t i = 0; i < spec.projector_scales.size(); ++i)
    {
        levels.push_back(
            addProjectorLevel(network, weights_map, features, static_cast<int>(i), spec, geometry, named_tensors));
    }
    return levels;
}

/**
 * @brief 生成一层 feature map 的 sine position encoding 常量。
 */
std::vector<float> makeSinePositionTable(int channels, int height, int width)
{
    const int          num_pos_feats = channels / 2;
    std::vector<float> table(static_cast<size_t>(channels) * height * width);
    std::vector<float> dim_t(static_cast<size_t>(num_pos_feats));
    for (int i = 0; i < num_pos_feats; ++i)
    {
        dim_t[static_cast<size_t>(i)] = std::pow(kRFDETRSinePositionTemp, 2.0F * std::floor(i / 2.0F) / num_pos_feats);
    }

    for (int y = 0; y < height; ++y)
    {
        const float y_embed = (static_cast<float>(y) + 1.0F) / static_cast<float>(height) * kRFDETRTwoPi;
        for (int x = 0; x < width; ++x)
        {
            const float x_embed = (static_cast<float>(x) + 1.0F) / static_cast<float>(width) * kRFDETRTwoPi;
            const int   spatial = y * width + x;
            for (int i = 0; i < num_pos_feats; ++i)
            {
                const bool  even                                         = (i % 2) == 0;
                const float py                                           = y_embed / dim_t[static_cast<size_t>(i)];
                const float px                                           = x_embed / dim_t[static_cast<size_t>(i)];
                table[static_cast<size_t>(i) * height * width + spatial] = even ? std::sin(py) : std::cos(py);
                table[static_cast<size_t>(num_pos_feats + i) * height * width + spatial]
                    = even ? std::sin(px) : std::cos(px);
            }
        }
    }
    return table;
}

/**
 * @brief NCHW feature flatten 为 [N,HW,C]。
 */
nvinfer1::ITensor *flattenFeature(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &feature, int channels,
                                  int height, int width)
{
    auto *shuffle = requireLayer(network->addShuffle(feature), "Failed to flatten RF-DETR feature");
    shuffle->setReshapeDimensions(nvinfer1::Dims3{0, channels, height * width});
    shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    return shuffle->getOutput(0);
}

/**
 * @brief 拼接 decoder memory、position encoding 与 proposal 常量。
 */
void addDecoderInputs(nvinfer1::INetworkDefinition *network, const std::vector<FeatureLevel> &levels,
                      const RFDETRSpec &spec, nvinfer1::ITensor *&memory, nvinfer1::ITensor *&pos,
                      nvinfer1::ITensor *&proposals)
{
    std::vector<nvinfer1::ITensor *> memory_parts;
    std::vector<nvinfer1::ITensor *> pos_parts;
    std::vector<float>               proposal_values;
    memory_parts.reserve(levels.size());
    pos_parts.reserve(levels.size());

    for (size_t level = 0; level < levels.size(); ++level)
    {
        const auto &feature = levels[level];
        memory_parts.push_back(
            flattenFeature(network, *feature.tensor, spec.hidden_dim, feature.height, feature.width));

        auto *pos_const
            = requireLayer(network->addConstant(
                               nvinfer1::Dims4{1, spec.hidden_dim, feature.height, feature.width},
                               ownedFloatVector(makeSinePositionTable(spec.hidden_dim, feature.height, feature.width))),
                           "Failed to add RF-DETR position encoding")
                  ->getOutput(0);
        pos_const = broadcastFirstDimLike(network, *pos_const, *feature.tensor);
        pos_parts.push_back(flattenFeature(network, *pos_const, spec.hidden_dim, feature.height, feature.width));

        for (int y = 0; y < feature.height; ++y)
        {
            for (int x = 0; x < feature.width; ++x)
            {
                proposal_values.push_back((static_cast<float>(x) + 0.5F) / static_cast<float>(feature.width));
                proposal_values.push_back((static_cast<float>(y) + 0.5F) / static_cast<float>(feature.height));
                const float wh = kProposalBaseBoxScale * std::pow(2.0F, static_cast<float>(level));
                proposal_values.push_back(wh);
                proposal_values.push_back(wh);
            }
        }
    }

    auto *memory_cat
        = requireLayer(network->addConcatenation(memory_parts.data(), static_cast<int32_t>(memory_parts.size())),
                       "Failed to concat RF-DETR memory");
    memory_cat->setAxis(1);
    memory = memory_cat->getOutput(0);

    auto *pos_cat = requireLayer(network->addConcatenation(pos_parts.data(), static_cast<int32_t>(pos_parts.size())),
                                 "Failed to concat RF-DETR position");
    pos_cat->setAxis(1);
    pos = pos_cat->getOutput(0);

    const int proposal_count = static_cast<int>(proposal_values.size() / 4);
    auto     *proposal_const = requireLayer(network->addConstant(nvinfer1::Dims3{1, proposal_count, 4},
                                                                 ownedFloatVector(std::move(proposal_values))),
                                            "Failed to add RF-DETR proposals")
                               ->getOutput(0);
    proposals = broadcastFirstDimLike(network, *proposal_const, *memory);
}

/**
 * @brief 按 batch 维 gather `[B,S,C]` 数据。
 */
nvinfer1::ITensor *gatherBatchSequence(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &data,
                                       nvinfer1::ITensor &indices)
{
    auto *gather = requireLayer(network->addGather(data, indices, 1), "Failed to add RF-DETR gather");
    gather->setNbElementWiseDims(1);
    return gather->getOutput(0);
}

/**
 * @brief 添加 bbox_reparam 公式。
 */
nvinfer1::ITensor *applyBBoxReparam(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &delta,
                                    nvinfer1::ITensor &reference, int tokens)
{
    auto *delta_xy = slicePreserveFirstDim(network, delta, nvinfer1::Dims3{0, 0, 0}, {tokens, 2});
    auto *delta_wh = slicePreserveFirstDim(network, delta, nvinfer1::Dims3{0, 0, 2}, {tokens, 2});
    auto *ref_xy   = slicePreserveFirstDim(network, reference, nvinfer1::Dims3{0, 0, 0}, {tokens, 2});
    auto *ref_wh   = slicePreserveFirstDim(network, reference, nvinfer1::Dims3{0, 0, 2}, {tokens, 2});

    auto *scaled_xy
        = requireLayer(network->addElementWise(*delta_xy, *ref_wh, E::kPROD), "Failed RF-DETR bbox xy scale")
              ->getOutput(0);
    auto *xy = requireLayer(network->addElementWise(*scaled_xy, *ref_xy, E::kSUM), "Failed RF-DETR bbox xy add")
                   ->getOutput(0);
    auto *exp_wh = requireLayer(network->addUnary(*delta_wh, nvinfer1::UnaryOperation::kEXP), "Failed RF-DETR bbox exp")
                       ->getOutput(0);
    auto *wh
        = requireLayer(network->addElementWise(*exp_wh, *ref_wh, E::kPROD), "Failed RF-DETR bbox wh")->getOutput(0);

    std::array<nvinfer1::ITensor *, 2> parts{xy, wh};
    auto *cat = requireLayer(network->addConcatenation(parts.data(), static_cast<int32_t>(parts.size())),
                             "Failed RF-DETR bbox concat");
    cat->setAxis(2);
    return cat->getOutput(0);
}

/**
 * @brief 添加 two-stage proposal TopK 初始化。
 */
void addTwoStageQueries(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &memory,
                        nvinfer1::ITensor &proposals, const RFDETRSpec &spec, int classes_with_background,
                        nvinfer1::ITensor *&query, nvinfer1::ITensor *&refpoints,
                        priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *enc_memory
        = addLinear3D(network, weights_map, memory, "transformer.enc_output.0", spec.hidden_dim, spec.hidden_dim);
    enc_memory
        = addLayerNormLastDim(network, weights_map, *enc_memory, "transformer.enc_output_norm.0", spec.hidden_dim);
    named_tensors["enc.memory"] = enc_memory;
    auto *enc_class             = addLinear3D(network, weights_map, *enc_memory, "transformer.enc_out_class_embed.0",
                                              spec.hidden_dim, classes_with_background);
    named_tensors["enc.class"]  = enc_class;
    auto *enc_delta             = addMLP(network, weights_map, *enc_memory, "transformer.enc_out_bbox_embed.0",
                                         {spec.hidden_dim, spec.hidden_dim, spec.hidden_dim, 4});
    named_tensors["enc.delta"]  = enc_delta;
    auto *enc_boxes = applyBBoxReparam(network, *enc_delta, proposals, static_cast<int>(memory.getDimensions().d[1]));
    named_tensors["enc.boxes"] = enc_boxes;

    auto *scores = requireLayer(network->addReduce(*enc_class, nvinfer1::ReduceOperation::kMAX, 1U << 2U, false),
                                "Failed RF-DETR encoder class max");
    named_tensors["enc.scores"] = scores->getOutput(0);
#if TRT_VERSION >= 10140
    auto *topk = requireLayer(network->addTopK(*scores->getOutput(0), nvinfer1::TopKOperation::kMAX, spec.num_queries,
                                               1U << 1U, nvinfer1::DataType::kINT32),
                              "Failed RF-DETR TopK");
#else
    auto *topk = requireLayer(
        network->addTopK(*scores->getOutput(0), nvinfer1::TopKOperation::kMAX, spec.num_queries, 1U << 1U),
        "Failed RF-DETR TopK");
#endif
    named_tensors["enc.topk.values"]  = topk->getOutput(0);
    named_tensors["enc.topk.indices"] = topk->getOutput(1);
    refpoints                         = gatherBatchSequence(network, *enc_boxes, *topk->getOutput(1));
    named_tensors["refpoints.enc"]    = refpoints;

    auto *query_const
        = requireLayer(network->addConstant(nvinfer1::Dims3{1, spec.num_queries, spec.hidden_dim},
                                            requireWeight(weights_map, "query_feat.weight", "RF-DETR query feature",
                                                          static_cast<int64_t>(spec.num_queries) * spec.hidden_dim)),
                       "Failed RF-DETR query feature")
              ->getOutput(0);
    query = broadcastFirstDimLike(network, *query_const, memory);

    auto *base_ref = requireLayer(network->addConstant(
                                      nvinfer1::Dims3{1, spec.num_queries, 4},
                                      requireWeight(weights_map, "refpoint_embed.weight", "RF-DETR refpoint embed",
                                                    static_cast<int64_t>(spec.num_queries) * 4)),
                                  "Failed RF-DETR base refpoint")
                         ->getOutput(0);
    base_ref                        = broadcastFirstDimLike(network, *base_ref, memory);
    named_tensors["refpoints.base"] = base_ref;
    refpoints                       = applyBBoxReparam(network, *base_ref, *refpoints, spec.num_queries);
}

/**
 * @brief 对单个坐标通道生成 alternating sin/cos embedding。
 */
nvinfer1::ITensor *addSineComponent(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &component, int dim)
{
    std::vector<float> dim_t(static_cast<size_t>(dim));
    std::vector<float> even_mask(static_cast<size_t>(dim));
    std::vector<float> odd_mask(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
    {
        dim_t[static_cast<size_t>(i)]     = std::pow(kRFDETRSinePositionTemp, 2.0F * std::floor(i / 2.0F) / dim);
        even_mask[static_cast<size_t>(i)] = (i % 2) == 0 ? 1.0F : 0.0F;
        odd_mask[static_cast<size_t>(i)]  = (i % 2) == 0 ? 0.0F : 1.0F;
    }

    auto *scale = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, dim}, ownedFloatVector(dim_t)),
                               "Failed RF-DETR sine dim_t")
                      ->getOutput(0);
    auto *two_pi = addScalar(network, component, kRFDETRTwoPi);
    auto *scaled = requireLayer(network->addElementWise(component, *two_pi, E::kPROD), "Failed RF-DETR sine scale")
                       ->getOutput(0);
    auto *angle
        = requireLayer(network->addElementWise(*scaled, *scale, E::kDIV), "Failed RF-DETR sine div")->getOutput(0);
    auto *sin_part
        = requireLayer(network->addUnary(*angle, nvinfer1::UnaryOperation::kSIN), "Failed RF-DETR sin")->getOutput(0);
    auto *cos_part
        = requireLayer(network->addUnary(*angle, nvinfer1::UnaryOperation::kCOS), "Failed RF-DETR cos")->getOutput(0);
    auto *even = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, dim}, ownedFloatVector(std::move(even_mask))),
                              "Failed RF-DETR sine even mask")
                     ->getOutput(0);
    auto *odd = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, dim}, ownedFloatVector(std::move(odd_mask))),
                             "Failed RF-DETR sine odd mask")
                    ->getOutput(0);
    auto *sin_even
        = requireLayer(network->addElementWise(*sin_part, *even, E::kPROD), "Failed RF-DETR sine mask")->getOutput(0);
    auto *cos_odd
        = requireLayer(network->addElementWise(*cos_part, *odd, E::kPROD), "Failed RF-DETR cos mask")->getOutput(0);
    return requireLayer(network->addElementWise(*sin_even, *cos_odd, E::kSUM), "Failed RF-DETR sine merge")
        ->getOutput(0);
}

/**
 * @brief 官方 `gen_sineembed_for_position` 的 TensorRT 版本。
 */
nvinfer1::ITensor *addReferenceSineEmbedding(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &refpoints,
                                             const RFDETRSpec &spec)
{
    const int dim = spec.hidden_dim / 2;
    auto     *x   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 0}, {spec.num_queries, 1});
    auto     *y   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 1}, {spec.num_queries, 1});
    auto     *w   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 2}, {spec.num_queries, 1});
    auto     *h   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 3}, {spec.num_queries, 1});

    std::array<nvinfer1::ITensor *, 4> parts{addSineComponent(network, *y, dim), addSineComponent(network, *x, dim),
                                             addSineComponent(network, *w, dim), addSineComponent(network, *h, dim)};
    auto *concat = requireLayer(network->addConcatenation(parts.data(), static_cast<int32_t>(parts.size())),
                                "Failed RF-DETR sine concat");
    concat->setAxis(2);
    return concat->getOutput(0);
}

/**
 * @brief 添加 decoder self-attention。
 */
nvinfer1::ITensor *addDecoderSelfAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                           nvinfer1::ITensor &target, nvinfer1::ITensor &query_pos,
                                           const std::string &prefix, const RFDETRSpec &spec)
{
    auto *qk_input
        = requireLayer(network->addElementWise(target, query_pos, E::kSUM), "Failed RF-DETR self q add")->getOutput(0);
    const auto &packed_weight
        = requireWeight(weights_map, prefix + ".self_attn.in_proj_weight", "RF-DETR self attention projection",
                        static_cast<int64_t>(3) * spec.hidden_dim * spec.hidden_dim);
    const auto &packed_bias
        = requireWeight(weights_map, prefix + ".self_attn.in_proj_bias", "RF-DETR self attention projection",
                        static_cast<int64_t>(3) * spec.hidden_dim);

    const int64_t projection_weight_count = static_cast<int64_t>(spec.hidden_dim) * spec.hidden_dim;
    auto          q_weight = sliceFloatWeights(packed_weight, 0, projection_weight_count, "self_attn.q.weight");
    auto          k_weight
        = sliceFloatWeights(packed_weight, projection_weight_count, projection_weight_count, "self_attn.k.weight");
    auto v_weight
        = sliceFloatWeights(packed_weight, 2 * projection_weight_count, projection_weight_count, "self_attn.v.weight");
    auto q_bias = sliceFloatWeights(packed_bias, 0, spec.hidden_dim, "self_attn.q.bias");
    auto k_bias = sliceFloatWeights(packed_bias, spec.hidden_dim, spec.hidden_dim, "self_attn.k.bias");
    auto v_bias = sliceFloatWeights(packed_bias, 2 * spec.hidden_dim, spec.hidden_dim, "self_attn.v.bias");

    auto *q = addLinear3DFromWeights(network, *qk_input, q_weight, &q_bias, spec.hidden_dim, spec.hidden_dim);
    auto *k = addLinear3DFromWeights(network, *qk_input, k_weight, &k_bias, spec.hidden_dim, spec.hidden_dim);
    auto *v = addLinear3DFromWeights(network, target, v_weight, &v_bias, spec.hidden_dim, spec.hidden_dim);

    const int head_dim = spec.hidden_dim / spec.self_attention_heads;
    auto     *q_heads  = reshapeToHeads(network, *q, 0, spec.num_queries, spec.self_attention_heads, head_dim);
    auto     *k_heads  = reshapeToHeads(network, *k, 0, spec.num_queries, spec.self_attention_heads, head_dim);
    auto     *v_heads  = reshapeToHeads(network, *v, 0, spec.num_queries, spec.self_attention_heads, head_dim);
    auto     *qk       = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                                      "Failed RF-DETR self qk");
    auto     *scale    = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto     *scaled
        = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD), "Failed RF-DETR self scale");
    auto *softmax = requireLayer(network->addSoftMax(*scaled->getOutput(0)), "Failed RF-DETR self softmax");
    softmax->setAxes(1U << 3U);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed RF-DETR self attended");
    auto *merged   = mergeHeads(network, *attended->getOutput(0), 0, spec.num_queries, spec.hidden_dim);
    return addLinear3D(network, weights_map, *merged, prefix + ".self_attn.out_proj", spec.hidden_dim, spec.hidden_dim);
}

/**
 * @brief 添加一个 feature level 的 deformable sampling。
 */
nvinfer1::ITensor *addDeformableSampleLevel(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &value,
                                            nvinfer1::ITensor &sampling_offsets, nvinfer1::ITensor &attention_weights,
                                            nvinfer1::ITensor &refpoints, int level_index, int level_start, int height,
                                            int width, const RFDETRSpec &spec)
{
    const int head_dim    = spec.hidden_dim / spec.cross_attention_heads;
    auto     *value_slice = slicePreserveFirstDim(network, value, nvinfer1::Dims4{0, 0, 0, level_start},
                                                  {spec.cross_attention_heads, head_dim, height * width});
    auto     *value_map   = requireLayer(network->addShuffle(*value_slice), "Failed RF-DETR value map");
    value_map->setReshapeDimensions(nvinfer1::Dims4{-1, head_dim, height, width});

    auto *offset_slice
        = slicePreserveFirstDim(network, sampling_offsets, makeDimsFromValues({0, 0, 0, level_index, 0, 0}),
                                {spec.num_queries, spec.cross_attention_heads, 1, spec.deform_points, 2});
    auto *offset_view = requireLayer(network->addShuffle(*offset_slice), "Failed RF-DETR offset level view");
    offset_view->setReshapeDimensions(
        makeDimsFromValues({0, spec.num_queries, spec.cross_attention_heads, spec.deform_points, 2}));
    auto *offset_l = offset_view->getOutput(0);
    auto *ref_xy   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 0}, {spec.num_queries, 2});
    auto *ref_wh   = slicePreserveFirstDim(network, refpoints, nvinfer1::Dims3{0, 0, 2}, {spec.num_queries, 2});

    auto *ref_xy_view = requireLayer(network->addShuffle(*ref_xy), "Failed RF-DETR ref xy view");
    ref_xy_view->setReshapeDimensions(makeDimsFromValues({0, spec.num_queries, 1, 1, 2}));
    auto *ref_wh_view = requireLayer(network->addShuffle(*ref_wh), "Failed RF-DETR ref wh view");
    ref_wh_view->setReshapeDimensions(makeDimsFromValues({0, spec.num_queries, 1, 1, 2}));

    auto *point_scale = addScalar(network, *offset_l, 0.5F / static_cast<float>(spec.deform_points));
    auto *offset_scaled
        = requireLayer(network->addElementWise(*offset_l, *point_scale, E::kPROD), "Failed RF-DETR offset scale")
              ->getOutput(0);
    auto *offset_box = requireLayer(network->addElementWise(*offset_scaled, *ref_wh_view->getOutput(0), E::kPROD),
                                    "Failed RF-DETR offset box")
                           ->getOutput(0);
    auto *loc = requireLayer(network->addElementWise(*offset_box, *ref_xy_view->getOutput(0), E::kSUM),
                             "Failed RF-DETR sampling loc")
                    ->getOutput(0);
    auto *two  = addScalar(network, *loc, 2.0F);
    auto *one  = addScalar(network, *loc, 1.0F);
    auto *grid = requireLayer(network->addElementWise(*loc, *two, E::kPROD), "Failed RF-DETR grid scale")->getOutput(0);
    grid       = requireLayer(network->addElementWise(*grid, *one, E::kSUB), "Failed RF-DETR grid shift")->getOutput(0);

    auto *grid_view = requireLayer(network->addShuffle(*grid), "Failed RF-DETR grid view");
    grid_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3, 4});
    grid_view->setReshapeDimensions(nvinfer1::Dims4{-1, spec.num_queries, spec.deform_points, 2});

    auto *sample = requireLayer(network->addGridSample(*value_map->getOutput(0), *grid_view->getOutput(0)),
                                "Failed RF-DETR GridSample");
    sample->setInterpolationMode(nvinfer1::InterpolationMode::kLINEAR);
    sample->setAlignCorners(false);
    sample->setSampleMode(nvinfer1::SampleMode::kFILL);

    auto *attn_l
        = slicePreserveFirstDim(network, attention_weights, nvinfer1::Dims4{0, 0, 0, level_index * spec.deform_points},
                                {spec.num_queries, spec.cross_attention_heads, spec.deform_points});
    auto *attn_view = requireLayer(network->addShuffle(*attn_l), "Failed RF-DETR attention weight view");
    attn_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    attn_view->setReshapeDimensions(nvinfer1::Dims4{-1, 1, spec.num_queries, spec.deform_points});

    auto *weighted = requireLayer(network->addElementWise(*sample->getOutput(0), *attn_view->getOutput(0), E::kPROD),
                                  "Failed RF-DETR deform weight")
                         ->getOutput(0);
    return requireLayer(network->addReduce(*weighted, nvinfer1::ReduceOperation::kSUM, 1U << 3U, false),
                        "Failed RF-DETR deform reduce")
        ->getOutput(0);
}

/**
 * @brief 添加 MSDeformAttn 的 TensorRT 展开实现。
 */
nvinfer1::ITensor *addMSDeformAttn(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &query, nvinfer1::ITensor &memory, nvinfer1::ITensor &refpoints,
                                   const std::vector<FeatureLevel> &levels, const std::string &prefix,
                                   const RFDETRSpec &spec)
{
    auto *value      = addLinear3D(network, weights_map, memory, prefix + ".cross_attn.value_proj", spec.hidden_dim,
                                   spec.hidden_dim);
    auto *value_view = requireLayer(network->addShuffle(*value), "Failed RF-DETR deform value view");
    value_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
    value_view->setReshapeDimensions(nvinfer1::Dims4{
        0, spec.cross_attention_heads, spec.hidden_dim / spec.cross_attention_heads, memory.getDimensions().d[1]});

    auto *sampling_offsets
        = addLinear3D(network, weights_map, query, prefix + ".cross_attn.sampling_offsets", spec.hidden_dim,
                      spec.cross_attention_heads * static_cast<int>(levels.size()) * spec.deform_points * 2);
    auto *offset_view = requireLayer(network->addShuffle(*sampling_offsets), "Failed RF-DETR sampling offset view");
    offset_view->setReshapeDimensions(makeDimsFromValues(
        {0, spec.num_queries, spec.cross_attention_heads, static_cast<int>(levels.size()), spec.deform_points, 2}));

    auto *attention_weights
        = addLinear3D(network, weights_map, query, prefix + ".cross_attn.attention_weights", spec.hidden_dim,
                      spec.cross_attention_heads * static_cast<int>(levels.size()) * spec.deform_points);
    auto *attn_view = requireLayer(network->addShuffle(*attention_weights), "Failed RF-DETR deform attention view");
    attn_view->setReshapeDimensions(nvinfer1::Dims4{0, spec.num_queries, spec.cross_attention_heads,
                                                    static_cast<int>(levels.size()) * spec.deform_points});
    auto *softmax = requireLayer(network->addSoftMax(*attn_view->getOutput(0)), "Failed RF-DETR deform softmax");
    softmax->setAxes(1U << 3U);

    std::vector<nvinfer1::ITensor *> sampled_levels;
    sampled_levels.reserve(levels.size());
    int level_start = 0;
    for (size_t level = 0; level < levels.size(); ++level)
    {
        sampled_levels.push_back(addDeformableSampleLevel(
            network, *value_view->getOutput(0), *offset_view->getOutput(0), *softmax->getOutput(0), refpoints,
            static_cast<int>(level), level_start, levels[level].height, levels[level].width, spec));
        level_start += levels[level].height * levels[level].width;
    }

    nvinfer1::ITensor *sum = sampled_levels.front();
    for (size_t i = 1; i < sampled_levels.size(); ++i)
    {
        sum = requireLayer(network->addElementWise(*sum, *sampled_levels[i], E::kSUM), "Failed RF-DETR deform sum")
                  ->getOutput(0);
    }

    auto *merged = requireLayer(network->addShuffle(*sum), "Failed RF-DETR deform merge");
    merged->setInput(1, *shapeWithFirstDimOf(network, memory,
                                             {spec.cross_attention_heads, spec.hidden_dim / spec.cross_attention_heads,
                                              spec.num_queries}));
    auto *out = requireLayer(network->addShuffle(*merged->getOutput(0)), "Failed RF-DETR deform output view");
    out->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    out->setReshapeDimensions(nvinfer1::Dims3{0, spec.num_queries, spec.hidden_dim});
    return addLinear3D(network, weights_map, *out->getOutput(0), prefix + ".cross_attn.output_proj", spec.hidden_dim,
                       spec.hidden_dim);
}

/**
 * @brief 添加一个 RF-DETR decoder layer。
 */
nvinfer1::ITensor *addDecoderLayer(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &target, nvinfer1::ITensor &query_pos, nvinfer1::ITensor &memory,
                                   nvinfer1::ITensor &refpoints, const std::vector<FeatureLevel> &levels, int index,
                                   const RFDETRSpec &spec)
{
    const auto prefix    = "transformer.decoder.layers." + std::to_string(index);
    auto      *self_attn = addDecoderSelfAttention(network, weights_map, target, query_pos, prefix, spec);
    auto      *res1
        = requireLayer(network->addElementWise(target, *self_attn, E::kSUM), "Failed RF-DETR decoder self residual")
              ->getOutput(0);
    auto *norm1 = addLayerNormLastDim(network, weights_map, *res1, prefix + ".norm1", spec.hidden_dim);

    auto *cross_query
        = requireLayer(network->addElementWise(*norm1, query_pos, E::kSUM), "Failed RF-DETR decoder cross query")
              ->getOutput(0);
    auto *cross = addMSDeformAttn(network, weights_map, *cross_query, memory, refpoints, levels, prefix, spec);
    auto *res2 = requireLayer(network->addElementWise(*norm1, *cross, E::kSUM), "Failed RF-DETR decoder cross residual")
                     ->getOutput(0);
    auto *norm2 = addLayerNormLastDim(network, weights_map, *res2, prefix + ".norm2", spec.hidden_dim);

    auto *ffn0 = addLinear3D(network, weights_map, *norm2, prefix + ".linear1", spec.hidden_dim, spec.hidden_dim * 8);
    auto *relu = requireLayer(network->addActivation(*ffn0, nvinfer1::ActivationType::kRELU),
                              "Failed RF-DETR decoder FFN ReLU")
                     ->getOutput(0);
    auto *ffn1 = addLinear3D(network, weights_map, *relu, prefix + ".linear2", spec.hidden_dim * 8, spec.hidden_dim);
    auto *res3 = requireLayer(network->addElementWise(*norm2, *ffn1, E::kSUM), "Failed RF-DETR decoder FFN residual")
                     ->getOutput(0);
    return addLayerNormLastDim(network, weights_map, *res3, prefix + ".norm3", spec.hidden_dim);
}

/**
 * @brief 添加完整 RF-DETR decoder。
 */
nvinfer1::ITensor *addDecoder(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                              nvinfer1::ITensor &query, nvinfer1::ITensor &memory, nvinfer1::ITensor &pos,
                              nvinfer1::ITensor &refpoints, const std::vector<FeatureLevel> &levels,
                              const RFDETRSpec &spec, nvinfer1::ITensor *&head_refpoints)
{
    auto *sine      = addReferenceSineEmbedding(network, refpoints, spec);
    auto *query_pos = addMLP(network, weights_map, *sine, "transformer.decoder.ref_point_head",
                             {2 * spec.hidden_dim, spec.hidden_dim, spec.hidden_dim});
    (void)pos;

    nvinfer1::ITensor *x = &query;
    head_refpoints       = &refpoints;
    for (int i = 0; i < spec.decoder_layers; ++i)
    {
        x = addDecoderLayer(network, weights_map, *x, *query_pos, memory, refpoints, levels, i, spec);
    }
    return addLayerNormLastDim(network, weights_map, *x, "transformer.decoder.norm", spec.hidden_dim);
}

/**
 * @brief 添加检测 box/class heads。
 */
void addDetectionHeads(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &hs,
                       nvinfer1::ITensor &refpoints, const RFDETRSpec &spec, int classes_with_background,
                       nvinfer1::ITensor *&boxes, nvinfer1::ITensor *&logits)
{
    auto *delta
        = addMLP(network, weights_map, hs, "bbox_embed", {spec.hidden_dim, spec.hidden_dim, spec.hidden_dim, 4});
    boxes  = applyBBoxReparam(network, *delta, refpoints, spec.num_queries);
    logits = addLinear3D(network, weights_map, hs, "class_embed", spec.hidden_dim, classes_with_background);
}

/**
 * @brief 添加 segmentation head 中的 DepthwiseConvBlock。
 */
nvinfer1::ITensor *addSegDepthwiseBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *dw = addConv2d(network, weights_map, input, prefix + ".dwconv", channels, channels, 3, 1, 1, channels, true);
    auto *nhwc = requireLayer(network->addShuffle(*dw), "Failed RF-DETR seg NCHW->NHWC");
    nhwc->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    auto *norm = addLayerNormLastDim(network, weights_map, *nhwc->getOutput(0), prefix + ".norm", channels);
    auto *proj = addLinearLastDim(network, weights_map, *norm, prefix + ".pwconv1", channels, channels);
    auto *gelu = addGeluExact(network, *proj);
    auto *nchw = requireLayer(network->addShuffle(*gelu), "Failed RF-DETR seg NHWC->NCHW");
    nchw->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    return requireLayer(network->addElementWise(input, *nchw->getOutput(0), E::kSUM), "Failed RF-DETR seg residual")
        ->getOutput(0);
}

/**
 * @brief 添加实例分割 mask head。
 */
nvinfer1::ITensor *addSegmentationHead(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &spatial, nvinfer1::ITensor &query, const RFDETRSpec &spec,
                                       const RFDETRGeometry &geometry)
{
    const int mask_h = geometry.height / kRFDETRMaskDownsample;
    const int mask_w = geometry.width / kRFDETRMaskDownsample;

    auto *resize = requireLayer(network->addResize(spatial), "Failed RF-DETR seg resize");
    resize->setResizeMode(nvinfer1::InterpolationMode::kLINEAR);
    resize->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    resize->setInput(1, *shapeWithFirstDimOf(network, spatial, {spec.hidden_dim, mask_h, mask_w}));

    nvinfer1::ITensor *spatial_features = resize->getOutput(0);
    for (int i = 0; i < spec.decoder_layers; ++i)
    {
        spatial_features = addSegDepthwiseBlock(network, weights_map, *spatial_features,
                                                "segmentation_head.blocks." + std::to_string(i), spec.hidden_dim);
    }
    spatial_features = addConv2d(network, weights_map, *spatial_features, "segmentation_head.spatial_features_proj",
                                 spec.hidden_dim, spec.hidden_dim, 1, 1, 0, 1, true);

    auto *q = addLayerNormLastDim(network, weights_map, query, "segmentation_head.query_features_block.norm_in",
                                  spec.hidden_dim);
    q       = addLinear3D(network, weights_map, *q, "segmentation_head.query_features_block.layers.0", spec.hidden_dim,
                          spec.hidden_dim * 4);
    q       = addGeluExact(network, *q);
    q = addLinear3D(network, weights_map, *q, "segmentation_head.query_features_block.layers.2", spec.hidden_dim * 4,
                    spec.hidden_dim);
    q = requireLayer(network->addElementWise(query, *q, E::kSUM), "Failed RF-DETR seg query residual")->getOutput(0);
    q = addLinear3D(network, weights_map, *q, "segmentation_head.query_features_proj", spec.hidden_dim,
                    spec.hidden_dim);

    auto *flat = requireLayer(network->addShuffle(*spatial_features), "Failed RF-DETR seg flatten spatial");
    flat->setReshapeDimensions(nvinfer1::Dims3{0, spec.hidden_dim, mask_h * mask_w});
    auto *masks = requireLayer(network->addMatrixMultiply(*q, M::kNONE, *flat->getOutput(0), M::kNONE),
                               "Failed RF-DETR seg einsum");
    auto *view  = requireLayer(network->addShuffle(*masks->getOutput(0)), "Failed RF-DETR seg mask view");
    view->setReshapeDimensions(nvinfer1::Dims4{0, spec.num_queries, mask_h, mask_w});

    auto *bias = requireLayer(
                     network->addConstant(nvinfer1::Dims4{1, 1, 1, 1},
                                          requireWeight(weights_map, "segmentation_head.bias", "RF-DETR seg bias", 1)),
                     "Failed RF-DETR seg bias")
                     ->getOutput(0);
    return requireLayer(network->addElementWise(*view->getOutput(0), *bias, E::kSUM), "Failed RF-DETR seg bias add")
        ->getOutput(0);
}

/**
 * @brief 生成 RF-DETR 检测模型规格。
 */
RFDETRSpec makeDetectionSpec(const char *display_name, int resolution, int patch_size, int num_windows, int encoder_dim,
                             int encoder_heads, int hidden_dim, int decoder_layers, int self_heads, int cross_heads,
                             int deform_points, std::vector<int> out_features,
                             std::vector<std::string> projector_scales, int num_queries = 300, int num_select = 300)
{
    return {display_name,
            resolution,
            patch_size,
            num_windows,
            encoder_dim,
            encoder_heads,
            hidden_dim,
            decoder_layers,
            self_heads,
            cross_heads,
            deform_points,
            num_queries,
            num_select,
            std::move(out_features),
            std::move(projector_scales),
            false};
}

/**
 * @brief 生成 RF-DETR 实例分割模型规格。
 */
RFDETRSpec makeSegmentationSpec(const char *display_name, int resolution, int patch_size, int num_windows,
                                int decoder_layers, int num_queries, int num_select)
{
    return {
        display_name, resolution, patch_size,  num_windows, 384,           6,      256, decoder_layers, 8,
        16,           2,          num_queries, num_select,  {3, 6, 9, 12},
                                {"P4"},
                                true
    };
}

class RFDETRBase : public RFDETRModel
{
public:
    RFDETRBase()
        : RFDETRModel(makeDetectionSpec("RFDETRBase", 560, 14, 4, 384, 6, 256, 3, 8, 16, 2, {2, 5, 8, 11}, {"P4"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_base";
    }
};

class RFDETRNano : public RFDETRModel
{
public:
    RFDETRNano()
        : RFDETRModel(makeDetectionSpec("RFDETRNano", 384, 16, 2, 384, 6, 256, 2, 8, 16, 2, {3, 6, 9, 12}, {"P4"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_nano";
    }
};

class RFDETRSmall : public RFDETRModel
{
public:
    RFDETRSmall()
        : RFDETRModel(makeDetectionSpec("RFDETRSmall", 512, 16, 2, 384, 6, 256, 3, 8, 16, 2, {3, 6, 9, 12}, {"P4"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_small";
    }
};

class RFDETRMedium : public RFDETRModel
{
public:
    RFDETRMedium()
        : RFDETRModel(makeDetectionSpec("RFDETRMedium", 576, 16, 2, 384, 6, 256, 4, 8, 16, 2, {3, 6, 9, 12}, {"P4"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_medium";
    }
};

class RFDETRLarge : public RFDETRModel
{
public:
    RFDETRLarge()
        : RFDETRModel(makeDetectionSpec("RFDETRLarge", 704, 16, 2, 384, 6, 256, 4, 8, 16, 2, {3, 6, 9, 12}, {"P4"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_large";
    }
};

class RFDETRLargeDeprecated : public RFDETRModel
{
public:
    RFDETRLargeDeprecated()
        : RFDETRModel(makeDetectionSpec("RFDETRLargeDeprecated", 560, 14, 4, 768, 12, 384, 3, 12, 24, 4, {2, 5, 8, 11},
                                        {"P3", "P5"}))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_large_deprecated";
    }
};

class RFDETRSegPreview : public RFDETRModel
{
public:
    RFDETRSegPreview()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegPreview", 432, 12, 2, 4, 200, 200))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_preview";
    }
};

class RFDETRSegNano : public RFDETRModel
{
public:
    RFDETRSegNano()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegNano", 312, 12, 1, 4, 100, 100))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_nano";
    }
};

class RFDETRSegSmall : public RFDETRModel
{
public:
    RFDETRSegSmall()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegSmall", 384, 12, 2, 4, 100, 100))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_small";
    }
};

class RFDETRSegMedium : public RFDETRModel
{
public:
    RFDETRSegMedium()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegMedium", 432, 12, 2, 5, 200, 200))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_medium";
    }
};

class RFDETRSegLarge : public RFDETRModel
{
public:
    RFDETRSegLarge()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegLarge", 504, 12, 2, 5, 200, 200))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_large";
    }
};

class RFDETRSegXLarge : public RFDETRModel
{
public:
    RFDETRSegXLarge()
        : RFDETRModel(makeSegmentationSpec("RFDETRSegXLarge", 624, 12, 2, 6, 300, 300))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_xlarge";
    }
};

class RFDETRSeg2XLarge : public RFDETRModel
{
public:
    RFDETRSeg2XLarge()
        : RFDETRModel(makeSegmentationSpec("RFDETRSeg2XLarge", 768, 12, 2, 6, 300, 300))
    {
    }

    static const char *key() noexcept
    {
        return "rfdetr_seg_2xlarge";
    }
};

#define INFERRT_RFDETR_ALIAS_CLASS(CLASS_NAME, BASE_CLASS, KEY_LITERAL) \
    class CLASS_NAME : public BASE_CLASS                                \
    {                                                                   \
    public:                                                             \
        static const char *key() noexcept                               \
        {                                                               \
            return KEY_LITERAL;                                         \
        }                                                               \
    }

INFERRT_RFDETR_ALIAS_CLASS(RFDETRBaseHyphenAlias, RFDETRBase, "rfdetr-base");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRNanoHyphenAlias, RFDETRNano, "rfdetr-nano");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSmallHyphenAlias, RFDETRSmall, "rfdetr-small");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRMediumHyphenAlias, RFDETRMedium, "rfdetr-medium");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRLargeHyphenAlias, RFDETRLarge, "rfdetr-large");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRLargeDeprecatedHyphenAlias, RFDETRLargeDeprecated, "rfdetr-large-deprecated");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegPreviewHyphenAlias, RFDETRSegPreview, "rfdetr-seg-preview");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegNanoHyphenAlias, RFDETRSegNano, "rfdetr-seg-nano");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegSmallHyphenAlias, RFDETRSegSmall, "rfdetr-seg-small");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegMediumHyphenAlias, RFDETRSegMedium, "rfdetr-seg-medium");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegLargeHyphenAlias, RFDETRSegLarge, "rfdetr-seg-large");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegXLargeHyphenAlias, RFDETRSegXLarge, "rfdetr-seg-xlarge");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSeg2XLargeHyphenAlias, RFDETRSeg2XLarge, "rfdetr-seg-2xlarge");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegXXLargeAlias, RFDETRSeg2XLarge, "rfdetr_seg_xxlarge");
INFERRT_RFDETR_ALIAS_CLASS(RFDETRSegXXLargeHyphenAlias, RFDETRSeg2XLarge, "rfdetr-seg-xxlarge");

} // namespace

void RFDETRModel::normalizeModelConfig(IModelConfig &config) const
{
    if (usesDefaultImageNetInputShape(config))
    {
        config.setInputShape(nvinfer1::Dims4{1, 3, spec_.resolution, spec_.resolution});
    }
    if (config.numClasses() == 1000)
    {
        config.setNumClasses(kRFDETRDefaultClasses);
    }
    if (config.outputTensorNames() == std::vector<std::string>{"output"})
    {
        config.setOutputTensorNames(spec_.segmentation ? std::vector<std::string>{"dets", "labels", "masks"}
                                                       : std::vector<std::string>{"dets", "labels"});
    }
}

void RFDETRModel::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }

    const auto geometry                = resolveGeometry(spec_, modelConfig());
    const int  classes_with_background = modelConfig().numClasses() + 1;

    priv::IModelImpl::NamedTensorMap named_tensors;
    const auto backbone_features = addDINOBackbone(*this, network, weights_map, spec_, geometry, named_tensors);
    const auto levels = addProjector(network, weights_map, backbone_features, spec_, geometry, named_tensors);
    for (size_t i = 0; i < backbone_features.size(); ++i)
    {
        named_tensors["backbone.feature" + std::to_string(i)] = backbone_features[i];
    }
    for (size_t i = 0; i < levels.size(); ++i)
    {
        named_tensors["projector.level" + std::to_string(i)] = levels[i].tensor;
    }

    nvinfer1::ITensor *memory    = nullptr;
    nvinfer1::ITensor *pos       = nullptr;
    nvinfer1::ITensor *proposals = nullptr;
    addDecoderInputs(network, levels, spec_, memory, pos, proposals);
    named_tensors["memory"]    = memory;
    named_tensors["pos"]       = pos;
    named_tensors["proposals"] = proposals;

    nvinfer1::ITensor *query     = nullptr;
    nvinfer1::ITensor *refpoints = nullptr;
    addTwoStageQueries(network, weights_map, *memory, *proposals, spec_, classes_with_background, query, refpoints,
                       named_tensors);
    named_tensors["query"]     = query;
    named_tensors["refpoints"] = refpoints;

    nvinfer1::ITensor *head_refpoints = nullptr;
    auto *hs = addDecoder(network, weights_map, *query, *memory, *pos, *refpoints, levels, spec_, head_refpoints);
    named_tensors["decoder"]           = hs;
    named_tensors["refpoints.decoder"] = head_refpoints;

    nvinfer1::ITensor *boxes  = nullptr;
    nvinfer1::ITensor *logits = nullptr;
    addDetectionHeads(network, weights_map, *hs, *head_refpoints, spec_, classes_with_background, boxes, logits);
    named_tensors["dets"]   = boxes;
    named_tensors["labels"] = logits;

    if (spec_.segmentation)
    {
        auto *masks = addSegmentationHead(network, weights_map, *levels.front().tensor, *hs, spec_, geometry);
        named_tensors["masks"] = masks;
        if (modelConfig().featureOnly())
        {
            markFeatureOutputTensors(network, named_tensors);
            return;
        }
        markOutputTensors(network, {boxes, logits, masks});
        return;
    }

    if (modelConfig().featureOnly())
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    markOutputTensors(network, {boxes, logits});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(RFDETRBase)
INFERRT_REGISTER_MODEL(RFDETRBaseHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRNano)
INFERRT_REGISTER_MODEL(RFDETRNanoHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSmall)
INFERRT_REGISTER_MODEL(RFDETRSmallHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRMedium)
INFERRT_REGISTER_MODEL(RFDETRMediumHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRLarge)
INFERRT_REGISTER_MODEL(RFDETRLargeHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRLargeDeprecated)
INFERRT_REGISTER_MODEL(RFDETRLargeDeprecatedHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegPreview)
INFERRT_REGISTER_MODEL(RFDETRSegPreviewHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegNano)
INFERRT_REGISTER_MODEL(RFDETRSegNanoHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegSmall)
INFERRT_REGISTER_MODEL(RFDETRSegSmallHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegMedium)
INFERRT_REGISTER_MODEL(RFDETRSegMediumHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegLarge)
INFERRT_REGISTER_MODEL(RFDETRSegLargeHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegXLarge)
INFERRT_REGISTER_MODEL(RFDETRSegXLargeHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSeg2XLarge)
INFERRT_REGISTER_MODEL(RFDETRSeg2XLargeHyphenAlias)
INFERRT_REGISTER_MODEL(RFDETRSegXXLargeAlias)
INFERRT_REGISTER_MODEL(RFDETRSegXXLargeHyphenAlias)
