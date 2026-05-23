#include "ViT.hpp"

#include <NvInfer.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace irt::model {
namespace {

using E = nvinfer1::ElementWiseOperation;
using M = nvinfer1::MatrixOperation;

/**
 * @brief ViT 权重命名格式。
 */
enum class WeightLayout
{
    Timm,        ///< timm: `patch_embed.proj.*`、`blocks.N.*`。
    HuggingFace, ///< tensorrtx/HF: `vit.embeddings.*`、`vit.encoder.layer.N.*`。
};

/**
 * @brief ViT 构建期派生出来的输入尺寸信息。
 */
struct InputGeometry
{
    int batch;       ///< 固定 batch 大小，当前手写网络要求为 1。
    int channels;    ///< 输入通道数，应为 3。
    int height;      ///< 输入高。
    int width;       ///< 输入宽。
    int grid_h;      ///< patch 网格高。
    int grid_w;      ///< patch 网格宽。
    int num_patches; ///< patch token 数量。
    int num_tokens;  ///< class token + patch token 数量。
};

/**
 * @brief 判断权重表中是否存在指定 key。
 * @param weights_map 权重表。
 * @param key 权重名称。
 * @return 存在返回 true。
 */
bool hasWeight(const WeightsMap &weights_map, const std::string &key)
{
    return weights_map.find(key) != weights_map.end();
}

/**
 * @brief 按期望元素数量读取权重。
 * @param weights_map 权重表。
 * @param key 权重名称。
 * @param expected_count 期望元素数量；负数表示不校验数量。
 * @return TensorRT 权重对象。
 */
const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key,
                                       int64_t expected_count = -1)
{
    const auto it = weights_map.find(key);
    if (it == weights_map.end())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Missing ViT weight: %s", key.c_str());
    }

    if (expected_count >= 0 && it->second.count != expected_count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Unexpected ViT weight element count for %s: got %lld, expected %lld", key.c_str(),
                             static_cast<long long>(it->second.count), static_cast<long long>(expected_count));
    }

    return it->second;
}

/**
 * @brief 根据权重 key 判断当前 `.wts` 来自 timm 还是 HuggingFace/tensorrtx。
 * @param weights_map 权重表。
 * @return 权重命名格式。
 */
WeightLayout detectWeightLayout(const WeightsMap &weights_map)
{
    if (hasWeight(weights_map, "patch_embed.proj.weight"))
    {
        return WeightLayout::Timm;
    }
    if (hasWeight(weights_map, "vit.embeddings.patch_embeddings.projection.weight"))
    {
        return WeightLayout::HuggingFace;
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                         "Unsupported ViT weight layout: expected timm or HuggingFace/tensorrtx keys");
}

/**
 * @brief 选择指定布局下的 patch embedding 权重前缀。
 */
std::string patchPrefix(WeightLayout layout)
{
    return layout == WeightLayout::Timm ? "patch_embed.proj" : "vit.embeddings.patch_embeddings.projection";
}

/**
 * @brief 选择指定布局下的 class token 权重名称。
 */
std::string classTokenKey(WeightLayout layout)
{
    return layout == WeightLayout::Timm ? "cls_token" : "vit.embeddings.cls_token";
}

/**
 * @brief 选择指定布局下的位置编码权重名称。
 */
std::string positionEmbeddingKey(WeightLayout layout)
{
    return layout == WeightLayout::Timm ? "pos_embed" : "vit.embeddings.position_embeddings";
}

/**
 * @brief 选择指定布局下的第 index 个 block 前缀。
 */
std::string blockPrefix(WeightLayout layout, int index)
{
    return layout == WeightLayout::Timm ? "blocks." + std::to_string(index)
                                        : "vit.encoder.layer." + std::to_string(index);
}

/**
 * @brief 选择指定布局下的最终 LayerNorm 前缀。
 */
std::string finalNormPrefix(WeightLayout layout)
{
    return layout == WeightLayout::Timm ? "norm" : "vit.layernorm";
}

/**
 * @brief 选择指定布局下的分类头前缀。
 */
std::string headPrefix(WeightLayout layout)
{
    return layout == WeightLayout::Timm ? "head" : "classifier";
}

/**
 * @brief 构造一个静态生命周期的标量权重。
 *
 * TensorRT 网络构建期间需要权重指针保持有效。这里使用进程静态存储保存小标量，
 * 避免临时变量生命周期问题。
 *
 * @param value 标量值。
 * @return TensorRT float 标量权重。
 */
nvinfer1::Weights ownedScalarWeight(float value)
{
    static std::vector<std::unique_ptr<float>> scalars;
    scalars.push_back(std::make_unique<float>(value));
    return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, scalars.back().get(), 1};
}

/**
 * @brief 生成与输入同 rank、每维为 1 的标量广播维度。
 * @param input 参考张量。
 * @return 广播用维度。
 */
nvinfer1::Dims scalarDimsLike(const nvinfer1::ITensor &input)
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

/**
 * @brief 添加 GeLU 激活的 tanh 近似实现。
 *
 * 公式与 PyTorch/timm 中常见近似一致：
 * `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`。
 *
 * @param network TensorRT 网络定义。
 * @param input 输入张量。
 * @return GeLU 输出张量。
 */
nvinfer1::ITensor *addGelu(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    const auto scalar_dims = scalarDimsLike(input);
    auto *half = network->addConstant(scalar_dims, ownedScalarWeight(0.5F));
    auto *one = network->addConstant(scalar_dims, ownedScalarWeight(1.0F));
    auto *sqrt_2_div_pi = network->addConstant(scalar_dims, ownedScalarWeight(std::sqrt(2.0F / 3.14159265358979323846F)));
    auto *coeff = network->addConstant(scalar_dims, ownedScalarWeight(0.044715F));

    auto *x2 = network->addElementWise(input, input, E::kPROD);
    auto *x3 = network->addElementWise(*x2->getOutput(0), input, E::kPROD);
    auto *scaled_x3 = network->addElementWise(*x3->getOutput(0), *coeff->getOutput(0), E::kPROD);
    auto *inner = network->addElementWise(input, *scaled_x3->getOutput(0), E::kSUM);
    auto *scaled = network->addElementWise(*inner->getOutput(0), *sqrt_2_div_pi->getOutput(0), E::kPROD);
    auto *tanh = network->addActivation(*scaled->getOutput(0), nvinfer1::ActivationType::kTANH);
    auto *one_plus_tanh = network->addElementWise(*tanh->getOutput(0), *one->getOutput(0), E::kSUM);
    auto *half_x = network->addElementWise(input, *half->getOutput(0), E::kPROD);
    return network->addElementWise(*half_x->getOutput(0), *one_plus_tanh->getOutput(0), E::kPROD)->getOutput(0);
}

/**
 * @brief 添加最后一维上的 LayerNorm。
 * @param network TensorRT 网络定义。
 * @param weights_map 权重表。
 * @param input 输入张量，形状为 NLC。
 * @param prefix LayerNorm 权重前缀。
 * @param embed_dim embedding 维度。
 * @param epsilon LayerNorm epsilon。
 * @return LayerNorm 输出张量。
 */
nvinfer1::ITensor *addLayerNorm(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, const std::string &prefix, int embed_dim, float epsilon)
{
    const auto scale_weights = requireWeight(weights_map, prefix + ".weight", embed_dim);
    const auto bias_weights  = requireWeight(weights_map, prefix + ".bias", embed_dim);
    auto *scale = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, scale_weights);
    auto *bias  = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim}, bias_weights);

    const auto dims = input.getDimensions();
    const auto axes = 1U << static_cast<uint32_t>(dims.nbDims - 1);
#if TRT_VERSION >= 11500
    auto *norm = network->addNormalizationV2(input, *scale->getOutput(0), *bias->getOutput(0), axes);
#else
    auto *norm = network->addNormalization(input, *scale->getOutput(0), *bias->getOutput(0), axes);
#endif
    norm->setEpsilon(epsilon);
    return norm->getOutput(0);
}

/**
 * @brief 添加面向 NLC 张量的全连接层。
 * @param network TensorRT 网络定义。
 * @param weights_map 权重表。
 * @param input 输入张量，形状为 `[N, L, in_features]`。
 * @param prefix 线性层权重前缀。
 * @param in_features 输入特征数。
 * @param out_features 输出特征数。
 * @param bias_required 是否强制要求 bias。
 * @return 线性层输出张量。
 */
nvinfer1::ITensor *addLinear3D(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                               nvinfer1::ITensor &input, const std::string &prefix, int in_features,
                               int out_features, bool bias_required = true)
{
    auto *weight = network
                       ->addConstant(nvinfer1::Dims3{1, out_features, in_features},
                                     requireWeight(weights_map, prefix + ".weight",
                                                   static_cast<int64_t>(out_features) * in_features))
                       ->getOutput(0);
    auto *matmul = network->addMatrixMultiply(input, M::kNONE, *weight, M::kTRANSPOSE);
    auto *output = matmul->getOutput(0);

    const auto bias_key = prefix + ".bias";
    if (hasWeight(weights_map, bias_key))
    {
        auto *bias = network
                         ->addConstant(nvinfer1::Dims3{1, 1, out_features},
                                       requireWeight(weights_map, bias_key, out_features))
                         ->getOutput(0);
        output = network->addElementWise(*output, *bias, E::kSUM)->getOutput(0);
    }
    else if (bias_required)
    {
        requireWeight(weights_map, bias_key, out_features);
    }

    return output;
}

/**
 * @brief 添加 timm 合并 qkv 权重格式的注意力输入投影。
 */
void addTimmQkv(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &input,
                int batch, int num_tokens, int embed_dim, nvinfer1::ITensor *&q, nvinfer1::ITensor *&k,
                nvinfer1::ITensor *&v, const std::string &prefix)
{
    auto *qkv = addLinear3D(network, weights_map, input, prefix + ".attn.qkv", embed_dim, 3 * embed_dim, true);
    q = network
            ->addSlice(*qkv, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{batch, num_tokens, embed_dim},
                       nvinfer1::Dims3{1, 1, 1})
            ->getOutput(0);
    k = network
            ->addSlice(*qkv, nvinfer1::Dims3{0, 0, embed_dim}, nvinfer1::Dims3{batch, num_tokens, embed_dim},
                       nvinfer1::Dims3{1, 1, 1})
            ->getOutput(0);
    v = network
            ->addSlice(*qkv, nvinfer1::Dims3{0, 0, 2 * embed_dim}, nvinfer1::Dims3{batch, num_tokens, embed_dim},
                       nvinfer1::Dims3{1, 1, 1})
            ->getOutput(0);
}

/**
 * @brief 添加 HuggingFace/tensorrtx 分离 q/k/v 权重格式的注意力输入投影。
 */
void addHuggingFaceQkv(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                       nvinfer1::ITensor &input, int embed_dim, nvinfer1::ITensor *&q, nvinfer1::ITensor *&k,
                       nvinfer1::ITensor *&v, const std::string &prefix)
{
    const auto attention_prefix = prefix + ".attention.attention";
    q = addLinear3D(network, weights_map, input, attention_prefix + ".query", embed_dim, embed_dim, true);
    k = addLinear3D(network, weights_map, input, attention_prefix + ".key", embed_dim, embed_dim, true);
    v = addLinear3D(network, weights_map, input, attention_prefix + ".value", embed_dim, embed_dim, true);
}

/**
 * @brief 将 `[N, L, D]` 投影结果 reshape 成 `[N, H, L, head_dim]`。
 */
nvinfer1::ITensor *reshapeToHeads(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                  int num_tokens, int num_heads, int head_dim)
{
    auto *shuffle = network->addShuffle(input);
    shuffle->setReshapeDimensions(nvinfer1::Dims4{batch, num_tokens, num_heads, head_dim});
    shuffle->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    return shuffle->getOutput(0);
}

/**
 * @brief 添加多头自注意力计算和输出投影。
 * @param network TensorRT 网络定义。
 * @param weights_map 权重表。
 * @param input 输入 token，形状为 NLC。
 * @param layout 权重命名格式。
 * @param prefix block 前缀。
 * @param geometry 输入几何信息。
 * @param spec ViT 结构参数。
 * @return 注意力输出张量，形状为 NLC。
 */
nvinfer1::ITensor *addAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, WeightLayout layout, const std::string &prefix,
                                const InputGeometry &geometry, const VisionTransformerSpec &spec)
{
    const int head_dim = spec.embed_dim / spec.num_heads;
    nvinfer1::ITensor *q = nullptr;
    nvinfer1::ITensor *k = nullptr;
    nvinfer1::ITensor *v = nullptr;

    if (layout == WeightLayout::Timm)
    {
        addTimmQkv(network, weights_map, input, geometry.batch, geometry.num_tokens, spec.embed_dim, q, k, v, prefix);
    }
    else
    {
        addHuggingFaceQkv(network, weights_map, input, spec.embed_dim, q, k, v, prefix);
    }

    auto *q_heads = reshapeToHeads(network, *q, geometry.batch, geometry.num_tokens, spec.num_heads, head_dim);
    auto *k_heads = reshapeToHeads(network, *k, geometry.batch, geometry.num_tokens, spec.num_heads, head_dim);
    auto *v_heads = reshapeToHeads(network, *v, geometry.batch, geometry.num_tokens, spec.num_heads, head_dim);

    auto *qk = network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE);
    auto *scale = network->addConstant(nvinfer1::Dims4{1, 1, 1, 1}, ownedScalarWeight(1.0F / std::sqrt(static_cast<float>(head_dim))));
    auto *scaled_qk = network->addElementWise(*qk->getOutput(0), *scale->getOutput(0), E::kPROD);
    auto *softmax = network->addSoftMax(*scaled_qk->getOutput(0));
    softmax->setAxes(1U << static_cast<uint32_t>(scaled_qk->getOutput(0)->getDimensions().nbDims - 1));

    auto *attended = network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE);
    auto *merge_heads = network->addShuffle(*attended->getOutput(0));
    merge_heads->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    merge_heads->setReshapeDimensions(nvinfer1::Dims3{geometry.batch, geometry.num_tokens, spec.embed_dim});

    const auto proj_prefix = layout == WeightLayout::Timm ? prefix + ".attn.proj" : prefix + ".attention.output.dense";
    return addLinear3D(network, weights_map, *merge_heads->getOutput(0), proj_prefix, spec.embed_dim, spec.embed_dim,
                       true);
}

/**
 * @brief 添加一个标准 Pre-LN ViT Transformer block。
 */
nvinfer1::ITensor *addTransformerBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, WeightLayout layout, int index,
                                       const InputGeometry &geometry, const VisionTransformerSpec &spec,
                                       float norm_epsilon)
{
    const auto prefix = blockPrefix(layout, index);
    const auto norm1_prefix = layout == WeightLayout::Timm ? prefix + ".norm1" : prefix + ".layernorm_before";
    const auto norm2_prefix = layout == WeightLayout::Timm ? prefix + ".norm2" : prefix + ".layernorm_after";

    auto *norm1 = addLayerNorm(network, weights_map, input, norm1_prefix, spec.embed_dim, norm_epsilon);
    auto *attn = addAttention(network, weights_map, *norm1, layout, prefix, geometry, spec);
    auto *attn_residual = network->addElementWise(input, *attn, E::kSUM)->getOutput(0);

    auto *norm2 = addLayerNorm(network, weights_map, *attn_residual, norm2_prefix, spec.embed_dim, norm_epsilon);

    const int mlp_hidden = static_cast<int>(std::lround(static_cast<float>(spec.embed_dim) * spec.mlp_ratio));
    const auto fc1_prefix = layout == WeightLayout::Timm ? prefix + ".mlp.fc1" : prefix + ".intermediate.dense";
    const auto fc2_prefix = layout == WeightLayout::Timm ? prefix + ".mlp.fc2" : prefix + ".output.dense";
    auto *fc1 = addLinear3D(network, weights_map, *norm2, fc1_prefix, spec.embed_dim, mlp_hidden, true);
    auto *gelu = addGelu(network, *fc1);
    auto *fc2 = addLinear3D(network, weights_map, *gelu, fc2_prefix, mlp_hidden, spec.embed_dim, true);

    return network->addElementWise(*attn_residual, *fc2, E::kSUM)->getOutput(0);
}

/**
 * @brief 校验 ViT 输入配置并计算 patch/token 数量。
 */
InputGeometry resolveInputGeometry(const VisionTransformerSpec &spec, const IModelConfig &config)
{
    const auto &shape = config.inputShape();
    InputGeometry geometry{};
    geometry.batch    = static_cast<int>(shape.d[0]);
    geometry.channels = static_cast<int>(shape.d[1]);
    geometry.height   = static_cast<int>(shape.d[2]);
    geometry.width    = static_cast<int>(shape.d[3]);

    if (geometry.batch != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "ViT handwritten TensorRT network currently requires batch=1, got %d", geometry.batch);
    }
    if (geometry.channels != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "ViT requires 3 input channels, got %d",
                             geometry.channels);
    }
    if (geometry.height % spec.patch_size != 0 || geometry.width % spec.patch_size != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "ViT input H/W must be divisible by patch size %d, got H=%d W=%d", spec.patch_size,
                             geometry.height, geometry.width);
    }
    if (spec.embed_dim % spec.num_heads != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "ViT embed_dim (%d) must be divisible by num_heads (%d)",
                             spec.embed_dim, spec.num_heads);
    }

    geometry.grid_h      = geometry.height / spec.patch_size;
    geometry.grid_w      = geometry.width / spec.patch_size;
    geometry.num_patches = geometry.grid_h * geometry.grid_w;
    geometry.num_tokens  = geometry.num_patches + 1;
    return geometry;
}

/**
 * @brief 构建 patch embedding、class token 和位置编码。
 */
nvinfer1::ITensor *addPatchAndPositionEmbedding(const VisionTransformer &impl, nvinfer1::INetworkDefinition *network,
                                                const WeightsMap &weights_map, WeightLayout layout,
                                                const InputGeometry &geometry,
                                                const VisionTransformerSpec &spec,
                                                priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *input = impl.addInputTensor(network);
    named_tensors["input"] = input;

    const auto patch_prefix = patchPrefix(layout);
    auto *patch = network->addConvolutionNd(
        *input, spec.embed_dim, nvinfer1::DimsHW{spec.patch_size, spec.patch_size},
        requireWeight(weights_map, patch_prefix + ".weight",
                      static_cast<int64_t>(spec.embed_dim) * geometry.channels * spec.patch_size * spec.patch_size),
        requireWeight(weights_map, patch_prefix + ".bias", spec.embed_dim));
    patch->setStrideNd(nvinfer1::DimsHW{spec.patch_size, spec.patch_size});

    auto *flatten = network->addShuffle(*patch->getOutput(0));
    flatten->setReshapeDimensions(nvinfer1::Dims3{geometry.batch, spec.embed_dim, geometry.num_patches});
    flatten->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    named_tensors["patch_embed"] = flatten->getOutput(0);

    auto *cls_token = network
                          ->addConstant(nvinfer1::Dims3{1, 1, spec.embed_dim},
                                        requireWeight(weights_map, classTokenKey(layout), spec.embed_dim))
                          ->getOutput(0);
    const std::vector<nvinfer1::ITensor *> tokens{cls_token, flatten->getOutput(0)};
    auto *concat = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
    concat->setAxis(1);

    auto *pos_embed = network
                          ->addConstant(nvinfer1::Dims3{1, geometry.num_tokens, spec.embed_dim},
                                        requireWeight(weights_map, positionEmbeddingKey(layout),
                                                      static_cast<int64_t>(geometry.num_tokens) * spec.embed_dim))
                          ->getOutput(0);
    auto *position_added = network->addElementWise(*concat->getOutput(0), *pos_embed, E::kSUM)->getOutput(0);
    named_tensors["tokens"] = position_added;
    return position_added;
}

} // namespace

void VisionTransformer::normalizeModelConfig(IModelConfig &config) const
{
    constexpr int kDefaultImageSize = 224;

    if (spec_.image_size == kDefaultImageSize || config.inputShapes().size() != 1)
    {
        return;
    }

    const auto &shape = config.inputShape();
    const bool  is_default_image_config
        = shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageSize && shape.d[3] == kDefaultImageSize;
    if (!is_default_image_config)
    {
        return;
    }

    // 当调用方没有显式设置输入尺寸时，使用变体名携带的默认分辨率，避免 pos_embed 数量不匹配。
    config.setInputShape(nvinfer1::Dims4{1, 3, spec_.image_size, spec_.image_size});
}

void VisionTransformer::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }

    const auto layout = detectWeightLayout(weights_map);
    const auto geometry = resolveInputGeometry(spec_, modelConfig());
    const float norm_epsilon = layout == WeightLayout::HuggingFace ? 1e-12F : 1e-6F;
    const bool feature_only = isBuildingFeatureEngine();

    priv::IModelImpl::NamedTensorMap named_tensors;
    auto *x = addPatchAndPositionEmbedding(*this, network, weights_map, layout, geometry, spec_, named_tensors);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    for (int i = 0; i < spec_.depth; ++i)
    {
        x = addTransformerBlock(network, weights_map, *x, layout, i, geometry, spec_, norm_epsilon);
        named_tensors["block" + std::to_string(i)] = x;
        named_tensors["blocks." + std::to_string(i)] = x;
        if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
        {
            return;
        }
    }

    x = addLayerNorm(network, weights_map, *x, finalNormPrefix(layout), spec_.embed_dim, norm_epsilon);
    named_tensors["norm"] = x;
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    auto *cls = network
                    ->addSlice(*x, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{geometry.batch, 1, spec_.embed_dim},
                               nvinfer1::Dims3{1, 1, 1})
                    ->getOutput(0);
    auto *pre_logits = network->addShuffle(*cls);
    pre_logits->setReshapeDimensions(nvinfer1::Dims2{geometry.batch, spec_.embed_dim});
    named_tensors["cls"] = pre_logits->getOutput(0);
    named_tensors["pre_logits"] = pre_logits->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    auto *head = addLinear3D(network, weights_map, *cls, headPrefix(layout), spec_.embed_dim, modelConfig().numClasses(),
                             true);
    auto *logits = network->addShuffle(*head);
    logits->setReshapeDimensions(nvinfer1::Dims2{geometry.batch, modelConfig().numClasses()});
    named_tensors["logits"] = logits->getOutput(0);
    if (feature_only)
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    markOutputTensors(network, {logits->getOutput(0)});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(ViT)
INFERRT_REGISTER_MODEL(ViTTinyPatch16_224)
INFERRT_REGISTER_MODEL(ViTTinyPatch16_384)
INFERRT_REGISTER_MODEL(ViTSmallPatch32_224)
INFERRT_REGISTER_MODEL(ViTSmallPatch32_384)
INFERRT_REGISTER_MODEL(ViTSmallPatch16_224)
INFERRT_REGISTER_MODEL(ViTSmallPatch16_384)
INFERRT_REGISTER_MODEL(ViTSmallPatch8_224)
INFERRT_REGISTER_MODEL(ViTBasePatch32_224)
INFERRT_REGISTER_MODEL(ViTBasePatch32_384)
INFERRT_REGISTER_MODEL(ViTBasePatch16_224)
INFERRT_REGISTER_MODEL(ViTBasePatch16_384)
INFERRT_REGISTER_MODEL(ViTBasePatch8_224)
INFERRT_REGISTER_MODEL(ViTLargePatch32_224)
INFERRT_REGISTER_MODEL(ViTLargePatch32_384)
INFERRT_REGISTER_MODEL(ViTLargePatch16_224)
INFERRT_REGISTER_MODEL(ViTLargePatch16_384)
INFERRT_REGISTER_MODEL(ViTLargePatch14_224)
INFERRT_REGISTER_MODEL(ViTHugePatch14_224)
INFERRT_REGISTER_MODEL(ViTGiantPatch14_224)
INFERRT_REGISTER_MODEL(ViTGiganticPatch14_224)
