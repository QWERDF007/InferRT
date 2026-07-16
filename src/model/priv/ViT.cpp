#include "ViT.hpp"

#include "Layers.hpp"

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

constexpr const char *kViTTag = "ViT";

inline const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key,
                                              int64_t expected_count = -1)
{
    return irt::model::requireWeight(weights_map, key, kViTTag, expected_count);
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
 * @brief 添加 timm 合并 qkv 权重格式的注意力输入投影。
 */
void addTimmQkv(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &input,
                int batch, int num_tokens, int embed_dim, nvinfer1::ITensor *&q, nvinfer1::ITensor *&k,
                nvinfer1::ITensor *&v, const std::string &prefix)
{
    auto *qkv = addLinear3D(network, weights_map, input, prefix + ".attn.qkv", embed_dim, 3 * embed_dim, true);
    splitQkv(network, *qkv, batch, num_tokens, embed_dim, q, k, v);
}

/**
 * @brief 添加 HuggingFace/tensorrtx 分离 q/k/v 权重格式的注意力输入投影。
 */
void addHuggingFaceQkv(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &input,
                       int embed_dim, nvinfer1::ITensor *&q, nvinfer1::ITensor *&k, nvinfer1::ITensor *&v,
                       const std::string &prefix)
{
    const auto attention_prefix = prefix + ".attention.attention";
    q = addLinear3D(network, weights_map, input, attention_prefix + ".query", embed_dim, embed_dim, true);
    k = addLinear3D(network, weights_map, input, attention_prefix + ".key", embed_dim, embed_dim, true);
    v = addLinear3D(network, weights_map, input, attention_prefix + ".value", embed_dim, embed_dim, true);
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
    const int          head_dim = spec.embed_dim / spec.num_heads;
    nvinfer1::ITensor *q        = nullptr;
    nvinfer1::ITensor *k        = nullptr;
    nvinfer1::ITensor *v        = nullptr;

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

    auto *qk        = network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE);
    auto *scale     = network->addConstant(nvinfer1::Dims4{1, 1, 1, 1},
                                           ownedScalarWeight(1.0F / std::sqrt(static_cast<float>(head_dim))));
    auto *scaled_qk = network->addElementWise(*qk->getOutput(0), *scale->getOutput(0), E::kPROD);
    auto *softmax   = network->addSoftMax(*scaled_qk->getOutput(0));
    softmax->setAxes(1U << static_cast<uint32_t>(scaled_qk->getOutput(0)->getDimensions().nbDims - 1));

    auto *attended = network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE);
    auto *attended_output
        = mergeHeads(network, *attended->getOutput(0), geometry.batch, geometry.num_tokens, spec.embed_dim);

    const auto proj_prefix = layout == WeightLayout::Timm ? prefix + ".attn.proj" : prefix + ".attention.output.dense";
    return addLinear3D(network, weights_map, *attended_output, proj_prefix, spec.embed_dim, spec.embed_dim, true);
}

/**
 * @brief 添加一个标准 Pre-LN ViT Transformer block。
 */
nvinfer1::ITensor *addTransformerBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, WeightLayout layout, int index,
                                       const InputGeometry &geometry, const VisionTransformerSpec &spec,
                                       float norm_epsilon)
{
    const auto prefix       = blockPrefix(layout, index);
    const auto norm1_prefix = layout == WeightLayout::Timm ? prefix + ".norm1" : prefix + ".layernorm_before";
    const auto norm2_prefix = layout == WeightLayout::Timm ? prefix + ".norm2" : prefix + ".layernorm_after";

    auto *norm1         = addLayerNorm(network, weights_map, input, norm1_prefix, spec.embed_dim, norm_epsilon);
    auto *attn          = addAttention(network, weights_map, *norm1, layout, prefix, geometry, spec);
    auto *attn_residual = network->addElementWise(input, *attn, E::kSUM)->getOutput(0);

    auto *norm2 = addLayerNorm(network, weights_map, *attn_residual, norm2_prefix, spec.embed_dim, norm_epsilon);

    const int  mlp_hidden = static_cast<int>(std::lround(static_cast<float>(spec.embed_dim) * spec.mlp_ratio));
    const auto fc1_prefix = layout == WeightLayout::Timm ? prefix + ".mlp.fc1" : prefix + ".intermediate.dense";
    const auto fc2_prefix = layout == WeightLayout::Timm ? prefix + ".mlp.fc2" : prefix + ".output.dense";
    auto      *fc1        = addLinear3D(network, weights_map, *norm2, fc1_prefix, spec.embed_dim, mlp_hidden, true);
    auto      *gelu       = addGeluExact(network, *fc1);
    auto      *fc2        = addLinear3D(network, weights_map, *gelu, fc2_prefix, mlp_hidden, spec.embed_dim, true);

    return network->addElementWise(*attn_residual, *fc2, E::kSUM)->getOutput(0);
}

/**
 * @brief 校验 ViT 输入配置并计算 patch/token 数量。
 */
InputGeometry resolveInputGeometry(const VisionTransformerSpec &spec, const IModelConfig &config)
{
    const auto   &shape = config.inputShape();
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
                                                const InputGeometry &geometry, const VisionTransformerSpec &spec,
                                                priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *input            = impl.addInputTensor(network);
    named_tensors["input"] = input;

    const auto patch_prefix = patchPrefix(layout);
    auto      *patch        = network->addConvolutionNd(
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
    auto *position_added    = network->addElementWise(*concat->getOutput(0), *pos_embed, E::kSUM)->getOutput(0);
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

    const auto  layout       = detectWeightLayout(weights_map);
    const auto  geometry     = resolveInputGeometry(spec_, modelConfig());
    const float norm_epsilon = layout == WeightLayout::HuggingFace ? 1e-12F : 1e-6F;
    const bool  feature_only = isBuildingFeatureEngine();

    priv::IModelImpl::NamedTensorMap named_tensors;
    auto *x = addPatchAndPositionEmbedding(*this, network, weights_map, layout, geometry, spec_, named_tensors);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    for (int i = 0; i < spec_.depth; ++i)
    {
        x = addTransformerBlock(network, weights_map, *x, layout, i, geometry, spec_, norm_epsilon);
        named_tensors["block" + std::to_string(i)]   = x;
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
    named_tensors["cls"]        = pre_logits->getOutput(0);
    named_tensors["pre_logits"] = pre_logits->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    auto *head   = addLinear3D(network, weights_map, *cls, headPrefix(layout), spec_.embed_dim,
                               modelConfig().numClasses(), true);
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
