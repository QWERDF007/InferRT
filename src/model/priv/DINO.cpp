#include "DINO.hpp"
#include "Layers.hpp"

#include <NvInfer.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/ModelFactory.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace irt::model {
namespace {

using E = nvinfer1::ElementWiseOperation;
using M = nvinfer1::MatrixOperation;

/**
 * @brief DINO 构建期派生出的输入几何信息。
 */
struct DINOInputGeometry
{
    int batch;         ///< 当前手写 TensorRT 网络固定支持 batch=1。
    int channels;      ///< 输入通道数，应为 3。
    int height;        ///< 输入图像高度。
    int width;         ///< 输入图像宽度。
    int grid_h;        ///< patch 网格高度。
    int grid_w;        ///< patch 网格宽度。
    int num_patches;   ///< patch token 数量。
    int prefix_tokens; ///< cls token + register/storage token 数量。
    int num_tokens;    ///< prefix token + patch token 总数。
    int head_dim;      ///< 单个 attention head 的通道维度。
};

/**
 * @brief RoPE 的 sin/cos 常量张量。
 */
struct DINORopeConstants
{
    nvinfer1::ITensor *sin; ///< sin 位置编码，形状为 `[1, 1, HW, head_dim]`。
    nvinfer1::ITensor *cos; ///< cos 位置编码，形状为 `[1, 1, HW, head_dim]`。
};

/**
 * @brief 读取权重并按需校验元素数量。
 */
constexpr const char *kDINOTag = "DINO";

inline const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key,
                                              int64_t expected_count = -1)
{
    return irt::model::requireWeight(weights_map, key, kDINOTag, expected_count);
}

/**
 * @brief 返回第一个存在的权重 key。
 */
const nvinfer1::Weights &requireAnyWeight(const WeightsMap &weights_map,
                                          const std::vector<std::string> &candidate_keys, int64_t expected_count)
{
    for (const auto &key : candidate_keys)
    {
        if (hasWeight(weights_map, key))
        {
            return requireWeight(weights_map, key, expected_count);
        }
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Missing DINO weight: %s", candidate_keys.front().c_str());
}

/**
 * @brief 获取 float 权重中的单个元素。
 */
float weightValue(const nvinfer1::Weights &weights, int64_t index)
{
    const auto *values = static_cast<const float *>(weights.values);
    return values[index];
}

// weightValue() is DINO-specific and is kept here.

// kPi, hasWeight, requireWeight (wrapped above), ownedScalarWeight, ownedFloatVector,
// scalarDimsLike, addGeluApprox, addSilu, addLayerNorm, addLinear3D, reshapeToHeads,
// mergeHeads are now provided by Layers.hpp.

/**
 * @brief 添加不自动附加 bias 的线性层，供 qkv 手动处理官方/timm 不同偏置格式。
 */
nvinfer1::ITensor *addLinear3DNoBias(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &input, const std::string &prefix, int in_features,
                                     int out_features)
{
    auto *weight = network
                       ->addConstant(nvinfer1::Dims3{1, out_features, in_features},
                                     requireWeight(weights_map, prefix + ".weight",
                                                   static_cast<int64_t>(out_features) * in_features))
                       ->getOutput(0);
    auto *matmul = network->addMatrixMultiply(input, M::kNONE, *weight, M::kTRANSPOSE);
    return matmul->getOutput(0);
}

/**
 * @brief 添加 LayerScale 缩放；权重不存在时保持输入不变，便于兼容无 LayerScale 变体。
 */
nvinfer1::ITensor *addLayerScale(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, const std::vector<std::string> &candidate_keys,
                                 int embed_dim)
{
    for (const auto &key : candidate_keys)
    {
        if (hasWeight(weights_map, key))
        {
            auto *gamma = network->addConstant(nvinfer1::Dims3{1, 1, embed_dim},
                                               requireWeight(weights_map, key, embed_dim));
            return network->addElementWise(input, *gamma->getOutput(0), E::kPROD)->getOutput(0);
        }
    }
    return &input;
}

/**
 * @brief 将 q/v bias 与全零 k bias 拼成 timm DINOv3 qkv 偏置。
 */
nvinfer1::Weights makeSplitQkvBias(const WeightsMap &weights_map, const std::string &prefix, int embed_dim)
{
    const auto &q_bias = requireWeight(weights_map, prefix + ".attn.q_bias", embed_dim);
    const auto &v_bias = requireWeight(weights_map, prefix + ".attn.v_bias", embed_dim);

    std::vector<float> values(static_cast<size_t>(3 * embed_dim), 0.0F);
    for (int i = 0; i < embed_dim; ++i)
    {
        values[static_cast<size_t>(i)] = weightValue(q_bias, i);
        values[static_cast<size_t>(2 * embed_dim + i)] = weightValue(v_bias, i);
    }
    return ownedFloatVector(std::move(values));
}

/**
 * @brief 复制 qkv.bias，并在官方 DINOv3 mask_k_bias 场景下清零 K bias。
 */
nvinfer1::Weights makeFullQkvBias(const nvinfer1::Weights &bias, int embed_dim, bool mask_k_bias)
{
    std::vector<float> values(static_cast<size_t>(3 * embed_dim));
    for (int i = 0; i < 3 * embed_dim; ++i)
    {
        values[static_cast<size_t>(i)] = weightValue(bias, i);
    }
    if (mask_k_bias)
    {
        std::fill(values.begin() + embed_dim, values.begin() + 2 * embed_dim, 0.0F);
    }
    return ownedFloatVector(std::move(values));
}

/**
 * @brief 添加 qkv 投影，自动适配官方 qkv.bias 与 timm q_bias/v_bias。
 */
nvinfer1::ITensor *addQkvProjection(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    nvinfer1::ITensor &input, const std::string &prefix,
                                    const DINOTransformerSpec &spec)
{
    auto *qkv = addLinear3DNoBias(network, weights_map, input, prefix + ".attn.qkv", spec.embed_dim,
                                  3 * spec.embed_dim);

    const auto qkv_bias_key = prefix + ".attn.qkv.bias";
    std::optional<nvinfer1::Weights> bias_weights;
    if (hasWeight(weights_map, qkv_bias_key))
    {
        const bool mask_k_bias = spec.version == DINOVersion::V3 && hasWeight(weights_map, qkv_bias_key + "_mask");
        bias_weights = makeFullQkvBias(requireWeight(weights_map, qkv_bias_key, 3 * spec.embed_dim), spec.embed_dim,
                                       mask_k_bias);
    }
    else if (hasWeight(weights_map, prefix + ".attn.q_bias") || hasWeight(weights_map, prefix + ".attn.v_bias"))
    {
        bias_weights = makeSplitQkvBias(weights_map, prefix, spec.embed_dim);
    }

    if (!bias_weights.has_value())
    {
        return qkv;
    }

    auto *bias = network->addConstant(nvinfer1::Dims3{1, 1, 3 * spec.embed_dim}, bias_weights.value());
    return network->addElementWise(*qkv, *bias->getOutput(0), E::kSUM)->getOutput(0);
}

/**
 * @brief 根据官方 DINOv3 RoPE 公式生成 periods。
 */
std::vector<float> resolveRopePeriods(const WeightsMap &weights_map, const DINOInputGeometry &geometry)
{
    const int period_count = geometry.head_dim / 4;
    if (hasWeight(weights_map, "rope_embed.periods"))
    {
        const auto &periods = requireWeight(weights_map, "rope_embed.periods", period_count);
        std::vector<float> values(static_cast<size_t>(period_count));
        for (int i = 0; i < period_count; ++i)
        {
            values[static_cast<size_t>(i)] = weightValue(periods, i);
        }
        return values;
    }

    std::vector<float> values(static_cast<size_t>(period_count));
    for (int i = 0; i < period_count; ++i)
    {
        values[static_cast<size_t>(i)] = std::pow(100.0F, 2.0F * static_cast<float>(i)
                                                            / static_cast<float>(geometry.head_dim / 2));
    }
    return values;
}

/**
 * @brief 生成 DINOv3 RoPE 的 sin 或 cos 表。
 */
std::vector<float> makeRopeTable(const DINOInputGeometry &geometry, const std::vector<float> &periods, bool make_sin)
{
    std::vector<float> table(static_cast<size_t>(geometry.num_patches * geometry.head_dim));
    size_t             offset = 0;

    for (int y = 0; y < geometry.grid_h; ++y)
    {
        const float coord_h = 2.0F * ((static_cast<float>(y) + 0.5F) / static_cast<float>(geometry.grid_h)) - 1.0F;
        for (int x = 0; x < geometry.grid_w; ++x)
        {
            const float coord_w = 2.0F * ((static_cast<float>(x) + 0.5F) / static_cast<float>(geometry.grid_w)) - 1.0F;

            std::vector<float> half_angles;
            half_angles.reserve(static_cast<size_t>(geometry.head_dim / 2));
            for (const float period : periods)
            {
                half_angles.push_back(2.0F * kPi * coord_h / period);
            }
            for (const float period : periods)
            {
                half_angles.push_back(2.0F * kPi * coord_w / period);
            }

            for (int repeat = 0; repeat < 2; ++repeat)
            {
                (void)repeat;
                for (const float angle : half_angles)
                {
                    table[offset++] = make_sin ? std::sin(angle) : std::cos(angle);
                }
            }
        }
    }
    return table;
}

/**
 * @brief 向网络中添加 RoPE sin/cos 常量。
 */
DINORopeConstants addRopeConstants(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   const DINOInputGeometry &geometry)
{
    const auto periods = resolveRopePeriods(weights_map, geometry);
    auto *sin = network
                    ->addConstant(nvinfer1::Dims4{1, 1, geometry.num_patches, geometry.head_dim},
                                  ownedFloatVector(makeRopeTable(geometry, periods, true)))
                    ->getOutput(0);
    auto *cos = network
                    ->addConstant(nvinfer1::Dims4{1, 1, geometry.num_patches, geometry.head_dim},
                                  ownedFloatVector(makeRopeTable(geometry, periods, false)))
                    ->getOutput(0);
    return {sin, cos};
}

/**
 * @brief 对 q/k 的 patch token 部分应用 DINOv3 RoPE，prefix token 保持不变。
 */
nvinfer1::ITensor *applyRope(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                             const DINORopeConstants &rope, const DINOInputGeometry &geometry,
                             const DINOTransformerSpec &spec)
{
    auto *prefix = network
                       ->addSlice(input, nvinfer1::Dims4{0, 0, 0, 0},
                                  nvinfer1::Dims4{geometry.batch, spec.num_heads, geometry.prefix_tokens,
                                                  geometry.head_dim},
                                  nvinfer1::Dims4{1, 1, 1, 1})
                       ->getOutput(0);
    auto *patch = network
                      ->addSlice(input, nvinfer1::Dims4{0, 0, geometry.prefix_tokens, 0},
                                 nvinfer1::Dims4{geometry.batch, spec.num_heads, geometry.num_patches,
                                                 geometry.head_dim},
                                 nvinfer1::Dims4{1, 1, 1, 1})
                      ->getOutput(0);

    auto *x1 = network
                   ->addSlice(*patch, nvinfer1::Dims4{0, 0, 0, 0},
                              nvinfer1::Dims4{geometry.batch, spec.num_heads, geometry.num_patches,
                                              geometry.head_dim / 2},
                              nvinfer1::Dims4{1, 1, 1, 1})
                   ->getOutput(0);
    auto *x2 = network
                   ->addSlice(*patch, nvinfer1::Dims4{0, 0, 0, geometry.head_dim / 2},
                              nvinfer1::Dims4{geometry.batch, spec.num_heads, geometry.num_patches,
                                              geometry.head_dim / 2},
                              nvinfer1::Dims4{1, 1, 1, 1})
                   ->getOutput(0);
    auto *neg_x2 = network->addUnary(*x2, nvinfer1::UnaryOperation::kNEG)->getOutput(0);

    const std::vector<nvinfer1::ITensor *> rotated_parts{neg_x2, x1};
    auto *rotated_half = network->addConcatenation(rotated_parts.data(), static_cast<int32_t>(rotated_parts.size()));
    rotated_half->setAxis(3);

    auto *x_cos = network->addElementWise(*patch, *rope.cos, E::kPROD);
    auto *rot_sin = network->addElementWise(*rotated_half->getOutput(0), *rope.sin, E::kPROD);
    auto *rotated = network->addElementWise(*x_cos->getOutput(0), *rot_sin->getOutput(0), E::kSUM)->getOutput(0);

    const std::vector<nvinfer1::ITensor *> tokens{prefix, rotated};
    auto *concat = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
    concat->setAxis(2);
    return concat->getOutput(0);
}

/**
 * @brief 添加多头自注意力。
 */
nvinfer1::ITensor *addAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, const std::string &prefix,
                                const DINOInputGeometry &geometry, const DINOTransformerSpec &spec)
{
    auto *qkv = addQkvProjection(network, weights_map, input, prefix, spec);
    nvinfer1::ITensor *q = nullptr;
    nvinfer1::ITensor *k = nullptr;
    nvinfer1::ITensor *v = nullptr;
    splitQkv(network, *qkv, geometry.batch, geometry.num_tokens, spec.embed_dim, q, k, v);

    auto *q_heads = reshapeToHeads(network, *q, geometry.batch, geometry.num_tokens, spec.num_heads,
                                    geometry.head_dim);
    auto *k_heads = reshapeToHeads(network, *k, geometry.batch, geometry.num_tokens, spec.num_heads,
                                    geometry.head_dim);
    auto *v_heads = reshapeToHeads(network, *v, geometry.batch, geometry.num_tokens, spec.num_heads,
                                    geometry.head_dim);

    if (spec.version == DINOVersion::V3)
    {
        const auto rope = addRopeConstants(network, weights_map, geometry);
        q_heads = applyRope(network, *q_heads, rope, geometry, spec);
        k_heads = applyRope(network, *k_heads, rope, geometry, spec);
    }

    auto *qk = network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE);
    auto *scale = network->addConstant(nvinfer1::Dims4{1, 1, 1, 1},
                                       ownedScalarWeight(1.0F / std::sqrt(static_cast<float>(geometry.head_dim))));
    auto *scaled_qk = network->addElementWise(*qk->getOutput(0), *scale->getOutput(0), E::kPROD);
    auto *softmax = network->addSoftMax(*scaled_qk->getOutput(0));
    softmax->setAxes(1U << static_cast<uint32_t>(scaled_qk->getOutput(0)->getDimensions().nbDims - 1));

    auto *attended = network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE);
    auto *attended_output = mergeHeads(network, *attended->getOutput(0), geometry.batch, geometry.num_tokens,
                                        spec.embed_dim);

    return addLinear3D(network, weights_map, *attended_output, prefix + ".attn.proj", spec.embed_dim,
                       spec.embed_dim, true);
}

/**
 * @brief 计算 DINOv2/DINOv3 SwiGLU 实际隐藏维度。
 */
int swigluHiddenDim(const DINOTransformerSpec &spec)
{
    const int raw_hidden = static_cast<int>(std::lround(static_cast<float>(spec.embed_dim) * spec.mlp_ratio));
    const int reduced = static_cast<int>(static_cast<float>(raw_hidden) * 2.0F / 3.0F);
    const int align = std::max(1, spec.swiglu_align);
    return reduced + ((align - (reduced % align)) % align);
}

/**
 * @brief 添加标准 MLP 子层。
 */
nvinfer1::ITensor *addStandardMlp(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix,
                                  const DINOTransformerSpec &spec)
{
    const int hidden = static_cast<int>(std::lround(static_cast<float>(spec.embed_dim) * spec.mlp_ratio));
    auto *fc1 = addLinear3D(network, weights_map, input, prefix + ".mlp.fc1", spec.embed_dim, hidden, true);
    auto *gelu = addGeluApprox(network, *fc1);
    return addLinear3D(network, weights_map, *gelu, prefix + ".mlp.fc2", hidden, spec.embed_dim, true);
}

/**
 * @brief 添加 DINOv2 Giant 的打包 SwiGLU 子层。
 */
nvinfer1::ITensor *addPackedSwiGLU(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &input, const std::string &prefix,
                                   const DINOTransformerSpec &spec)
{
    const int hidden = swigluHiddenDim(spec);
    const auto fc1_prefix = hasWeight(weights_map, prefix + ".mlp.w12.weight") ? prefix + ".mlp.w12"
                                                                               : prefix + ".mlp.fc1";
    const auto fc2_prefix = hasWeight(weights_map, prefix + ".mlp.w3.weight") ? prefix + ".mlp.w3"
                                                                              : prefix + ".mlp.fc2";

    auto *packed = addLinear3D(network, weights_map, input, fc1_prefix, spec.embed_dim, 2 * hidden, true);
    auto *gate = network
                     ->addSlice(*packed, nvinfer1::Dims3{0, 0, 0},
                                nvinfer1::Dims3{input.getDimensions().d[0], input.getDimensions().d[1], hidden},
                                nvinfer1::Dims3{1, 1, 1})
                     ->getOutput(0);
    auto *value = network
                      ->addSlice(*packed, nvinfer1::Dims3{0, 0, hidden},
                                 nvinfer1::Dims3{input.getDimensions().d[0], input.getDimensions().d[1], hidden},
                                 nvinfer1::Dims3{1, 1, 1})
                      ->getOutput(0);
    auto *activated = addSilu(network, *gate);
    auto *hidden_tensor = network->addElementWise(*activated, *value, E::kPROD)->getOutput(0);
    return addLinear3D(network, weights_map, *hidden_tensor, fc2_prefix, hidden, spec.embed_dim, true);
}

/**
 * @brief 添加 DINOv3 Plus/7B 的拆分 SwiGLU 子层。
 */
nvinfer1::ITensor *addSplitSwiGLU(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix,
                                  const DINOTransformerSpec &spec)
{
    const int hidden = swigluHiddenDim(spec);

    const auto gate_prefix = hasWeight(weights_map, prefix + ".mlp.w1.weight") ? prefix + ".mlp.w1"
                                                                               : prefix + ".mlp.fc1_g";
    const auto value_prefix = hasWeight(weights_map, prefix + ".mlp.w2.weight") ? prefix + ".mlp.w2"
                                                                                : prefix + ".mlp.fc1_x";
    const auto fc2_prefix = hasWeight(weights_map, prefix + ".mlp.w3.weight") ? prefix + ".mlp.w3"
                                                                              : prefix + ".mlp.fc2";

    auto *gate = addLinear3D(network, weights_map, input, gate_prefix, spec.embed_dim, hidden, true);
    auto *value = addLinear3D(network, weights_map, input, value_prefix, spec.embed_dim, hidden, true);
    auto *activated = addSilu(network, *gate);
    auto *hidden_tensor = network->addElementWise(*activated, *value, E::kPROD)->getOutput(0);
    return addLinear3D(network, weights_map, *hidden_tensor, fc2_prefix, hidden, spec.embed_dim, true);
}

/**
 * @brief 根据结构参数添加 FFN 子层。
 */
nvinfer1::ITensor *addMlp(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                          nvinfer1::ITensor &input, const std::string &prefix, const DINOTransformerSpec &spec)
{
    switch (spec.mlp_kind)
    {
    case DINOMlpKind::Mlp:
        return addStandardMlp(network, weights_map, input, prefix, spec);
    case DINOMlpKind::PackedSwiGLU:
        return addPackedSwiGLU(network, weights_map, input, prefix, spec);
    case DINOMlpKind::SplitSwiGLU:
        return addSplitSwiGLU(network, weights_map, input, prefix, spec);
    default:
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported DINO MLP kind");
    }
}

/**
 * @brief 添加一个 DINO Pre-LN Transformer block。
 */
nvinfer1::ITensor *addDINOBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, int index, const DINOInputGeometry &geometry,
                                const DINOTransformerSpec &spec)
{
    const auto prefix = "blocks." + std::to_string(index);
    auto *norm1 = addLayerNorm(network, weights_map, input, prefix + ".norm1", spec.embed_dim, spec.norm_epsilon);
    auto *attn = addAttention(network, weights_map, *norm1, prefix, geometry, spec);
    auto *scaled_attn = addLayerScale(network, weights_map, *attn,
                                      {prefix + ".ls1.gamma", prefix + ".gamma_1"}, spec.embed_dim);
    auto *attn_residual = network->addElementWise(input, *scaled_attn, E::kSUM)->getOutput(0);

    auto *norm2 = addLayerNorm(network, weights_map, *attn_residual, prefix + ".norm2", spec.embed_dim,
                               spec.norm_epsilon);
    auto *mlp = addMlp(network, weights_map, *norm2, prefix, spec);
    auto *scaled_mlp = addLayerScale(network, weights_map, *mlp,
                                     {prefix + ".ls2.gamma", prefix + ".gamma_2"}, spec.embed_dim);
    return network->addElementWise(*attn_residual, *scaled_mlp, E::kSUM)->getOutput(0);
}

/**
 * @brief 校验输入配置并计算 token 数量。
 */
DINOInputGeometry resolveInputGeometry(const DINOTransformerSpec &spec, const IModelConfig &config)
{
    const auto &shape = config.inputShape();
    DINOInputGeometry geometry{};
    geometry.batch = static_cast<int>(shape.d[0]);
    geometry.channels = static_cast<int>(shape.d[1]);
    geometry.height = static_cast<int>(shape.d[2]);
    geometry.width = static_cast<int>(shape.d[3]);

    if (geometry.batch != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "DINO handwritten TensorRT network currently requires batch=1, got %d",
                             geometry.batch);
    }
    if (geometry.channels != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "DINO requires 3 input channels, got %d",
                             geometry.channels);
    }
    if (geometry.height % spec.patch_size != 0 || geometry.width % spec.patch_size != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "DINO input H/W must be divisible by patch size %d, got H=%d W=%d", spec.patch_size,
                             geometry.height, geometry.width);
    }
    if (spec.embed_dim % spec.num_heads != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "DINO embed_dim (%d) must be divisible by num_heads (%d)",
                             spec.embed_dim, spec.num_heads);
    }

    geometry.grid_h = geometry.height / spec.patch_size;
    geometry.grid_w = geometry.width / spec.patch_size;
    geometry.num_patches = geometry.grid_h * geometry.grid_w;
    geometry.prefix_tokens = 1 + spec.extra_tokens;
    geometry.num_tokens = geometry.prefix_tokens + geometry.num_patches;
    geometry.head_dim = spec.embed_dim / spec.num_heads;

    if (spec.version == DINOVersion::V3 && geometry.head_dim % 4 != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "DINOv3 RoPE requires head_dim divisible by 4, got %d", geometry.head_dim);
    }
    return geometry;
}

/**
 * @brief 添加 patch embedding 并返回 NLC patch token。
 */
nvinfer1::ITensor *addPatchEmbedding(const DINOTransformer &impl, nvinfer1::INetworkDefinition *network,
                                     const WeightsMap &weights_map, const DINOInputGeometry &geometry,
                                     const DINOTransformerSpec &spec,
                                     priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *input = impl.addInputTensor(network);
    named_tensors["input"] = input;

    auto *patch = network->addConvolutionNd(
        *input, spec.embed_dim, nvinfer1::DimsHW{spec.patch_size, spec.patch_size},
        requireWeight(weights_map, "patch_embed.proj.weight",
                      static_cast<int64_t>(spec.embed_dim) * geometry.channels * spec.patch_size * spec.patch_size),
        requireWeight(weights_map, "patch_embed.proj.bias", spec.embed_dim));
    patch->setStrideNd(nvinfer1::DimsHW{spec.patch_size, spec.patch_size});

    auto *flatten = network->addShuffle(*patch->getOutput(0));
    flatten->setReshapeDimensions(nvinfer1::Dims3{geometry.batch, spec.embed_dim, geometry.num_patches});
    flatten->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    named_tensors["patch_embed"] = flatten->getOutput(0);
    return flatten->getOutput(0);
}

/**
 * @brief 添加 register/storage token 常量；无额外 token 时返回空指针。
 */
nvinfer1::ITensor *addExtraTokens(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  const DINOTransformerSpec &spec)
{
    if (spec.extra_tokens <= 0)
    {
        return nullptr;
    }

    const auto &weights = requireAnyWeight(weights_map, {"register_tokens", "reg_token", "storage_tokens"},
                                           static_cast<int64_t>(spec.extra_tokens) * spec.embed_dim);
    return network->addConstant(nvinfer1::Dims3{1, spec.extra_tokens, spec.embed_dim}, weights)->getOutput(0);
}

/**
 * @brief 添加 DINOv2 的 class/register token 与学习式位置编码。
 */
nvinfer1::ITensor *addDINOv2Tokens(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &patch_tokens, const DINOInputGeometry &geometry,
                                   const DINOTransformerSpec &spec)
{
    auto *cls_token = network
                          ->addConstant(nvinfer1::Dims3{1, 1, spec.embed_dim},
                                        requireWeight(weights_map, "cls_token", spec.embed_dim))
                          ->getOutput(0);
    auto *extra_tokens = addExtraTokens(network, weights_map, spec);
    const auto &pos_weight = requireWeight(weights_map, "pos_embed");
    if (pos_weight.count % spec.embed_dim != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "DINOv2 pos_embed count is not divisible by embed_dim");
    }

    const int pos_tokens = static_cast<int>(pos_weight.count / spec.embed_dim);
    if (pos_tokens == geometry.num_patches + 1)
    {
        const std::vector<nvinfer1::ITensor *> cls_and_patch{cls_token, &patch_tokens};
        auto *concat = network->addConcatenation(cls_and_patch.data(), static_cast<int32_t>(cls_and_patch.size()));
        concat->setAxis(1);

        auto *pos = network
                        ->addConstant(nvinfer1::Dims3{1, geometry.num_patches + 1, spec.embed_dim}, pos_weight)
                        ->getOutput(0);
        auto *position_added = network->addElementWise(*concat->getOutput(0), *pos, E::kSUM)->getOutput(0);
        if (extra_tokens == nullptr)
        {
            return position_added;
        }

        auto *cls_with_pos = network
                                 ->addSlice(*position_added, nvinfer1::Dims3{0, 0, 0},
                                            nvinfer1::Dims3{geometry.batch, 1, spec.embed_dim},
                                            nvinfer1::Dims3{1, 1, 1})
                                 ->getOutput(0);
        auto *patch_with_pos = network
                                   ->addSlice(*position_added, nvinfer1::Dims3{0, 1, 0},
                                              nvinfer1::Dims3{geometry.batch, geometry.num_patches, spec.embed_dim},
                                              nvinfer1::Dims3{1, 1, 1})
                                   ->getOutput(0);
        const std::vector<nvinfer1::ITensor *> tokens{cls_with_pos, extra_tokens, patch_with_pos};
        auto *with_registers = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
        with_registers->setAxis(1);
        return with_registers->getOutput(0);
    }

    if (pos_tokens == geometry.num_patches)
    {
        auto *pos = network
                        ->addConstant(nvinfer1::Dims3{1, geometry.num_patches, spec.embed_dim}, pos_weight)
                        ->getOutput(0);
        auto *patch_with_pos = network->addElementWise(patch_tokens, *pos, E::kSUM)->getOutput(0);

        std::vector<nvinfer1::ITensor *> tokens{cls_token};
        if (extra_tokens != nullptr)
        {
            tokens.push_back(extra_tokens);
        }
        tokens.push_back(patch_with_pos);
        auto *concat = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
        concat->setAxis(1);
        return concat->getOutput(0);
    }

    if (pos_tokens == geometry.num_tokens)
    {
        std::vector<nvinfer1::ITensor *> tokens{cls_token};
        if (extra_tokens != nullptr)
        {
            tokens.push_back(extra_tokens);
        }
        tokens.push_back(&patch_tokens);
        auto *concat = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
        concat->setAxis(1);

        auto *pos = network->addConstant(nvinfer1::Dims3{1, geometry.num_tokens, spec.embed_dim}, pos_weight)
                        ->getOutput(0);
        return network->addElementWise(*concat->getOutput(0), *pos, E::kSUM)->getOutput(0);
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                         "DINOv2 pos_embed token count mismatch: got %d, expected %d, %d or %d", pos_tokens,
                         geometry.num_patches, geometry.num_patches + 1, geometry.num_tokens);
}

/**
 * @brief 添加 DINOv3 的 class/storage token；位置编码由每个 block 的 RoPE 处理。
 */
nvinfer1::ITensor *addDINOv3Tokens(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &patch_tokens, const DINOTransformerSpec &spec)
{
    auto *cls_token = network
                          ->addConstant(nvinfer1::Dims3{1, 1, spec.embed_dim},
                                        requireWeight(weights_map, "cls_token", spec.embed_dim))
                          ->getOutput(0);
    auto *extra_tokens = addExtraTokens(network, weights_map, spec);

    std::vector<nvinfer1::ITensor *> tokens{cls_token};
    if (extra_tokens != nullptr)
    {
        tokens.push_back(extra_tokens);
    }
    tokens.push_back(&patch_tokens);

    auto *concat = network->addConcatenation(tokens.data(), static_cast<int32_t>(tokens.size()));
    concat->setAxis(1);
    return concat->getOutput(0);
}

/**
 * @brief 添加版本专属 token 准备逻辑。
 */
nvinfer1::ITensor *addInputTokens(const DINOTransformer &impl, nvinfer1::INetworkDefinition *network,
                                  const WeightsMap &weights_map, const DINOInputGeometry &geometry,
                                  const DINOTransformerSpec &spec,
                                  priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *patch_tokens = addPatchEmbedding(impl, network, weights_map, geometry, spec, named_tensors);
    auto *tokens = spec.version == DINOVersion::V2 ? addDINOv2Tokens(network, weights_map, *patch_tokens, geometry,
                                                                      spec)
                                                   : addDINOv3Tokens(network, weights_map, *patch_tokens, spec);
    named_tensors["tokens"] = tokens;
    return tokens;
}

/**
 * @brief 添加最终归一化并登记常用 DINO 特征张量名称。
 */
nvinfer1::ITensor *addFinalFeatureOutputs(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                          nvinfer1::ITensor &prenorm, const DINOInputGeometry &geometry,
                                          const DINOTransformerSpec &spec,
                                          priv::IModelImpl::NamedTensorMap &named_tensors)
{
    named_tensors["x_prenorm"] = &prenorm;
    auto *norm = addLayerNorm(network, weights_map, prenorm, "norm", spec.embed_dim, spec.norm_epsilon);
    named_tensors["norm"] = norm;

    auto *cls = network
                    ->addSlice(*norm, nvinfer1::Dims3{0, 0, 0},
                               nvinfer1::Dims3{geometry.batch, 1, spec.embed_dim}, nvinfer1::Dims3{1, 1, 1})
                    ->getOutput(0);
    auto *cls_flatten = network->addShuffle(*cls);
    cls_flatten->setReshapeDimensions(nvinfer1::Dims2{geometry.batch, spec.embed_dim});
    named_tensors["cls"] = cls_flatten->getOutput(0);
    named_tensors["pre_logits"] = cls_flatten->getOutput(0);
    named_tensors["x_norm_clstoken"] = cls_flatten->getOutput(0);

    if (spec.extra_tokens > 0)
    {
        auto *extra = network
                          ->addSlice(*norm, nvinfer1::Dims3{0, 1, 0},
                                     nvinfer1::Dims3{geometry.batch, spec.extra_tokens, spec.embed_dim},
                                     nvinfer1::Dims3{1, 1, 1})
                          ->getOutput(0);
        named_tensors[spec.version == DINOVersion::V2 ? "x_norm_regtokens" : "x_storage_tokens"] = extra;
        named_tensors["extra_tokens"] = extra;
    }

    auto *patch = network
                      ->addSlice(*norm, nvinfer1::Dims3{0, geometry.prefix_tokens, 0},
                                 nvinfer1::Dims3{geometry.batch, geometry.num_patches, spec.embed_dim},
                                 nvinfer1::Dims3{1, 1, 1})
                      ->getOutput(0);
    named_tensors["x_norm_patchtokens"] = patch;
    return cls_flatten->getOutput(0);
}

/**
 * @brief 生成 DINOv2 标准结构参数。
 */
DINOTransformerSpec makeDINOv2Spec(const char *display_name, int image_size, int embed_dim, int depth, int num_heads,
                                   DINOMlpKind mlp_kind, int extra_tokens)
{
    return {display_name, image_size, 14, embed_dim, depth, num_heads, extra_tokens, 4.0F,
            DINOVersion::V2, mlp_kind, 8, 1e-6F};
}

/**
 * @brief 生成 DINOv3 标准结构参数。
 */
DINOTransformerSpec makeDINOv3Spec(const char *display_name, int image_size, int embed_dim, int depth, int num_heads,
                                   float mlp_ratio, DINOMlpKind mlp_kind, int swiglu_align)
{
    return {display_name, image_size, 16, embed_dim, depth, num_heads, 4, mlp_ratio,
            DINOVersion::V3, mlp_kind, swiglu_align, 1e-5F};
}

class DINOv2ViTS14 : public DINOTransformer
{
public:
    DINOv2ViTS14()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTS14", 518, 384, 12, 6, DINOMlpKind::Mlp, 0))
    {
    }
    static const char *key() noexcept { return "dinov2_vits14"; }
};

class DINOv2ViTB14 : public DINOTransformer
{
public:
    DINOv2ViTB14()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTB14", 518, 768, 12, 12, DINOMlpKind::Mlp, 0))
    {
    }
    static const char *key() noexcept { return "dinov2_vitb14"; }
};

class DINOv2ViTL14 : public DINOTransformer
{
public:
    DINOv2ViTL14()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTL14", 518, 1024, 24, 16, DINOMlpKind::Mlp, 0))
    {
    }
    static const char *key() noexcept { return "dinov2_vitl14"; }
};

class DINOv2ViTG14 : public DINOTransformer
{
public:
    DINOv2ViTG14()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTG14", 518, 1536, 40, 24, DINOMlpKind::PackedSwiGLU, 0))
    {
    }
    static const char *key() noexcept { return "dinov2_vitg14"; }
};

class DINOv2ViTS14Reg : public DINOTransformer
{
public:
    DINOv2ViTS14Reg()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTS14Reg4", 518, 384, 12, 6, DINOMlpKind::Mlp, 4))
    {
    }
    static const char *key() noexcept { return "dinov2_vits14_reg"; }
};

class DINOv2ViTB14Reg : public DINOTransformer
{
public:
    DINOv2ViTB14Reg()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTB14Reg4", 518, 768, 12, 12, DINOMlpKind::Mlp, 4))
    {
    }
    static const char *key() noexcept { return "dinov2_vitb14_reg"; }
};

class DINOv2ViTL14Reg : public DINOTransformer
{
public:
    DINOv2ViTL14Reg()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTL14Reg4", 518, 1024, 24, 16, DINOMlpKind::Mlp, 4))
    {
    }
    static const char *key() noexcept { return "dinov2_vitl14_reg"; }
};

class DINOv2ViTG14Reg : public DINOTransformer
{
public:
    DINOv2ViTG14Reg()
        : DINOTransformer(makeDINOv2Spec("DINOv2ViTG14Reg4", 518, 1536, 40, 24, DINOMlpKind::PackedSwiGLU, 4))
    {
    }
    static const char *key() noexcept { return "dinov2_vitg14_reg"; }
};

class DINOv3ViTS16 : public DINOTransformer
{
public:
    DINOv3ViTS16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTS16", 224, 384, 12, 6, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "dinov3_vits16"; }
};

class DINOv3ViTS16Plus : public DINOTransformer
{
public:
    DINOv3ViTS16Plus()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTS16Plus", 224, 384, 12, 6, 6.0F, DINOMlpKind::SplitSwiGLU, 8))
    {
    }
    static const char *key() noexcept { return "dinov3_vits16plus"; }
};

class DINOv3ViTB16 : public DINOTransformer
{
public:
    DINOv3ViTB16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTB16", 224, 768, 12, 12, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "dinov3_vitb16"; }
};

class DINOv3ViTL16 : public DINOTransformer
{
public:
    DINOv3ViTL16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTL16", 224, 1024, 24, 16, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "dinov3_vitl16"; }
};

class DINOv3ViTL16Plus : public DINOTransformer
{
public:
    DINOv3ViTL16Plus()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTL16Plus", 224, 1024, 24, 16, 6.0F, DINOMlpKind::SplitSwiGLU, 8))
    {
    }
    static const char *key() noexcept { return "dinov3_vitl16plus"; }
};

class DINOv3ViTH16Plus : public DINOTransformer
{
public:
    DINOv3ViTH16Plus()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTH16Plus", 224, 1280, 32, 20, 6.0F, DINOMlpKind::SplitSwiGLU, 8))
    {
    }
    static const char *key() noexcept { return "dinov3_vith16plus"; }
};

class DINOv3ViT7B16 : public DINOTransformer
{
public:
    DINOv3ViT7B16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViT7B16", 224, 4096, 40, 32, 3.0F, DINOMlpKind::SplitSwiGLU, 64))
    {
    }
    static const char *key() noexcept { return "dinov3_vit7b16"; }
};

/**
 * @brief 轻量别名类：只复用结构参数并绑定额外注册 key。
 */
#define INFERRT_DINO_ALIAS_CLASS(CLASS_NAME, BASE_CLASS, KEY_LITERAL)                                             \
    class CLASS_NAME : public BASE_CLASS                                                                          \
    {                                                                                                             \
    public:                                                                                                       \
        static const char *key() noexcept { return KEY_LITERAL; }                                                 \
    }

INFERRT_DINO_ALIAS_CLASS(DINOv2ViTS14Reg4Alias, DINOv2ViTS14Reg, "dinov2_vits14_reg4");
INFERRT_DINO_ALIAS_CLASS(DINOv2ViTB14Reg4Alias, DINOv2ViTB14Reg, "dinov2_vitb14_reg4");
INFERRT_DINO_ALIAS_CLASS(DINOv2ViTL14Reg4Alias, DINOv2ViTL14Reg, "dinov2_vitl14_reg4");
INFERRT_DINO_ALIAS_CLASS(DINOv2ViTG14Reg4Alias, DINOv2ViTG14Reg, "dinov2_vitg14_reg4");

INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTS14, DINOv2ViTS14, "vit_small_patch14_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTB14, DINOv2ViTB14, "vit_base_patch14_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTL14, DINOv2ViTL14, "vit_large_patch14_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTG14, DINOv2ViTG14, "vit_giant_patch14_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTS14Reg, DINOv2ViTS14Reg, "vit_small_patch14_reg4_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTB14Reg, DINOv2ViTB14Reg, "vit_base_patch14_reg4_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTL14Reg, DINOv2ViTL14Reg, "vit_large_patch14_reg4_dinov2");
INFERRT_DINO_ALIAS_CLASS(TimmDINOv2ViTG14Reg, DINOv2ViTG14Reg, "vit_giant_patch14_reg4_dinov2");

class TimmDINOv3ViTS16 : public DINOTransformer
{
public:
    TimmDINOv3ViTS16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTS16", 256, 384, 12, 6, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "vit_small_patch16_dinov3"; }
};

INFERRT_DINO_ALIAS_CLASS(TimmDINOv3ViTS16Qkvb, TimmDINOv3ViTS16, "vit_small_patch16_dinov3_qkvb");

class TimmDINOv3ViTS16Plus : public DINOTransformer
{
public:
    TimmDINOv3ViTS16Plus()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTS16Plus", 256, 384, 12, 6, 6.0F, DINOMlpKind::SplitSwiGLU, 8))
    {
    }
    static const char *key() noexcept { return "vit_small_plus_patch16_dinov3"; }
};

INFERRT_DINO_ALIAS_CLASS(TimmDINOv3ViTS16PlusQkvb, TimmDINOv3ViTS16Plus,
                         "vit_small_plus_patch16_dinov3_qkvb");

class TimmDINOv3ViTB16 : public DINOTransformer
{
public:
    TimmDINOv3ViTB16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTB16", 256, 768, 12, 12, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "vit_base_patch16_dinov3"; }
};

INFERRT_DINO_ALIAS_CLASS(TimmDINOv3ViTB16Qkvb, TimmDINOv3ViTB16, "vit_base_patch16_dinov3_qkvb");

class TimmDINOv3ViTL16 : public DINOTransformer
{
public:
    TimmDINOv3ViTL16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTL16", 256, 1024, 24, 16, 4.0F, DINOMlpKind::Mlp, 1))
    {
    }
    static const char *key() noexcept { return "vit_large_patch16_dinov3"; }
};

INFERRT_DINO_ALIAS_CLASS(TimmDINOv3ViTL16Qkvb, TimmDINOv3ViTL16, "vit_large_patch16_dinov3_qkvb");

class TimmDINOv3ViTH16Plus : public DINOTransformer
{
public:
    TimmDINOv3ViTH16Plus()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViTH16Plus", 256, 1280, 32, 20, 6.0F, DINOMlpKind::SplitSwiGLU, 8))
    {
    }
    static const char *key() noexcept { return "vit_huge_plus_patch16_dinov3"; }
};

INFERRT_DINO_ALIAS_CLASS(TimmDINOv3ViTH16PlusQkvb, TimmDINOv3ViTH16Plus,
                         "vit_huge_plus_patch16_dinov3_qkvb");

class TimmDINOv3ViT7B16 : public DINOTransformer
{
public:
    TimmDINOv3ViT7B16()
        : DINOTransformer(makeDINOv3Spec("DINOv3ViT7B16", 256, 4096, 40, 32, 3.0F, DINOMlpKind::SplitSwiGLU, 64))
    {
    }
    static const char *key() noexcept { return "vit_7b_patch16_dinov3"; }
};

} // namespace

void DINOTransformer::normalizeModelConfig(IModelConfig &config) const
{
    constexpr int kDefaultImageSize = 224;
    if (spec_.image_size == kDefaultImageSize || config.inputShapes().size() != 1)
    {
        return;
    }

    const auto &shape = config.inputShape();
    const bool is_default_image_config = shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageSize
                                      && shape.d[3] == kDefaultImageSize;
    if (is_default_image_config)
    {
        config.setInputShape(nvinfer1::Dims4{1, 3, spec_.image_size, spec_.image_size});
    }
}

void DINOTransformer::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }

    const auto geometry = resolveInputGeometry(spec_, modelConfig());
    const bool feature_only = isBuildingFeatureEngine();

    priv::IModelImpl::NamedTensorMap named_tensors;
    auto *x = addInputTokens(*this, network, weights_map, geometry, spec_, named_tensors);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    for (int i = 0; i < spec_.depth; ++i)
    {
        x = addDINOBlock(network, weights_map, *x, i, geometry, spec_);
        named_tensors["block" + std::to_string(i)] = x;
        named_tensors["blocks." + std::to_string(i)] = x;
        if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
        {
            return;
        }
    }

    auto *cls = addFinalFeatureOutputs(network, weights_map, *x, geometry, spec_, named_tensors);
    if (feature_only)
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    markOutputTensors(network, {cls});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(DINOv2ViTS14)
INFERRT_REGISTER_MODEL(DINOv2ViTB14)
INFERRT_REGISTER_MODEL(DINOv2ViTL14)
INFERRT_REGISTER_MODEL(DINOv2ViTG14)
INFERRT_REGISTER_MODEL(DINOv2ViTS14Reg)
INFERRT_REGISTER_MODEL(DINOv2ViTB14Reg)
INFERRT_REGISTER_MODEL(DINOv2ViTL14Reg)
INFERRT_REGISTER_MODEL(DINOv2ViTG14Reg)
INFERRT_REGISTER_MODEL(DINOv2ViTS14Reg4Alias)
INFERRT_REGISTER_MODEL(DINOv2ViTB14Reg4Alias)
INFERRT_REGISTER_MODEL(DINOv2ViTL14Reg4Alias)
INFERRT_REGISTER_MODEL(DINOv2ViTG14Reg4Alias)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTS14)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTB14)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTL14)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTG14)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTS14Reg)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTB14Reg)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTL14Reg)
INFERRT_REGISTER_MODEL(TimmDINOv2ViTG14Reg)
INFERRT_REGISTER_MODEL(DINOv3ViTS16)
INFERRT_REGISTER_MODEL(DINOv3ViTS16Plus)
INFERRT_REGISTER_MODEL(DINOv3ViTB16)
INFERRT_REGISTER_MODEL(DINOv3ViTL16)
INFERRT_REGISTER_MODEL(DINOv3ViTL16Plus)
INFERRT_REGISTER_MODEL(DINOv3ViTH16Plus)
INFERRT_REGISTER_MODEL(DINOv3ViT7B16)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTS16)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTS16Qkvb)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTS16Plus)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTS16PlusQkvb)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTB16)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTB16Qkvb)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTL16)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTL16Qkvb)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTH16Plus)
INFERRT_REGISTER_MODEL(TimmDINOv3ViTH16PlusQkvb)
INFERRT_REGISTER_MODEL(TimmDINOv3ViT7B16)
