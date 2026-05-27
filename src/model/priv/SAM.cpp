#include "SAM.hpp"
#include "Layers.hpp"

#include <NvInferVersion.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/ModelFactory.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace irt::model {
namespace {

using E = nvinfer1::ElementWiseOperation;
using M = nvinfer1::MatrixOperation;
using U = nvinfer1::UnaryOperation;

constexpr int   kDefaultImageNetSize = 224;
constexpr int   kDefaultNumClasses   = 1000;
constexpr int   kSamImageSize        = 1024;
constexpr int   kSamPatchSize        = 16;
constexpr int   kSam2PatchSize       = 4;
constexpr int   kSamPromptDim        = 256;
constexpr int   kSamMaskSize         = 256;
constexpr int   kSamEmbedGrid        = 64;
constexpr int   kSamMaxPoints        = 16;
constexpr int   kSamPointTokens      = kSamMaxPoints + 1;
constexpr int   kSamMaskTokens       = 4;
constexpr int   kSamOutputMasks      = 3;
constexpr int   kSamTwoWayDepth      = 2;
constexpr int   kSamTwoWayHeads      = 8;
constexpr int   kSamTwoWayMlpDim     = 2048;
constexpr float kLayerNormEps        = 1.0e-6F;

constexpr std::array<const char *, 5> kDefaultInputNames{
    "image",
    "point_coords",
    "point_labels",
    "mask_input",
    "has_mask_input",
};

constexpr std::array<const char *, 3> kDefaultOutputNames{
    "masks",
    "iou_predictions",
    "low_res_masks",
};

/**
 * @brief SAM v1 官方 ViT image encoder 的结构参数。
 */
struct SAMViTSpec
{
    int              embed_dim;            ///< ViT token 维度。
    int              depth;                ///< Transformer block 数量。
    int              num_heads;            ///< 注意力头数量。
    std::set<int>    global_attn_indexes;  ///< 使用全局注意力的 block 下标。
};

/**
 * @brief SAM2 官方 Hiera image encoder 的结构参数。
 */
struct SAM2HieraSpec
{
    int              embed_dim;            ///< Hiera 初始 token 维度。
    int              num_heads;            ///< Hiera 初始注意力头数量。
    std::vector<int> stages;               ///< 各 stage 的 block 数量。
    std::vector<int> window_spec;          ///< 各 stage 的窗口大小。
    std::set<int>    global_attn_indexes;  ///< 使用全局注意力的 block 下标。
    std::vector<int> backbone_channels;    ///< FPN neck 输入通道，按低分辨率到高分辨率排列。
    int              pos_embed_size;       ///< 背景位置编码参数的空间边长。
    int              q_pool;               ///< 执行 q pooling 的 stage 数。
};

/**
 * @brief SAM prompt/mask head 的权重前缀。
 */
struct SAMHeadPrefixes
{
    std::string prompt; ///< PromptEncoder 在 state_dict 中的前缀。
    std::string mask;   ///< MaskDecoder 在 state_dict 中的前缀。
};

/**
 * @brief SAM mask decoder 的构建选项。
 */
struct SAMMaskDecoderOptions
{
    SAMHeadPrefixes prefixes;              ///< 权重前缀。
    int             image_grid;            ///< image embedding 的空间边长。
    bool            use_high_res_features; ///< 是否接入 SAM2 high-res FPN 特征。
    bool            pred_obj_scores;       ///< 是否包含 SAM2 object score token。
    bool            sigmoid_iou;           ///< 是否对 IoU head 输出做 sigmoid。
};

/**
 * @brief 构建期解析出的固定输入几何信息。
 */
struct SAMGeometry
{
    int batch;      ///< 当前实现固定为 1。
    int channels;   ///< 输入图像通道数，必须为 3。
    int image_h;    ///< 输入图像高度。
    int image_w;    ///< 输入图像宽度。
    int grid_h;     ///< patch 后的网格高度。
    int grid_w;     ///< patch 后的网格宽度。
    int grid_tokens; ///< 图像 token 数量。
};

/**
 * @brief 构造 TensorRT Dims。
 */
nvinfer1::Dims makeDims(std::initializer_list<int32_t> values)
{
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(values.size());
    int32_t i = 0;
    for (const auto value : values)
    {
        dims.d[i++] = value;
    }
    return dims;
}

/**
 * @brief 计算静态 Dims 的元素数量。
 */
int64_t volume(const nvinfer1::Dims &dims)
{
    int64_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        count *= dims.d[i];
    }
    return count;
}

/**
 * @brief 从候选线性层前缀中选择实际存在的权重命名。
 *
 * SAM v1 与 SAM2 的 TwoWayTransformer 结构一致，但 MLP 命名分别为
 * ``mlp.lin*`` 与 ``mlp.layers.*``。统一在这里解析可减少后续构图分支。
 */
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

constexpr const char *kSAMTag = "SAM official";

inline const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key,
                                              int64_t expected_count = -1)
{
    return irt::model::requireWeight(weights_map, key, kSAMTag, expected_count);
}

/**
 * @brief 断言 TensorRT layer 创建成功。
 */
template <typename T>
T *requireLayer(T *layer, const char *message)
{
    if (layer == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "%s", message);
    }
    return layer;
}

/**
 * @brief 添加一个可广播标量常量。
 */
nvinfer1::ITensor *addScalar(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &like, float value)
{
    auto *constant = requireLayer(network->addConstant(scalarDimsLike(like), ownedScalarWeight(value)),
                                  "Failed to add SAM scalar constant");
    return constant->getOutput(0);
}

/**
 * @brief 添加最后一维 LayerNorm，覆盖官方 ViT block 和 TwoWayTransformer。
 */
nvinfer1::ITensor *addLayerNormLastDim(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto dims = scalarDimsLike(input);
    dims.d[dims.nbDims - 1] = channels;
    auto *scale = requireLayer(network->addConstant(dims, requireWeight(weights_map, prefix + ".weight", channels)),
                               "Failed to add SAM LayerNorm scale");
    auto *bias = requireLayer(network->addConstant(dims, requireWeight(weights_map, prefix + ".bias", channels)),
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

/**
 * @brief 添加官方 LayerNorm2d，按 NCHW 的 C 维归一化。
 */
nvinfer1::ITensor *addLayerNorm2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *mean = requireLayer(network->addReduce(input, nvinfer1::ReduceOperation::kAVG, 1U << 1, true),
                              "Failed to add SAM LayerNorm2d mean");
    auto *centered = requireLayer(network->addElementWise(input, *mean->getOutput(0), E::kSUB),
                                  "Failed to add SAM LayerNorm2d center");
    auto *square = requireLayer(network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), E::kPROD),
                                "Failed to add SAM LayerNorm2d square");
    auto *var = requireLayer(network->addReduce(*square->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true),
                             "Failed to add SAM LayerNorm2d variance");
    auto *eps = addScalar(network, *var->getOutput(0), kLayerNormEps);
    auto *var_eps = requireLayer(network->addElementWise(*var->getOutput(0), *eps, E::kSUM),
                                 "Failed to add SAM LayerNorm2d eps");
    auto *std = requireLayer(network->addUnary(*var_eps->getOutput(0), U::kSQRT),
                             "Failed to add SAM LayerNorm2d sqrt");
    auto *normalized = requireLayer(network->addElementWise(*centered->getOutput(0), *std->getOutput(0), E::kDIV),
                                    "Failed to add SAM LayerNorm2d div");

    auto *scale = requireLayer(
        network->addConstant(nvinfer1::Dims4{1, channels, 1, 1}, requireWeight(weights_map, prefix + ".weight", channels)),
        "Failed to add SAM LayerNorm2d scale");
    auto *bias = requireLayer(
        network->addConstant(nvinfer1::Dims4{1, channels, 1, 1}, requireWeight(weights_map, prefix + ".bias", channels)),
        "Failed to add SAM LayerNorm2d bias");
    auto *scaled = requireLayer(network->addElementWise(*normalized->getOutput(0), *scale->getOutput(0), E::kPROD),
                                "Failed to add SAM LayerNorm2d scale product");
    return requireLayer(network->addElementWise(*scaled->getOutput(0), *bias->getOutput(0), E::kSUM),
                        "Failed to add SAM LayerNorm2d bias add")
        ->getOutput(0);
}

/**
 * @brief 将 NHWC 图像 token 展平为 `[B, H*W, C]`。
 */
nvinfer1::ITensor *flattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                               int width, int channels)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC flatten");
    shuffle->setReshapeDimensions(nvinfer1::Dims3{batch, height * width, channels});
    return shuffle->getOutput(0);
}

/**
 * @brief 将 `[B, H*W, C]` 还原为 NHWC。
 */
nvinfer1::ITensor *unflattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                 int height, int width, int channels)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC unflatten");
    shuffle->setReshapeDimensions(nvinfer1::Dims4{batch, height, width, channels});
    return shuffle->getOutput(0);
}

/**
 * @brief NCHW 转 NHWC。
 */
nvinfer1::ITensor *nchwToNhwc(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NCHW->NHWC");
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    return shuffle->getOutput(0);
}

/**
 * @brief NHWC 转 NCHW。
 */
nvinfer1::ITensor *nhwcToNchw(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *shuffle = requireLayer(network->addShuffle(input), "Failed to add SAM NHWC->NCHW");
    shuffle->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    return shuffle->getOutput(0);
}

/**
 * @brief 添加标准缩放点积注意力。
 *
 * 该函数用于 mask decoder 的 token 注意力；image encoder 需要额外叠加
 * ViTDet decomposed relative positional embedding，因此走独立的图构建函数。
 */
nvinfer1::ITensor *addTokenAttention(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &q,
                                     nvinfer1::ITensor &k, nvinfer1::ITensor &v, int batch, int q_tokens,
                                     int k_tokens, int num_heads, int internal_dim)
{
    const int head_dim = internal_dim / num_heads;
    auto     *q_heads  = reshapeToHeads(network, q, batch, q_tokens, num_heads, head_dim);
    auto     *k_heads  = reshapeToHeads(network, k, batch, k_tokens, num_heads, head_dim);
    auto     *v_heads  = reshapeToHeads(network, v, batch, k_tokens, num_heads, head_dim);

    auto *qk = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                            "Failed to add SAM attention qk");
    auto *scale = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto *scaled = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD),
                                "Failed to add SAM attention scale");
    auto *softmax = requireLayer(network->addSoftMax(*scaled->getOutput(0)), "Failed to add SAM attention softmax");
    softmax->setAxes(1U << 3);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed to add SAM attention value matmul");
    return mergeHeads(network, *attended->getOutput(0), batch, q_tokens, internal_dim);
}

/**
 * @brief 根据官方坐标规则展开一维相对位置权重。
 *
 * @param weights_map `.wts` 权重表。
 * @param key `rel_pos_h` 或 `rel_pos_w` 的权重 key。
 * @param q_size query 轴长度。
 * @param k_size key 轴长度。
 * @param head_dim 单个注意力头的通道数。
 * @return 布局为 `[q_size, head_dim, k_size]` 的 TensorRT 常量权重。
 */
nvinfer1::Weights makeRelativePositionWeight(const WeightsMap &weights_map, const std::string &key, int q_size,
                                             int k_size, int head_dim)
{
    const auto &weight           = requireWeight(weights_map, key);
    const int   expected_pos_len = 2 * std::max(q_size, k_size) - 1;
    if (weight.count != static_cast<int64_t>(expected_pos_len) * head_dim)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Unexpected SAM relative position shape for %s: got %lld values, expected %d x %d",
                             key.c_str(), static_cast<long long>(weight.count), expected_pos_len, head_dim);
    }

    const auto *source = static_cast<const float *>(weight.values);
    std::vector<float> values(static_cast<size_t>(q_size) * head_dim * k_size);
    const float        q_scale = std::max(static_cast<float>(k_size) / static_cast<float>(q_size), 1.0F);
    const float        k_scale = std::max(static_cast<float>(q_size) / static_cast<float>(k_size), 1.0F);
    const float        offset  = static_cast<float>(k_size - 1) * k_scale;
    for (int q = 0; q < q_size; ++q)
    {
        for (int k = 0; k < k_size; ++k)
        {
            const int rel_index = static_cast<int>((static_cast<float>(q) * q_scale - static_cast<float>(k) * k_scale)
                                                   + offset);
            for (int d = 0; d < head_dim; ++d)
            {
                values[(static_cast<size_t>(q) * head_dim + d) * k_size + k]
                    = source[static_cast<size_t>(rel_index) * head_dim + d];
            }
        }
    }
    return ownedFloatVector(std::move(values));
}

/**
 * @brief 叠加官方 decomposed relative positional embedding。
 *
 * 官方实现对高度和宽度分别做 `einsum`，再广播到注意力矩阵：
 * `attn[B,H,qh,qw,kh,kw] += rel_h[...,kh,None] + rel_w[...,None,kw]`。
 */
nvinfer1::ITensor *addImageRelativePosition(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                            nvinfer1::ITensor &attn, nvinfer1::ITensor &q_heads,
                                            const std::string &prefix, int batch, int q_h, int q_w, int num_heads,
                                            int head_dim)
{
    auto *attn_view = requireLayer(network->addShuffle(attn), "Failed to reshape SAM relative position attention");
    attn_view->setReshapeDimensions(makeDims({batch, num_heads, q_h, q_w, q_h, q_w}));

    auto *q_view = requireLayer(network->addShuffle(q_heads), "Failed to reshape SAM relative position query");
    q_view->setReshapeDimensions(makeDims({batch, num_heads, q_h, q_w, head_dim}));

    auto *rel_h = requireLayer(
        network->addConstant(makeDims({1, 1, q_h, head_dim, q_h}),
                             makeRelativePositionWeight(weights_map, prefix + ".attn.rel_pos_h", q_h, q_h, head_dim)),
        "Failed to add SAM rel_pos_h constant");
    auto *rel_h_scores = requireLayer(
        network->addMatrixMultiply(*q_view->getOutput(0), M::kNONE, *rel_h->getOutput(0), M::kNONE),
        "Failed to add SAM rel_pos_h scores");
    auto *rel_h_view = requireLayer(network->addShuffle(*rel_h_scores->getOutput(0)),
                                    "Failed to expand SAM rel_pos_h scores");
    rel_h_view->setReshapeDimensions(makeDims({batch, num_heads, q_h, q_w, q_h, 1}));

    auto *q_w_view = requireLayer(network->addShuffle(*q_view->getOutput(0)),
                                  "Failed to transpose SAM relative position query");
    q_w_view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4});
    auto *rel_w = requireLayer(
        network->addConstant(makeDims({1, 1, q_w, head_dim, q_w}),
                             makeRelativePositionWeight(weights_map, prefix + ".attn.rel_pos_w", q_w, q_w, head_dim)),
        "Failed to add SAM rel_pos_w constant");
    auto *rel_w_scores = requireLayer(
        network->addMatrixMultiply(*q_w_view->getOutput(0), M::kNONE, *rel_w->getOutput(0), M::kNONE),
        "Failed to add SAM rel_pos_w scores");
    auto *rel_w_transpose = requireLayer(network->addShuffle(*rel_w_scores->getOutput(0)),
                                         "Failed to transpose SAM rel_pos_w scores");
    rel_w_transpose->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4});
    auto *rel_w_view = requireLayer(network->addShuffle(*rel_w_transpose->getOutput(0)),
                                    "Failed to expand SAM rel_pos_w scores");
    rel_w_view->setReshapeDimensions(makeDims({batch, num_heads, q_h, q_w, 1, q_w}));

    auto *with_h = requireLayer(network->addElementWise(*attn_view->getOutput(0), *rel_h_view->getOutput(0), E::kSUM),
                                "Failed to add SAM rel_pos_h to attention");
    auto *with_hw
        = requireLayer(network->addElementWise(*with_h->getOutput(0), *rel_w_view->getOutput(0), E::kSUM),
                       "Failed to add SAM rel_pos_w to attention");
    auto *scores = requireLayer(network->addShuffle(*with_hw->getOutput(0)),
                                "Failed to flatten SAM relative position attention");
    scores->setReshapeDimensions(nvinfer1::Dims4{batch, num_heads, q_h * q_w, q_h * q_w});
    return scores->getOutput(0);
}

/**
 * @brief 添加 SAM image encoder 中带相对位置项的 qkv 注意力。
 */
nvinfer1::ITensor *addImageAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &tokens, const std::string &prefix, int batch, int height,
                                     int width, const SAMViTSpec &spec)
{
    const int token_count = height * width;
    const int head_dim    = spec.embed_dim / spec.num_heads;
    auto *qkv = addLinear3D(network, weights_map, tokens, prefix + ".attn.qkv", spec.embed_dim, 3 * spec.embed_dim);
    nvinfer1::ITensor *q = nullptr;
    nvinfer1::ITensor *k = nullptr;
    nvinfer1::ITensor *v = nullptr;
    splitQkv(network, *qkv, batch, token_count, spec.embed_dim, q, k, v);
    auto *q_heads = reshapeToHeads(network, *q, batch, token_count, spec.num_heads, head_dim);
    auto *k_heads = reshapeToHeads(network, *k, batch, token_count, spec.num_heads, head_dim);
    auto *v_heads = reshapeToHeads(network, *v, batch, token_count, spec.num_heads, head_dim);

    auto *qk = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                            "Failed to add SAM image attention qk");
    auto *scale = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto *scaled = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD),
                                "Failed to add SAM image attention scale");
    auto *with_rel_pos = addImageRelativePosition(network, weights_map, *scaled->getOutput(0), *q_heads, prefix, batch,
                                                  height, width, spec.num_heads, head_dim);
    auto *softmax = requireLayer(network->addSoftMax(*with_rel_pos), "Failed to add SAM image attention softmax");
    softmax->setAxes(1U << 3);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed to add SAM image attention value matmul");
    auto *attn = mergeHeads(network, *attended->getOutput(0), batch, token_count, spec.embed_dim);
    return addLinear3D(network, weights_map, *attn, prefix + ".attn.proj", spec.embed_dim, spec.embed_dim);
}

/**
 * @brief 将 NHWC token padding 到窗口注意力需要的尺寸。
 */
nvinfer1::ITensor *padNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                           int width, int channels, int padded_h, int padded_w)
{
    if (height == padded_h && width == padded_w)
    {
        return &input;
    }
    auto *slice = requireLayer(network->addSlice(input, nvinfer1::Dims4{0, 0, 0, 0},
                                                 nvinfer1::Dims4{batch, padded_h, padded_w, channels},
                                                 nvinfer1::Dims4{1, 1, 1, 1}),
                               "Failed to add SAM window padding");
    slice->setMode(nvinfer1::SampleMode::kFILL);
    auto *zero = requireLayer(network->addConstant(nvinfer1::Dims4{1, 1, 1, 1}, ownedScalarWeight(0.0F)),
                              "Failed to add SAM padding zero");
    slice->setInput(4, *zero->getOutput(0));
    return slice->getOutput(0);
}

/**
 * @brief 官方 window_partition：`[B,H,W,C] -> [B*num_windows, ws, ws, C]`。
 */
nvinfer1::ITensor *partitionWindows(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                    int height, int width, int channels, int window_size, int &padded_h,
                                    int &padded_w)
{
    padded_h = ((height + window_size - 1) / window_size) * window_size;
    padded_w = ((width + window_size - 1) / window_size) * window_size;
    auto *padded = padNHWC(network, input, batch, height, width, channels, padded_h, padded_w);

    const int windows_h = padded_h / window_size;
    const int windows_w = padded_w / window_size;
    auto     *view = requireLayer(network->addShuffle(*padded), "Failed to add SAM window view");
    view->setReshapeDimensions(makeDims({batch, windows_h, window_size, windows_w, window_size, channels}));
    view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *windows = requireLayer(network->addShuffle(*view->getOutput(0)), "Failed to add SAM window flatten");
    windows->setReshapeDimensions(nvinfer1::Dims4{batch * windows_h * windows_w, window_size, window_size, channels});
    return windows->getOutput(0);
}

/**
 * @brief 官方 window_unpartition：恢复到原始 NHWC 尺寸。
 */
nvinfer1::ITensor *unpartitionWindows(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &windows, int batch,
                                      int height, int width, int channels, int window_size, int padded_h,
                                      int padded_w)
{
    const int windows_h = padded_h / window_size;
    const int windows_w = padded_w / window_size;
    auto     *view = requireLayer(network->addShuffle(windows), "Failed to add SAM window restore view");
    view->setReshapeDimensions(makeDims({batch, windows_h, windows_w, window_size, window_size, channels}));
    view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *merged = requireLayer(network->addShuffle(*view->getOutput(0)), "Failed to add SAM window restore merge");
    merged->setReshapeDimensions(nvinfer1::Dims4{batch, padded_h, padded_w, channels});
    if (height == padded_h && width == padded_w)
    {
        return merged->getOutput(0);
    }
    return requireLayer(network->addSlice(*merged->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                                          nvinfer1::Dims4{batch, height, width, channels},
                                          nvinfer1::Dims4{1, 1, 1, 1}),
                        "Failed to add SAM window crop")
        ->getOutput(0);
}

/**
 * @brief 添加 SAM image encoder 的一个官方 Transformer block。
 */
nvinfer1::ITensor *addImageBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, int index, const SAMGeometry &geometry,
                                 const SAMViTSpec &spec)
{
    const std::string prefix = "image_encoder.blocks." + std::to_string(index);
    auto             *norm1  = addLayerNormLastDim(network, weights_map, input, prefix + ".norm1", spec.embed_dim);

    nvinfer1::ITensor *attn_out = nullptr;
    if (spec.global_attn_indexes.count(index) != 0)
    {
        auto *tokens = flattenNHWC(network, *norm1, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
        auto *attn_tokens
            = addImageAttention(network, weights_map, *tokens, prefix, geometry.batch, geometry.grid_h, geometry.grid_w,
                                spec);
        attn_out = unflattenNHWC(network, *attn_tokens, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    }
    else
    {
        int padded_h = 0;
        int padded_w = 0;
        auto *windows = partitionWindows(network, *norm1, geometry.batch, geometry.grid_h, geometry.grid_w,
                                         spec.embed_dim, 14, padded_h, padded_w);
        const int window_batch = (padded_h / 14) * (padded_w / 14) * geometry.batch;
        auto     *tokens       = flattenNHWC(network, *windows, window_batch, 14, 14, spec.embed_dim);
        auto     *attn_tokens  = addImageAttention(network, weights_map, *tokens, prefix, window_batch, 14, 14, spec);
        auto     *attn_windows = unflattenNHWC(network, *attn_tokens, window_batch, 14, 14, spec.embed_dim);
        attn_out = unpartitionWindows(network, *attn_windows, geometry.batch, geometry.grid_h, geometry.grid_w,
                                      spec.embed_dim, 14, padded_h, padded_w);
    }

    auto *attn_residual = requireLayer(network->addElementWise(input, *attn_out, E::kSUM),
                                       "Failed to add SAM image attention residual")
                              ->getOutput(0);

    auto *norm2 = addLayerNormLastDim(network, weights_map, *attn_residual, prefix + ".norm2", spec.embed_dim);
    auto *tokens = flattenNHWC(network, *norm2, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    auto *fc1 = addLinear3D(network, weights_map, *tokens, prefix + ".mlp.lin1", spec.embed_dim, spec.embed_dim * 4);
    auto *gelu = addGeluExact(network, *fc1);
    auto *fc2 = addLinear3D(network, weights_map, *gelu, prefix + ".mlp.lin2", spec.embed_dim * 4, spec.embed_dim);
    auto *mlp = unflattenNHWC(network, *fc2, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    return requireLayer(network->addElementWise(*attn_residual, *mlp, E::kSUM), "Failed to add SAM image MLP residual")
        ->getOutput(0);
}

/**
 * @brief 判断整数集合中是否包含指定值。
 */
bool containsIndex(const std::vector<int> &values, int target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

/**
 * @brief 面向 NHWC 特征图添加线性层。
 */
nvinfer1::ITensor *addLinearNHWC(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, const std::string &prefix, int batch, int height,
                                 int width, int in_channels, int out_channels)
{
    auto *tokens = flattenNHWC(network, input, batch, height, width, in_channels);
    auto *linear = addLinear3D(network, weights_map, *tokens, prefix, in_channels, out_channels);
    return unflattenNHWC(network, *linear, batch, height, width, out_channels);
}

/**
 * @brief 对 NHWC 特征图执行官方 Hiera q pooling。
 */
nvinfer1::ITensor *addMaxPoolNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *nchw = nhwcToNchw(network, input);
    auto *pool = requireLayer(network->addPoolingNd(*nchw, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2}),
                              "Failed to add SAM2 Hiera max pool");
    pool->setStrideNd(nvinfer1::DimsHW{2, 2});
    return nchwToNhwc(network, *pool->getOutput(0));
}

/**
 * @brief 将 NCHW 特征图按最近邻方式上采样到参考特征图大小。
 */
nvinfer1::ITensor *addNearestResizeLike(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                        const nvinfer1::ITensor &reference)
{
    auto *resize = requireLayer(network->addResize(input), "Failed to add SAM2 nearest resize");
    auto  dims   = input.getDimensions();
    const auto ref_dims = reference.getDimensions();
    if (dims.nbDims != 4 || ref_dims.nbDims != 4)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM2 nearest resize expects NCHW tensors");
    }
    dims.d[2] = ref_dims.d[2];
    dims.d[3] = ref_dims.d[3];
    resize->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    resize->setOutputDimensions(dims);
    return resize->getOutput(0);
}

/**
 * @brief 生成 SAM2 Hiera 官方窗口位置编码。
 */
nvinfer1::ITensor *addSAM2HieraPositionEmbedding(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                                 const SAM2HieraSpec &spec, int grid_h, int grid_w)
{
    const std::string prefix = "image_encoder.trunk";
    auto *base = requireLayer(
        network->addConstant(nvinfer1::Dims4{1, spec.embed_dim, spec.pos_embed_size, spec.pos_embed_size},
                             requireWeight(weights_map, prefix + ".pos_embed",
                                           static_cast<int64_t>(spec.embed_dim) * spec.pos_embed_size
                                               * spec.pos_embed_size)),
        "Failed to add SAM2 Hiera base position embedding");
    auto *resize = requireLayer(network->addResize(*base->getOutput(0)), "Failed to resize SAM2 Hiera pos_embed");
    resize->setResizeMode(nvinfer1::InterpolationMode::kCUBIC);
    /**
     * @brief 对齐官方 PyTorch 的 bicubic 插值语义。
     *
     * SAM2 Hiera 在官方实现中使用
     * ``F.interpolate(..., mode="bicubic", align_corners=False)`` 对绝对位置编码做插值。
     * TensorRT Resize 默认使用 ASYMMETRIC 坐标，若不显式设置会导致 image encoder
     * 从第一个 block 开始出现数值偏差；HALF_PIXEL 与 PyTorch 的 align_corners=False 对齐。
     */
    resize->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    resize->setCubicCoeff(-0.75F);
    resize->setOutputDimensions(nvinfer1::Dims4{1, spec.embed_dim, grid_h, grid_w});

    const int window_size = spec.window_spec.front();
    const auto &window_weight = requireWeight(weights_map, prefix + ".pos_embed_window",
                                             static_cast<int64_t>(spec.embed_dim) * window_size * window_size);
    const auto *window_values = static_cast<const float *>(window_weight.values);
    std::vector<float> tiled(static_cast<size_t>(spec.embed_dim) * grid_h * grid_w);
    for (int c = 0; c < spec.embed_dim; ++c)
    {
        for (int y = 0; y < grid_h; ++y)
        {
            for (int x = 0; x < grid_w; ++x)
            {
                const auto dst = (static_cast<size_t>(c) * grid_h + y) * grid_w + x;
                const auto src = (static_cast<size_t>(c) * window_size + (y % window_size)) * window_size
                               + (x % window_size);
                tiled[dst] = window_values[src];
            }
        }
    }
    auto *window = requireLayer(network->addConstant(nvinfer1::Dims4{1, spec.embed_dim, grid_h, grid_w},
                                                     ownedFloatVector(std::move(tiled))),
                                "Failed to add SAM2 Hiera window position embedding");
    auto *sum = requireLayer(network->addElementWise(*resize->getOutput(0), *window->getOutput(0), E::kSUM),
                             "Failed to add SAM2 Hiera position embedding");
    return nchwToNhwc(network, *sum->getOutput(0));
}

/**
 * @brief 添加 SAM2 Hiera 多尺度注意力。
 */
nvinfer1::ITensor *addSAM2HieraAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                         nvinfer1::ITensor &input, const std::string &prefix, int batch, int height,
                                         int width, int dim_in, int dim_out, int num_heads, bool q_pool,
                                         int &out_h, int &out_w)
{
    const int token_count = height * width;
    auto     *tokens      = flattenNHWC(network, input, batch, height, width, dim_in);
    auto *qkv = addLinear3D(network, weights_map, *tokens, prefix + ".attn.qkv", dim_in, 3 * dim_out);
    nvinfer1::ITensor *q = nullptr;
    nvinfer1::ITensor *k = nullptr;
    nvinfer1::ITensor *v = nullptr;
    splitQkv(network, *qkv, batch, token_count, dim_out, q, k, v);

    out_h = height;
    out_w = width;
    if (q_pool)
    {
        auto *q_map    = unflattenNHWC(network, *q, batch, height, width, dim_out);
        auto *pooled_q = addMaxPoolNHWC(network, *q_map);
        out_h /= 2;
        out_w /= 2;
        q = flattenNHWC(network, *pooled_q, batch, out_h, out_w, dim_out);
    }

    auto *attn = addTokenAttention(network, *q, *k, *v, batch, out_h * out_w, token_count, num_heads, dim_out);
    auto *proj = addLinear3D(network, weights_map, *attn, prefix + ".attn.proj", dim_out, dim_out);
    return unflattenNHWC(network, *proj, batch, out_h, out_w, dim_out);
}

/**
 * @brief 添加 SAM2 Hiera 的一个 MultiScaleBlock。
 */
nvinfer1::ITensor *addSAM2HieraBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &input, int index, int batch, int &height, int &width,
                                     int dim_in, int dim_out, int num_heads, int window_size, bool q_pool)
{
    const std::string prefix = "image_encoder.trunk.blocks." + std::to_string(index);
    auto             *norm1  = addLayerNormLastDim(network, weights_map, input, prefix + ".norm1", dim_in);

    nvinfer1::ITensor *shortcut = &input;
    if (dim_in != dim_out)
    {
        shortcut = addLinearNHWC(network, weights_map, *norm1, prefix + ".proj", batch, height, width, dim_in, dim_out);
        if (q_pool)
        {
            shortcut = addMaxPoolNHWC(network, *shortcut);
        }
    }

    nvinfer1::ITensor *attn_input = norm1;
    int                attn_batch = batch;
    int                attn_h     = height;
    int                attn_w     = width;
    int                padded_h   = height;
    int                padded_w   = width;
    if (window_size > 0)
    {
        attn_input = partitionWindows(network, *norm1, batch, height, width, dim_in, window_size, padded_h, padded_w);
        attn_batch = batch * (padded_h / window_size) * (padded_w / window_size);
        attn_h = window_size;
        attn_w = window_size;
    }

    int out_h = attn_h;
    int out_w = attn_w;
    auto *attn = addSAM2HieraAttention(network, weights_map, *attn_input, prefix, attn_batch, attn_h, attn_w, dim_in,
                                       dim_out, num_heads, q_pool, out_h, out_w);
    if (window_size > 0)
    {
        const int restore_window = q_pool ? window_size / 2 : window_size;
        const int target_h       = q_pool ? height / 2 : height;
        const int target_w       = q_pool ? width / 2 : width;
        const int restore_h      = q_pool ? padded_h / 2 : padded_h;
        const int restore_w      = q_pool ? padded_w / 2 : padded_w;
        attn = unpartitionWindows(network, *attn, batch, target_h, target_w, dim_out, restore_window, restore_h,
                                  restore_w);
        out_h = target_h;
        out_w = target_w;
    }

    auto *x = requireLayer(network->addElementWise(*shortcut, *attn, E::kSUM),
                           "Failed to add SAM2 Hiera attention residual")
                  ->getOutput(0);
    auto *norm2 = addLayerNormLastDim(network, weights_map, *x, prefix + ".norm2", dim_out);
    auto *mlp0 = addLinearNHWC(network, weights_map, *norm2, prefix + ".mlp.layers.0", batch, out_h, out_w, dim_out,
                               dim_out * 4);
    auto *gelu = addGeluExact(network, *mlp0);
    auto *mlp1 = addLinearNHWC(network, weights_map, *gelu, prefix + ".mlp.layers.1", batch, out_h, out_w,
                               dim_out * 4, dim_out);
    height = out_h;
    width  = out_w;
    return requireLayer(network->addElementWise(*x, *mlp1, E::kSUM), "Failed to add SAM2 Hiera MLP residual")
        ->getOutput(0);
}

/**
 * @brief 构建官方 SAM2 Hiera trunk，返回从高分辨率到低分辨率的 stage 输出。
 */
std::vector<nvinfer1::ITensor *> addSAM2HieraTrunk(const SAMSegmentationModel &impl,
                                                   nvinfer1::INetworkDefinition *network,
                                                   const WeightsMap &weights_map, const SAMGeometry &geometry,
                                                   const SAM2HieraSpec &spec,
                                                   priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *image = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 0);
    named_tensors["image"] = image;

    auto *patch = requireLayer(
        network->addConvolutionNd(*image, spec.embed_dim, nvinfer1::DimsHW{7, 7},
                                  requireWeight(weights_map, "image_encoder.trunk.patch_embed.proj.weight",
                                                static_cast<int64_t>(spec.embed_dim) * geometry.channels * 7 * 7),
                                  requireWeight(weights_map, "image_encoder.trunk.patch_embed.proj.bias",
                                                spec.embed_dim)),
        "Failed to add SAM2 Hiera patch embedding");
    patch->setStrideNd(nvinfer1::DimsHW{kSam2PatchSize, kSam2PatchSize});
    patch->setPaddingNd(nvinfer1::DimsHW{3, 3});

    int   height = geometry.image_h / kSam2PatchSize;
    int   width  = geometry.image_w / kSam2PatchSize;
    auto *x      = nchwToNhwc(network, *patch->getOutput(0));
    auto *pos    = addSAM2HieraPositionEmbedding(network, weights_map, spec, height, width);
    x = requireLayer(network->addElementWise(*x, *pos, E::kSUM), "Failed to add SAM2 Hiera position embedding")
            ->getOutput(0);

    std::vector<int> stage_ends;
    stage_ends.reserve(spec.stages.size());
    int depth = 0;
    for (const int blocks : spec.stages)
    {
        depth += blocks;
        stage_ends.push_back(depth - 1);
    }
    std::vector<int> q_pool_blocks;
    for (size_t i = 0; i + 1 < stage_ends.size() && static_cast<int>(q_pool_blocks.size()) < spec.q_pool; ++i)
    {
        q_pool_blocks.push_back(stage_ends[i] + 1);
    }

    std::vector<nvinfer1::ITensor *> outputs;
    int cur_stage = 1;
    int dim       = spec.embed_dim;
    int heads     = spec.num_heads;
    for (int i = 0; i < depth; ++i)
    {
        int window_size = spec.window_spec.at(static_cast<size_t>(cur_stage - 1));
        if (spec.global_attn_indexes.count(i) != 0)
        {
            window_size = 0;
        }

        int dim_out = dim;
        if (containsIndex(stage_ends, i - 1))
        {
            dim_out = dim * 2;
            heads *= 2;
            ++cur_stage;
        }
        const bool q_pool = containsIndex(q_pool_blocks, i);
        x = addSAM2HieraBlock(network, weights_map, *x, i, geometry.batch, height, width, dim, dim_out, heads,
                              window_size, q_pool);
        dim = dim_out;
        named_tensors["image_encoder.trunk.blocks." + std::to_string(i)] = x;
        if (containsIndex(stage_ends, i))
        {
            auto *stage = nhwcToNchw(network, *x);
            outputs.push_back(stage);
            named_tensors["image_encoder.trunk.stage" + std::to_string(outputs.size())] = stage;
        }
    }
    return outputs;
}

/**
 * @brief 构建官方 SAM2 Hiera + FPN image encoder。
 */
nvinfer1::ITensor *addSAM2ImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                       const WeightsMap &weights_map, const SAMGeometry &geometry,
                                       const SAM2HieraSpec &spec, nvinfer1::ITensor *&high_res_s0,
                                       nvinfer1::ITensor *&high_res_s1,
                                       priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto trunk_outputs = addSAM2HieraTrunk(impl, network, weights_map, geometry, spec, named_tensors);
    if (trunk_outputs.size() != 4 || spec.backbone_channels.size() != 4)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "SAM2 Hiera expects four FPN levels");
    }

    std::vector<nvinfer1::ITensor *> fpn(4, nullptr);
    nvinfer1::ITensor *prev = nullptr;
    const int n = static_cast<int>(trunk_outputs.size()) - 1;
    for (int i = n; i >= 0; --i)
    {
        const int conv_index = n - i;
        const int channels   = spec.backbone_channels.at(static_cast<size_t>(conv_index));
        auto *lateral = requireLayer(
            network->addConvolutionNd(*trunk_outputs.at(static_cast<size_t>(i)), kSamPromptDim,
                                      nvinfer1::DimsHW{1, 1},
                                      requireWeight(weights_map,
                                                    "image_encoder.neck.convs." + std::to_string(conv_index)
                                                        + ".conv.weight",
                                                    static_cast<int64_t>(kSamPromptDim) * channels),
                                      requireWeight(weights_map,
                                                    "image_encoder.neck.convs." + std::to_string(conv_index)
                                                        + ".conv.bias",
                                                    kSamPromptDim)),
            "Failed to add SAM2 FPN lateral conv");
        nvinfer1::ITensor *out = lateral->getOutput(0);
        if ((i == 2 || i == 3) && prev != nullptr)
        {
            auto *top_down = addNearestResizeLike(network, *prev, *out);
            out = requireLayer(network->addElementWise(*out, *top_down, E::kSUM),
                               "Failed to add SAM2 FPN top-down feature")
                      ->getOutput(0);
        }
        fpn.at(static_cast<size_t>(i)) = out;
        prev = out;
        named_tensors["image_encoder.fpn." + std::to_string(i)] = out;
    }

    auto *conv_s0 = requireLayer(
        network->addConvolutionNd(*fpn[0], kSamPromptDim / 8, nvinfer1::DimsHW{1, 1},
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s0.weight",
                                                static_cast<int64_t>(kSamPromptDim / 8) * kSamPromptDim),
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s0.bias", kSamPromptDim / 8)),
        "Failed to add SAM2 high-res s0 projection");
    auto *conv_s1 = requireLayer(
        network->addConvolutionNd(*fpn[1], kSamPromptDim / 4, nvinfer1::DimsHW{1, 1},
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s1.weight",
                                                static_cast<int64_t>(kSamPromptDim / 4) * kSamPromptDim),
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s1.bias", kSamPromptDim / 4)),
        "Failed to add SAM2 high-res s1 projection");
    high_res_s0 = conv_s0->getOutput(0);
    high_res_s1 = conv_s1->getOutput(0);

    named_tensors["high_res_s0"]     = high_res_s0;
    named_tensors["high_res_s1"]     = high_res_s1;
    named_tensors["image_embedding"] = fpn[2];
    return fpn[2];
}

/**
 * @brief 解析并校验 SAM 输入配置。
 */
SAMGeometry resolveSAMGeometry(const SAMSpec &spec, const IModelConfig &config)
{
    const auto &input_shapes = config.inputShapes();
    if (input_shapes.size() != kDefaultInputNames.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM expects %zu input tensors, got %zu",
                             kDefaultInputNames.size(), input_shapes.size());
    }
    if (config.inputTensorNames().size() != input_shapes.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM input tensor name count must match input shape count");
    }
    if (config.outputTensorNames().size() != kDefaultOutputNames.size() && !config.featureOnly())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM expects %zu output tensors, got %zu",
                             kDefaultOutputNames.size(), config.outputTensorNames().size());
    }

    const auto &image_shape = input_shapes[0];
    SAMGeometry geometry{};
    geometry.batch    = static_cast<int>(image_shape.d[0]);
    geometry.channels = static_cast<int>(image_shape.d[1]);
    geometry.image_h  = static_cast<int>(image_shape.d[2]);
    geometry.image_w  = static_cast<int>(image_shape.d[3]);
    if (geometry.batch != 1 || geometry.channels != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM image input must be 1x3xHxW, got %dx%dx%dx%d",
                             image_shape.d[0], image_shape.d[1], image_shape.d[2], image_shape.d[3]);
    }
    if (geometry.image_h != spec.image_size || geometry.image_w != spec.image_size)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s expects fixed image size %dx%d, got %dx%d",
                             spec.display_name, spec.image_size, spec.image_size, geometry.image_h, geometry.image_w);
    }
    if (geometry.image_h % kSamPatchSize != 0 || geometry.image_w % kSamPatchSize != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM image size must be divisible by patch size %d",
                             kSamPatchSize);
    }
    geometry.grid_h      = geometry.image_h / kSamPatchSize;
    geometry.grid_w      = geometry.image_w / kSamPatchSize;
    geometry.grid_tokens = geometry.grid_h * geometry.grid_w;

    const auto &coords_shape = input_shapes[1];
    if (coords_shape.d[0] != 1 || coords_shape.d[1] != spec.max_points || coords_shape.d[2] != 2
        || coords_shape.d[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM point_coords must be 1x%dx2x1", spec.max_points);
    }

    const auto &labels_shape = input_shapes[2];
    if (labels_shape.d[0] != 1 || labels_shape.d[1] != spec.max_points || labels_shape.d[2] != 1
        || labels_shape.d[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM point_labels must be 1x%dx1x1", spec.max_points);
    }

    const auto &mask_shape = input_shapes[3];
    if (mask_shape.d[0] != 1 || mask_shape.d[1] != 1 || mask_shape.d[2] != spec.mask_size
        || mask_shape.d[3] != spec.mask_size)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM mask_input must be 1x1x%dx%d", spec.mask_size,
                             spec.mask_size);
    }

    const auto &has_mask_shape = input_shapes[4];
    if (has_mask_shape.d[0] != 1 || has_mask_shape.d[1] != 1 || has_mask_shape.d[2] != 1 || has_mask_shape.d[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM has_mask_input must be 1x1x1x1");
    }

    return geometry;
}

/**
 * @brief 构建官方 SAM v1 image encoder。
 */
nvinfer1::ITensor *addSAMImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                       const WeightsMap &weights_map, const SAMGeometry &geometry,
                                       const SAMViTSpec &spec,
                                       priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *image = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 0);
    named_tensors["image"] = image;

    auto *patch = requireLayer(
        network->addConvolutionNd(
            *image, spec.embed_dim, nvinfer1::DimsHW{kSamPatchSize, kSamPatchSize},
            requireWeight(weights_map, "image_encoder.patch_embed.proj.weight",
                          static_cast<int64_t>(spec.embed_dim) * geometry.channels * kSamPatchSize * kSamPatchSize),
            requireWeight(weights_map, "image_encoder.patch_embed.proj.bias", spec.embed_dim)),
        "Failed to add SAM patch embedding");
    patch->setStrideNd(nvinfer1::DimsHW{kSamPatchSize, kSamPatchSize});
    auto *x = nchwToNhwc(network, *patch->getOutput(0));

    auto *pos = requireLayer(network->addConstant(
                                 nvinfer1::Dims4{1, geometry.grid_h, geometry.grid_w, spec.embed_dim},
                                 requireWeight(weights_map, "image_encoder.pos_embed",
                                               static_cast<int64_t>(geometry.grid_tokens) * spec.embed_dim)),
                             "Failed to add SAM image pos_embed");
    x = requireLayer(network->addElementWise(*x, *pos->getOutput(0), E::kSUM),
                     "Failed to add SAM image pos embedding")
            ->getOutput(0);
    named_tensors["image_tokens"] = x;

    for (int i = 0; i < spec.depth; ++i)
    {
        x = addImageBlock(network, weights_map, *x, i, geometry, spec);
        named_tensors["image_encoder.block" + std::to_string(i)] = x;
    }

    auto *nchw = nhwcToNchw(network, *x);
    auto *neck0 = requireLayer(network->addConvolutionNd(
                                   *nchw, kSamPromptDim, nvinfer1::DimsHW{1, 1},
                                   requireWeight(weights_map, "image_encoder.neck.0.weight",
                                                 static_cast<int64_t>(kSamPromptDim) * spec.embed_dim),
                                   emptyWeights()),
                               "Failed to add SAM neck conv0");
    auto *neck1 = addLayerNorm2d(network, weights_map, *neck0->getOutput(0), "image_encoder.neck.1", kSamPromptDim);
    auto *neck2 = requireLayer(network->addConvolutionNd(
                                   *neck1, kSamPromptDim, nvinfer1::DimsHW{3, 3},
                                   requireWeight(weights_map, "image_encoder.neck.2.weight",
                                                 static_cast<int64_t>(kSamPromptDim) * kSamPromptDim * 3 * 3),
                                   emptyWeights()),
                               "Failed to add SAM neck conv1");
    neck2->setPaddingNd(nvinfer1::DimsHW{1, 1});
    auto *embedding = addLayerNorm2d(network, weights_map, *neck2->getOutput(0), "image_encoder.neck.3", kSamPromptDim);
    named_tensors["image_embedding"] = embedding;
    return embedding;
}

/**
 * @brief 根据官方随机傅里叶位置编码权重生成 dense PE 常量。
 */
nvinfer1::ITensor *addDensePromptPE(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    const std::string &prompt_prefix)
{
    const auto &gaussian = requireWeight(weights_map, prompt_prefix + ".pe_layer.positional_encoding_gaussian_matrix",
                                         2 * (kSamPromptDim / 2));
    const auto *g = static_cast<const float *>(gaussian.values);

    std::vector<float> values(static_cast<size_t>(kSamPromptDim) * kSamEmbedGrid * kSamEmbedGrid);
    for (int y = 0; y < kSamEmbedGrid; ++y)
    {
        for (int x = 0; x < kSamEmbedGrid; ++x)
        {
            const float nx = (static_cast<float>(x) + 0.5F) / static_cast<float>(kSamEmbedGrid);
            const float ny = (static_cast<float>(y) + 0.5F) / static_cast<float>(kSamEmbedGrid);
            const float px = 2.0F * nx - 1.0F;
            const float py = 2.0F * ny - 1.0F;
            for (int c = 0; c < kSamPromptDim / 2; ++c)
            {
                const float projected = 2.0F * kPi * (px * g[c] + py * g[(kSamPromptDim / 2) + c]);
                const auto  base      = static_cast<size_t>(y) * kSamEmbedGrid + x;
                values[static_cast<size_t>(c) * kSamEmbedGrid * kSamEmbedGrid + base] = std::sin(projected);
                values[static_cast<size_t>(c + kSamPromptDim / 2) * kSamEmbedGrid * kSamEmbedGrid + base]
                    = std::cos(projected);
            }
        }
    }

    auto *pe = requireLayer(network->addConstant(nvinfer1::Dims4{1, kSamPromptDim, kSamEmbedGrid, kSamEmbedGrid},
                                                 ownedFloatVector(std::move(values))),
                            "Failed to add SAM dense prompt PE");
    return pe->getOutput(0);
}

/**
 * @brief 生成 label 等于 target 时的浮点 gate。
 */
nvinfer1::ITensor *addLabelGate(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &labels, float target)
{
    auto *target_const = addScalar(network, labels, target);
    auto *diff = requireLayer(network->addElementWise(labels, *target_const, E::kSUB), "Failed to add SAM label diff");
    auto *abs = requireLayer(network->addUnary(*diff->getOutput(0), U::kABS), "Failed to add SAM label abs");
    auto *one = addScalar(network, labels, 1.0F);
    auto *raw = requireLayer(network->addElementWise(*one, *abs->getOutput(0), E::kSUB), "Failed to add SAM label gate");
    auto *zero = addScalar(network, labels, 0.0F);
    return requireLayer(network->addElementWise(*raw->getOutput(0), *zero, E::kMAX), "Failed to add SAM label clamp")
        ->getOutput(0);
}

/**
 * @brief 添加一个由 label gate 选择的 embedding。
 */
nvinfer1::ITensor *addGatedEmbedding(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &gate, const std::string &key)
{
    auto *embedding = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim},
                                                        requireWeight(weights_map, key, kSamPromptDim)),
                                   "Failed to add SAM prompt embedding");
    return requireLayer(network->addElementWise(*embedding->getOutput(0), gate, E::kPROD),
                        "Failed to add SAM gated prompt embedding")
        ->getOutput(0);
}

/**
 * @brief 构建官方点 prompt sparse embedding。
 */
nvinfer1::ITensor *addPointPromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map, const SAMGeometry &geometry,
                                           const std::string &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *point_coords = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 1);
    auto *point_labels = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 2);
    named_tensors["point_coords"] = point_coords;
    named_tensors["point_labels"] = point_labels;

    auto *coords = requireLayer(network->addShuffle(*point_coords), "Failed to reshape SAM point coords");
    coords->setReshapeDimensions(nvinfer1::Dims3{1, kSamMaxPoints, 2});
    auto *half = addScalar(network, *coords->getOutput(0), 0.5F);
    auto *shifted = requireLayer(network->addElementWise(*coords->getOutput(0), *half, E::kSUM),
                                 "Failed to shift SAM point coords");
    std::vector<float> inv_size{1.0F / static_cast<float>(geometry.image_w), 1.0F / static_cast<float>(geometry.image_h)};
    auto *norm_const = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 2}, ownedFloatVector(inv_size)),
                                    "Failed to add SAM point norm");
    auto *normalized = requireLayer(network->addElementWise(*shifted->getOutput(0), *norm_const->getOutput(0), E::kPROD),
                                    "Failed to normalize SAM point coords");
    auto *pad_coord = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 2}, ownedFloatVector({0.0F, 0.0F})),
                                   "Failed to add SAM padding point");
    std::array<nvinfer1::ITensor *, 2> coord_tensors{normalized->getOutput(0), pad_coord->getOutput(0)};
    auto *coords_cat = requireLayer(network->addConcatenation(coord_tensors.data(), 2), "Failed to concat SAM coords");
    coords_cat->setAxis(1);

    auto *two = addScalar(network, *coords_cat->getOutput(0), 2.0F);
    auto *one = addScalar(network, *coords_cat->getOutput(0), 1.0F);
    auto *double_coords = requireLayer(network->addElementWise(*coords_cat->getOutput(0), *two, E::kPROD),
                                       "Failed to double SAM coords");
    auto *pe_coords = requireLayer(network->addElementWise(*double_coords->getOutput(0), *one, E::kSUB),
                                   "Failed to center SAM coords");

    auto *gaussian = requireLayer(
        network->addConstant(nvinfer1::Dims3{1, 2, kSamPromptDim / 2},
                             requireWeight(weights_map, prompt_prefix + ".pe_layer.positional_encoding_gaussian_matrix",
                                           2 * (kSamPromptDim / 2))),
        "Failed to add SAM point PE gaussian");
    auto *projected = requireLayer(network->addMatrixMultiply(*pe_coords->getOutput(0), M::kNONE,
                                                              *gaussian->getOutput(0), M::kNONE),
                                   "Failed to add SAM point PE projection");
    auto *two_pi = addScalar(network, *projected->getOutput(0), 2.0F * kPi);
    auto *phase = requireLayer(network->addElementWise(*projected->getOutput(0), *two_pi, E::kPROD),
                               "Failed to scale SAM point PE");
    auto *sin = requireLayer(network->addUnary(*phase->getOutput(0), U::kSIN), "Failed to add SAM point PE sin");
    auto *cos = requireLayer(network->addUnary(*phase->getOutput(0), U::kCOS), "Failed to add SAM point PE cos");
    std::array<nvinfer1::ITensor *, 2> pe_parts{sin->getOutput(0), cos->getOutput(0)};
    auto *pe = requireLayer(network->addConcatenation(pe_parts.data(), 2), "Failed to concat SAM point PE");
    pe->setAxis(2);

    auto *labels = requireLayer(network->addShuffle(*point_labels), "Failed to reshape SAM point labels");
    labels->setReshapeDimensions(nvinfer1::Dims3{1, kSamMaxPoints, 1});
    auto *pad_label = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 1}, ownedScalarWeight(-1.0F)),
                                   "Failed to add SAM padding label");
    std::array<nvinfer1::ITensor *, 2> label_tensors{labels->getOutput(0), pad_label->getOutput(0)};
    auto *labels_cat = requireLayer(network->addConcatenation(label_tensors.data(), 2), "Failed to concat SAM labels");
    labels_cat->setAxis(1);

    auto *neg_gate  = addLabelGate(network, *labels_cat->getOutput(0), -1.0F);
    auto *zero_gate = addLabelGate(network, *labels_cat->getOutput(0), 0.0F);
    auto *pos_gate  = addLabelGate(network, *labels_cat->getOutput(0), 1.0F);
    auto *box_tl_gate = addLabelGate(network, *labels_cat->getOutput(0), 2.0F);
    auto *box_br_gate = addLabelGate(network, *labels_cat->getOutput(0), 3.0F);
    auto *keep_pe = requireLayer(network->addElementWise(*addScalar(network, *neg_gate, 1.0F), *neg_gate, E::kSUB),
                                 "Failed to add SAM point PE keep");
    auto *pe_kept = requireLayer(network->addElementWise(*pe->getOutput(0), *keep_pe->getOutput(0), E::kPROD),
                                 "Failed to apply SAM point PE keep");

    auto *not_point = addGatedEmbedding(network, weights_map, *neg_gate, prompt_prefix + ".not_a_point_embed.weight");
    auto *neg_point = addGatedEmbedding(network, weights_map, *zero_gate, prompt_prefix + ".point_embeddings.0.weight");
    auto *pos_point = addGatedEmbedding(network, weights_map, *pos_gate, prompt_prefix + ".point_embeddings.1.weight");
    auto *box_tl = addGatedEmbedding(network, weights_map, *box_tl_gate, prompt_prefix + ".point_embeddings.2.weight");
    auto *box_br = addGatedEmbedding(network, weights_map, *box_br_gate, prompt_prefix + ".point_embeddings.3.weight");
    auto *tmp = requireLayer(network->addElementWise(*pe_kept->getOutput(0), *not_point, E::kSUM),
                             "Failed to add SAM not-a-point embedding");
    tmp = requireLayer(network->addElementWise(*tmp->getOutput(0), *neg_point, E::kSUM),
                       "Failed to add SAM negative point embedding");
    tmp = requireLayer(network->addElementWise(*tmp->getOutput(0), *pos_point, E::kSUM),
                       "Failed to add SAM positive point embedding");
    tmp = requireLayer(network->addElementWise(*tmp->getOutput(0), *box_tl, E::kSUM),
                       "Failed to add SAM box top-left embedding");
    auto *sparse = requireLayer(network->addElementWise(*tmp->getOutput(0), *box_br, E::kSUM),
                                "Failed to add SAM box bottom-right embedding")
                       ->getOutput(0);
    named_tensors["sparse_prompt_embedding"] = sparse;
    return sparse;
}

/**
 * @brief 构建官方 mask prompt dense embedding。
 */
nvinfer1::ITensor *addDensePromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map,
                                           const std::string &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *mask_input     = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 3);
    auto *has_mask_input = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 4);
    named_tensors["mask_input"]     = mask_input;
    named_tensors["has_mask_input"] = has_mask_input;

    auto *conv0 = requireLayer(network->addConvolutionNd(
                                   *mask_input, 4, nvinfer1::DimsHW{2, 2},
                                   requireWeight(weights_map, prompt_prefix + ".mask_downscaling.0.weight", 4 * 1 * 2 * 2),
                                   requireWeight(weights_map, prompt_prefix + ".mask_downscaling.0.bias", 4)),
                               "Failed to add SAM mask prompt conv0");
    conv0->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto *norm0 = addLayerNorm2d(network, weights_map, *conv0->getOutput(0), prompt_prefix + ".mask_downscaling.1", 4);
    auto *gelu0 = addGeluExact(network, *norm0);
    auto *conv1 = requireLayer(network->addConvolutionNd(
                                   *gelu0, 16, nvinfer1::DimsHW{2, 2},
                                   requireWeight(weights_map, prompt_prefix + ".mask_downscaling.3.weight", 16 * 4 * 2 * 2),
                                   requireWeight(weights_map, prompt_prefix + ".mask_downscaling.3.bias", 16)),
                               "Failed to add SAM mask prompt conv1");
    conv1->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto *norm1 = addLayerNorm2d(network, weights_map, *conv1->getOutput(0), prompt_prefix + ".mask_downscaling.4", 16);
    auto *gelu1 = addGeluExact(network, *norm1);
    auto *mask_embedding = requireLayer(
        network->addConvolutionNd(*gelu1, kSamPromptDim, nvinfer1::DimsHW{1, 1},
                                  requireWeight(weights_map, prompt_prefix + ".mask_downscaling.6.weight",
                                                static_cast<int64_t>(kSamPromptDim) * 16),
                                  requireWeight(weights_map, prompt_prefix + ".mask_downscaling.6.bias", kSamPromptDim)),
        "Failed to add SAM mask prompt conv2");

    auto *has_part = requireLayer(network->addElementWise(*mask_embedding->getOutput(0), *has_mask_input, E::kPROD),
                                  "Failed to apply SAM has-mask gate");
    auto *one = addScalar(network, *has_mask_input, 1.0F);
    auto *no_mask_gate = requireLayer(network->addElementWise(*one, *has_mask_input, E::kSUB),
                                      "Failed to add SAM no-mask gate");
    auto *no_mask = requireLayer(network->addConstant(nvinfer1::Dims4{1, kSamPromptDim, 1, 1},
                                                      requireWeight(weights_map, prompt_prefix + ".no_mask_embed.weight",
                                                                    kSamPromptDim)),
                                 "Failed to add SAM no-mask embedding");
    auto *no_mask_part = requireLayer(network->addElementWise(*no_mask->getOutput(0), *no_mask_gate->getOutput(0),
                                                              E::kPROD),
                                      "Failed to apply SAM no-mask gate");
    auto *dense = requireLayer(network->addElementWise(*has_part->getOutput(0), *no_mask_part->getOutput(0), E::kSUM),
                               "Failed to add SAM dense prompt embedding")
                      ->getOutput(0);
    named_tensors["dense_prompt_embedding"] = dense;
    return dense;
}

/**
 * @brief 添加 TwoWayTransformer 中的 Attention 子层。
 */
nvinfer1::ITensor *addDecoderAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &q_input, nvinfer1::ITensor &k_input,
                                       nvinfer1::ITensor &v_input, const std::string &prefix, int q_tokens,
                                       int k_tokens, int downsample_rate)
{
    const int internal_dim = kSamPromptDim / downsample_rate;
    auto     *q = addLinear3D(network, weights_map, q_input, prefix + ".q_proj", kSamPromptDim, internal_dim);
    auto     *k = addLinear3D(network, weights_map, k_input, prefix + ".k_proj", kSamPromptDim, internal_dim);
    auto     *v = addLinear3D(network, weights_map, v_input, prefix + ".v_proj", kSamPromptDim, internal_dim);
    auto     *attn = addTokenAttention(network, *q, *k, *v, 1, q_tokens, k_tokens, kSamTwoWayHeads, internal_dim);
    return addLinear3D(network, weights_map, *attn, prefix + ".out_proj", internal_dim, kSamPromptDim);
}

/**
 * @brief 添加 TwoWayTransformer 的一个官方 block。
 */
void addTwoWayBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor *&queries,
                    nvinfer1::ITensor *&keys, nvinfer1::ITensor &query_pe, nvinfer1::ITensor &key_pe, int index,
                    int query_tokens, int image_tokens, const std::string &mask_prefix)
{
    const std::string prefix = mask_prefix + ".transformer.layers." + std::to_string(index);
    if (index == 0)
    {
        auto *self_attn = addDecoderAttention(network, weights_map, *queries, *queries, *queries,
                                              prefix + ".self_attn", query_tokens, query_tokens, 1);
        /**
         * @brief 对齐官方 skip_first_layer_pe=True 的第一层 self-attention。
         *
         * 官方 SAM/SAM2 在第一层 TwoWayAttentionBlock 中既不叠加 query PE，
         * 也不做残差加回，而是直接令 ``queries = self_attn(...)``。
         */
        queries = self_attn;
    }
    else
    {
        auto *q = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM),
                               "Failed to add SAM decoder query PE")
                      ->getOutput(0);
        auto *self_attn = addDecoderAttention(network, weights_map, *q, *q, *queries, prefix + ".self_attn",
                                              query_tokens, query_tokens, 1);
        queries = requireLayer(network->addElementWise(*queries, *self_attn, E::kSUM),
                               "Failed to add SAM decoder self-attn residual")
                      ->getOutput(0);
    }
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm1", kSamPromptDim);

    auto *q_cross = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM),
                                 "Failed to add SAM token->image q")
                        ->getOutput(0);
    auto *k_cross = requireLayer(network->addElementWise(*keys, key_pe, E::kSUM), "Failed to add SAM token->image k")
                        ->getOutput(0);
    auto *cross = addDecoderAttention(network, weights_map, *q_cross, *k_cross, *keys,
                                      prefix + ".cross_attn_token_to_image", query_tokens, image_tokens, 2);
    queries = requireLayer(network->addElementWise(*queries, *cross, E::kSUM),
                           "Failed to add SAM token->image residual")
                  ->getOutput(0);
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm2", kSamPromptDim);

    const auto mlp0_prefix
        = resolveLinearPrefix(weights_map, {prefix + ".mlp.lin1", prefix + ".mlp.layers.0"});
    const auto mlp1_prefix
        = resolveLinearPrefix(weights_map, {prefix + ".mlp.lin2", prefix + ".mlp.layers.1"});
    auto *mlp0 = addLinear3D(network, weights_map, *queries, mlp0_prefix, kSamPromptDim, kSamTwoWayMlpDim);
    auto *relu = requireLayer(network->addActivation(*mlp0, nvinfer1::ActivationType::kRELU),
                              "Failed to add SAM decoder MLP ReLU");
    auto *mlp1
        = addLinear3D(network, weights_map, *relu->getOutput(0), mlp1_prefix, kSamTwoWayMlpDim, kSamPromptDim);
    queries = requireLayer(network->addElementWise(*queries, *mlp1, E::kSUM), "Failed to add SAM decoder MLP residual")
                  ->getOutput(0);
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm3", kSamPromptDim);

    auto *q_image = requireLayer(network->addElementWise(*keys, key_pe, E::kSUM), "Failed to add SAM image->token q")
                        ->getOutput(0);
    auto *k_token = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM),
                                 "Failed to add SAM image->token k")
                        ->getOutput(0);
    auto *image_cross = addDecoderAttention(network, weights_map, *q_image, *k_token, *queries,
                                            prefix + ".cross_attn_image_to_token", image_tokens, query_tokens, 2);
    keys = requireLayer(network->addElementWise(*keys, *image_cross, E::kSUM), "Failed to add SAM image token residual")
               ->getOutput(0);
    keys = addLayerNormLastDim(network, weights_map, *keys, prefix + ".norm4", kSamPromptDim);
}

/**
 * @brief 构建官方 mask decoder，返回 low-res masks 与 IoU 预测。
 */
void addSAMMaskDecoder(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                       nvinfer1::ITensor &image_embedding, nvinfer1::ITensor &image_pe,
                       nvinfer1::ITensor &sparse_prompt, nvinfer1::ITensor &dense_prompt,
                       const SAMMaskDecoderOptions &options,
                       nvinfer1::ITensor *high_res_s0, nvinfer1::ITensor *high_res_s1,
                       nvinfer1::ITensor *&masks, nvinfer1::ITensor *&iou_predictions,
                       priv::IModelImpl::NamedTensorMap &named_tensors)
{
    const auto &mask_prefix = options.prefixes.mask;
    if (options.use_high_res_features && (high_res_s0 == nullptr || high_res_s1 == nullptr))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "SAM2 mask decoder requires high-resolution FPN features");
    }

    std::vector<nvinfer1::ITensor *> output_token_parts;
    output_token_parts.reserve(options.pred_obj_scores ? 3 : 2);
    if (options.pred_obj_scores)
    {
        auto *obj_token = requireLayer(
            network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim},
                                 requireWeight(weights_map, mask_prefix + ".obj_score_token.weight", kSamPromptDim)),
            "Failed to add SAM2 object score token");
        output_token_parts.push_back(obj_token->getOutput(0));
    }
    auto *iou_token = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim},
                                                        requireWeight(weights_map, mask_prefix + ".iou_token.weight",
                                                                      kSamPromptDim)),
                                   "Failed to add SAM iou token");
    output_token_parts.push_back(iou_token->getOutput(0));
    auto *mask_tokens = requireLayer(
        network->addConstant(nvinfer1::Dims3{1, kSamMaskTokens, kSamPromptDim},
                             requireWeight(weights_map, mask_prefix + ".mask_tokens.weight",
                                           static_cast<int64_t>(kSamMaskTokens) * kSamPromptDim)),
        "Failed to add SAM mask tokens");
    output_token_parts.push_back(mask_tokens->getOutput(0));
    auto *out_cat = requireLayer(
        network->addConcatenation(output_token_parts.data(), static_cast<int32_t>(output_token_parts.size())),
        "Failed to concat SAM output tokens");
    out_cat->setAxis(1);
    std::array<nvinfer1::ITensor *, 2> all_tokens{out_cat->getOutput(0), &sparse_prompt};
    auto *token_cat = requireLayer(network->addConcatenation(all_tokens.data(), 2), "Failed to concat SAM decoder tokens");
    token_cat->setAxis(1);
    auto *queries = token_cat->getOutput(0);

    auto *src = requireLayer(network->addElementWise(image_embedding, dense_prompt, E::kSUM),
                             "Failed to add SAM dense prompt to image embedding")
                    ->getOutput(0);
    auto *keys = requireLayer(network->addShuffle(*src), "Failed to flatten SAM decoder image tokens");
    keys->setReshapeDimensions(nvinfer1::Dims3{1, kSamPromptDim, options.image_grid * options.image_grid});
    keys->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});

    auto *pe_tokens = requireLayer(network->addShuffle(image_pe), "Failed to flatten SAM decoder image PE");
    pe_tokens->setReshapeDimensions(nvinfer1::Dims3{1, kSamPromptDim, options.image_grid * options.image_grid});
    pe_tokens->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    nvinfer1::ITensor *key_tokens = keys->getOutput(0);
    nvinfer1::ITensor *query_tokens_tensor = queries;

    const int output_token_count = (options.pred_obj_scores ? 1 : 0) + 1 + kSamMaskTokens;
    const int iou_token_index    = options.pred_obj_scores ? 1 : 0;
    const int mask_token_index   = iou_token_index + 1;
    const int query_tokens       = output_token_count + kSamPointTokens;
    const int image_tokens       = options.image_grid * options.image_grid;
    for (int i = 0; i < kSamTwoWayDepth; ++i)
    {
        addTwoWayBlock(network, weights_map, query_tokens_tensor, key_tokens, *queries, *pe_tokens->getOutput(0), i,
                       query_tokens, image_tokens, mask_prefix);
    }

    auto *q_final = requireLayer(network->addElementWise(*query_tokens_tensor, *queries, E::kSUM),
                                 "Failed to add SAM final query PE")
                        ->getOutput(0);
    auto *k_final = requireLayer(network->addElementWise(*key_tokens, *pe_tokens->getOutput(0), E::kSUM),
                                 "Failed to add SAM final key PE")
                        ->getOutput(0);
    auto *final_attn = addDecoderAttention(network, weights_map, *q_final, *k_final, *key_tokens,
                                           mask_prefix + ".transformer.final_attn_token_to_image", query_tokens,
                                           image_tokens, 2);
    query_tokens_tensor = requireLayer(network->addElementWise(*query_tokens_tensor, *final_attn, E::kSUM),
                                       "Failed to add SAM final attention residual")
                              ->getOutput(0);
    query_tokens_tensor = addLayerNormLastDim(network, weights_map, *query_tokens_tensor,
                                              mask_prefix + ".transformer.norm_final_attn", kSamPromptDim);
    named_tensors["mask_decoder_tokens"] = query_tokens_tensor;

    auto *iou_token_out = requireLayer(network->addSlice(*query_tokens_tensor, nvinfer1::Dims3{0, iou_token_index, 0},
                                                         nvinfer1::Dims3{1, 1, kSamPromptDim},
                                                         nvinfer1::Dims3{1, 1, 1}),
                                       "Failed to slice SAM iou token out")
                              ->getOutput(0);
    auto *mask_tokens_out = requireLayer(network->addSlice(*query_tokens_tensor, nvinfer1::Dims3{0, mask_token_index, 0},
                                                           nvinfer1::Dims3{1, kSamMaskTokens, kSamPromptDim},
                                                           nvinfer1::Dims3{1, 1, 1}),
                                         "Failed to slice SAM mask tokens out")
                                ->getOutput(0);

    auto *src_view = requireLayer(network->addShuffle(*key_tokens), "Failed to restore SAM decoder image tokens");
    src_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
    src_view->setReshapeDimensions(nvinfer1::Dims4{1, kSamPromptDim, options.image_grid, options.image_grid});

    auto *up0 = requireLayer(network->addDeconvolutionNd(
                                 *src_view->getOutput(0), kSamPromptDim / 4, nvinfer1::DimsHW{2, 2},
                                 requireWeight(weights_map, mask_prefix + ".output_upscaling.0.weight",
                                               static_cast<int64_t>(kSamPromptDim) * (kSamPromptDim / 4) * 2 * 2),
                                 requireWeight(weights_map, mask_prefix + ".output_upscaling.0.bias", kSamPromptDim / 4)),
                             "Failed to add SAM mask upscaling deconv0");
    up0->setStrideNd(nvinfer1::DimsHW{2, 2});
    nvinfer1::ITensor *up0_input = up0->getOutput(0);
    if (options.use_high_res_features)
    {
        up0_input = requireLayer(network->addElementWise(*up0_input, *high_res_s1, E::kSUM),
                                 "Failed to add SAM2 high-res s1 feature")
                        ->getOutput(0);
    }
    auto *up_norm = addLayerNorm2d(network, weights_map, *up0_input, mask_prefix + ".output_upscaling.1",
                                   kSamPromptDim / 4);
    auto *up_gelu = addGeluExact(network, *up_norm);
    auto *up1 = requireLayer(network->addDeconvolutionNd(
                                 *up_gelu, kSamPromptDim / 8, nvinfer1::DimsHW{2, 2},
                                 requireWeight(weights_map, mask_prefix + ".output_upscaling.3.weight",
                                               static_cast<int64_t>(kSamPromptDim / 4) * (kSamPromptDim / 8) * 2 * 2),
                                 requireWeight(weights_map, mask_prefix + ".output_upscaling.3.bias", kSamPromptDim / 8)),
                             "Failed to add SAM mask upscaling deconv1");
    up1->setStrideNd(nvinfer1::DimsHW{2, 2});
    nvinfer1::ITensor *up1_input = up1->getOutput(0);
    if (options.use_high_res_features)
    {
        up1_input = requireLayer(network->addElementWise(*up1_input, *high_res_s0, E::kSUM),
                                 "Failed to add SAM2 high-res s0 feature")
                        ->getOutput(0);
    }
    auto *upscaled = addGeluExact(network, *up1_input);
    named_tensors["upscaled_embedding"] = upscaled;

    std::vector<nvinfer1::ITensor *> hyper_outputs;
    hyper_outputs.reserve(kSamMaskTokens);
    for (int i = 0; i < kSamMaskTokens; ++i)
    {
        const auto prefix = mask_prefix + ".output_hypernetworks_mlps." + std::to_string(i) + ".layers.";
        auto *token = requireLayer(network->addSlice(*mask_tokens_out, nvinfer1::Dims3{0, i, 0},
                                                     nvinfer1::Dims3{1, 1, kSamPromptDim},
                                                     nvinfer1::Dims3{1, 1, 1}),
                                   "Failed to slice SAM hyper token")
                          ->getOutput(0);
        auto *h0 = addLinear3D(network, weights_map, *token, prefix + "0", kSamPromptDim, kSamPromptDim);
        auto *r0 = requireLayer(network->addActivation(*h0, nvinfer1::ActivationType::kRELU),
                                "Failed to add SAM hyper ReLU0");
        auto *h1 = addLinear3D(network, weights_map, *r0->getOutput(0), prefix + "1", kSamPromptDim, kSamPromptDim);
        auto *r1 = requireLayer(network->addActivation(*h1, nvinfer1::ActivationType::kRELU),
                                "Failed to add SAM hyper ReLU1");
        hyper_outputs.push_back(addLinear3D(network, weights_map, *r1->getOutput(0), prefix + "2", kSamPromptDim,
                                            kSamPromptDim / 8));
    }
    auto *hyper = requireLayer(network->addConcatenation(hyper_outputs.data(), static_cast<int32_t>(hyper_outputs.size())),
                               "Failed to concat SAM hyper outputs");
    hyper->setAxis(1);

    auto *up_flat = requireLayer(network->addShuffle(*upscaled), "Failed to flatten SAM upscaled embedding");
    up_flat->setReshapeDimensions(nvinfer1::Dims3{1, kSamPromptDim / 8, kSamMaskSize * kSamMaskSize});
    auto *mask_logits = requireLayer(network->addMatrixMultiply(*hyper->getOutput(0), M::kNONE, *up_flat->getOutput(0),
                                                                M::kNONE),
                                     "Failed to add SAM hyper mask matmul");
    auto *mask_view = requireLayer(network->addShuffle(*mask_logits->getOutput(0)), "Failed to reshape SAM masks");
    mask_view->setReshapeDimensions(nvinfer1::Dims4{1, kSamMaskTokens, kSamMaskSize, kSamMaskSize});
    auto *selected_masks = requireLayer(network->addSlice(*mask_view->getOutput(0), nvinfer1::Dims4{0, 1, 0, 0},
                                                          nvinfer1::Dims4{1, kSamOutputMasks, kSamMaskSize, kSamMaskSize},
                                                          nvinfer1::Dims4{1, 1, 1, 1}),
                                      "Failed to slice SAM multimask outputs")
                               ->getOutput(0);

    auto *iou0 = addLinear3D(network, weights_map, *iou_token_out, mask_prefix + ".iou_prediction_head.layers.0",
                             kSamPromptDim, kSamPromptDim);
    auto *iou_r0 = requireLayer(network->addActivation(*iou0, nvinfer1::ActivationType::kRELU),
                                "Failed to add SAM iou ReLU0");
    auto *iou1 = addLinear3D(network, weights_map, *iou_r0->getOutput(0), mask_prefix + ".iou_prediction_head.layers.1",
                             kSamPromptDim, kSamPromptDim);
    auto *iou_r1 = requireLayer(network->addActivation(*iou1, nvinfer1::ActivationType::kRELU),
                                "Failed to add SAM iou ReLU1");
    auto *iou2 = addLinear3D(network, weights_map, *iou_r1->getOutput(0), mask_prefix + ".iou_prediction_head.layers.2",
                             kSamPromptDim, kSamMaskTokens);
    nvinfer1::ITensor *iou_logits = iou2;
    if (options.sigmoid_iou)
    {
        iou_logits = requireLayer(network->addActivation(*iou_logits, nvinfer1::ActivationType::kSIGMOID),
                                  "Failed to add SAM2 iou sigmoid")
                         ->getOutput(0);
    }
    auto *iou_slice = requireLayer(network->addSlice(*iou_logits, nvinfer1::Dims3{0, 0, 1},
                                                     nvinfer1::Dims3{1, 1, kSamOutputMasks},
                                                     nvinfer1::Dims3{1, 1, 1}),
                                   "Failed to slice SAM iou predictions");
    auto *iou_view = requireLayer(network->addShuffle(*iou_slice->getOutput(0)), "Failed to reshape SAM iou predictions");
    iou_view->setReshapeDimensions(nvinfer1::Dims4{1, kSamOutputMasks, 1, 1});

    masks           = selected_masks;
    iou_predictions = iou_view->getOutput(0);
    named_tensors["low_res_masks"]   = selected_masks;
    named_tensors["masks"]           = selected_masks;
    named_tensors["iou_predictions"] = iou_predictions;
}

/**
 * @brief 生成 SAM 输入张量默认形状。
 */
std::vector<nvinfer1::Dims4> defaultInputShapes(const SAMSpec &spec)
{
    return {
        nvinfer1::Dims4{1, 3, spec.image_size, spec.image_size},
        nvinfer1::Dims4{1, spec.max_points, 2, 1},
        nvinfer1::Dims4{1, spec.max_points, 1, 1},
        nvinfer1::Dims4{1, 1, spec.mask_size, spec.mask_size},
        nvinfer1::Dims4{1, 1, 1, 1},
    };
}

/**
 * @brief 将静态 C 字符串数组复制为字符串数组。
 */
template <size_t N>
std::vector<std::string> toStringVector(const std::array<const char *, N> &values)
{
    std::vector<std::string> result;
    result.reserve(values.size());
    for (const char *value : values)
    {
        result.emplace_back(value);
    }
    return result;
}

/**
 * @brief 判断配置是否仍为默认 ImageNet 单输入形状。
 */
bool usesDefaultSingleImageShape(const IModelConfig &config)
{
    if (config.inputShapes().size() != 1)
    {
        return false;
    }
    const auto &shape = config.inputShape();
    return shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageNetSize
        && shape.d[3] == kDefaultImageNetSize;
}

/**
 * @brief 构造 SAM v1 结构参数。
 */
SAMSpec makeSAMSpec(const char *display_name, int embed_dim, int depth, int heads,
                    std::initializer_list<int> global_attn)
{
    return {display_name, kSamImageSize, kSamMaskSize, kSamMaxPoints, kSamOutputMasks, embed_dim, depth, heads,
            std::set<int>(global_attn.begin(), global_attn.end()), {}, {}, {}, 0, 0, SAMFamily::SAM};
}

/**
 * @brief 构造 SAM2 Hiera 结构参数。
 */
SAMSpec makeSAM2Spec(const char *display_name, int embed_dim, int heads, std::vector<int> stages,
                     std::initializer_list<int> global_attn, std::vector<int> window_spec, int pos_embed_size)
{
    const int depth = std::accumulate(stages.begin(), stages.end(), 0);
    return {display_name,
            kSamImageSize,
            kSamMaskSize,
            kSamMaxPoints,
            kSamOutputMasks,
            embed_dim,
            depth,
            heads,
            std::set<int>(global_attn.begin(), global_attn.end()),
            std::move(stages),
            std::move(window_spec),
            {embed_dim * 8, embed_dim * 4, embed_dim * 2, embed_dim},
            pos_embed_size,
            3,
            SAMFamily::SAM2};
}

/**
 * @brief 构造尚未完整接入的 SAM3 结构参数。
 */
SAMSpec makeSAM3Spec(const char *display_name)
{
    return {display_name, 1008, kSamMaskSize, kSamMaxPoints, kSamOutputMasks, 1024, 0, 0, {}, {}, {}, {}, 0, 0,
            SAMFamily::SAM3};
}

} // namespace

SAMSegmentationModel::SAMSegmentationModel(SAMSpec spec)
    : spec_(std::move(spec))
{
    auto config = std::make_unique<IModelConfig>();
    normalizeModelConfig(*config);
    setModelConfig(std::move(config));
}

std::string SAMSegmentationModel::name() const noexcept
{
    return spec_.display_name;
}

void SAMSegmentationModel::normalizeModelConfig(IModelConfig &config) const
{
    if (config.inputTensorNames() == std::vector<std::string>{"input"})
    {
        config.setInputTensorNames(toStringVector(kDefaultInputNames));
    }
    if (usesDefaultSingleImageShape(config))
    {
        config.setInputShapes(defaultInputShapes(spec_));
    }
    if (config.outputTensorNames() == std::vector<std::string>{"output"})
    {
        config.setOutputTensorNames(toStringVector(kDefaultOutputNames));
    }
    if (config.numClasses() == kDefaultNumClasses)
    {
        config.setNumClasses(spec_.multimask_outputs);
    }
}

void SAMSegmentationModel::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    const auto geometry = resolveSAMGeometry(spec_, modelConfig());
    if (spec_.family == SAMFamily::SAM3)
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED,
                             "%s official TensorRT graph requires its native SAM3 image encoder and is not "
                             "silently mapped to SAM v1",
                             spec_.display_name);
    }

    NamedTensorMap named_tensors;
    nvinfer1::ITensor *image_embedding = nullptr;
    nvinfer1::ITensor *high_res_s0     = nullptr;
    nvinfer1::ITensor *high_res_s1     = nullptr;
    SAMMaskDecoderOptions decoder_options{};
    if (spec_.family == SAMFamily::SAM)
    {
        SAMViTSpec vit_spec{spec_.encoder_embed_dim, spec_.encoder_depth, spec_.encoder_num_heads,
                            spec_.global_attn_indexes};
        image_embedding = addSAMImageEncoder(*this, network, weights_map, geometry, vit_spec, named_tensors);
        decoder_options = {{"prompt_encoder", "mask_decoder"}, kSamEmbedGrid, false, false, false};
    }
    else
    {
        SAM2HieraSpec hiera_spec{spec_.encoder_embed_dim,
                                 spec_.encoder_num_heads,
                                 spec_.hiera_stages,
                                 spec_.hiera_window_spec,
                                 spec_.global_attn_indexes,
                                 spec_.backbone_channels,
                                 spec_.pos_embed_size,
                                 spec_.q_pool};
        image_embedding = addSAM2ImageEncoder(*this, network, weights_map, geometry, hiera_spec, high_res_s0,
                                              high_res_s1, named_tensors);
        decoder_options = {{"sam_prompt_encoder", "sam_mask_decoder"}, kSamEmbedGrid, true, true, true};
    }

    auto *sparse_prompt = addPointPromptEmbedding(*this, network, weights_map, geometry, decoder_options.prefixes.prompt,
                                                 named_tensors);
    auto *dense_prompt = addDensePromptEmbedding(*this, network, weights_map, decoder_options.prefixes.prompt,
                                                named_tensors);
    auto *image_pe = addDensePromptPE(network, weights_map, decoder_options.prefixes.prompt);
    named_tensors["dense_pe"] = image_pe;

    nvinfer1::ITensor *masks           = nullptr;
    nvinfer1::ITensor *iou_predictions = nullptr;
    addSAMMaskDecoder(network, weights_map, *image_embedding, *image_pe, *sparse_prompt, *dense_prompt,
                      decoder_options, high_res_s0, high_res_s1, masks, iou_predictions, named_tensors);

    if (isBuildingFeatureEngine())
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    auto *low_res_clone = requireLayer(network->addShuffle(*masks), "Failed to clone SAM low_res_masks output");
    low_res_clone->setReshapeDimensions(nvinfer1::Dims4{1, kSamOutputMasks, kSamMaskSize, kSamMaskSize});
    markOutputTensors(network, {masks, iou_predictions, low_res_clone->getOutput(0)});
}

SAM::SAM()
    : SAMSegmentationModel(makeSAMSpec("SAMViTH", 1280, 32, 16, {7, 15, 23, 31}))
{
}

SAMViTB::SAMViTB()
    : SAMSegmentationModel(makeSAMSpec("SAMViTB", 768, 12, 12, {2, 5, 8, 11}))
{
}

SAMViTL::SAMViTL()
    : SAMSegmentationModel(makeSAMSpec("SAMViTL", 1024, 24, 16, {5, 11, 17, 23}))
{
}

SAMViTH::SAMViTH()
    : SAMSegmentationModel(makeSAMSpec("SAMViTH", 1280, 32, 16, {7, 15, 23, 31}))
{
}

SAM2::SAM2()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM2HieraTiny::SAM2HieraTiny()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraTiny", 96, 1, {1, 2, 7, 2}, {5, 7, 9}, {8, 4, 14, 7}, 7))
{
}

SAM2HieraSmall::SAM2HieraSmall()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraSmall", 96, 1, {1, 2, 11, 2}, {7, 10, 13}, {8, 4, 14, 7}, 7))
{
}

SAM2HieraBasePlus::SAM2HieraBasePlus()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraBasePlus", 112, 2, {2, 3, 16, 3}, {12, 16, 20}, {8, 4, 14, 7}, 14))
{
}

SAM2HieraLarge::SAM2HieraLarge()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM21HieraTiny::SAM21HieraTiny()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraTiny", 96, 1, {1, 2, 7, 2}, {5, 7, 9}, {8, 4, 14, 7}, 7))
{
}

SAM21HieraSmall::SAM21HieraSmall()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraSmall", 96, 1, {1, 2, 11, 2}, {7, 10, 13}, {8, 4, 14, 7}, 7))
{
}

SAM21HieraBasePlus::SAM21HieraBasePlus()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraBasePlus", 112, 2, {2, 3, 16, 3}, {12, 16, 20}, {8, 4, 14, 7}, 14))
{
}

SAM21HieraLarge::SAM21HieraLarge()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM3::SAM3()
    : SAMSegmentationModel(makeSAM3Spec("SAM3Image"))
{
}

SAM3Image::SAM3Image()
    : SAMSegmentationModel(makeSAM3Spec("SAM3Image"))
{
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(SAM)
INFERRT_REGISTER_MODEL(SAMViTB)
INFERRT_REGISTER_MODEL(SAMViTL)
INFERRT_REGISTER_MODEL(SAMViTH)
INFERRT_REGISTER_MODEL(SAM2)
INFERRT_REGISTER_MODEL(SAM2HieraTiny)
INFERRT_REGISTER_MODEL(SAM2HieraSmall)
INFERRT_REGISTER_MODEL(SAM2HieraBasePlus)
INFERRT_REGISTER_MODEL(SAM2HieraLarge)
INFERRT_REGISTER_MODEL(SAM21HieraTiny)
INFERRT_REGISTER_MODEL(SAM21HieraSmall)
INFERRT_REGISTER_MODEL(SAM21HieraBasePlus)
INFERRT_REGISTER_MODEL(SAM21HieraLarge)
INFERRT_REGISTER_MODEL(SAM3)
INFERRT_REGISTER_MODEL(SAM3Image)
