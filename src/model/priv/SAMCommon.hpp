#pragma once

#include "Layers.hpp"
#include "SAM.hpp"

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
constexpr int   kSamOutputMasks      = kSamMaskTokens;
constexpr int   kSamTwoWayDepth      = 2;
constexpr int   kSamTwoWayHeads      = 8;
constexpr int   kSamTwoWayMlpDim     = 2048;
constexpr float kLayerNormEps        = 1.0e-6F;
constexpr float kBatchNormEps        = 1.0e-5F;

constexpr std::array<const char *, 5> kDefaultInputNames{
    "image", "point_coords", "point_labels", "mask_input", "has_mask_input",
};

constexpr std::array<const char *, 3> kDefaultOutputNames{
    "masks",
    "iou_predictions",
    "low_res_masks",
};

struct SAMViTSpec
{
    int           embed_dim;           ///< ViT token 维度。
    int           depth;               ///< Transformer block 数量。
    int           num_heads;           ///< 注意力头数量。
    std::set<int> global_attn_indexes; ///< 使用全局注意力的 block 下标。
};

struct SAM2HieraSpec
{
    int              embed_dim;           ///< Hiera 初始 token 维度。
    int              num_heads;           ///< Hiera 初始注意力头数量。
    std::vector<int> stages;              ///< 各 stage 的 block 数量。
    std::vector<int> window_spec;         ///< 各 stage 的窗口大小。
    std::set<int>    global_attn_indexes; ///< 使用全局注意力的 block 下标。
    std::vector<int> backbone_channels;   ///< FPN neck 输入通道，按低分辨率到高分辨率排列。
    int              pos_embed_size;      ///< 背景位置编码参数的空间边长。
    int              q_pool;              ///< 执行 q pooling 的 stage 数。
};

struct EdgeSAMRepViTBlockSpec
{
    int  kernel_size;  ///< depthwise token mixer 的卷积核边长。
    int  expansion;    ///< channel mixer 的扩展倍率。
    int  out_channels; ///< block 输出通道数。
    bool use_se;       ///< 是否启用 Squeeze-Excite。
    int  stride;       ///< token mixer 的空间步长。
};

struct SAMHeadPrefixes
{
    std::string prompt; ///< PromptEncoder 在 state_dict 中的前缀。
    std::string mask;   ///< MaskDecoder 在 state_dict 中的前缀。
};

struct SAMMaskDecoderOptions
{
    SAMHeadPrefixes prefixes;              ///< 权重前缀。
    int             image_grid;            ///< image embedding 的空间边长。
    bool            use_high_res_features; ///< 是否接入 SAM2 high-res FPN 特征。
    bool            pred_obj_scores;       ///< 是否包含 SAM2 object score token。
    bool            sigmoid_iou;           ///< 是否对 IoU head 输出做 sigmoid。
};

struct SAMGeometry
{
    int batch;       ///< 配置中的默认 batch；TensorRT 动态 profile 可在运行时覆盖第 0 维。
    int channels;    ///< 输入图像通道数，必须为 3。
    int image_h;     ///< 输入图像高度。
    int image_w;     ///< 输入图像宽度。
    int grid_h;      ///< patch 后的网格高度。
    int grid_w;      ///< patch 后的网格宽度。
    int grid_tokens; ///< 图像 token 数量。
};

nvinfer1::Dims makeDims(std::initializer_list<int32_t> values);
int64_t        volume(const nvinfer1::Dims &dims);

std::string resolveLinearPrefix(const WeightsMap &weights_map, std::initializer_list<std::string> candidates);

constexpr const char *kSAMTag = "SAM official";

inline const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key,
                                              int64_t expected_count = -1)
{
    return irt::model::requireWeight(weights_map, key, kSAMTag, expected_count);
}

template<typename T>
T *requireLayer(T *layer, const char *message)
{
    if (layer == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "%s", message);
    }
    return layer;
}

nvinfer1::ITensor *addScalar(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &like, float value);

nvinfer1::ITensor *addLayerNormLastDim(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int channels);

nvinfer1::ITensor *addLayerNorm2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix, int channels);

nvinfer1::Weights makeFusedConvBNWeight(const WeightsMap &weights_map, const std::string &prefix, int64_t conv_count,
                                        int out_channels);

nvinfer1::Weights makeFusedConvBNBias(const WeightsMap &weights_map, const std::string &prefix, int out_channels);

nvinfer1::ITensor *addEdgeSAMConvBN(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    nvinfer1::ITensor &input, const std::string &prefix, int in_channels,
                                    int out_channels, int kernel_size, int stride, int padding, int groups = 1);

nvinfer1::ITensor *addEdgeSAMConvNoBias(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const std::string &key, int in_channels,
                                        int out_channels, int kernel_size, int stride = 1, int padding = 0);

nvinfer1::ITensor *flattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                               int width, int channels);

nvinfer1::ITensor *unflattenNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                                 int width, int channels);

nvinfer1::ITensor *nchwToNhwc(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input);

nvinfer1::ITensor *nhwcToNchw(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input);

nvinfer1::ITensor *addTokenAttention(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &q, nvinfer1::ITensor &k,
                                     nvinfer1::ITensor &v, int batch, int q_tokens, int k_tokens, int num_heads,
                                     int internal_dim);

} // namespace irt::model
