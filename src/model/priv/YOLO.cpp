#include "YOLO.hpp"

#include "BatchNorm.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace irt::model {

namespace {

constexpr int   kDefaultImageSize   = 224;
constexpr int   kYoloInputSize      = 640;
constexpr int   kYoloDefaultClasses = 80;
constexpr int   kYoloDefaultOutputs = 3;
constexpr int   kYoloV8SegOutputs   = 4;
constexpr int   kYoloV8RegMax       = 16;
constexpr int   kYoloV8BoxChannels  = 4 * kYoloV8RegMax;
constexpr int   kYoloV8MaskChannels = 32;
constexpr float kBatchNormEps       = 1.0e-3F;

struct YOLOv8FeatureMaps
{
    nvinfer1::ITensor *p3{nullptr};
    nvinfer1::ITensor *p4{nullptr};
    nvinfer1::ITensor *p5{nullptr};
};

struct DFLHeadOutputs
{
    nvinfer1::ITensor *boxes{nullptr};
    nvinfer1::ITensor *classes{nullptr};
    int                grid{0};
};

/**
 * @brief 计算 YOLO 系列的宽度缩放通道数。
 *
 * tensorrtx 与 Ultralytics 都将通道数对齐到 8 的倍数；YOLOv8 的 m/l/x 变体还会
 * 对缩放后的通道数设置上限，避免高层通道超出官方结构。
 */
int scaledWidth(int channels, float width, int max_channels = 0, int divisor = 8)
{
    const int scaled = static_cast<int>(std::ceil((static_cast<float>(channels) * width) / divisor)) * divisor;
    return max_channels > 0 ? std::min(scaled, max_channels) : scaled;
}

/**
 * @brief 计算 YOLO 系列的深度缩放重复次数。
 */
int scaledDepth(int repeats, float depth)
{
    if (repeats == 1)
    {
        return 1;
    }

    int        scaled = static_cast<int>(std::round(static_cast<float>(repeats) * depth));
    const auto exact  = static_cast<float>(repeats) * depth;
    if (exact - static_cast<int>(exact) == 0.5F && (static_cast<int>(exact) % 2) == 0)
    {
        --scaled;
    }
    return std::max(scaled, 1);
}

/**
 * @brief 添加 YOLO 中常用的 Conv2d + BatchNorm2d + SiLU。
 *
 * @param padding 显式 padding；小于 0 时按常规 `kernel_size / 2` 自动推导。
 */
nvinfer1::ITensor *addConvBnSiLU(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, int out_channels, int kernel_size, int stride,
                                 const std::string &prefix, int groups = 1, int padding = -1)
{
    using namespace nvinfer1;

    Weights empty_weights{DataType::kFLOAT, nullptr, 0};
    auto   *conv = network->addConvolutionNd(input, out_channels, DimsHW{kernel_size, kernel_size},
                                             weights_map.at(prefix + ".conv.weight"), empty_weights);
    if (conv == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO convolution: %s", prefix.c_str());
    }
    const int layer_padding = padding >= 0 ? padding : kernel_size / 2;
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{layer_padding, layer_padding});
    conv->setNbGroups(groups);
    conv->setName((prefix + ".conv").c_str());

    auto *bn      = addBatchNorm2d(network, weights_map, *conv->getOutput(0), prefix + ".bn", kBatchNormEps);
    auto *sigmoid = network->addActivation(*bn->getOutput(0), ActivationType::kSIGMOID);
    auto *silu    = network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    if (silu == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO SiLU: %s", prefix.c_str());
    }
    silu->getOutput(0)->setName(prefix.c_str());
    return silu->getOutput(0);
}

/**
 * @brief 添加不带 BN/激活的 1x1 检测头卷积。
 */
nvinfer1::ITensor *addLinearConv1x1(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    nvinfer1::ITensor &input, int out_channels, const std::string &prefix)
{
    auto *conv = network->addConvolutionNd(input, out_channels, nvinfer1::DimsHW{1, 1},
                                           weights_map.at(prefix + ".weight"), weights_map.at(prefix + ".bias"));
    if (conv == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO head convolution: %s", prefix.c_str());
    }
    conv->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv->getOutput(0)->setName(prefix.c_str());
    return conv->getOutput(0);
}

/**
 * @brief 添加 YOLOv8-Seg Proto 中的 ConvTranspose2d 上采样。
 */
nvinfer1::ITensor *addDeconv2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                               nvinfer1::ITensor &input, int out_channels, int kernel_size, int stride,
                               const std::string &prefix)
{
    auto *deconv = network->addDeconvolutionNd(input, out_channels, nvinfer1::DimsHW{kernel_size, kernel_size},
                                               weights_map.at(prefix + ".weight"), weights_map.at(prefix + ".bias"));
    if (deconv == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO deconvolution: %s", prefix.c_str());
    }
    deconv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    deconv->setPaddingNd(nvinfer1::DimsHW{0, 0});
    deconv->getOutput(0)->setName(prefix.c_str());
    return deconv->getOutput(0);
}

/**
 * @brief 沿通道维拼接 NCHW 张量。
 */
nvinfer1::ITensor *concatChannels(nvinfer1::INetworkDefinition *network, const std::vector<nvinfer1::ITensor *> &inputs)
{
    auto *concat = network->addConcatenation(inputs.data(), static_cast<int32_t>(inputs.size()));
    if (concat == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO concatenation");
    }
    concat->setAxis(1);
    return concat->getOutput(0);
}

/**
 * @brief 添加 YOLOv5 C3 内部 Bottleneck，结构为 1x1 Conv 后接 3x3 Conv。
 */
nvinfer1::ITensor *addYOLOv5Bottleneck(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, int channels, bool shortcut, const std::string &prefix)
{
    auto *cv1 = addConvBnSiLU(network, weights_map, input, channels, 1, 1, prefix + ".cv1");
    auto *cv2 = addConvBnSiLU(network, weights_map, *cv1, channels, 3, 1, prefix + ".cv2");
    if (!shortcut)
    {
        return cv2;
    }

    auto *sum = network->addElementWise(input, *cv2, nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLOv5 bottleneck shortcut: %s", prefix.c_str());
    }
    return sum->getOutput(0);
}

/**
 * @brief 添加 YOLOv5 C3 模块。
 */
nvinfer1::ITensor *addC3(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &input,
                         int out_channels, int repeats, bool shortcut, const std::string &prefix)
{
    const int hidden = static_cast<int>(static_cast<float>(out_channels) * 0.5F);
    auto     *left   = addConvBnSiLU(network, weights_map, input, hidden, 1, 1, prefix + ".cv1");
    auto     *right  = addConvBnSiLU(network, weights_map, input, hidden, 1, 1, prefix + ".cv2");

    for (int i = 0; i < repeats; ++i)
    {
        left = addYOLOv5Bottleneck(network, weights_map, *left, hidden, shortcut, prefix + ".m." + std::to_string(i));
    }

    auto *cat = concatChannels(network, {left, right});
    return addConvBnSiLU(network, weights_map, *cat, out_channels, 1, 1, prefix + ".cv3");
}

/**
 * @brief 添加 YOLOv8 C2f 内部 Bottleneck，结构为 3x3 Conv 后接 3x3 Conv。
 */
nvinfer1::ITensor *addYOLOv8Bottleneck(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, int channels, bool shortcut, const std::string &prefix)
{
    auto *cv1 = addConvBnSiLU(network, weights_map, input, channels, 3, 1, prefix + ".cv1");
    auto *cv2 = addConvBnSiLU(network, weights_map, *cv1, channels, 3, 1, prefix + ".cv2");
    if (!shortcut)
    {
        return cv2;
    }

    auto *sum = network->addElementWise(input, *cv2, nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLOv8 bottleneck shortcut: %s", prefix.c_str());
    }
    return sum->getOutput(0);
}

/**
 * @brief 添加 YOLOv8 C2f 模块。
 */
nvinfer1::ITensor *addC2f(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                          nvinfer1::ITensor &input, int out_channels, int repeats, bool shortcut,
                          const std::string &prefix)
{
    const int  hidden = static_cast<int>(static_cast<float>(out_channels) * 0.5F);
    auto      *cv1    = addConvBnSiLU(network, weights_map, input, 2 * hidden, 1, 1, prefix + ".cv1");
    const auto dims   = cv1->getDimensions();
    if (dims.nbDims != 4 || dims.d[1] != 2 * hidden)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unexpected C2f tensor shape for %s", prefix.c_str());
    }

    auto *split0
        = network->addSlice(*cv1, nvinfer1::Dims4{0, 0, 0, 0}, nvinfer1::Dims4{dims.d[0], hidden, dims.d[2], dims.d[3]},
                            nvinfer1::Dims4{1, 1, 1, 1});
    auto *split1
        = network->addSlice(*cv1, nvinfer1::Dims4{0, hidden, 0, 0},
                            nvinfer1::Dims4{dims.d[0], hidden, dims.d[2], dims.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    std::vector<nvinfer1::ITensor *> branches{split0->getOutput(0), split1->getOutput(0)};

    auto *tail = split1->getOutput(0);
    for (int i = 0; i < repeats; ++i)
    {
        tail = addYOLOv8Bottleneck(network, weights_map, *tail, hidden, shortcut, prefix + ".m." + std::to_string(i));
        branches.push_back(tail);
    }

    auto *cat = concatChannels(network, branches);
    return addConvBnSiLU(network, weights_map, *cat, out_channels, 1, 1, prefix + ".cv2");
}

/**
 * @brief 添加 YOLOv5/YOLOv8 共享的 SPPF 模块。
 */
nvinfer1::ITensor *addSPPF(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                           nvinfer1::ITensor &input, int in_channels, int out_channels, int kernel_size,
                           const std::string &prefix)
{
    const int hidden = in_channels / 2;
    auto     *cv1    = addConvBnSiLU(network, weights_map, input, hidden, 1, 1, prefix + ".cv1");

    auto *pool1 = network->addPoolingNd(*cv1, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{kernel_size, kernel_size});
    auto *pool2 = network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX,
                                        nvinfer1::DimsHW{kernel_size, kernel_size});
    auto *pool3 = network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX,
                                        nvinfer1::DimsHW{kernel_size, kernel_size});
    for (auto *pool : {pool1, pool2, pool3})
    {
        pool->setStrideNd(nvinfer1::DimsHW{1, 1});
        pool->setPaddingNd(nvinfer1::DimsHW{kernel_size / 2, kernel_size / 2});
    }

    auto *cat = concatChannels(network, {cv1, pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)});
    return addConvBnSiLU(network, weights_map, *cat, out_channels, 1, 1, prefix + ".cv2");
}

/**
 * @brief 按目标张量的 H/W 尺寸进行最近邻上采样。
 */
nvinfer1::ITensor *addNearestResizeLike(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                        const nvinfer1::ITensor &reference)
{
    auto *resize = network->addResize(input);
    if (resize == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to add YOLO nearest resize");
    }
    auto       output_dims = input.getDimensions();
    const auto ref_dims    = reference.getDimensions();
    if (output_dims.nbDims == 4 && ref_dims.nbDims == 4)
    {
        output_dims.d[2] = ref_dims.d[2];
        output_dims.d[3] = ref_dims.d[3];
    }
    else if (output_dims.nbDims == 3 && ref_dims.nbDims == 3)
    {
        output_dims.d[1] = ref_dims.d[1];
        output_dims.d[2] = ref_dims.d[2];
    }
    else
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unexpected YOLO resize tensor rank");
    }
    resize->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    resize->setOutputDimensions(output_dims);
    return resize->getOutput(0);
}

/**
 * @brief YOLOv8 Detect 头中的 DFL 积分层。
 */
nvinfer1::ITensor *addDFL(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                          nvinfer1::ITensor &input, int grid, const std::string &weight_key)
{
    const auto dims = input.getDimensions();
    if (dims.nbDims != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "DFL expects [N, 64, grid] input");
    }

    auto *reshape = network->addShuffle(input);
    reshape->setReshapeDimensions(nvinfer1::Dims4{dims.d[0], 4, kYoloV8RegMax, grid});
    reshape->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});

    auto *softmax = network->addSoftMax(*reshape->getOutput(0));
    softmax->setAxes(1U << 1U);

    nvinfer1::Weights empty_weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto             *conv = network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{1, 1},
                                                       weights_map.at(weight_key), empty_weights);
    conv->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv->setPaddingNd(nvinfer1::DimsHW{0, 0});

    auto *out = network->addShuffle(*conv->getOutput(0));
    out->setReshapeDimensions(nvinfer1::Dims3{dims.d[0], 4, grid});
    return out->getOutput(0);
}

/**
 * @brief 将 DFL 检测头的某一尺度 box/class 分支转换为 [N, 4+classes, grid]。
 *
 * Ultralytics 当前 YOLOv5u 与 YOLOv8 都使用 `cv2/cv3 + DFL` 检测头，仅头部层号不同：
 * YOLOv5u 为 `model.24`，YOLOv8 为 `model.22`。该 helper 统一处理两者，避免重复实现。
 */
DFLHeadOutputs addDFLBoxClassHeads(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                   nvinfer1::ITensor &input, int branch_index, const std::string &head_prefix,
                                   int in_channels, int class_channels, int num_classes,
                                   const nvinfer1::Dims4 &input_shape)
{
    const std::string index = std::to_string(branch_index);
    const int grid_h = static_cast<int>(input_shape.d[2] / (branch_index == 0 ? 8 : branch_index == 1 ? 16 : 32));
    const int grid_w = static_cast<int>(input_shape.d[3] / (branch_index == 0 ? 8 : branch_index == 1 ? 16 : 32));
    const int grid   = grid_h * grid_w;

    auto *box0 = addConvBnSiLU(network, weights_map, input, in_channels, 3, 1, head_prefix + ".cv2." + index + ".0");
    auto *box1 = addConvBnSiLU(network, weights_map, *box0, in_channels, 3, 1, head_prefix + ".cv2." + index + ".1");
    auto *box = addLinearConv1x1(network, weights_map, *box1, kYoloV8BoxChannels, head_prefix + ".cv2." + index + ".2");

    auto *cls0 = addConvBnSiLU(network, weights_map, input, class_channels, 3, 1, head_prefix + ".cv3." + index + ".0");
    auto *cls1 = addConvBnSiLU(network, weights_map, *cls0, class_channels, 3, 1, head_prefix + ".cv3." + index + ".1");
    auto *cls  = addLinearConv1x1(network, weights_map, *cls1, num_classes, head_prefix + ".cv3." + index + ".2");

    auto *box_shuffle = network->addShuffle(*box);
    box_shuffle->setReshapeDimensions(nvinfer1::Dims3{input_shape.d[0], kYoloV8BoxChannels, grid});
    auto *cls_shuffle = network->addShuffle(*cls);
    cls_shuffle->setReshapeDimensions(nvinfer1::Dims3{input_shape.d[0], num_classes, grid});

    auto *dfl = addDFL(network, weights_map, *box_shuffle->getOutput(0), grid, head_prefix + ".dfl.conv.weight");
    return DFLHeadOutputs{dfl, cls_shuffle->getOutput(0), grid};
}

/**
 * @brief 将 DFL 检测头的某一尺度 box/class 分支转换为 [N, 4+classes, grid]。
 *
 * Ultralytics 当前 YOLOv5u 和 YOLOv8 都使用 `cv2/cv3 + DFL` 检测头，仅头部层号不同：
 * YOLOv5u 为 `model.24`，YOLOv8 为 `model.22`。该 helper 统一处理两者，避免重复实现。
 */
nvinfer1::ITensor *addDFLDetectBranch(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                      nvinfer1::ITensor &input, int branch_index, const std::string &head_prefix,
                                      int in_channels, int class_channels, int num_classes,
                                      const nvinfer1::Dims4 &input_shape)
{
    auto head = addDFLBoxClassHeads(network, weights_map, input, branch_index, head_prefix, in_channels, class_channels,
                                    num_classes, input_shape);
    std::array<nvinfer1::ITensor *, 2> cat_inputs{head.boxes, head.classes};
    auto *cat = network->addConcatenation(cat_inputs.data(), static_cast<int32_t>(cat_inputs.size()));
    cat->setAxis(1);
    return cat->getOutput(0);
}

/**
 * @brief 添加 YOLOv8-Seg 的 mask coefficient 分支并展平为 [N, nm, grid]。
 */
nvinfer1::ITensor *addMaskCoeffBranch(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                      nvinfer1::ITensor &input, int branch_index, const std::string &head_prefix,
                                      int hidden_channels, const nvinfer1::Dims4 &input_shape)
{
    const std::string index = std::to_string(branch_index);
    const int grid_h = static_cast<int>(input_shape.d[2] / (branch_index == 0 ? 8 : branch_index == 1 ? 16 : 32));
    const int grid_w = static_cast<int>(input_shape.d[3] / (branch_index == 0 ? 8 : branch_index == 1 ? 16 : 32));
    const int grid   = grid_h * grid_w;

    auto *mask0
        = addConvBnSiLU(network, weights_map, input, hidden_channels, 3, 1, head_prefix + ".cv4." + index + ".0");
    auto *mask1
        = addConvBnSiLU(network, weights_map, *mask0, hidden_channels, 3, 1, head_prefix + ".cv4." + index + ".1");
    auto *mask
        = addLinearConv1x1(network, weights_map, *mask1, kYoloV8MaskChannels, head_prefix + ".cv4." + index + ".2");

    auto *shuffle = network->addShuffle(*mask);
    shuffle->setReshapeDimensions(nvinfer1::Dims3{input_shape.d[0], kYoloV8MaskChannels, grid});
    return shuffle->getOutput(0);
}

/**
 * @brief 将 YOLOv8-Seg 某一尺度输出拼接为 [N, 4+classes+nm, grid]。
 */
nvinfer1::ITensor *addDFLSegmentBranch(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, int branch_index, const std::string &head_prefix,
                                       int box_channels, int class_channels, int mask_hidden_channels, int num_classes,
                                       const nvinfer1::Dims4 &input_shape)
{
    auto  head = addDFLBoxClassHeads(network, weights_map, input, branch_index, head_prefix, box_channels,
                                     class_channels, num_classes, input_shape);
    auto *mask
        = addMaskCoeffBranch(network, weights_map, input, branch_index, head_prefix, mask_hidden_channels, input_shape);

    std::array<nvinfer1::ITensor *, 3> cat_inputs{head.boxes, head.classes, mask};
    auto *cat = network->addConcatenation(cat_inputs.data(), static_cast<int32_t>(cat_inputs.size()));
    cat->setAxis(1);
    return cat->getOutput(0);
}

/**
 * @brief 添加 YOLOv8-Seg 的 Proto 分支，输出低分辨率 mask prototypes。
 */
nvinfer1::ITensor *addProtoBranch(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &p3, const std::string &head_prefix, int proto_channels)
{
    auto *cv1 = addConvBnSiLU(network, weights_map, p3, proto_channels, 3, 1, head_prefix + ".proto.cv1");
    auto *up  = addDeconv2d(network, weights_map, *cv1, proto_channels, 2, 2, head_prefix + ".proto.upsample");
    auto *cv2 = addConvBnSiLU(network, weights_map, *up, proto_channels, 3, 1, head_prefix + ".proto.cv2");
    return addConvBnSiLU(network, weights_map, *cv2, kYoloV8MaskChannels, 1, 1, head_prefix + ".proto.cv3");
}

/**
 * @brief 添加 YOLOv8 backbone 与 PAN/FPN neck，返回 P3/P4/P5 三个输出特征。
 */
YOLOv8FeatureMaps addYOLOv8BackboneNeck(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                        nvinfer1::ITensor &input, const YOLOv8Spec &spec)
{
    const float depth        = spec.depth;
    const float width        = spec.width;
    const int   max_channels = spec.max_channels;

    auto sw = [width, max_channels](int channels)
    {
        return scaledWidth(channels, width, max_channels);
    };

    auto *conv0 = addConvBnSiLU(network, weights_map, input, sw(64), 3, 2, "model.0");
    auto *conv1 = addConvBnSiLU(network, weights_map, *conv0, sw(128), 3, 2, "model.1");
    auto *c2f2  = addC2f(network, weights_map, *conv1, sw(128), scaledDepth(3, depth), true, "model.2");
    auto *conv3 = addConvBnSiLU(network, weights_map, *c2f2, sw(256), 3, 2, "model.3");
    auto *c2f4  = addC2f(network, weights_map, *conv3, sw(256), scaledDepth(6, depth), true, "model.4");
    auto *conv5 = addConvBnSiLU(network, weights_map, *c2f4, sw(512), 3, 2, "model.5");
    auto *c2f6  = addC2f(network, weights_map, *conv5, sw(512), scaledDepth(6, depth), true, "model.6");
    auto *conv7 = addConvBnSiLU(network, weights_map, *c2f6, sw(1024), 3, 2, "model.7");
    auto *c2f8  = addC2f(network, weights_map, *conv7, sw(1024), scaledDepth(3, depth), true, "model.8");
    auto *sppf9 = addSPPF(network, weights_map, *c2f8, sw(1024), sw(1024), 5, "model.9");

    auto *up10  = addNearestResizeLike(network, *sppf9, *c2f6);
    auto *cat11 = concatChannels(network, {up10, c2f6});
    auto *c2f12 = addC2f(network, weights_map, *cat11, sw(512), scaledDepth(3, depth), false, "model.12");

    auto *up13  = addNearestResizeLike(network, *c2f12, *c2f4);
    auto *cat14 = concatChannels(network, {up13, c2f4});
    auto *c2f15 = addC2f(network, weights_map, *cat14, sw(256), scaledDepth(3, depth), false, "model.15");

    auto *conv16 = addConvBnSiLU(network, weights_map, *c2f15, sw(256), 3, 2, "model.16");
    auto *cat17  = concatChannels(network, {conv16, c2f12});
    auto *c2f18  = addC2f(network, weights_map, *cat17, sw(512), scaledDepth(3, depth), false, "model.18");

    auto *conv19 = addConvBnSiLU(network, weights_map, *c2f18, sw(512), 3, 2, "model.19");
    auto *cat20  = concatChannels(network, {conv19, sppf9});
    auto *c2f21  = addC2f(network, weights_map, *cat20, sw(1024), scaledDepth(3, depth), false, "model.21");

    return YOLOv8FeatureMaps{c2f15, c2f18, c2f21};
}

} // namespace

void YOLOModelBase::normalizeModelConfig(IModelConfig &config) const
{
    if (config.inputShapes().size() == 1)
    {
        const auto &shape = config.inputShape();
        const bool  is_default_shape
            = shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageSize && shape.d[3] == kDefaultImageSize;
        if (is_default_shape)
        {
            config.setInputShape(nvinfer1::Dims4{1, 3, kYoloInputSize, kYoloInputSize});
        }
    }

    if (config.numClasses() == 1000)
    {
        config.setNumClasses(kYoloDefaultClasses);
    }

    if (config.outputTensorNames() == std::vector<std::string>{"output"})
    {
        config.setOutputTensorNames({"output0", "output1", "output2"});
    }
}

std::string YOLOModelBase::generateSuffix(const IModelConfig &config) const noexcept
{
    std::string suffix;
    for (const auto &input_shape : config.inputShapes())
    {
        suffix += "_" + std::to_string(input_shape.d[0]) + "x" + std::to_string(input_shape.d[1]) + "x"
                + std::to_string(input_shape.d[2]) + "x" + std::to_string(input_shape.d[3]);
    }
    suffix += "_" + std::to_string(config.numClasses()) + "cls";
    suffix += "_3det";
    return suffix;
}

void YOLOModelBase::validateDetectionConfig() const
{
    const auto &config = modelConfig();
    if (config.inputShapes().size() != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "YOLO detection expects exactly one input tensor");
    }

    const auto &shape = config.inputShape();
    if (shape.nbDims != 4 || shape.d[0] <= 0 || shape.d[1] != 3 || shape.d[2] <= 0 || shape.d[3] <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "YOLO input shape must be NCHW with 3 channels, got [%d,%d,%d,%d]", shape.d[0], shape.d[1],
                             shape.d[2], shape.d[3]);
    }
    if (shape.d[2] % 32 != 0 || shape.d[3] % 32 != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "YOLO input height/width must be divisible by 32, got H=%d W=%d", shape.d[2], shape.d[3]);
    }
    if (config.outputTensorNames().size() != kYoloDefaultOutputs)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "YOLO detection expects exactly 3 output tensor names");
    }
}

void YOLOv5Detector::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }
    validateDetectionConfig();

    const auto   &shape             = modelConfig().inputShape();
    const int     classes           = modelConfig().numClasses();
    const float   depth             = spec_.depth;
    const float   width             = spec_.width;
    constexpr int anchors_per_scale = 3;

    auto *input = addInputTensor(network);

    auto *conv0 = addConvBnSiLU(network, weights_map, *input, scaledWidth(64, width), 6, 2, "model.0", 1, 2);
    auto *conv1 = addConvBnSiLU(network, weights_map, *conv0, scaledWidth(128, width), 3, 2, "model.1");
    auto *c3_2  = addC3(network, weights_map, *conv1, scaledWidth(128, width), scaledDepth(3, depth), true, "model.2");
    auto *conv3 = addConvBnSiLU(network, weights_map, *c3_2, scaledWidth(256, width), 3, 2, "model.3");
    auto *c3_4  = addC3(network, weights_map, *conv3, scaledWidth(256, width), scaledDepth(6, depth), true, "model.4");
    auto *conv5 = addConvBnSiLU(network, weights_map, *c3_4, scaledWidth(512, width), 3, 2, "model.5");
    auto *c3_6  = addC3(network, weights_map, *conv5, scaledWidth(512, width), scaledDepth(9, depth), true, "model.6");
    auto *conv7 = addConvBnSiLU(network, weights_map, *c3_6, scaledWidth(1024, width), 3, 2, "model.7");
    auto *c3_8  = addC3(network, weights_map, *conv7, scaledWidth(1024, width), scaledDepth(3, depth), true, "model.8");
    auto *sppf9
        = addSPPF(network, weights_map, *c3_8, scaledWidth(1024, width), scaledWidth(1024, width), 5, "model.9");

    auto *conv10 = addConvBnSiLU(network, weights_map, *sppf9, scaledWidth(512, width), 1, 1, "model.10");
    auto *up11   = addNearestResizeLike(network, *conv10, *c3_6);
    auto *cat12  = concatChannels(network, {up11, c3_6});
    auto *c3_13
        = addC3(network, weights_map, *cat12, scaledWidth(512, width), scaledDepth(3, depth), false, "model.13");

    auto *conv14 = addConvBnSiLU(network, weights_map, *c3_13, scaledWidth(256, width), 1, 1, "model.14");
    auto *up15   = addNearestResizeLike(network, *conv14, *c3_4);
    auto *cat16  = concatChannels(network, {up15, c3_4});
    auto *c3_17
        = addC3(network, weights_map, *cat16, scaledWidth(256, width), scaledDepth(3, depth), false, "model.17");

    auto *conv18 = addConvBnSiLU(network, weights_map, *c3_17, scaledWidth(256, width), 3, 2, "model.18");
    auto *cat19  = concatChannels(network, {conv18, conv14});
    auto *c3_20
        = addC3(network, weights_map, *cat19, scaledWidth(512, width), scaledDepth(3, depth), false, "model.20");

    auto *conv21 = addConvBnSiLU(network, weights_map, *c3_20, scaledWidth(512, width), 3, 2, "model.21");
    auto *cat22  = concatChannels(network, {conv21, conv10});
    auto *c3_23
        = addC3(network, weights_map, *cat22, scaledWidth(1024, width), scaledDepth(3, depth), false, "model.23");

    const bool has_dfl_head    = weights_map.find("model.24.cv2.0.0.conv.weight") != weights_map.end();
    const bool has_legacy_head = weights_map.find("model.24.m.0.weight") != weights_map.end();
    if (has_dfl_head)
    {
        const int box_branch_channels = width == 1.25F ? 80 : 64;
        const int class_branch_channels
            = width == 0.25F ? std::max(64, std::min(classes, 100)) : scaledWidth(256, width);

        auto *out0 = addDFLDetectBranch(network, weights_map, *c3_17, 0, "model.24", box_branch_channels,
                                        class_branch_channels, classes, shape);
        auto *out1 = addDFLDetectBranch(network, weights_map, *c3_20, 1, "model.24", box_branch_channels,
                                        class_branch_channels, classes, shape);
        auto *out2 = addDFLDetectBranch(network, weights_map, *c3_23, 2, "model.24", box_branch_channels,
                                        class_branch_channels, classes, shape);
        markOutputTensors(network, {out0, out1, out2});
        return;
    }
    if (!has_legacy_head)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "YOLOv5 weights must contain either model.24.cv2.* DFL head or model.24.m.* legacy head");
    }

    const int head_channels = anchors_per_scale * (classes + 5);
    auto     *out0          = addLinearConv1x1(network, weights_map, *c3_17, head_channels, "model.24.m.0");
    auto     *out1          = addLinearConv1x1(network, weights_map, *c3_20, head_channels, "model.24.m.1");
    auto     *out2          = addLinearConv1x1(network, weights_map, *c3_23, head_channels, "model.24.m.2");
    markOutputTensors(network, {out0, out1, out2});
}

void YOLOv8Detector::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }
    validateDetectionConfig();

    const auto &shape   = modelConfig().inputShape();
    const int   classes = modelConfig().numClasses();
    const auto  spec    = this->spec();
    auto       *input   = addInputTensor(network);
    const auto  neck    = addYOLOv8BackboneNeck(network, weights_map, *input, spec);

    auto sw = [&spec](int channels)
    {
        return scaledWidth(channels, spec.width, spec.max_channels);
    };

    const int box_branch_channels   = spec.width == 1.25F ? 80 : 64;
    const int class_branch_channels = spec.width == 0.25F ? std::max(64, std::min(classes, 100)) : sw(256);

    auto *out0 = addDFLDetectBranch(network, weights_map, *neck.p3, 0, "model.22", box_branch_channels,
                                    class_branch_channels, classes, shape);
    auto *out1 = addDFLDetectBranch(network, weights_map, *neck.p4, 1, "model.22", box_branch_channels,
                                    class_branch_channels, classes, shape);
    auto *out2 = addDFLDetectBranch(network, weights_map, *neck.p5, 2, "model.22", box_branch_channels,
                                    class_branch_channels, classes, shape);

    markOutputTensors(network, {out0, out1, out2});
}

void YOLOv8Segmenter::normalizeModelConfig(IModelConfig &config) const
{
    if (config.inputShapes().size() == 1)
    {
        const auto &shape = config.inputShape();
        const bool  is_default_shape
            = shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == kDefaultImageSize && shape.d[3] == kDefaultImageSize;
        if (is_default_shape)
        {
            config.setInputShape(nvinfer1::Dims4{1, 3, kYoloInputSize, kYoloInputSize});
        }
    }

    if (config.numClasses() == 1000)
    {
        config.setNumClasses(kYoloDefaultClasses);
    }

    if (config.outputTensorNames() == std::vector<std::string>{"output"})
    {
        config.setOutputTensorNames({"output0", "output1", "output2", "proto"});
    }
}

std::string YOLOv8Segmenter::generateSuffix(const IModelConfig &config) const noexcept
{
    std::string suffix;
    for (const auto &input_shape : config.inputShapes())
    {
        suffix += "_" + std::to_string(input_shape.d[0]) + "x" + std::to_string(input_shape.d[1]) + "x"
                + std::to_string(input_shape.d[2]) + "x" + std::to_string(input_shape.d[3]);
    }
    suffix += "_" + std::to_string(config.numClasses()) + "cls";
    suffix += "_3det_1proto";
    return suffix;
}

void YOLOv8Segmenter::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    if (network == nullptr)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "network must not be null");
    }

    const auto &config = modelConfig();
    if (config.inputShapes().size() != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "YOLO segmentation expects exactly one input tensor");
    }

    const auto &shape = config.inputShape();
    if (shape.nbDims != 4 || shape.d[0] <= 0 || shape.d[1] != 3 || shape.d[2] <= 0 || shape.d[3] <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "YOLO segmentation input shape must be NCHW with 3 channels, got [%d,%d,%d,%d]",
                             shape.d[0], shape.d[1], shape.d[2], shape.d[3]);
    }
    if (shape.d[2] % 32 != 0 || shape.d[3] % 32 != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "YOLO segmentation input height/width must be divisible by 32, got H=%d W=%d", shape.d[2],
                             shape.d[3]);
    }
    if (config.outputTensorNames().size() != kYoloV8SegOutputs)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "YOLO segmentation expects exactly 4 output tensor names");
    }

    const int  classes = config.numClasses();
    const auto spec    = this->spec();
    auto      *input   = addInputTensor(network);
    const auto neck    = addYOLOv8BackboneNeck(network, weights_map, *input, spec);

    auto sw = [&spec](int channels)
    {
        return scaledWidth(channels, spec.width, spec.max_channels);
    };

    const int             box_branch_channels   = spec.width == 1.25F ? 80 : 64;
    const int             class_branch_channels = spec.width == 0.25F ? std::max(64, std::min(classes, 100)) : sw(256);
    const int             mask_hidden_channels  = std::max(sw(256) / 4, kYoloV8MaskChannels);
    const int             proto_hidden_channels = sw(256);
    constexpr const char *head_prefix           = "model.22";

    auto *out0  = addDFLSegmentBranch(network, weights_map, *neck.p3, 0, head_prefix, box_branch_channels,
                                      class_branch_channels, mask_hidden_channels, classes, shape);
    auto *out1  = addDFLSegmentBranch(network, weights_map, *neck.p4, 1, head_prefix, box_branch_channels,
                                      class_branch_channels, mask_hidden_channels, classes, shape);
    auto *out2  = addDFLSegmentBranch(network, weights_map, *neck.p5, 2, head_prefix, box_branch_channels,
                                      class_branch_channels, mask_hidden_channels, classes, shape);
    auto *proto = addProtoBranch(network, weights_map, *neck.p3, head_prefix, proto_hidden_channels);

    markOutputTensors(network, {out0, out1, out2, proto});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(YOLOv5)
INFERRT_REGISTER_MODEL(YOLOv5n)
INFERRT_REGISTER_MODEL(YOLOv5s)
INFERRT_REGISTER_MODEL(YOLOv5m)
INFERRT_REGISTER_MODEL(YOLOv5l)
INFERRT_REGISTER_MODEL(YOLOv5x)
INFERRT_REGISTER_MODEL(YOLOv8)
INFERRT_REGISTER_MODEL(YOLOv8n)
INFERRT_REGISTER_MODEL(YOLOv8s)
INFERRT_REGISTER_MODEL(YOLOv8m)
INFERRT_REGISTER_MODEL(YOLOv8l)
INFERRT_REGISTER_MODEL(YOLOv8x)
INFERRT_REGISTER_MODEL(YOLOv8Seg)
INFERRT_REGISTER_MODEL(YOLOv8nSeg)
INFERRT_REGISTER_MODEL(YOLOv8sSeg)
INFERRT_REGISTER_MODEL(YOLOv8mSeg)
INFERRT_REGISTER_MODEL(YOLOv8lSeg)
INFERRT_REGISTER_MODEL(YOLOv8xSeg)
INFERRT_REGISTER_MODEL(YOLOv8SegHyphenAlias)
INFERRT_REGISTER_MODEL(YOLOv8nSegHyphenAlias)
INFERRT_REGISTER_MODEL(YOLOv8sSegHyphenAlias)
INFERRT_REGISTER_MODEL(YOLOv8mSegHyphenAlias)
INFERRT_REGISTER_MODEL(YOLOv8lSegHyphenAlias)
INFERRT_REGISTER_MODEL(YOLOv8xSegHyphenAlias)
