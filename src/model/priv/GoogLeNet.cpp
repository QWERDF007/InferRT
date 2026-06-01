#include "GoogLeNet.hpp"

#include "BatchNorm.hpp"
#include "Layers.hpp"
#include "Weights.hpp"

#include <inferrt/model/IModel.h>

#include <array>
#include <initializer_list>
#include <string>

namespace irt::model {

namespace {

/**
 * @brief Inception 模块各分支的通道配置。
 */
struct InceptionSpec
{
    const char *prefix;       ///< 权重文件中的模块前缀，例如 `inception3a.`。
    const char *feature_name; ///< featureOnly 使用的稳定特征名，例如 `inception3a`。
    int         ch1x1;        ///< branch1 的 1x1 卷积输出通道数。
    int         ch3x3_reduce; ///< branch2 的 1x1 降维通道数。
    int         ch3x3;        ///< branch2 的 3x3 卷积输出通道数。
    int         ch5x5_reduce; ///< branch3 的 1x1 降维通道数。
    int         ch5x5;        ///< branch3 的 3x3 卷积输出通道数，等价于 torchvision 的 5x5 分支。
    int         pool_proj;    ///< branch4 池化后的 1x1 投影通道数。
};

/**
 * @brief 全局平均池化与展平结果。
 */
struct PoolFlatten
{
    nvinfer1::ITensor *pooled;    ///< 保留 NCHW 维度的全局平均池化输出。
    nvinfer1::ITensor *flattened; ///< 展平后的二维输出。
};

/**
 * @brief 添加 GoogLeNet 中重复使用的 Conv2d + BatchNorm2d + ReLU 结构。
 *
 * 权重命名遵循 torchvision GoogLeNet：`prefix.conv.weight` 和
 * `prefix.bn.{weight,bias,running_mean,running_var}`。
 *
 * @param network TensorRT 网络定义。
 * @param weights_map 权重映射表。
 * @param input 输入张量。
 * @param prefix 权重名前缀。
 * @param out_channels 输出通道数。
 * @param kernel 卷积核尺寸。
 * @param stride 卷积步长。
 * @param padding 卷积 padding。
 * @return ReLU 后的输出张量。
 */
nvinfer1::ITensor *addBasicConv2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                  nvinfer1::ITensor &input, const std::string &prefix, int out_channels, int kernel,
                                  int stride = 1, int padding = 0)
{
    using namespace nvinfer1;

    auto *conv = network->addConvolutionNd(input, out_channels, DimsHW{kernel, kernel},
                                           weights_map.at(prefix + ".conv.weight"), emptyWeights());
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});

    auto *bn   = addBatchNorm2d(network, weights_map, *conv->getOutput(0), prefix + ".bn", 1e-3f);
    auto *relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    return relu->getOutput(0);
}

/**
 * @brief 添加 GoogLeNet 的 max-pooling 层。
 *
 * torchvision GoogLeNet 的池化层使用 `ceil_mode=True`，这里通过
 * TensorRT 的 `kEXPLICIT_ROUND_UP` 对齐输出尺寸。
 *
 * @param network TensorRT 网络定义。
 * @param input 输入张量。
 * @param kernel 池化核尺寸。
 * @param stride 池化步长。
 * @param padding 显式 padding。
 * @return 池化输出张量。
 */
nvinfer1::ITensor *addCeilMaxPool(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int kernel,
                                  int stride, int padding = 0)
{
    using namespace nvinfer1;

    auto *pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{kernel, kernel});
    pool->setStrideNd(DimsHW{stride, stride});
    pool->setPaddingNd(DimsHW{padding, padding});
    pool->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    return pool->getOutput(0);
}

/**
 * @brief 添加一个 Inception v1 模块。
 *
 * 模块包含四条分支：1x1、1x1+3x3、1x1+3x3 以及 3x3 max-pool+1x1。
 * torchvision 的 GoogLeNet 用 3x3 卷积实现原论文中的 5x5 分支，本实现与其保持一致。
 *
 * @param network TensorRT 网络定义。
 * @param weights_map 权重映射表。
 * @param input 输入张量。
 * @param spec Inception 分支配置。
 * @return 按通道拼接后的输出张量。
 */
nvinfer1::ITensor *addInception(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, const InceptionSpec &spec)
{
    using namespace nvinfer1;

    const std::string prefix = spec.prefix;

    auto *branch1 = addBasicConv2d(network, weights_map, input, prefix + "branch1", spec.ch1x1, 1);

    auto *branch2_reduce = addBasicConv2d(network, weights_map, input, prefix + "branch2.0", spec.ch3x3_reduce, 1);
    auto *branch2 = addBasicConv2d(network, weights_map, *branch2_reduce, prefix + "branch2.1", spec.ch3x3, 3, 1, 1);

    auto *branch3_reduce = addBasicConv2d(network, weights_map, input, prefix + "branch3.0", spec.ch5x5_reduce, 1);
    auto *branch3 = addBasicConv2d(network, weights_map, *branch3_reduce, prefix + "branch3.1", spec.ch5x5, 3, 1, 1);

    auto *branch4_pool = addCeilMaxPool(network, input, 3, 1, 1);
    auto *branch4      = addBasicConv2d(network, weights_map, *branch4_pool, prefix + "branch4.1", spec.pool_proj, 1);

    std::array<ITensor *, 4> branches{branch1, branch2, branch3, branch4};
    auto                    *concat = network->addConcatenation(branches.data(), static_cast<int>(branches.size()));
    concat->setAxis(1);
    return concat->getOutput(0);
}

/**
 * @brief 记录可导出的中间张量，并在 featureOnly 请求满足时立即标记输出。
 *
 * @param impl 模型实现对象。
 * @param network TensorRT 网络定义。
 * @param named_tensors 已记录的命名张量表。
 * @param names 当前张量的一个或多个别名。
 * @param tensor 当前张量。
 * @param feature_only 是否正在构建仅特征提取网络。
 * @return 若 featureOnly 输出已全部标记则返回 true，否则返回 false。
 */
bool recordFeature(const priv::IModelImpl &impl, nvinfer1::INetworkDefinition *network,
                   priv::IModelImpl::NamedTensorMap &named_tensors, std::initializer_list<const char *> names,
                   nvinfer1::ITensor *tensor, bool feature_only)
{
    for (const char *name : names)
    {
        named_tensors[name] = tensor;
    }
    return feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors);
}

/**
 * @brief 添加空间维度全局平均池化并展平为二维张量。
 *
 * @param network TensorRT 网络定义。
 * @param input 输入张量。
 * @return 同时返回池化张量与展平张量。
 */
PoolFlatten addGlobalAvgFlatten(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *avgpool = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, (1U << 2U) | (1U << 3U), true);
    auto *flatten = flattenPreserveBatch(network, *avgpool->getOutput(0));
    return {avgpool->getOutput(0), flatten};
}

/**
 * @brief 添加 GoogLeNet 最后的全连接分类层。
 *
 * @param impl 模型实现对象，用于读取类别数配置。
 * @param network TensorRT 网络定义。
 * @param weights_map 权重映射表。
 * @param input 输入张量。
 * @return logits 输出张量。
 */
nvinfer1::ITensor *addClassifier(const priv::IModelImpl &impl, nvinfer1::INetworkDefinition *network,
                                 const WeightsMap &weights_map, nvinfer1::ITensor &input)
{
    using namespace nvinfer1;

    constexpr int fc_in_features = 1024;
    const int     num_classes    = impl.modelConfig().numClasses();

    auto *fcw = network->addConstant(DimsHW{num_classes, fc_in_features}, weights_map.at("fc.weight"))->getOutput(0);
    auto *fcb = network->addConstant(DimsHW{1, num_classes}, weights_map.at("fc.bias"))->getOutput(0);
    auto *mm  = network->addMatrixMultiply(input, MatrixOperation::kNONE, *fcw, MatrixOperation::kTRANSPOSE);
    return network->addElementWise(*mm->getOutput(0), *fcb, ElementWiseOperation::kSUM)->getOutput(0);
}

} // namespace

void GoogLeNet::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    using namespace nvinfer1;

    const bool                       feature_only = isBuildingFeatureEngine();
    ITensor                         *x            = addInputTensor(network);
    priv::IModelImpl::NamedTensorMap named_tensors{
        {"input", x}
    };

    x = addBasicConv2d(network, weights_map, *x, "conv1", 64, 7, 2, 3);
    if (recordFeature(*this, network, named_tensors, {"conv1", "stem.conv1"}, x, feature_only))
    {
        return;
    }

    x = addCeilMaxPool(network, *x, 3, 2);
    if (recordFeature(*this, network, named_tensors, {"pool1", "stem.pool1"}, x, feature_only))
    {
        return;
    }

    x = addBasicConv2d(network, weights_map, *x, "conv2", 64, 1);
    if (recordFeature(*this, network, named_tensors, {"conv2", "stem.conv2"}, x, feature_only))
    {
        return;
    }

    x = addBasicConv2d(network, weights_map, *x, "conv3", 192, 3, 1, 1);
    if (recordFeature(*this, network, named_tensors, {"conv3", "stem.conv3"}, x, feature_only))
    {
        return;
    }

    x = addCeilMaxPool(network, *x, 3, 2);
    if (recordFeature(*this, network, named_tensors, {"pool2", "stem.pool2"}, x, feature_only))
    {
        return;
    }

    constexpr std::array<InceptionSpec, 2> inception3{
        {{"inception3a.", "inception3a", 64, 96, 128, 16, 32, 32},
         {"inception3b.", "inception3b", 128, 128, 192, 32, 96, 64}}
    };
    for (const auto &spec : inception3)
    {
        x = addInception(network, weights_map, *x, spec);
        if (recordFeature(*this, network, named_tensors, {spec.feature_name}, x, feature_only))
        {
            return;
        }
    }

    x = addCeilMaxPool(network, *x, 3, 2);
    if (recordFeature(*this, network, named_tensors, {"pool3"}, x, feature_only))
    {
        return;
    }

    constexpr std::array<InceptionSpec, 5> inception4{
        {{"inception4a.", "inception4a", 192, 96, 208, 16, 48, 64},
         {"inception4b.", "inception4b", 160, 112, 224, 24, 64, 64},
         {"inception4c.", "inception4c", 128, 128, 256, 24, 64, 64},
         {"inception4d.", "inception4d", 112, 144, 288, 32, 64, 64},
         {"inception4e.", "inception4e", 256, 160, 320, 32, 128, 128}}
    };
    for (const auto &spec : inception4)
    {
        x = addInception(network, weights_map, *x, spec);
        if (recordFeature(*this, network, named_tensors, {spec.feature_name}, x, feature_only))
        {
            return;
        }
    }

    x = addCeilMaxPool(network, *x, 2, 2);
    if (recordFeature(*this, network, named_tensors, {"pool4"}, x, feature_only))
    {
        return;
    }

    constexpr std::array<InceptionSpec, 2> inception5{
        {{"inception5a.", "inception5a", 256, 160, 320, 32, 128, 128},
         {"inception5b.", "inception5b", 384, 192, 384, 48, 128, 128}}
    };
    for (const auto &spec : inception5)
    {
        x = addInception(network, weights_map, *x, spec);
        if (recordFeature(*this, network, named_tensors, {spec.feature_name}, x, feature_only))
        {
            return;
        }
    }

    const auto pooled = addGlobalAvgFlatten(network, *x);
    if (recordFeature(*this, network, named_tensors, {"avgpool"}, pooled.pooled, feature_only))
    {
        return;
    }

    x = pooled.flattened;
    if (recordFeature(*this, network, named_tensors, {"flatten"}, x, feature_only))
    {
        return;
    }

    x                       = addClassifier(*this, network, weights_map, *x);
    named_tensors["logits"] = x;
    if (feature_only)
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    markOutputTensors(network, {x});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(GoogLeNet)
