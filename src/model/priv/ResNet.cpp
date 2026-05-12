#include "ResNet.hpp"

#include "BatchNorm.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <array>

namespace irt::model {

namespace {

// 统一描述残差块构建函数的签名，便于 makeLayer 同时兼容 BasicBlock / Bottleneck。
using BlockBuilder = nvinfer1::IActivationLayer *(*)(nvinfer1::INetworkDefinition *, const WeightsMap &,
                                                     nvinfer1::ITensor &, int, int, int, int, std::string);

/**
 * @brief 按照 torchvision.models.resnet._make_layer 的思路构建一个 stage。
 *
 * 一个 stage 由多个残差块串联组成：
 * 1. 第一个 block 负责处理 stride 变化以及输入/输出通道对齐。
 * 2. 后续 block 保持 stride=1，并复用更新后的 inplanes。
 */
nvinfer1::IActivationLayer *makeLayer(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                      nvinfer1::ITensor &input, int &inplanes, int planes, int blocks, int stride,
                                      int expansion, int base_width, const std::string &lname, BlockBuilder block_builder)
{
    nvinfer1::IActivationLayer *output
        = block_builder(network, weights_map, input, inplanes, planes, stride, base_width, lname + "0.");

    inplanes = planes * expansion;
    for (int i = 1; i < blocks; ++i)
    {
        output = block_builder(network, weights_map, *output->getOutput(0), inplanes, planes, 1, base_width,
                               lname + std::to_string(i) + ".");
    }

    return output;
}

/**
 * @brief 构建 ResNet-18 / ResNet-34 使用的 BasicBlock。
 *
 * 结构与 torchvision 中的 BasicBlock 对齐：
 * conv3x3 -> bn -> relu -> conv3x3 -> bn -> add(shortcut) -> relu
 */
nvinfer1::IActivationLayer *BasicBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, int inch, int outch, int stride, int /* base_width */,
                                       std::string lname)
{
    using namespace nvinfer1;

    Weights empty_weights{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1
        = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weights_map.at(lname + "conv1.weight"), empty_weights);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer      *bn1   = addBatchNorm2d(network, weights_map, *conv1->getOutput(0), lname + "bn1", 1e-5f);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3},
                                                         weights_map.at(lname + "conv2.weight"), empty_weights);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn2 = addBatchNorm2d(network, weights_map, *conv2->getOutput(0), lname + "bn2", 1e-5f);

    IElementWiseLayer *ew1;
    if (stride != 1 || inch != outch)
    {
        IConvolutionLayer *conv3 = network->addConvolutionNd(
            input, outch, DimsHW{1, 1}, weights_map.at(lname + "downsample.0.weight"), empty_weights);
        conv3->setStrideNd(DimsHW{stride, stride});

        IScaleLayer *bn3 = addBatchNorm2d(network, weights_map, *conv3->getOutput(0), lname + "downsample.1", 1e-5f);
        ew1              = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }

    return network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
}

/**
 * @brief 构建 ResNet-50 / 101 / 152 使用的 Bottleneck 块。
 *
 * 当前实现与 torchvision 的 ResNet V1.5 保持一致：
 * stride 放在中间的 3x3 卷积上，而不是第一个 1x1 卷积上。
 */
nvinfer1::IActivationLayer *Bottleneck(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, int inch, int outch, int stride, int base_width,
                                       std::string lname)
{
    using namespace nvinfer1;

    Weights empty_weights{DataType::kFLOAT, nullptr, 0};
    const int width = outch * base_width / 64;

    IConvolutionLayer *conv1
        = network->addConvolutionNd(input, width, DimsHW{1, 1}, weights_map.at(lname + "conv1.weight"), empty_weights);

    IScaleLayer      *bn1   = addBatchNorm2d(network, weights_map, *conv1->getOutput(0), lname + "bn1", 1e-5f);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), width, DimsHW{3, 3},
                                                         weights_map.at(lname + "conv2.weight"), empty_weights);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer      *bn2   = addBatchNorm2d(network, weights_map, *conv2->getOutput(0), lname + "bn2", 1e-5f);
    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1},
                                                         weights_map.at(lname + "conv3.weight"), empty_weights);

    IScaleLayer *bn3 = addBatchNorm2d(network, weights_map, *conv3->getOutput(0), lname + "bn3", 1e-5f);

    IElementWiseLayer *ew1;
    if (stride != 1 || inch != outch * 4)
    {
        IConvolutionLayer *conv4 = network->addConvolutionNd(
            input, outch * 4, DimsHW{1, 1}, weights_map.at(lname + "downsample.0.weight"), empty_weights);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer *bn4 = addBatchNorm2d(network, weights_map, *conv4->getOutput(0), lname + "downsample.1", 1e-5f);
        ew1              = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }

    return network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
}

/**
 * @brief 构建一个完整的 torchvision 风格 ResNet 主干与分类头。
 *
 * @param layers     四个 stage 的 block 数量，例如 ResNet18 为 {2, 2, 2, 2}。
 * @param expansion  残差块通道扩张倍率，BasicBlock=1，Bottleneck=4。
 * @param block      具体的 block 构建函数。
 */
void buildResNet(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, const std::array<int, 4> &layers,
                 int expansion, int base_width, int num_classes, const InputShape &input_shape, BlockBuilder block)
{
    using namespace nvinfer1;

    Weights empty_weights{DataType::kFLOAT, nullptr, 0};
    int     inplanes = 64;

    ITensor *input
        = network->addInput("input", DataType::kFLOAT, Dims4{1, input_shape.channels, input_shape.height, input_shape.width});

    // stem: 7x7 conv -> bn -> relu -> 3x3 maxpool
    IConvolutionLayer *conv1
        = network->addConvolutionNd(*input, 64, DimsHW{7, 7}, weights_map.at("conv1.weight"), empty_weights);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer      *bn1   = addBatchNorm2d(network, weights_map, *conv1->getOutput(0), "bn1", 1e-5f);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    // 四个残差 stage，通道数分别为 64 / 128 / 256 / 512。
    IActivationLayer *layer1 = makeLayer(network, weights_map, *pool1->getOutput(0), inplanes, 64, layers[0], 1,
                                         expansion, base_width, "layer1.", block);
    IActivationLayer *layer2 = makeLayer(network, weights_map, *layer1->getOutput(0), inplanes, 128, layers[1], 2,
                                         expansion, base_width, "layer2.", block);
    IActivationLayer *layer3 = makeLayer(network, weights_map, *layer2->getOutput(0), inplanes, 256, layers[2], 2,
                                         expansion, base_width, "layer3.", block);
    IActivationLayer *layer4 = makeLayer(network, weights_map, *layer3->getOutput(0), inplanes, 512, layers[3], 2,
                                         expansion, base_width, "layer4.", block);

    // 对于固定输入 224x224，layer4 输出空间尺寸为 7x7，可直接做全局平均池化。
    IReduceLayer *avgpool = network->addReduce(*layer4->getOutput(0), ReduceOperation::kAVG,
                                               (1U << 2U) | (1U << 3U), true);

    IShuffleLayer *shuffle = network->addShuffle(*avgpool->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{1, -1});

    const int fc_in_channels = 512 * expansion;
    ITensor  *fcw
        = network->addConstant(DimsHW{num_classes, fc_in_channels}, weights_map.at("fc.weight"))->getOutput(0);
    ITensor  *fcb = network->addConstant(DimsHW{1, num_classes}, weights_map.at("fc.bias"))->getOutput(0);

    IMatrixMultiplyLayer *fc0
        = network->addMatrixMultiply(*shuffle->getOutput(0), MatrixOperation::kNONE, *fcw, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc1 = network->addElementWise(*fc0->getOutput(0), *fcb, ElementWiseOperation::kSUM);

    fc1->getOutput(0)->setName("output");
    network->markOutput(*fc1->getOutput(0));
}

} // namespace

void ResNet18::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {2, 2, 2, 2}, 1, 64, numClasses(), inputShape(), BasicBlock);
}

void ResNet34::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 4, 6, 3}, 1, 64, numClasses(), inputShape(), BasicBlock);
}

void ResNet50::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 4, 6, 3}, 4, 64, numClasses(), inputShape(), Bottleneck);
}

void ResNet101::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 4, 23, 3}, 4, 64, numClasses(), inputShape(), Bottleneck);
}

void ResNet152::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 8, 36, 3}, 4, 64, numClasses(), inputShape(), Bottleneck);
}

void WideResNet50_2::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 4, 6, 3}, 4, 128, numClasses(), inputShape(), Bottleneck);
}

void WideResNet101_2::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildResNet(network, weights_map, {3, 4, 23, 3}, 4, 128, numClasses(), inputShape(), Bottleneck);
}

void ResNet::infer(const std::vector<void *> &buffers)
{
    auto &trt_params = trtParams();

    if (!trt_params.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    if (buffers.size() != 2)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Expected 2 buffers (input and output), got %zu",
                             buffers.size());
    }

    if (!trt_params.context->setTensorAddress("input", buffers[0]))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set input tensor address");
    }

    if (!trt_params.context->setTensorAddress("output", buffers[1]))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set output tensor address");
    }

    if (!trt_params.stream)
    {
        trt_params.stream = MakeCudaStream();
        if (!trt_params.stream)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create CUDA stream");
        }
    }

    bool status = trt_params.context->enqueueV3(*trt_params.stream);
    if (!status)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute inference");
    }

    cudaStreamSynchronize(*trt_params.stream);
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(ResNet18)
INFERRT_REGISTER_MODEL(ResNet34)
INFERRT_REGISTER_MODEL(ResNet50)
INFERRT_REGISTER_MODEL(ResNet101)
INFERRT_REGISTER_MODEL(ResNet152)
INFERRT_REGISTER_MODEL(WideResNet50_2)
INFERRT_REGISTER_MODEL(WideResNet101_2)
