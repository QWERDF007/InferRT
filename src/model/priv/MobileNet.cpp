#include "MobileNet.hpp"

#include "BatchNorm.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <array>
#include <string>
#include <vector>

namespace irt::model {

namespace {

enum class Act
{
    None,
    Relu,
    Relu6,
    HSwish,
};

struct ConvSpec
{
    int kernel;
    int stride;
    int groups;
    int padding;
    Act act;
};

struct V3Block
{
    int in;
    int kernel;
    int hidden;
    int out;
    bool se;
    Act act;
    int stride;
};

nvinfer1::Weights emptyWeights()
{
    return {nvinfer1::DataType::kFLOAT, nullptr, 0};
}

nvinfer1::ITensor *addActivation(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, Act act)
{
    using namespace nvinfer1;

    if (act == Act::None)
    {
        return &input;
    }
    if (act == Act::HSwish)
    {
        auto *sigmoid = network->addActivation(input, ActivationType::kHARD_SIGMOID);
        sigmoid->setAlpha(1.0f / 6.0f);
        sigmoid->setBeta(0.5f);
        return network->addElementWise(input, *sigmoid->getOutput(0), ElementWiseOperation::kPROD)->getOutput(0);
    }

    if (act == Act::Relu6)
    {
        auto *relu  = network->addActivation(input, ActivationType::kRELU);
        auto *shift = reinterpret_cast<float *>(malloc(sizeof(float)));
        auto *scale = reinterpret_cast<float *>(malloc(sizeof(float)));
        auto *power = reinterpret_cast<float *>(malloc(sizeof(float)));
        *shift      = -6.0f;
        *scale      = 1.0f;
        *power      = 1.0f;

        auto *minus6 = network->addScale(input, ScaleMode::kUNIFORM, {DataType::kFLOAT, shift, 1},
                                         {DataType::kFLOAT, scale, 1}, {DataType::kFLOAT, power, 1});
        auto *relu6  = network->addActivation(*minus6->getOutput(0), ActivationType::kRELU);
        return network->addElementWise(*relu->getOutput(0), *relu6->getOutput(0), ElementWiseOperation::kSUB)
            ->getOutput(0);
    }
    auto *relu = network->addActivation(input, ActivationType::kRELU);
    return relu->getOutput(0);
}

nvinfer1::ITensor *addConvBnAct(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, const std::string &prefix, int out_channels,
                                const ConvSpec &spec, float eps)
{
    using namespace nvinfer1;

    auto *conv = network->addConvolutionNd(input, out_channels, DimsHW{spec.kernel, spec.kernel},
                                           weights_map.at(prefix + "0.weight"), emptyWeights());
    conv->setStrideNd(DimsHW{spec.stride, spec.stride});
    conv->setPaddingNd(DimsHW{spec.padding, spec.padding});
    conv->setNbGroups(spec.groups);

    auto *bn = addBatchNorm2d(network, weights_map, *conv->getOutput(0), prefix + "1", eps);
    return addActivation(network, *bn->getOutput(0), spec.act);
}

nvinfer1::ITensor *addLinear(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                             const WeightsMap &weights_map, const std::string &prefix, int out_features,
                             int in_features)
{
    using namespace nvinfer1;

    auto *weight = network->addConstant(DimsHW{out_features, in_features}, weights_map.at(prefix + ".weight"))->getOutput(0);
    auto *bias   = network->addConstant(DimsHW{1, out_features}, weights_map.at(prefix + ".bias"))->getOutput(0);
    auto *mm = network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight, MatrixOperation::kTRANSPOSE);
    return network->addElementWise(*mm->getOutput(0), *bias, ElementWiseOperation::kSUM)->getOutput(0);
}

nvinfer1::ITensor *addAvgFlatten(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *pool = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, (1U << 2U) | (1U << 3U), true);
    auto *flat = network->addShuffle(*pool->getOutput(0));
    flat->setReshapeDimensions(nvinfer1::Dims2{1, -1});
    return flat->getOutput(0);
}

nvinfer1::ITensor *addMobileNetV2Block(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, int in_channels,
                                       int out_channels, int stride, int expand_ratio)
{
    using namespace nvinfer1;

    const int hidden = in_channels * expand_ratio;
    ITensor  *tensor = &input;

    if (expand_ratio != 1)
    {
        tensor = addConvBnAct(network, weights_map, *tensor, prefix + "conv.0.", hidden, {1, 1, 1, 0, Act::Relu6}, 1e-5f);
        tensor = addConvBnAct(network, weights_map, *tensor, prefix + "conv.1.", hidden, {3, stride, hidden, 1, Act::Relu6},
                              1e-5f);

        auto *conv = network->addConvolutionNd(*tensor, out_channels, DimsHW{1, 1},
                                               weights_map.at(prefix + "conv.2.weight"), emptyWeights());
        tensor = addBatchNorm2d(network, weights_map, *conv->getOutput(0), prefix + "conv.3", 1e-5f)->getOutput(0);
    }
    else
    {
        tensor = addConvBnAct(network, weights_map, *tensor, prefix + "conv.0.", hidden, {3, stride, hidden, 1, Act::Relu6},
                              1e-5f);

        auto *conv = network->addConvolutionNd(*tensor, out_channels, DimsHW{1, 1},
                                               weights_map.at(prefix + "conv.1.weight"), emptyWeights());
        tensor = addBatchNorm2d(network, weights_map, *conv->getOutput(0), prefix + "conv.2", 1e-5f)->getOutput(0);
    }

    if (stride == 1 && in_channels == out_channels)
    {
        tensor = network->addElementWise(input, *tensor, ElementWiseOperation::kSUM)->getOutput(0);
    }
    return tensor;
}

nvinfer1::ITensor *addSE(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor &input,
                         const std::string &prefix, int channels, int squeeze_channels)
{
    using namespace nvinfer1;

    auto *pool = network->addReduce(input, ReduceOperation::kAVG, (1U << 2U) | (1U << 3U), true);

    auto *fc1 = network->addConvolutionNd(*pool->getOutput(0), squeeze_channels, DimsHW{1, 1},
                                          weights_map.at(prefix + "fc1.weight"), weights_map.at(prefix + "fc1.bias"));
    auto *relu = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    auto *fc2 = network->addConvolutionNd(*relu->getOutput(0), channels, DimsHW{1, 1}, weights_map.at(prefix + "fc2.weight"),
                                          weights_map.at(prefix + "fc2.bias"));
    auto *sigmoid = network->addActivation(*fc2->getOutput(0), ActivationType::kHARD_SIGMOID);
    sigmoid->setAlpha(1.0f / 6.0f);
    sigmoid->setBeta(0.5f);
    return network->addElementWise(input, *sigmoid->getOutput(0), ElementWiseOperation::kPROD)->getOutput(0);
}

nvinfer1::ITensor *addMobileNetV3Block(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &input, const std::string &prefix, const V3Block &block)
{
    using namespace nvinfer1;

    ITensor *tensor = &input;
    int      idx    = 0;

    if (block.hidden != block.in)
    {
        tensor = addConvBnAct(network, weights_map, *tensor, prefix + "block." + std::to_string(idx++) + ".",
                              block.hidden, {1, 1, 1, 0, block.act}, 1e-3f);
    }

    tensor = addConvBnAct(network, weights_map, *tensor, prefix + "block." + std::to_string(idx++) + ".", block.hidden,
                          {block.kernel, block.stride, block.hidden, (block.kernel - 1) / 2, block.act}, 1e-3f);

    if (block.se)
    {
        tensor = addSE(network, weights_map, *tensor, prefix + "block." + std::to_string(idx++) + ".", block.hidden,
                       ((block.hidden / 4 + 7) / 8) * 8);
    }

    tensor = addConvBnAct(network, weights_map, *tensor, prefix + "block." + std::to_string(idx) + ".", block.out,
                          {1, 1, 1, 0, Act::None}, 1e-3f);

    if (block.stride == 1 && block.in == block.out)
    {
        tensor = network->addElementWise(input, *tensor, ElementWiseOperation::kSUM)->getOutput(0);
    }
    return tensor;
}

void buildMobileNetV2(const priv::IModelImpl &impl, nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                      bool feature_only)
{
    using namespace nvinfer1;

    ITensor *x = impl.addInputTensor(network);
    priv::IModelImpl::NamedTensorMap named_tensors{{"input", x}};
    x = addConvBnAct(network, weights_map, *x, "features.0.", 32, {3, 2, 1, 1, Act::Relu6}, 1e-5f);
    named_tensors["stem"] = x;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    const std::array<std::array<int, 4>, 7> cfg = {{{1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2},
                                                   {6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1}}};
    int in_channels = 32;
    int feature     = 1;
    for (const auto &[expand, out_channels, repeats, stride] : cfg)
    {
        for (int i = 0; i < repeats; ++i)
        {
            x = addMobileNetV2Block(network, weights_map, *x, "features." + std::to_string(feature++) + ".",
                                    in_channels, out_channels, i == 0 ? stride : 1, expand);
            named_tensors["features." + std::to_string(feature - 1)] = x;
            in_channels = out_channels;
            if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
            {
                return;
            }
        }
    }

    x = addConvBnAct(network, weights_map, *x, "features.18.", 1280, {1, 1, 1, 0, Act::Relu6}, 1e-5f);
    named_tensors["features.18"] = x;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }
    auto *flatten = addAvgFlatten(network, *x);
    named_tensors["flatten"] = flatten;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }
    x = addLinear(network, *flatten, weights_map, "classifier.1", impl.modelConfig().numClasses(), 1280);
    named_tensors["logits"] = x;
    if (feature_only)
    {
        impl.markFeatureOutputTensors(network, named_tensors);
        return;
    }

    impl.markOutputTensors(network, {x});
}

void buildMobileNetV3(const priv::IModelImpl &impl, nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                      const std::vector<V3Block> &blocks, int last_conv_channels, int last_channel, bool feature_only)
{
    using namespace nvinfer1;

    ITensor *x = impl.addInputTensor(network);
    priv::IModelImpl::NamedTensorMap named_tensors{{"input", x}};
    x = addConvBnAct(network, weights_map, *x, "features.0.", 16, {3, 2, 1, 1, Act::HSwish}, 1e-3f);
    named_tensors["stem"] = x;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    for (size_t i = 0; i < blocks.size(); ++i)
    {
        x = addMobileNetV3Block(network, weights_map, *x, "features." + std::to_string(i + 1) + ".", blocks[i]);
        named_tensors["features." + std::to_string(i + 1)] = x;
        if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
        {
            return;
        }
    }

    x = addConvBnAct(network, weights_map, *x, "features." + std::to_string(blocks.size() + 1) + ".", last_conv_channels,
                     {1, 1, 1, 0, Act::HSwish}, 1e-3f);
    named_tensors["features." + std::to_string(blocks.size() + 1)] = x;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }
    auto *flatten = addAvgFlatten(network, *x);
    named_tensors["flatten"] = flatten;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }
    x = addActivation(network, *addLinear(network, *flatten, weights_map, "classifier.0", last_channel,
                                          last_conv_channels), Act::HSwish);
    named_tensors["classifier.0"] = x;
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }
    x = addLinear(network, *x, weights_map, "classifier.3", impl.modelConfig().numClasses(), last_channel);
    named_tensors["logits"] = x;
    if (feature_only)
    {
        impl.markFeatureOutputTensors(network, named_tensors);
        return;
    }
    impl.markOutputTensors(network, {x});
}

void enqueue(priv::IModelImpl &impl, const std::vector<void *> &buffers)
{
    impl.ensurePrimaryInferenceReady();
    auto &trt_params = impl.trtParams();
    impl.bindTensorAddresses(buffers);

    if (!trt_params.stream)
    {
        trt_params.stream = MakeCudaStream();
        if (!trt_params.stream)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create CUDA stream");
        }
    }

    if (!trt_params.context->enqueueV3(*trt_params.stream))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute inference");
    }

    cudaStreamSynchronize(*trt_params.stream);
}

} // namespace

void MobileNetV2::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildMobileNetV2(*this, network, weights_map, isBuildingFeatureEngine());
}

void MobileNetV3Large::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildMobileNetV3(*this, network, weights_map,
                     {{16, 3, 16, 16, false, Act::Relu, 1},
                      {16, 3, 64, 24, false, Act::Relu, 2},
                      {24, 3, 72, 24, false, Act::Relu, 1},
                      {24, 5, 72, 40, true, Act::Relu, 2},
                      {40, 5, 120, 40, true, Act::Relu, 1},
                      {40, 5, 120, 40, true, Act::Relu, 1},
                      {40, 3, 240, 80, false, Act::HSwish, 2},
                      {80, 3, 200, 80, false, Act::HSwish, 1},
                      {80, 3, 184, 80, false, Act::HSwish, 1},
                      {80, 3, 184, 80, false, Act::HSwish, 1},
                              {80, 3, 480, 112, true, Act::HSwish, 1},
                              {112, 3, 672, 112, true, Act::HSwish, 1},
                              {112, 5, 672, 160, true, Act::HSwish, 2},
                              {160, 5, 960, 160, true, Act::HSwish, 1},
                              {160, 5, 960, 160, true, Act::HSwish, 1}},
                     960, 1280, isBuildingFeatureEngine());
}

void MobileNetV3Small::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildMobileNetV3(*this, network, weights_map,
                     {{16, 3, 16, 16, true, Act::Relu, 2},
                      {16, 3, 72, 24, false, Act::Relu, 2},
                      {24, 3, 88, 24, false, Act::Relu, 1},
                      {24, 5, 96, 40, true, Act::HSwish, 2},
                      {40, 5, 240, 40, true, Act::HSwish, 1},
                      {40, 5, 240, 40, true, Act::HSwish, 1},
                      {40, 5, 120, 48, true, Act::HSwish, 1},
                      {48, 5, 144, 48, true, Act::HSwish, 1},
                      {48, 5, 288, 96, true, Act::HSwish, 2},
                      {96, 5, 576, 96, true, Act::HSwish, 1},
                      {96, 5, 576, 96, true, Act::HSwish, 1}},
                     576, 1024, isBuildingFeatureEngine());
}

void MobileNet::infer(const std::vector<void *> &buffers)
{
    enqueue(*this, buffers);
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(MobileNetV2)
INFERRT_REGISTER_MODEL(MobileNetV3Large)
INFERRT_REGISTER_MODEL(MobileNetV3Small)
