#include "VGG.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <string>
#include <vector>

namespace irt::model {

namespace {

struct VGGConfig
{
    std::vector<int> feature_weight_indices;
    std::vector<int> conv_channels;
    std::vector<int> block_depths;
    std::vector<int> classifier_weight_indices;
};

nvinfer1::IActivationLayer *addConvRelu(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                        const WeightsMap &weights_map, int layer_index, int out_channels)
{
    using namespace nvinfer1;

    const auto prefix = "features." + std::to_string(layer_index);
    IConvolutionLayer *conv = network->addConvolutionNd(input, out_channels, DimsHW{3, 3},
                                                        weights_map.at(prefix + ".weight"), weights_map.at(prefix + ".bias"));
    conv->setPaddingNd(DimsHW{1, 1});

    return network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
}

nvinfer1::ITensor *addMaxPool(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    using namespace nvinfer1;

    IPoolingLayer *pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool->setStrideNd(DimsHW{2, 2});
    return pool->getOutput(0);
}

void buildVGG(const priv::IModelImpl &impl, nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
              const VGGConfig &config, bool feature_only = false)
{
    using namespace nvinfer1;

    ITensor *tensor = impl.addInputTensor(network);
    priv::IModelImpl::NamedTensorMap named_tensors{{"input", tensor}};

    int feature_index = 0;
    int block_index   = 1;
    for (int block_depth : config.block_depths)
    {
        for (int i = 0; i < block_depth; ++i)
        {
            IActivationLayer *relu
                = addConvRelu(network, *tensor, weights_map, config.feature_weight_indices[feature_index],
                              config.conv_channels[feature_index]);
            tensor = relu->getOutput(0);
            ++feature_index;
        }
        tensor = addMaxPool(network, *tensor);
        named_tensors["block" + std::to_string(block_index++)] = tensor;
        if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
        {
            return;
        }
    }

    IResizeLayer *adaptive_pool = network->addResize(*tensor);
    adaptive_pool->setOutputDimensions(Dims4{1, 512, 7, 7});
    named_tensors["avgpool"] = adaptive_pool->getOutput(0);
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IShuffleLayer *shuffle = network->addShuffle(*adaptive_pool->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{1, -1});
    named_tensors["flatten"] = shuffle->getOutput(0);
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    constexpr int fc1_in_features = 512 * 7 * 7;

    ITensor *fc1w = network
                        ->addConstant(DimsHW{4096, fc1_in_features},
                                      weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[0]) + ".weight"))
                        ->getOutput(0);
    ITensor *fc1b = network
                        ->addConstant(DimsHW{1, 4096},
                                      weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[0]) + ".bias"))
                        ->getOutput(0);
    ITensor *fc2w = network
                        ->addConstant(DimsHW{4096, 4096},
                                      weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[1]) + ".weight"))
                        ->getOutput(0);
    ITensor *fc2b = network
                        ->addConstant(DimsHW{1, 4096},
                                      weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[1]) + ".bias"))
                        ->getOutput(0);
    const int num_classes = impl.modelConfig().numClasses();
    ITensor  *fc3w        = network
                       ->addConstant(DimsHW{num_classes, 4096},
                                     weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[2]) + ".weight"))
                       ->getOutput(0);
    ITensor *fc3b = network
                        ->addConstant(DimsHW{1, num_classes},
                                      weights_map.at("classifier." + std::to_string(config.classifier_weight_indices[2]) + ".bias"))
                        ->getOutput(0);

    IMatrixMultiplyLayer *fc1_0 = network->addMatrixMultiply(*shuffle->getOutput(0), MatrixOperation::kNONE, *fc1w,
                                                             MatrixOperation::kTRANSPOSE);
    IElementWiseLayer    *fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    IActivationLayer     *fc1_2 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    named_tensors["fc1"] = fc1_2->getOutput(0);
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IMatrixMultiplyLayer *fc2_0 = network->addMatrixMultiply(*fc1_2->getOutput(0), MatrixOperation::kNONE, *fc2w,
                                                             MatrixOperation::kTRANSPOSE);
    IElementWiseLayer    *fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, ElementWiseOperation::kSUM);
    IActivationLayer     *fc2_2 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    named_tensors["fc2"] = fc2_2->getOutput(0);
    if (feature_only && impl.tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IMatrixMultiplyLayer *fc3_0 = network->addMatrixMultiply(*fc2_2->getOutput(0), MatrixOperation::kNONE, *fc3w,
                                                             MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, ElementWiseOperation::kSUM);
    named_tensors["logits"] = fc3_1->getOutput(0);
    if (feature_only)
    {
        impl.markFeatureOutputTensors(network, named_tensors);
        return;
    }

    impl.markOutputTensors(network, {fc3_1->getOutput(0)});
}

} // namespace

void VGG11::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 3, 6, 8, 11, 13, 16, 18},
                                           {64, 128, 256, 256, 512, 512, 512, 512},
                                           {1, 1, 2, 2, 2},
                                           {0, 3, 6}});
}

void VGG11::buildFeatureNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 3, 6, 8, 11, 13, 16, 18},
                                           {64, 128, 256, 256, 512, 512, 512, 512},
                                           {1, 1, 2, 2, 2},
                                           {0, 3, 6}}, true);
}

void VGG13::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 2, 5, 7, 10, 12, 15, 17, 20, 22},
                                           {64, 64, 128, 128, 256, 256, 512, 512, 512, 512},
                                           {2, 2, 2, 2, 2},
                                           {0, 3, 6}});
}

void VGG13::buildFeatureNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 2, 5, 7, 10, 12, 15, 17, 20, 22},
                                           {64, 64, 128, 128, 256, 256, 512, 512, 512, 512},
                                           {2, 2, 2, 2, 2},
                                           {0, 3, 6}}, true);
}

void VGG16::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28},
                                           {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512},
                                           {2, 2, 3, 3, 3},
                                           {0, 3, 6}});
}

void VGG16::buildFeatureNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map, {{0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28},
                                           {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512},
                                           {2, 2, 3, 3, 3},
                                           {0, 3, 6}}, true);
}

void VGG19::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map,
             {{0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34},
              {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512},
              {2, 2, 4, 4, 4},
              {0, 3, 6}});
}

void VGG19::buildFeatureNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    buildVGG(*this, network, weights_map,
             {{0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34},
              {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512},
              {2, 2, 4, 4, 4},
              {0, 3, 6}}, true);
}

void VGG::infer(const std::vector<void *> &buffers)
{
    auto &trt_params = trtParams();
    bindTensorAddresses(buffers);

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

INFERRT_REGISTER_MODEL(VGG11)
INFERRT_REGISTER_MODEL(VGG13)
INFERRT_REGISTER_MODEL(VGG16)
INFERRT_REGISTER_MODEL(VGG19)
