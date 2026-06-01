#include "AlexNet.hpp"

#include "Layers.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace irt::model {

void AlexNet::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    using namespace nvinfer1;
    const bool                       feature_only = isBuildingFeatureEngine();
    ITensor                         *input        = addInputTensor(network);
    priv::IModelImpl::NamedTensorMap named_tensors{
        {"input", input}
    };

    // features
    // CRP (Conv-Relu-Pool)
    IConvolutionLayer *conv1 = network->addConvolutionNd(
        *input, 64, DimsHW{11, 11}, weights_map.at("features.0.weight"), weights_map.at("features.0.bias"));
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});

    IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    named_tensors["conv1"]  = conv1->getOutput(0);
    named_tensors["relu1"]  = relu1->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    named_tensors["pool1"] = pool1->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    // CRP
    IConvolutionLayer *conv2
        = network->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5}, weights_map.at("features.3.weight"),
                                    weights_map.at("features.3.bias"));
    conv2->setPaddingNd(DimsHW{2, 2});

    IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    named_tensors["conv2"]  = conv2->getOutput(0);
    named_tensors["relu2"]  = relu2->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IPoolingLayer *pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool2->setStrideNd(DimsHW{2, 2});
    named_tensors["pool2"] = pool2->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    // CR
    IConvolutionLayer *conv3
        = network->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weights_map.at("features.6.weight"),
                                    weights_map.at("features.6.bias"));
    conv3->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    named_tensors["conv3"]  = conv3->getOutput(0);
    named_tensors["relu3"]  = relu3->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    // CR
    IConvolutionLayer *conv4
        = network->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weights_map.at("features.8.weight"),
                                    weights_map.at("features.8.bias"));
    conv4->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    named_tensors["conv4"]  = conv4->getOutput(0);
    named_tensors["relu4"]  = relu4->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    // CRP
    IConvolutionLayer *conv5
        = network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weights_map.at("features.10.weight"),
                                    weights_map.at("features.10.bias"));
    conv5->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    named_tensors["conv5"]  = conv5->getOutput(0);
    named_tensors["relu5"]  = relu5->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IPoolingLayer *pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool3->setStrideNd(DimsHW{2, 2});
    named_tensors["pool3"] = pool3->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    const auto fc1_in_features = static_cast<int>(weights_map.at("classifier.1.weight").count / 4096);
    if (fc1_in_features <= 0 || fc1_in_features % 256 != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unexpected AlexNet classifier.1.weight shape");
    }

    const auto pooled_area = fc1_in_features / 256;
    const auto pooled_hw   = static_cast<int>(std::sqrt(static_cast<double>(pooled_area)));
    if (pooled_hw <= 0 || pooled_hw * pooled_hw != pooled_area)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "AlexNet flattened feature size is not square");
    }

    IResizeLayer *adaptive_pool = network->addResize(*pool3->getOutput(0));
    adaptive_pool->setOutputDimensions(Dims4{1, 256, pooled_hw, pooled_hw});
    named_tensors["avgpool"] = adaptive_pool->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    ITensor *flatten         = flattenPreserveBatch(network, *adaptive_pool->getOutput(0));
    named_tensors["flatten"] = flatten;
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    // classifier
    ITensor *fc1w
        = network->addConstant(DimsHW{4096, fc1_in_features}, weights_map.at("classifier.1.weight"))->getOutput(0);
    ITensor  *fc1b = network->addConstant(DimsHW{1, 4096}, weights_map.at("classifier.1.bias"))->getOutput(0);
    ITensor  *fc2w = network->addConstant(DimsHW{4096, 4096}, weights_map.at("classifier.4.weight"))->getOutput(0);
    ITensor  *fc2b = network->addConstant(DimsHW{1, 4096}, weights_map.at("classifier.4.bias"))->getOutput(0);
    const int num_classes = modelConfig().numClasses();
    ITensor  *fc3w
        = network->addConstant(DimsHW{num_classes, 4096}, weights_map.at("classifier.6.weight"))->getOutput(0);
    ITensor *fc3b = network->addConstant(DimsHW{1, num_classes}, weights_map.at("classifier.6.bias"))->getOutput(0);

    IMatrixMultiplyLayer *fc1_0
        = network->addMatrixMultiply(*flatten, MatrixOperation::kNONE, *fc1w, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    IActivationLayer  *relu6 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    named_tensors["fc1"]     = fc1_1->getOutput(0);
    named_tensors["relu6"]   = relu6->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IMatrixMultiplyLayer *fc2_0
        = network->addMatrixMultiply(*relu6->getOutput(0), MatrixOperation::kNONE, *fc2w, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, ElementWiseOperation::kSUM);
    IActivationLayer  *relu7 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    named_tensors["fc2"]     = fc2_1->getOutput(0);
    named_tensors["relu7"]   = relu7->getOutput(0);
    if (feature_only && tryMarkFeatureOutputTensors(network, named_tensors))
    {
        return;
    }

    IMatrixMultiplyLayer *fc3_0
        = network->addMatrixMultiply(*relu7->getOutput(0), MatrixOperation::kNONE, *fc3w, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, ElementWiseOperation::kSUM);
    named_tensors["logits"]  = fc3_1->getOutput(0);
    if (feature_only)
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    markOutputTensors(network, {fc3_1->getOutput(0)});
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(AlexNet)
