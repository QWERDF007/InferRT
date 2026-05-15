#include "AlexNet.hpp"

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
    ITensor *input = addInputTensor(network);

    // features
    // CRP (Conv-Relu-Pool)
    IConvolutionLayer *conv1 = network->addConvolutionNd(
        *input, 64, DimsHW{11, 11}, weights_map.at("features.0.weight"), weights_map.at("features.0.bias"));
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});

    IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});

    // CRP
    IConvolutionLayer *conv2
        = network->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5}, weights_map.at("features.3.weight"),
                                    weights_map.at("features.3.bias"));
    conv2->setPaddingNd(DimsHW{2, 2});

    IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool2->setStrideNd(DimsHW{2, 2});

    // CR
    IConvolutionLayer *conv3
        = network->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weights_map.at("features.6.weight"),
                                    weights_map.at("features.6.bias"));
    conv3->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);

    // CR
    IConvolutionLayer *conv4
        = network->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weights_map.at("features.8.weight"),
                                    weights_map.at("features.8.bias"));
    conv4->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);

    // CRP
    IConvolutionLayer *conv5
        = network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weights_map.at("features.10.weight"),
                                    weights_map.at("features.10.bias"));
    conv5->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool3->setStrideNd(DimsHW{2, 2});

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

    IShuffleLayer *shuffle = network->addShuffle(*adaptive_pool->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{1, -1});

    // classifier
    ITensor *fc1w
        = network->addConstant(DimsHW{4096, fc1_in_features}, weights_map.at("classifier.1.weight"))->getOutput(0);
    ITensor *fc1b = network->addConstant(DimsHW{1, 4096}, weights_map.at("classifier.1.bias"))->getOutput(0);
    ITensor *fc2w = network->addConstant(DimsHW{4096, 4096}, weights_map.at("classifier.4.weight"))->getOutput(0);
    ITensor *fc2b = network->addConstant(DimsHW{1, 4096}, weights_map.at("classifier.4.bias"))->getOutput(0);
    const int num_classes = modelConfig().numClasses();
    ITensor  *fc3w
        = network->addConstant(DimsHW{num_classes, 4096}, weights_map.at("classifier.6.weight"))->getOutput(0);
    ITensor *fc3b = network->addConstant(DimsHW{1, num_classes}, weights_map.at("classifier.6.bias"))->getOutput(0);

    // IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool3->getOutput(0), 4096, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    IMatrixMultiplyLayer *fc1_0 = network->addMatrixMultiply(*shuffle->getOutput(0), MatrixOperation::kNONE, *fc1w,
                                                             MatrixOperation::kTRANSPOSE);
    IElementWiseLayer    *fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    IActivationLayer     *relu6 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    // fc1_0->setName("fc1_0");  // set name here, only for debug purpose

    IMatrixMultiplyLayer *fc2_0
        = network->addMatrixMultiply(*relu6->getOutput(0), MatrixOperation::kNONE, *fc2w, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, ElementWiseOperation::kSUM);
    IActivationLayer  *relu7 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    // fc2_0->setName("fc2_0");

    IMatrixMultiplyLayer *fc3_0
        = network->addMatrixMultiply(*relu7->getOutput(0), MatrixOperation::kNONE, *fc3w, MatrixOperation::kTRANSPOSE);
    IElementWiseLayer *fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, ElementWiseOperation::kSUM);
    // fc3_0->setName("fc3_0");

    markOutputTensors(network, {fc3_1->getOutput(0)});
}

void AlexNet::infer(const std::vector<void *> &buffers)
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

    // 同步等待执行完成
    cudaStreamSynchronize(*trt_params.stream);
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(AlexNet)
