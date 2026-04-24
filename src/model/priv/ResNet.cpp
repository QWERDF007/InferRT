#include "ResNet.hpp"

#include "BatchNorm.hpp"

namespace irt::model {

class BasicBlock
{
public:
    explicit BasicBlock(nvinfer1::INetworkDefinition *network, WeightsMap *weights_map)
        : network_(network)
        , weights_map_(weights_map)
    {
    }

    ~BasicBlock() = default;

    nvinfer1::IActivationLayer *operator()();

private:
    nvinfer1::INetworkDefinition *network_{nullptr};

    WeightsMap *weights_map_{nullptr};
};

inline nvinfer1::IActivationLayer *BasicBlock::operator()()
{
    return nullptr;
}

void ResNet18::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    using namespace nvinfer1;
    constexpr int N = 1;

    ITensor *input{nullptr};
    input = network->addInput("input", DataType::kFLOAT, Dims4{1, 3, 224, 224});

    IConvolutionLayer *conv1 = network->addConvolutionNd(*input, 64, DimsHW{7, 7}, weights_map.at("conv1.weight"),
                                                         weights_map.at("conv1.bias"));
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer *bn1 = addBatchNorm2d(network, weights_map, *conv1->getOutput(0), "bn1", 1e-5f);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});
}

} // namespace irt::model
