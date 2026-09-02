#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <inferrt/engine/EngineConfig.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace irt::engine::test {

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity, const char *) noexcept override {}
};

struct TensorRtDeleter
{
    template<typename T>
    void operator()(T *object) const
    {
        delete object;
    }
};

inline std::filesystem::path buildIdentityEngine(const bool two_inputs = false)
{
    TensorRtLogger                                       logger;
    std::unique_ptr<nvinfer1::IBuilder, TensorRtDeleter> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        throw std::runtime_error("Failed to create TensorRT builder");
    }
    std::unique_ptr<nvinfer1::INetworkDefinition, TensorRtDeleter> network(builder->createNetworkV2(0U));
    if (!network)
    {
        throw std::runtime_error("Failed to create TensorRT network");
    }
    auto *image = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 3, 2, 2});
    if (!image)
    {
        throw std::runtime_error("Failed to add image input");
    }

    nvinfer1::ITensor *output = nullptr;
    if (two_inputs)
    {
        auto *aux = network->addInput("aux", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 3, 2, 2});
        if (!aux)
        {
            throw std::runtime_error("Failed to add auxiliary input");
        }
        auto *sum = network->addElementWise(*image, *aux, nvinfer1::ElementWiseOperation::kSUM);
        if (!sum)
        {
            throw std::runtime_error("Failed to add TensorRT sum layer");
        }
        output = sum->getOutput(0);
    }
    else
    {
        auto *identity = network->addIdentity(*image);
        if (!identity)
        {
            throw std::runtime_error("Failed to add TensorRT identity layer");
        }
        output = identity->getOutput(0);
    }
    output->setName("output");
    network->markOutput(*output);

    std::unique_ptr<nvinfer1::IBuilderConfig, TensorRtDeleter> config(builder->createBuilderConfig());
    if (!config)
    {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }
    auto *profile = builder->createOptimizationProfile();
    if (!profile)
    {
        throw std::runtime_error("Failed to create TensorRT optimization profile");
    }
    for (const char *name : two_inputs ? std::vector<const char *>{"image", "aux"} : std::vector<const char *>{"image"})
    {
        if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 2, 2})
            || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{2, 3, 2, 2})
            || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{4, 3, 2, 2}))
        {
            throw std::runtime_error("Failed to configure TensorRT optimization profile");
        }
    }
    if (config->addOptimizationProfile(profile) == -1)
    {
        throw std::runtime_error("Failed to add TensorRT optimization profile");
    }
    std::unique_ptr<nvinfer1::IHostMemory, TensorRtDeleter> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
    {
        throw std::runtime_error("Failed to serialize TensorRT test engine");
    }

    const auto path = std::filesystem::temp_directory_path()
                    / (two_inputs ? "inferrt_gpu_multi_input.engine" : "inferrt_gpu_identity.engine");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.good())
    {
        throw std::runtime_error("Failed to open TensorRT test engine output");
    }
    file.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
    if (!file.good())
    {
        throw std::runtime_error("Failed to write TensorRT test engine");
    }
    return path;
}

} // namespace irt::engine::test
