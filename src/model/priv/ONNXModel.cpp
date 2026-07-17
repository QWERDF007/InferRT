#include "ONNXModel.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <fstream>
#include <memory>
#include <vector>

namespace irt::model {

namespace {

void ValidateConfig(const ONNXModel &model)
{
    const auto &config  = model.modelConfig();
    const auto &shapes  = config.inputShapes();
    const auto &inputs  = config.inputTensorNames();
    const auto &outputs = config.outputTensorNames();

    if (shapes.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input shapes must not be empty");
    }

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        const auto &shape = shapes[i];
        if (shape.d[0] <= 0 || shape.d[1] <= 0 || shape.d[2] <= 0 || shape.d[3] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "input shape at index %zu must be positive, got N=%d C=%d H=%d W=%d", i, shape.d[0],
                                 shape.d[1], shape.d[2], shape.d[3]);
        }
    }

    if (inputs.empty() || outputs.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input and output tensor names must not be empty");
    }

    if (inputs.size() != shapes.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "input tensor name count (%zu) must match input shape count (%zu)", inputs.size(),
                             shapes.size());
    }

    if (!config.featureTensorNames().empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION,
                             "ONNXModel does not support selecting intermediate feature tensors");
    }

    if (config.featureOnly())
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "ONNXModel does not support featureOnly configuration");
    }
}

void SyncModelMetadataFromEngine(ONNXModel &model)
{
    const auto &current_config = model.modelConfig();
    const int   num_classes    = current_config.numClasses();
    auto        input_shapes   = current_config.inputShapes();
    auto        input_names    = current_config.inputTensorNames();
    auto        output_names   = current_config.outputTensorNames();

    const auto engine_input_names  = model.ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    const auto engine_output_names = model.ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    if (!engine_input_names.empty())
    {
        input_names = engine_input_names;

        std::vector<nvinfer1::Dims4> engine_input_shapes;
        engine_input_shapes.reserve(input_names.size());
        for (const auto &input_name : input_names)
        {
            const auto input_dims = model.tensorShape(input_name);
            if (input_dims.nbDims >= 3)
            {
                int batch = 1;
                if (input_dims.nbDims >= 4 && input_dims.d[input_dims.nbDims - 4] > 0)
                {
                    batch = static_cast<int>(input_dims.d[input_dims.nbDims - 4]);
                }
                const int c = static_cast<int>(input_dims.d[input_dims.nbDims - 3]);
                const int h = static_cast<int>(input_dims.d[input_dims.nbDims - 2]);
                const int w = static_cast<int>(input_dims.d[input_dims.nbDims - 1]);
                if (c > 0 && h > 0 && w > 0)
                {
                    engine_input_shapes.emplace_back(batch, c, h, w);
                }
            }
        }

        if (engine_input_shapes.size() == input_names.size())
        {
            input_shapes = std::move(engine_input_shapes);
        }
    }

    if (!engine_output_names.empty())
    {
        output_names = engine_output_names;
    }

    auto config = std::make_unique<IModelConfig>();
    config->setNumClasses(num_classes);
    config->setInputShapes(std::move(input_shapes));
    config->setInputTensorNames(std::move(input_names));
    config->setOutputTensorNames(std::move(output_names));
    config->setFeatureOnly(current_config.featureOnly());
    if (current_config.dynamicBatch())
    {
        config->setDynamicBatchRange(current_config.minBatchSize(), current_config.optBatchSize(),
                                     current_config.maxBatchSize());
    }
    config->setBackend(current_config.backend());
    config->setDevice(current_config.device());
    config->setDeviceId(current_config.deviceId());
    config->setPrecision(current_config.precision());
    model.replaceModelConfigWithoutReset(std::move(config));
}

} // namespace

void ONNXModel::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    static_cast<void>(network);
    static_cast<void>(weights_map);
    throw irt::Exception(Status::ERROR_INVALID_OPERATION, "ONNXModel does not use manual buildNetwork");
}

void ONNXModel::build(const std::string &onnx_file)
{
    using namespace nvinfer1;

    if (!usesTensorRTBackend())
    {
        priv::IModelImpl::build(onnx_file);
        return;
    }

    ValidateConfig(*this);

    auto &trt_params = trtParams();
    if (trt_params.logger == nullptr)
    {
        initLogger();
    }

    LOG_INFO(*trt_params.logger) << "Loading ONNX file: " << onnx_file << std::endl;

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*trt_params.logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    const auto flags   = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto       network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create NetworkDefinition");
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *trt_params.logger));
    if (!parser)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create ONNX parser");
    }

    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(logLevel())))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to parse ONNX file: %s", onnx_file.c_str());
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create BuilderConfig");
    }

    LOG_INFO(*trt_params.logger) << "Building TensorRT engine from ONNX..." << std::endl;
    auto buffer = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!buffer)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to build serialized network from ONNX");
    }

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(*trt_params.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    trt_params.engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer->data(), buffer->size()),
                                                     [](ICudaEngine *engine) { delete engine; });
    if (!trt_params.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    trt_params.context.reset(trt_params.engine->createExecutionContext());
    if (!trt_params.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    SyncModelMetadataFromEngine(*this);

    LOG_INFO(*trt_params.logger) << "TensorRT engine built successfully from ONNX" << std::endl;
}

void ONNXModel::load(const std::string &engine_file)
{
    if (!usesTensorRTBackend())
    {
        priv::IModelImpl::load(engine_file);
        return;
    }

    priv::IModelImpl::load(engine_file);
    SyncModelMetadataFromEngine(*this);
}

void ONNXModel::buildOrLoad(const std::string &onnx_file)
{
    if (!usesTensorRTBackend())
    {
        priv::IModelImpl::buildOrLoad(onnx_file);
        return;
    }

    ValidateConfig(*this);
    priv::IModelImpl::buildOrLoad(onnx_file);
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(ONNXModel)
