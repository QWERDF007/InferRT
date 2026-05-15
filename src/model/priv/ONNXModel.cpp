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

std::string BuildEngineFileName(const ONNXModel &model, const std::string &onnx_file)
{
    std::string engine_file = onnx_file;
    size_t      pos         = engine_file.rfind(model.wtsExtension());
    if (pos != std::string::npos)
    {
        engine_file.replace(pos, model.wtsExtension().size(), model.engineExtension());
    }
    else
    {
        engine_file += model.engineExtension();
    }

    const auto ext_pos = engine_file.rfind(model.engineExtension());
    const auto suffix  = model.generateSuffix(model.modelConfig());
    if (ext_pos != std::string::npos)
    {
        engine_file.insert(ext_pos, suffix);
    }
    else
    {
        engine_file += suffix;
    }

    return engine_file;
}

void ValidateConfig(const ONNXModel &model)
{
    const auto &config  = model.modelConfig();
    const auto &shape   = config.inputShape();
    const auto &inputs  = config.inputTensorNames();
    const auto &outputs = config.outputTensorNames();

    if (shape.channels <= 0 || shape.height <= 0 || shape.width <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input shape must be positive, got C=%d H=%d W=%d",
                             shape.channels, shape.height, shape.width);
    }

    if (inputs.empty() || outputs.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input and output tensor names must not be empty");
    }
}

void SyncModelMetadataFromEngine(ONNXModel &model)
{
    const auto input_names  = model.ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    const auto output_names = model.ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);

    if (!input_names.empty())
    {
        model.setInputTensorNames(input_names);

        const auto input_dims = model.tensorShape(input_names.front());
        if (input_dims.nbDims >= 3)
        {
            const int c = static_cast<int>(input_dims.d[input_dims.nbDims - 3]);
            const int h = static_cast<int>(input_dims.d[input_dims.nbDims - 2]);
            const int w = static_cast<int>(input_dims.d[input_dims.nbDims - 1]);
            if (c > 0 && h > 0 && w > 0)
            {
                model.setInputShape(c, h, w);
            }
        }
    }

    if (!output_names.empty())
    {
        model.setOutputTensorNames(output_names);
    }
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
    priv::IModelImpl::load(engine_file);
    SyncModelMetadataFromEngine(*this);
}

void ONNXModel::buildOrLoad(const std::string &onnx_file)
{
    ValidateConfig(*this);

    auto &trt_params = trtParams();
    if (trt_params.logger == nullptr)
    {
        initLogger();
    }

    std::string engine_file = BuildEngineFileName(*this, onnx_file);

    std::ifstream file(engine_file);
    bool          engine_exists = file.good();
    file.close();

    if (engine_exists)
    {
        LOG_INFO(*trt_params.logger) << "Found existing engine file: " << engine_file << ", loading..." << std::endl;
        try
        {
            load(engine_file);
            return;
        }
        catch (const std::exception &e)
        {
            LOG_WARN(*trt_params.logger) << "Failed to load engine: " << e.what() << std::endl;
            LOG_INFO(*trt_params.logger) << "Will rebuild engine from ONNX file: " << onnx_file << std::endl;
        }
    }

    LOG_INFO(*trt_params.logger) << "Building engine from ONNX file: " << onnx_file << std::endl;
    build(onnx_file);

    try
    {
        save(engine_file);
    }
    catch (const std::exception &e)
    {
        LOG_WARN(*trt_params.logger) << "Failed to save engine: " << e.what() << std::endl;
    }
}

void ONNXModel::infer(const std::vector<void *> &buffers)
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

    if (!trt_params.context->enqueueV3(*trt_params.stream))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute inference");
    }

    cudaStreamSynchronize(*trt_params.stream);
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(ONNXModel)
