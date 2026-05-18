#include "IModelImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <cctype>
#include <fstream>
#include <functional>
#include <unordered_set>

namespace irt::model::priv {

namespace {

bool HasTensor(const nvinfer1::ICudaEngine *engine, const std::string &tensor_name)
{
    if (!engine)
    {
        return false;
    }

    const int32_t num_io_tensors = engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (name && tensor_name == name)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief 在构建或加载模型前验证配置合法性。
 * @param impl 模型内部实现对象。
 */
void ValidateModelConfig(const IModelImpl &impl)
{
    const auto &config                       = impl.modelConfig();
    const auto &input_shapes                 = config.inputShapes();
    const auto &input_tensor_names           = config.inputTensorNames();
    const auto &output_tensor_names          = config.outputTensorNames();
    const auto &feature_tensor_names         = config.featureTensorNames();
    const auto &feature_output_tensor_names = config.featureOutputTensorNames();

    if (config.numClasses() <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "num_classes must be positive, got %d",
                             config.numClasses());
    }

    if (input_shapes.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "at least one input shape is required");
    }

    for (size_t i = 0; i < input_shapes.size(); ++i)
    {
        const auto &input_shape = input_shapes[i];
        if (input_shape.d[0] <= 0 || input_shape.d[1] <= 0 || input_shape.d[2] <= 0 || input_shape.d[3] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "input shape at index %zu must be positive, got N=%d C=%d H=%d W=%d", i,
                                 input_shape.d[0], input_shape.d[1], input_shape.d[2], input_shape.d[3]);
        }
    }

    if (input_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "at least one input tensor name is required");
    }

    if (input_tensor_names.size() != input_shapes.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "input tensor name count (%zu) must match input shape count (%zu)",
                             input_tensor_names.size(), input_shapes.size());
    }

    if (output_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "at least one output tensor name is required");
    }

    for (const auto &input_tensor_name : input_tensor_names)
    {
        if (input_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input tensor name must not be empty");
        }
    }

    for (const auto &output_tensor_name : output_tensor_names)
    {
        if (output_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "output tensor name must not be empty");
        }
    }

    for (const auto &feature_tensor_name : feature_tensor_names)
    {
        if (feature_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "feature tensor name must not be empty");
        }
    }

    if (!feature_output_tensor_names.empty() && feature_output_tensor_names.size() != feature_tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "feature output tensor name count (%zu) must match feature tensor count (%zu)",
                             feature_output_tensor_names.size(), feature_tensor_names.size());
    }

    for (const auto &feature_output_tensor_name : feature_output_tensor_names)
    {
        if (feature_output_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "feature output tensor name must not be empty");
        }
    }

    if (config.featureOnly() && feature_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "featureTensorNames must not be empty when featureOnly is enabled");
    }

    std::unordered_set<std::string> unique_output_names;
    for (const auto &output_tensor_name : output_tensor_names)
    {
        if (!unique_output_names.insert(output_tensor_name).second)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "duplicate output tensor name: %s",
                                 output_tensor_name.c_str());
        }
    }

    std::unordered_set<std::string> unique_feature_output_names;
    const auto &feature_export_names = feature_output_tensor_names.empty() ? feature_tensor_names : feature_output_tensor_names;
    for (const auto &feature_export_name : feature_export_names)
    {
        if (!unique_feature_output_names.insert(feature_export_name).second)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "duplicate feature output tensor name: %s",
                                 feature_export_name.c_str());
        }
    }
}

/**
 * @brief 根据权重文件路径和模型配置生成 engine 文件路径。
 * @param impl 模型内部实现对象。
 * @param weights_file 权重文件路径。
 * @return engine 文件路径。
 */
std::string BuildEngineFileName(const IModelImpl &impl, const std::string &weights_file)
{
    std::string engine_file = weights_file;
    size_t      pos         = engine_file.rfind(impl.wtsExtension());
    if (pos != std::string::npos)
    {
        engine_file.replace(pos, impl.wtsExtension().size(), impl.engineExtension());
    }
    else
    {
        engine_file += impl.engineExtension();
    }

    const auto &config  = impl.modelConfig();
    const auto  ext_pos = engine_file.rfind(impl.engineExtension());
    auto        base_config = IModelConfig{};
    base_config.setNumClasses(config.numClasses());
    base_config.setInputShapes(config.inputShapes());
    base_config.setInputTensorNames(config.inputTensorNames());
    base_config.setOutputTensorNames(config.outputTensorNames());
    const auto suffix = impl.generateSuffix(base_config);

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

std::string BuildFeatureEngineSuffix(const IModelImpl &impl)
{
    auto sanitize = [](const std::string &value) {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
        }
        return result;
    };

    std::string suffix = "_features";
    for (const auto &feature_tensor_name : impl.modelConfig().featureTensorNames())
    {
        suffix += "_feat_" + sanitize(feature_tensor_name);
    }
    return suffix;
}

} // namespace

nvinfer1::ILogger::Severity IModelImpl::logLevel() const noexcept
{
    return trt_params_.log_level;
}

std::string IModelImpl::generateSuffix(const IModelConfig &config) const noexcept
{
    auto sanitize = [](const std::string &value) {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
        }
        return result;
    };

    std::string suffix;
    for (const auto &input_shape : config.inputShapes())
    {
        suffix += "_" + std::to_string(input_shape.d[0]) + "x" + std::to_string(input_shape.d[1]) + "x"
                + std::to_string(input_shape.d[2]) + "x" + std::to_string(input_shape.d[3]);
    }
    suffix += "_" + std::to_string(config.numClasses());
    for (const auto &feature_tensor_name : config.featureTensorNames())
    {
        suffix += "_feat_" + sanitize(feature_tensor_name);
    }
    return suffix;
}

void IModelImpl::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    config_ = config ? std::move(config) : std::make_unique<IModelConfig>();
    trt_params_.context.reset();
    trt_params_.engine.reset();
    trt_params_.stream.reset();
    trt_params_.feature_only = false;
    feature_trt_params_.context.reset();
    feature_trt_params_.engine.reset();
    feature_trt_params_.stream.reset();
    feature_trt_params_.feature_only = false;
}

const IModelConfig &IModelImpl::modelConfig() const noexcept
{
    return *config_;
}

std::vector<std::string> IModelImpl::ioTensorNames(nvinfer1::TensorIOMode mode) const
{
    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    std::vector<std::string> names;
    const int32_t            num_io_tensors = trt_params_.engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char *tensor_name = trt_params_.engine->getIOTensorName(i);
        if (trt_params_.engine->getTensorIOMode(tensor_name) == mode)
        {
            names.emplace_back(tensor_name);
        }
    }

    return names;
}

nvinfer1::Dims IModelImpl::tensorShape(const std::string &tensor_name) const
{
    if (trt_params_.context && HasTensor(trt_params_.engine.get(), tensor_name))
    {
        return trt_params_.context->getTensorShape(tensor_name.c_str());
    }

    if (trt_params_.engine && HasTensor(trt_params_.engine.get(), tensor_name))
    {
        return trt_params_.engine->getTensorShape(tensor_name.c_str());
    }

    if (feature_trt_params_.context && HasTensor(feature_trt_params_.engine.get(), tensor_name))
    {
        return feature_trt_params_.context->getTensorShape(tensor_name.c_str());
    }

    if (feature_trt_params_.engine && HasTensor(feature_trt_params_.engine.get(), tensor_name))
    {
        return feature_trt_params_.engine->getTensorShape(tensor_name.c_str());
    }

    if (!trt_params_.engine && !feature_trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

nvinfer1::DataType IModelImpl::tensorDataType(const std::string &tensor_name) const
{
    if (trt_params_.engine && HasTensor(trt_params_.engine.get(), tensor_name))
    {
        return trt_params_.engine->getTensorDataType(tensor_name.c_str());
    }

    if (feature_trt_params_.engine && HasTensor(feature_trt_params_.engine.get(), tensor_name))
    {
        return feature_trt_params_.engine->getTensorDataType(tensor_name.c_str());
    }

    if (!trt_params_.engine && !feature_trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor not found: %s", tensor_name.c_str());
}

void IModelImpl::setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims)
{
    if (!trt_params_.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    if (!trt_params_.context->setInputShape(tensor_name.c_str(), dims))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to set input tensor shape: %s",
                             tensor_name.c_str());
    }

    if (feature_trt_params_.context
        && !feature_trt_params_.context->setInputShape(tensor_name.c_str(), dims))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to set feature input tensor shape: %s",
                             tensor_name.c_str());
    }
}

void IModelImpl::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    trt_params_.log_level = severity;
    feature_trt_params_.log_level = severity;
    if (trt_params_.logger)
    {
        trt_params_.logger->setReportableSeverity(severity);
    }
    if (feature_trt_params_.logger)
    {
        feature_trt_params_.logger->setReportableSeverity(severity);
    }
}

nvinfer1::ITensor *IModelImpl::addInputTensor(nvinfer1::INetworkDefinition *network, const nvinfer1::Dims &dims,
                                              nvinfer1::DataType data_type, size_t input_index) const
{
    const auto &tensor_names = modelConfig().inputTensorNames();
    if (input_index >= tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "input index %zu is out of range for %zu configured input tensor names", input_index,
                             tensor_names.size());
    }

    return network->addInput(tensor_names[input_index].c_str(), data_type, dims);
}

nvinfer1::ITensor *IModelImpl::addInputTensor(nvinfer1::INetworkDefinition *network, nvinfer1::DataType data_type,
                                              size_t input_index) const
{
    const auto &shapes = modelConfig().inputShapes();
    if (input_index >= shapes.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "input index %zu is out of range for %zu configured input shapes", input_index,
                             shapes.size());
    }

    return addInputTensor(network, shapes[input_index], data_type, input_index);
}

void IModelImpl::markOutputTensors(nvinfer1::INetworkDefinition *network,
                                   const std::vector<nvinfer1::ITensor *> &outputs) const
{
    const auto tensor_names = primaryOutputTensorNames();
    if (outputs.size() != tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "output tensor count mismatch: got %zu tensors but %zu names were configured",
                             outputs.size(), tensor_names.size());
    }

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i] == nullptr)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "output tensor at index %zu is null", i);
        }

        outputs[i]->setName(tensor_names[i].c_str());
        network->markOutput(*outputs[i]);
    }
}

void IModelImpl::markFeatureOutputTensors(nvinfer1::INetworkDefinition *network, const NamedTensorMap &named_tensors) const
{
    const auto feature_outputs = resolveFeatureTensors(named_tensors);
    const auto tensor_names    = featureOutputTensorNames();
    if (feature_outputs.size() != tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "feature output tensor count mismatch: got %zu tensors but %zu names were configured",
                             feature_outputs.size(), tensor_names.size());
    }

    for (size_t i = 0; i < feature_outputs.size(); ++i)
    {
        if (feature_outputs[i] == nullptr)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "feature output tensor at index %zu is null", i);
        }

        feature_outputs[i]->setName(tensor_names[i].c_str());
        network->markOutput(*feature_outputs[i]);
    }
}

bool IModelImpl::tryMarkFeatureOutputTensors(nvinfer1::INetworkDefinition *network, const NamedTensorMap &named_tensors) const
{
    const auto &feature_tensor_names = modelConfig().featureTensorNames();
    if (feature_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "feature tensor names must not be empty");
    }

    for (const auto &feature_tensor_name : feature_tensor_names)
    {
        if (named_tensors.find(feature_tensor_name) == named_tensors.end())
        {
            return false;
        }
    }

    markFeatureOutputTensors(network, named_tensors);
    return true;
}

void IModelImpl::bindTensorAddresses(const std::vector<void *> &buffers)
{
    auto &trt_params = trtParams();
    if (!trt_params.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    const auto &input_names  = modelConfig().inputTensorNames();
    const auto  output_names = primaryOutputTensorNames();
    const auto  expected     = input_names.size() + output_names.size();
    if (buffers.size() != expected)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Expected %zu buffers (%zu inputs and %zu outputs), got %zu", expected, input_names.size(),
                             output_names.size(), buffers.size());
    }

    size_t buffer_index = 0;
    for (const auto &input_name : input_names)
    {
        if (!trt_params.context->setTensorAddress(input_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set input tensor address: %s", input_name.c_str());
        }
    }

    for (const auto &output_name : output_names)
    {
        if (!trt_params.context->setTensorAddress(output_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set output tensor address: %s",
                                 output_name.c_str());
        }
    }
}

void IModelImpl::bindFeatureTensorAddresses(const std::vector<void *> &buffers)
{
    auto &trt_params = featureExecutionParams();
    if (!trt_params.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Feature extraction context is not initialized");
    }

    const auto &input_names   = modelConfig().inputTensorNames();
    const auto  output_names  = featureOutputTensorNames();
    const auto  expected      = input_names.size() + output_names.size();
    if (buffers.size() != expected)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Expected %zu buffers (%zu inputs and %zu feature outputs), got %zu", expected,
                             input_names.size(), output_names.size(), buffers.size());
    }

    size_t buffer_index = 0;
    for (const auto &input_name : input_names)
    {
        if (!trt_params.context->setTensorAddress(input_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set feature input tensor address: %s",
                                 input_name.c_str());
        }
    }

    for (const auto &output_name : output_names)
    {
        if (!trt_params.context->setTensorAddress(output_name.c_str(), buffers[buffer_index++]))
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "Failed to set feature output tensor address: %s",
                                 output_name.c_str());
        }
    }
}

void IModelImpl::initLogger()
{
    trt_params_.logger = std::make_shared<Logger>(name(), logLevel());
    feature_trt_params_.logger = trt_params_.logger;
}

std::vector<std::string> IModelImpl::primaryOutputTensorNames() const
{
    return modelConfig().outputTensorNames();
}

std::vector<std::string> IModelImpl::featureOutputTensorNames() const
{
    const auto &feature_output_names = modelConfig().featureOutputTensorNames();
    if (!feature_output_names.empty())
    {
        return feature_output_names;
    }

    return modelConfig().featureTensorNames();
}

std::vector<nvinfer1::ITensor *> IModelImpl::resolveFeatureTensors(const NamedTensorMap &named_tensors) const
{
    std::vector<nvinfer1::ITensor *> outputs;
    const auto &feature_tensor_names = modelConfig().featureTensorNames();
    outputs.reserve(feature_tensor_names.size());

    for (const auto &feature_tensor_name : feature_tensor_names)
    {
        const auto it = named_tensors.find(feature_tensor_name);
        if (it == named_tensors.end())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Requested feature tensor is not available: %s",
                                 feature_tensor_name.c_str());
        }
        if (it->second == nullptr)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Requested feature tensor is null: %s",
                                 feature_tensor_name.c_str());
        }
        outputs.push_back(it->second);
    }

    return outputs;
}

void IModelImpl::forwardFeatures(const std::vector<void *> &buffers)
{
    ensureFeatureExtractionReady();
    auto &trt_params = featureExecutionParams();
    bindFeatureTensorAddresses(buffers);

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
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to execute feature extraction");
    }

    cudaStreamSynchronize(*trt_params.stream);
}

void IModelImpl::buildRuntimeFromWeights(const std::string &weights_file, const WeightsMap &,
                                         const std::function<void(nvinfer1::INetworkDefinition *)> &build_fn,
                                         TRTParams &params)
{
    using namespace nvinfer1;

    if (params.logger == nullptr)
    {
        if (trt_params_.logger)
        {
            params.logger = trt_params_.logger;
        }
        else
        {
            params.logger = std::make_shared<Logger>(name(), logLevel());
        }
    }

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*params.logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    const auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network     = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create NetworkDefinition");
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create BuilderConfig");
    }

    LOG_INFO(*params.logger) << "Building TensorRT network from: " << weights_file << std::endl;
    build_fn(network.get());

    auto buffer = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!buffer)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to build serialized network");
    }

    auto runtime = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(*params.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    params.engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer->data(), buffer->size()),
                                                 [](ICudaEngine *engine) { delete engine; });
    if (!params.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    params.context.reset(params.engine->createExecutionContext());
    if (!params.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }
}

void IModelImpl::loadRuntimeFromFile(const std::string &engine_file, TRTParams &params)
{
    params.context.reset();
    params.engine.reset();

    if (params.logger == nullptr)
    {
        if (trt_params_.logger)
        {
            params.logger = trt_params_.logger;
        }
        else
        {
            params.logger = std::make_shared<Logger>(name(), logLevel());
        }
    }

    LOG_INFO(*params.logger) << "Loading TensorRT engine from: " << engine_file << std::endl;

    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open engine file: %s", engine_file.c_str());
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*params.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    params.engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), file_size),
                                                           [](nvinfer1::ICudaEngine *engine) { delete engine; });
    if (!params.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    params.context.reset(params.engine->createExecutionContext());
    if (!params.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }
}

void IModelImpl::saveRuntimeToFile(const std::string &engine_file, const TRTParams &params)
{
    if (!params.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized, cannot save");
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(params.engine->serialize());
    if (!serialized)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to serialize engine");
    }

    std::ofstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open file for writing: %s",
                             engine_file.c_str());
    }

    file.write(static_cast<const char *>(serialized->data()), serialized->size());
    file.close();
}

std::string IModelImpl::featureEngineFileName(const std::string &base_engine_file) const
{
    std::string engine_file = base_engine_file;
    const auto  ext_pos     = engine_file.rfind(engineExtension());
    const auto  suffix      = BuildFeatureEngineSuffix(*this);

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

void IModelImpl::ensurePrimaryInferenceReady() const
{
    if (!trt_params_.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Inference engine is not initialized");
    }
    if (trt_params_.feature_only)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION,
                             "Current engine is feature-only; use forwardFeatures instead of infer");
    }
}

void IModelImpl::ensureFeatureExtractionReady() const
{
    const auto &params = featureExecutionParams();
    if (!params.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Feature extraction engine is not initialized");
    }
    if (!params.feature_only)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION,
                             "Current engine is not a feature-only network; feature extraction is unavailable");
    }
}

void IModelImpl::build(const std::string &weights_file)
{
    ValidateModelConfig(*this);
    feature_trt_params_.context.reset();
    feature_trt_params_.engine.reset();
    feature_trt_params_.stream.reset();
    feature_trt_params_.feature_only = false;

    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }

    LOG_INFO(*trt_params_.logger) << "Loading weights file: " << weights_file << std::endl;
    auto weights_map = loadWeights(weights_file);
    build_variant_ = isFeatureOnlyConfig() ? BuildVariant::Feature : BuildVariant::Primary;
    buildRuntimeFromWeights(weights_file, weights_map, [&](nvinfer1::INetworkDefinition *network) {
        buildNetwork(network, weights_map);
    }, trt_params_);
    trt_params_.feature_only = (build_variant_ == BuildVariant::Feature);

    if (!isFeatureOnlyConfig() && !modelConfig().featureTensorNames().empty())
    {
        build_variant_ = BuildVariant::Feature;
        buildRuntimeFromWeights(weights_file, weights_map, [&](nvinfer1::INetworkDefinition *network) {
            buildNetwork(network, weights_map);
        }, feature_trt_params_);
        feature_trt_params_.feature_only = true;
    }
    build_variant_ = BuildVariant::Primary;

    for (auto &wt : weights_map)
    {
        delete[] static_cast<const uint32_t *>(wt.second.values);
    }
}

/**
 * @brief 将当前 engine 序列化保存到磁盘。
 * @param engine_file 输出文件路径。
 */
void IModelImpl::save(const std::string &engine_file)
{
    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }
    LOG_INFO(*trt_params_.logger) << "Saving TensorRT engine to: " << engine_file << std::endl;
    saveRuntimeToFile(engine_file, trt_params_);

    if (!isFeatureOnlyConfig() && feature_trt_params_.engine)
    {
        saveRuntimeToFile(featureEngineFileName(engine_file), feature_trt_params_);
    }
}

/**
 * @brief 从磁盘加载并反序列化 engine。
 * @param engine_file engine 文件路径。
 */
void IModelImpl::load(const std::string &engine_file)
{
    feature_trt_params_.context.reset();
    feature_trt_params_.engine.reset();
    feature_trt_params_.stream.reset();
    feature_trt_params_.feature_only = false;
    loadRuntimeFromFile(engine_file, trt_params_);
    trt_params_.feature_only = isFeatureOnlyConfig();

    if (!isFeatureOnlyConfig() && !modelConfig().featureTensorNames().empty())
    {
        const auto feature_engine_file = featureEngineFileName(engine_file);
        std::ifstream feature_file(feature_engine_file, std::ios::binary);
        if (feature_file.good())
        {
            feature_file.close();
            loadRuntimeFromFile(feature_engine_file, feature_trt_params_);
            feature_trt_params_.feature_only = true;
        }
    }
}

/**
 * @brief 优先加载已有 engine，失败时回退到重新构建。
 * @param weights_file 权重文件路径。
 */
void IModelImpl::buildOrLoad(const std::string &weights_file)
{
    ValidateModelConfig(*this);

    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }

    std::string engine_file = BuildEngineFileName(*this, weights_file);
    if (isFeatureOnlyConfig())
    {
        engine_file = featureEngineFileName(engine_file);
    }

    std::ifstream file(engine_file);
    bool          engine_exists = file.good();
    file.close();
    const bool need_feature_engine = !isFeatureOnlyConfig() && !modelConfig().featureTensorNames().empty();
    const auto feature_engine_file = featureEngineFileName(engine_file);
    bool       feature_engine_loaded = false;

    if (engine_exists)
    {
        LOG_INFO(*trt_params_.logger) << "Found existing engine file: " << engine_file << ", loading..." << std::endl;
        try
        {
            load(engine_file);
            feature_engine_loaded = !need_feature_engine || static_cast<bool>(feature_trt_params_.context);
            if (feature_engine_loaded)
            {
                return;
            }

            LOG_INFO(*trt_params_.logger) << "Feature engine is missing, will build it from weights file: "
                                          << weights_file << std::endl;
        }
        catch (const std::exception &e)
        {
            LOG_WARN(*trt_params_.logger) << "Failed to load engine: " << e.what() << std::endl;
            LOG_INFO(*trt_params_.logger) << "Will rebuild engine from weights file: " << weights_file << std::endl;
        }
    }

    LOG_INFO(*trt_params_.logger) << "Building engine from weights file: " << weights_file << std::endl;
    build(weights_file);

    try
    {
        save(engine_file);
    }
    catch (const std::exception &e)
    {
        LOG_WARN(*trt_params_.logger) << "Failed to save engine: " << e.what() << std::endl;
    }

    if (need_feature_engine)
    {
        std::ifstream feature_file(feature_engine_file);
        const bool feature_exists = feature_file.good();
        feature_file.close();

        if (feature_exists)
        {
            try
            {
                loadRuntimeFromFile(feature_engine_file, feature_trt_params_);
            }
            catch (const std::exception &e)
            {
                LOG_WARN(*trt_params_.logger) << "Failed to load feature engine: " << e.what() << std::endl;
            }
        }
    }
}

} // namespace irt::model::priv
