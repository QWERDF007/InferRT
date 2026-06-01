#include "IModelImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <cctype>
#include <fstream>
#include <functional>
#include <unordered_set>

namespace irt::model::priv {

namespace {

/**
 * @brief 校验模型的类别数量和输入尺寸配置是否合法。
 * @param config 待校验的模型配置。
 */
void ValidatePositiveModelDimensions(const IModelConfig &config)
{
    if (config.numClasses() <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "num_classes must be positive, got %d",
                             config.numClasses());
    }

    const auto &input_shapes = config.inputShapes();
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
}

/**
 * @brief 校验输入张量名称列表是否合法，并确认其数量与输入尺寸一一对应。
 * @param config 待校验的模型配置。
 */
void ValidateInputTensorConfig(const IModelConfig &config)
{
    const auto &input_shapes       = config.inputShapes();
    const auto &input_tensor_names = config.inputTensorNames();

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

    for (const auto &input_tensor_name : input_tensor_names)
    {
        if (input_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input tensor name must not be empty");
        }
    }
}

/**
 * @brief 校验主输出张量名称列表是否合法。
 * @param config 待校验的模型配置。
 */
void ValidatePrimaryOutputTensorConfig(const IModelConfig &config)
{
    const auto &output_tensor_names = config.outputTensorNames();

    if (output_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "at least one output tensor name is required");
    }

    for (const auto &output_tensor_name : output_tensor_names)
    {
        if (output_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "output tensor name must not be empty");
        }
    }
}

/**
 * @brief 校验特征提取相关配置是否合法。
 *
 * 包括特征张量名称、导出名称数量匹配关系，以及 featureOnly 模式下的必填项约束。
 *
 * @param config 待校验的模型配置。
 */
void ValidateFeatureTensorConfig(const IModelConfig &config)
{
    const auto &feature_tensor_names = config.featureTensorNames();
    const auto &output_tensor_names  = config.outputTensorNames();

    for (const auto &feature_tensor_name : feature_tensor_names)
    {
        if (feature_tensor_name.empty())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "feature tensor name must not be empty");
        }
    }

    if (!config.featureOnly() && !feature_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "featureTensorNames requires featureOnly to be enabled");
    }

    if (!config.featureOnly())
    {
        return;
    }

    if (feature_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "featureTensorNames must not be empty when featureOnly is enabled");
    }

    if (output_tensor_names.size() != feature_tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "output tensor name count (%zu) must match feature tensor count (%zu) when featureOnly "
                             "is enabled",
                             output_tensor_names.size(), feature_tensor_names.size());
    }
}

/**
 * @brief 校验动态 batch profile 配置是否可用于当前模型。
 * @param impl 模型内部实现对象。
 */
void ValidateDynamicBatchConfig(const IModelImpl &impl)
{
    const auto &config = impl.modelConfig();
    if (config.backend() != ModelBackend::TensorRT || !config.dynamicBatch())
    {
        return;
    }

    if (!impl.supportsDynamicBatch())
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "%s TensorRT network does not support dynamic batch yet",
                             impl.name().c_str());
    }

    if (config.minBatchSize() <= 0 || config.optBatchSize() <= 0 || config.maxBatchSize() <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "dynamic batch range must be positive, got min=%d opt=%d max=%d", config.minBatchSize(),
                             config.optBatchSize(), config.maxBatchSize());
    }

    if (config.minBatchSize() > config.optBatchSize() || config.optBatchSize() > config.maxBatchSize())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "dynamic batch range must satisfy min <= opt <= max, got min=%d opt=%d max=%d",
                             config.minBatchSize(), config.optBatchSize(), config.maxBatchSize());
    }

    for (size_t i = 0; i < config.inputShapes().size(); ++i)
    {
        const auto &shape = config.inputShapes()[i];
        if (shape.d[0] < config.minBatchSize() || shape.d[0] > config.maxBatchSize())
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "input shape batch at index %zu must be within dynamic batch range [%d, %d], got %lld",
                                 i, config.minBatchSize(), config.maxBatchSize(), static_cast<long long>(shape.d[0]));
        }
    }
}

/**
 * @brief 校验主输出张量名称不存在重复项。
 * @param config 待校验的模型配置。
 */
void ValidateUniqueOutputTensorNames(const IModelConfig &config)
{
    std::unordered_set<std::string> unique_output_names;
    for (const auto &output_tensor_name : config.outputTensorNames())
    {
        if (!unique_output_names.insert(output_tensor_name).second)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "duplicate output tensor name: %s",
                                 output_tensor_name.c_str());
        }
    }
}

/**
 * @brief 在构建或加载模型前验证配置合法性。
 * @param impl 模型内部实现对象。
 */
void ValidateModelConfig(const IModelImpl &impl)
{
    const auto &config = impl.modelConfig();

    ValidatePositiveModelDimensions(config);
    ValidateInputTensorConfig(config);
    ValidatePrimaryOutputTensorConfig(config);
    ValidateFeatureTensorConfig(config);
    ValidateUniqueOutputTensorNames(config);
    ValidateDynamicBatchConfig(impl);

    if (config.backend() == ModelBackend::TensorRT && config.device() == ModelDevice::CPU)
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "TensorRT backend requires GPU device");
    }
}

/**
 * @brief 根据权重文件路径和模型基础配置生成主 engine 文件路径。
 * @param impl 模型内部实现对象。
 * @param weights_file 权重文件路径。
 * @return 对应的 engine 文件路径。
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

    const auto &config       = impl.modelConfig();
    const auto  ext_pos      = engine_file.rfind(impl.engineExtension());
    auto        cache_config = IModelConfig{};
    cache_config.setNumClasses(config.numClasses());
    cache_config.setInputShapes(config.inputShapes());
    cache_config.setInputTensorNames(config.inputTensorNames());
    cache_config.setOutputTensorNames(config.outputTensorNames());
    if (config.dynamicBatch())
    {
        cache_config.setDynamicBatchRange(config.minBatchSize(), config.optBatchSize(), config.maxBatchSize());
    }
    if (config.featureOnly())
    {
        cache_config.setFeatureTensorNames(config.featureTensorNames());
        cache_config.setFeatureOnly(true);
    }
    const auto suffix = impl.generateSuffix(cache_config);

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

} // namespace

IModelImpl::~IModelImpl() = default;

nvinfer1::ILogger::Severity IModelImpl::logLevel() const noexcept
{
    return backend_runtime_ ? backend_runtime_->logLevel() : nvinfer1::ILogger::Severity::kWARNING;
}

std::string IModelImpl::generateSuffix(const IModelConfig &config) const noexcept
{
    auto sanitize = [](const std::string &value)
    {
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
    if (config.dynamicBatch())
    {
        suffix += "_dynb_" + std::to_string(config.minBatchSize()) + "x" + std::to_string(config.optBatchSize()) + "x"
                + std::to_string(config.maxBatchSize());
    }
    if (config.featureOnly())
    {
        for (const auto &output_tensor_name : config.outputTensorNames())
        {
            suffix += "_out_" + sanitize(output_tensor_name);
        }
    }
    return suffix;
}

void IModelImpl::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    const auto severity = logLevel();
    config_             = config ? std::move(config) : std::make_unique<IModelConfig>();
    normalizeModelConfig(*config_);
    backend_runtime_ = CreateBackendRuntime(config_->backend());
    backend_runtime_->setLogLevel(severity);
}

void IModelImpl::replaceModelConfigWithoutReset(std::unique_ptr<IModelConfig> config)
{
    config_ = config ? std::move(config) : std::make_unique<IModelConfig>();
    normalizeModelConfig(*config_);
}

const IModelConfig &IModelImpl::modelConfig() const noexcept
{
    return *config_;
}

TRTParams &IModelImpl::trtParams()
{
    return tensorRTBackend().params();
}

const TRTParams &IModelImpl::trtParams() const
{
    return tensorRTBackend().params();
}

std::vector<std::string> IModelImpl::ioTensorNames(nvinfer1::TensorIOMode mode) const
{
    if (!backend_runtime_)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
    }
    return backend_runtime_->ioTensorNames(mode);
}

nvinfer1::Dims IModelImpl::tensorShape(const std::string &tensor_name) const
{
    if (!backend_runtime_)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
    }
    return backend_runtime_->tensorShape(tensor_name);
}

nvinfer1::DataType IModelImpl::tensorDataType(const std::string &tensor_name) const
{
    if (!backend_runtime_)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
    }
    return backend_runtime_->tensorDataType(tensor_name);
}

void IModelImpl::setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims)
{
    if (!backend_runtime_)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
    }
    backend_runtime_->setTensorShape(tensor_name, dims);
}

void IModelImpl::setStream(cudaStream_t stream)
{
    backend_runtime_->setStream(stream);
}

void IModelImpl::clearStream()
{
    backend_runtime_->clearStream();
}

void IModelImpl::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    backend_runtime_->setLogLevel(severity);
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

    auto network_dims = dims;
    if (modelConfig().dynamicBatch() && network_dims.nbDims > 0)
    {
        network_dims.d[0] = -1;
    }

    return network->addInput(tensor_names[input_index].c_str(), data_type, network_dims);
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

void IModelImpl::markOutputTensors(nvinfer1::INetworkDefinition           *network,
                                   const std::vector<nvinfer1::ITensor *> &outputs) const
{
    const auto &tensor_names = modelConfig().outputTensorNames();
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

void IModelImpl::markFeatureOutputTensors(nvinfer1::INetworkDefinition *network,
                                          const NamedTensorMap         &named_tensors) const
{
    const auto  feature_outputs = resolveFeatureTensors(named_tensors);
    const auto &tensor_names    = modelConfig().outputTensorNames();
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

bool IModelImpl::tryMarkFeatureOutputTensors(nvinfer1::INetworkDefinition *network,
                                             const NamedTensorMap         &named_tensors) const
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

void IModelImpl::initLogger()
{
    tensorRTBackend().initLogger(name());
}

std::vector<nvinfer1::ITensor *> IModelImpl::resolveFeatureTensors(const NamedTensorMap &named_tensors) const
{
    std::vector<nvinfer1::ITensor *> outputs;
    const auto                      &feature_tensor_names = modelConfig().featureTensorNames();
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

void IModelImpl::infer(const std::vector<void *> &buffers, cudaStream_t stream, bool non_blocking)
{
    if (!usesTensorRTBackend())
    {
        if (!backend_runtime_)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
        }
        (void)stream;
        (void)non_blocking;
        backend_runtime_->infer(buffers);
        return;
    }

    execute(buffers, stream, non_blocking);
}

void IModelImpl::forwardFeatures(const std::vector<void *> &buffers, cudaStream_t stream, bool non_blocking)
{
    if (!usesTensorRTBackend())
    {
        if (!backend_runtime_)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Backend runtime is not initialized");
        }
        (void)stream;
        (void)non_blocking;
        backend_runtime_->infer(buffers);
        return;
    }

    execute(buffers, stream, non_blocking);
}

void IModelImpl::buildRuntimeFromWeights(const std::string                                         &weights_file,
                                         const std::function<void(nvinfer1::INetworkDefinition *)> &build_fn)
{
    tensorRTBackend().buildFromNetwork(weights_file, name(), modelConfig(), build_fn);
}

void IModelImpl::loadRuntimeFromFile(const std::string &engine_file)
{
    tensorRTBackend().load(engine_file, modelConfig(), name());
}

void IModelImpl::saveRuntimeToFile(const std::string &engine_file) const
{
    tensorRTBackend().save(engine_file);
}

bool IModelImpl::usesTensorRTBackend() const noexcept
{
    return modelConfig().backend() == ModelBackend::TensorRT;
}

TensorRTBackend &IModelImpl::tensorRTBackend()
{
    auto *backend = backend_runtime_ ? backend_runtime_->asTensorRT() : nullptr;
    if (!backend)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "TensorRT backend runtime is not initialized");
    }
    return *backend;
}

const TensorRTBackend &IModelImpl::tensorRTBackend() const
{
    auto *backend = backend_runtime_ ? backend_runtime_->asTensorRT() : nullptr;
    if (!backend)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "TensorRT backend runtime is not initialized");
    }
    return *backend;
}

void IModelImpl::ensureBackendRuntime()
{
    if (backend_runtime_ && backend_runtime_->backend() == modelConfig().backend())
    {
        return;
    }

    backend_runtime_ = CreateBackendRuntime(modelConfig().backend());
}

void IModelImpl::syncModelConfigFromBackendRuntime()
{
    if (!backend_runtime_)
    {
        return;
    }

    const auto input_names  = backend_runtime_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    const auto output_names = backend_runtime_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
    config_->setInputTensorNames(input_names);
    config_->setOutputTensorNames(output_names);

    std::vector<nvinfer1::Dims4> input_shapes;
    input_shapes.reserve(input_names.size());
    bool all_inputs_are_4d = !input_names.empty();
    for (const auto &input_name : input_names)
    {
        const auto dims = backend_runtime_->tensorShape(input_name);
        if (dims.nbDims != 4)
        {
            all_inputs_are_4d = false;
            break;
        }
        input_shapes.emplace_back(dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
    }
    if (all_inputs_are_4d)
    {
        config_->setInputShapes(std::move(input_shapes));
    }
}

void IModelImpl::buildBackendRuntimeFromFile(const std::string &model_file)
{
    ValidateModelConfig(*this);
    ensureBackendRuntime();
    backend_runtime_->load(model_file, modelConfig(), name());
    syncModelConfigFromBackendRuntime();
}

cudaStream_t IModelImpl::resolveExecutionStream(cudaStream_t stream_override)
{
    return backend_runtime_->resolveExecutionStream(stream_override);
}

void IModelImpl::execute(const std::vector<void *> &buffers, cudaStream_t stream_override, bool non_blocking)
{
    tensorRTBackend().execute(buffers, stream_override, non_blocking);
}

void IModelImpl::build(const std::string &weights_file)
{
    if (!usesTensorRTBackend())
    {
        buildBackendRuntimeFromFile(weights_file);
        return;
    }

    ValidateModelConfig(*this);

    auto &trt_params = trtParams();
    if (trt_params.logger == nullptr)
    {
        initLogger();
    }

    LOG_INFO(*trt_params.logger) << "Loading weights file: " << weights_file << std::endl;
    auto weights_map = loadWeights(weights_file);
    build_variant_   = isFeatureOnlyConfig() ? BuildVariant::Feature : BuildVariant::Primary;
    buildRuntimeFromWeights(weights_file,
                            [&](nvinfer1::INetworkDefinition *network) { buildNetwork(network, weights_map); });
    tensorRTBackend().setFeatureOnly(build_variant_ == BuildVariant::Feature);
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
    if (!usesTensorRTBackend())
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "save() is only supported for TensorRT serialized engines");
    }

    auto &trt_params = trtParams();
    if (trt_params.logger == nullptr)
    {
        initLogger();
    }
    LOG_INFO(*trt_params.logger) << "Saving TensorRT engine to: " << engine_file << std::endl;
    saveRuntimeToFile(engine_file);
}

/**
 * @brief 从磁盘加载并反序列化 engine。
 * @param engine_file engine 文件路径。
 */
void IModelImpl::load(const std::string &engine_file)
{
    if (!usesTensorRTBackend())
    {
        buildBackendRuntimeFromFile(engine_file);
        return;
    }

    loadRuntimeFromFile(engine_file);
    tensorRTBackend().setFeatureOnly(isFeatureOnlyConfig());
}

/**
 * @brief 优先加载已有 engine，失败时回退到重新构建。
 * @param weights_file 权重文件路径。
 */
void IModelImpl::buildOrLoad(const std::string &weights_file)
{
    if (!usesTensorRTBackend())
    {
        buildBackendRuntimeFromFile(weights_file);
        return;
    }

    ValidateModelConfig(*this);

    auto &trt_params = trtParams();
    if (trt_params.logger == nullptr)
    {
        initLogger();
    }

    const std::string engine_file = BuildEngineFileName(*this, weights_file);

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
            LOG_INFO(*trt_params.logger) << "Will rebuild engine from weights file: " << weights_file << std::endl;
        }
    }

    LOG_INFO(*trt_params.logger) << "Building engine from weights file: " << weights_file << std::endl;
    build(weights_file);

    try
    {
        save(engine_file);
    }
    catch (const std::exception &e)
    {
        LOG_WARN(*trt_params.logger) << "Failed to save engine: " << e.what() << std::endl;
    }
}

} // namespace irt::model::priv
