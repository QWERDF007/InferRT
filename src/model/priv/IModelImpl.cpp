#include "IModelImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <fstream>

namespace irt::model::priv {

namespace {

/**
 * @brief 在构建或加载模型前验证配置合法性。
 * @param impl 模型内部实现对象。
 */
void ValidateModelConfig(const IModelImpl &impl)
{
    const auto &config             = impl.modelConfig();
    const auto &input_shape        = config.inputShape();
    const auto &input_tensor_names = config.inputTensorNames();
    const auto &output_tensor_names = config.outputTensorNames();

    if (config.numClasses() <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "num_classes must be positive, got %d",
                             config.numClasses());
    }

    if (input_shape.channels <= 0 || input_shape.height <= 0 || input_shape.width <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "input shape must be positive, got C=%d H=%d W=%d",
                             input_shape.channels, input_shape.height, input_shape.width);
    }

    if (input_tensor_names.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "at least one input tensor name is required");
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
    const auto  suffix  = impl.generateSuffix(config);

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

nvinfer1::ILogger::Severity IModelImpl::logLevel() const noexcept
{
    return trt_params_.log_level;
}

std::string IModelImpl::generateSuffix(const IModelConfig &config) const noexcept
{
    const auto &input_shape = config.inputShape();
    return "_" + std::to_string(input_shape.channels) + "x" + std::to_string(input_shape.height) + "x"
         + std::to_string(input_shape.width) + "_" + std::to_string(config.numClasses());
}

void IModelImpl::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    config_ = config ? std::move(config) : std::make_unique<IModelConfig>();
}

void IModelImpl::setNumClasses(int num_classes)
{
    config_->setNumClasses(num_classes);
}

void IModelImpl::setInputShape(const InputShape &shape)
{
    config_->setInputShape(shape);
}

void IModelImpl::setInputShape(int channels, int height, int width)
{
    config_->setInputShape(channels, height, width);
}

void IModelImpl::setInputTensorNames(std::vector<std::string> input_tensor_names)
{
    config_->setInputTensorNames(std::move(input_tensor_names));
}

void IModelImpl::setOutputTensorNames(std::vector<std::string> output_tensor_names)
{
    config_->setOutputTensorNames(std::move(output_tensor_names));
}

const IModelConfig &IModelImpl::modelConfig() const noexcept
{
    return *config_;
}

int IModelImpl::numClasses() const noexcept
{
    return config_->numClasses();
}

const InputShape &IModelImpl::inputShape() const noexcept
{
    return config_->inputShape();
}

const std::vector<std::string> &IModelImpl::inputTensorNames() const noexcept
{
    return config_->inputTensorNames();
}

const std::vector<std::string> &IModelImpl::outputTensorNames() const noexcept
{
    return config_->outputTensorNames();
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
    if (trt_params_.context)
    {
        return trt_params_.context->getTensorShape(tensor_name.c_str());
    }

    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    return trt_params_.engine->getTensorShape(tensor_name.c_str());
}

nvinfer1::DataType IModelImpl::tensorDataType(const std::string &tensor_name) const
{
    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized");
    }

    return trt_params_.engine->getTensorDataType(tensor_name.c_str());
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
}

void IModelImpl::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    trt_params_.log_level = severity;
    if (trt_params_.logger)
    {
        trt_params_.logger->setReportableSeverity(severity);
    }
}

nvinfer1::ITensor *IModelImpl::addInputTensor(nvinfer1::INetworkDefinition *network, const nvinfer1::Dims &dims,
                                              nvinfer1::DataType data_type, size_t input_index) const
{
    const auto &tensor_names = inputTensorNames();
    if (input_index >= tensor_names.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "input index %zu is out of range for %zu configured input tensor names", input_index,
                             tensor_names.size());
    }

    return network->addInput(tensor_names[input_index].c_str(), data_type, dims);
}

void IModelImpl::markOutputTensors(nvinfer1::INetworkDefinition *network,
                                   const std::vector<nvinfer1::ITensor *> &outputs) const
{
    const auto &tensor_names = outputTensorNames();
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

void IModelImpl::bindTensorAddresses(const std::vector<void *> &buffers)
{
    auto &trt_params = trtParams();
    if (!trt_params.context)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Execution context is not initialized");
    }

    const auto &input_names  = inputTensorNames();
    const auto &output_names = outputTensorNames();
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

void IModelImpl::initLogger()
{
    trt_params_.logger = std::make_unique<Logger>(name(), logLevel());
}

void IModelImpl::build(const std::string &weights_file)
{
    using namespace nvinfer1;

    ValidateModelConfig(*this);

    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }

    LOG_INFO(*trt_params_.logger) << "Loading weights file: " << weights_file << std::endl;
    auto weights_map = loadWeights(weights_file);

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*trt_params_.logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    const auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create NetworkDefinition");
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create BuilderConfig");
    }

    LOG_INFO(*trt_params_.logger) << "Building TensorRT network..." << std::endl;

    buildNetwork(network.get(), weights_map);

    LOG_INFO(*trt_params_.logger) << "Building TensorRT engine..." << std::endl;
    auto buffer = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!buffer)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to build serialized network");
    }

    auto runtime = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(*trt_params_.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    trt_params_.engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buffer->data(), buffer->size()),
                                                      [](ICudaEngine *engine) { delete engine; });
    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    trt_params_.context.reset(trt_params_.engine->createExecutionContext());
    if (!trt_params_.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    LOG_INFO(*trt_params_.logger) << "TensorRT engine built successfully" << std::endl;

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
    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Engine is not initialized, cannot save");
    }

    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }
    LOG_INFO(*trt_params_.logger) << "Saving TensorRT engine to: " << engine_file << std::endl;

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(trt_params_.engine->serialize());
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

    LOG_INFO(*trt_params_.logger) << "Engine saved successfully, size: " << serialized->size() / (1024.0 * 1024.0)
                                  << " MiB" << std::endl;
}

/**
 * @brief 从磁盘加载并反序列化 engine。
 * @param engine_file engine 文件路径。
 */
void IModelImpl::load(const std::string &engine_file)
{
    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }
    LOG_INFO(*trt_params_.logger) << "Loading TensorRT engine from: " << engine_file << std::endl;

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

    LOG_INFO(*trt_params_.logger) << "Engine file loaded, size: " << file_size / (1024.0 * 1024.0) << " MiB"
                                  << std::endl;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*trt_params_.logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    trt_params_.engine
        = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), file_size),
                                                 [](nvinfer1::ICudaEngine *engine) { delete engine; });
    if (!trt_params_.engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }

    trt_params_.context.reset(trt_params_.engine->createExecutionContext());
    if (!trt_params_.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    LOG_INFO(*trt_params_.logger) << "TensorRT engine loaded successfully" << std::endl;
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

    std::ifstream file(engine_file);
    bool          engine_exists = file.good();
    file.close();

    if (engine_exists)
    {
        LOG_INFO(*trt_params_.logger) << "Found existing engine file: " << engine_file << ", loading..." << std::endl;
        try
        {
            load(engine_file);
            return;
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
}

} // namespace irt::model::priv
