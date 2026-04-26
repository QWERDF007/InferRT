#include "IModelImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <fstream>

namespace irt::model::priv {

namespace {

void ValidateModelConfig(const IModelImpl &impl)
{
    const auto &config      = impl.modelConfig();
    const auto &input_shape = config.inputShape();

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
}

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

    const auto &config      = impl.modelConfig();
    const auto &input_shape = config.inputShape();
    const auto  ext_pos     = engine_file.rfind(impl.engineExtension());
    const auto  suffix = "." + std::to_string(input_shape.channels) + "x" + std::to_string(input_shape.height) + "x"
                       + std::to_string(input_shape.width) + ".cls" + std::to_string(config.numClasses());

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

void IModelImpl::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    trt_params_.log_level = severity;
    if (trt_params_.logger)
    {
        trt_params_.logger->setReportableSeverity(severity);
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
