#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>

namespace irt::model {

namespace {

using ModelRegistry = std::map<std::string, ModelCreator>;

ModelRegistry &GetModelRegistry()
{
    static ModelRegistry registry;
    return registry;
}

} // namespace

ModelRegistrar::ModelRegistrar(const std::string &name, ModelCreator creator)
{
    RegisterModel(name, creator);
}

bool RegisterModel(const std::string &name, ModelCreator creator)
{
    return GetModelRegistry().emplace(name, creator).second;
}

std::unique_ptr<IModel> CreateModel(const std::string &name)
{
    std::string normalized_name = name;
    std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    const auto it = GetModelRegistry().find(normalized_name);
    return it == GetModelRegistry().end() ? nullptr : it->second();
}

void IModel::build(const std::string &weights_file)
{
    using namespace nvinfer1;

    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }

    LOG_INFO(*trt_params_.logger) << "Loading weights file: " << weights_file << std::endl;
    // 加载权重
    auto weights_map = loadWeights(weights_file);

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*trt_params_.logger));
    if (!builder)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferBuilder");
    }

    // kEXPLICIT_BATCH 在 TensorRT 10.0 被废弃
    // NetworkDefinitionCreationFlags flags = (1 << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    //                                      | (1 << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));

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

    // 释放权重内存
    for (auto &wt : weights_map)
    {
        delete[] static_cast<const uint32_t *>(wt.second.values);
    }
}

void IModel::save(const std::string &engine_file)
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

    // 序列化引擎
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(trt_params_.engine->serialize());
    if (!serialized)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to serialize engine");
    }

    // 写入文件
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

void IModel::load(const std::string &engine_file)
{
    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }
    LOG_INFO(*trt_params_.logger) << "Loading TensorRT engine from: " << engine_file << std::endl;

    // 读取引擎文件
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open engine file: %s", engine_file.c_str());
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取文件内容
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    LOG_INFO(*trt_params_.logger) << "Engine file loaded, size: " << file_size / (1024.0 * 1024.0) << " MiB"
                                  << std::endl;

    // 创建运行时并反序列化引擎
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

    // 创建执行上下文
    trt_params_.context.reset(trt_params_.engine->createExecutionContext());
    if (!trt_params_.context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }

    LOG_INFO(*trt_params_.logger) << "TensorRT engine loaded successfully" << std::endl;
}

void IModel::buildOrLoad(const std::string &weights_file)
{
    if (trt_params_.logger == nullptr)
    {
        initLogger();
    }

    // 生成引擎文件名
    std::string engine_file = weights_file;
    size_t      pos         = engine_file.rfind(wtsExtension());
    if (pos != std::string::npos)
    {
        engine_file.replace(pos, 4, engineExtension());
    }
    else
    {
        engine_file += engineExtension();
    }

    // 检查引擎文件是否存在
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

    // 引擎文件不存在或加载失败，从权重构建
    LOG_INFO(*trt_params_.logger) << "Building engine from weights file: " << weights_file << std::endl;
    build(weights_file);

    // 保存引擎以便下次使用
    try
    {
        save(engine_file);
    }
    catch (const std::exception &e)
    {
        LOG_WARN(*trt_params_.logger) << "Failed to save engine: " << e.what() << std::endl;
    }
}

} // namespace irt::model
