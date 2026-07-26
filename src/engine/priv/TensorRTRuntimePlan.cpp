#include "TensorRTRuntimePlan.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace irt::engine::priv {
namespace {

std::vector<char> readEngineFile(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open TensorRT engine: %s",
                             path.string().c_str());
    }

    const auto size = file.tellg();
    if (size <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "TensorRT engine is empty: %s",
                             path.string().c_str());
    }
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(size)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read TensorRT engine: %s",
                             path.string().c_str());
    }
    return bytes;
}

bool contains(const std::vector<std::string> &names, const std::string &name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

} // namespace

TensorRTRuntimePlan::TensorRTRuntimePlan(const EngineConfig &config, const int device_id)
    : logger_(config.model_name, nvinfer1::ILogger::Severity::kWARNING)
{
    irt::model::setCudaDevice(device_id);
    const auto serialized = readEngineFile(config.engine_file);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to create TensorRT runtime");
    }
    engine_
        = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()),
                                                 [](nvinfer1::ICudaEngine *engine) { delete engine; });
    if (!engine_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to deserialize TensorRT engine: %s",
                             config.engine_file.string().c_str());
    }

    for (int32_t index = 0; index < engine_->getNbIOTensors(); ++index)
    {
        const char *name = engine_->getIOTensorName(index);
        if (name == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "TensorRT engine has an unnamed I/O tensor");
        }
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input_names_.emplace_back(name);
        }
        else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output_names_.emplace_back(name);
        }
    }
    if (input_names_.empty() || output_names_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT engine must expose at least one input and one output");
    }
    for (const auto &input_name : input_names_)
    {
        if (tensorDataType(input_name) != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Engine input %s must be float32",
                                 input_name.c_str());
        }
        const auto input_shape = engine_->getTensorShape(input_name.c_str());
        if (input_shape.nbDims != 4)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Engine input %s must be NCHW",
                                 input_name.c_str());
        }
        if (input_shape.d[0] > 0)
        {
            if (fixed_batch_size_ != 0 && fixed_batch_size_ != input_shape.d[0])
            {
                throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                     "TensorRT inputs must use the same fixed batch size");
            }
            fixed_batch_size_ = input_shape.d[0];
        }
    }
    for (const auto &name : output_names_)
    {
        if (tensorDataType(name) != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Engine output %s must be float32", name.c_str());
        }
    }
    for (const auto &name : config.output_tensor_names)
    {
        if (!contains(output_names_, name))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured output tensor does not exist in TensorRT engine: %s", name.c_str());
        }
    }
}

std::unique_ptr<nvinfer1::IExecutionContext> TensorRTRuntimePlan::createSession() const
{
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to create TensorRT execution context");
    }
    return context;
}

void TensorRTRuntimePlan::setInputShape(nvinfer1::IExecutionContext &context, const std::string &input_name,
                                        const nvinfer1::Dims4 &shape) const
{
    if (!contains(input_names_, input_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "TensorRT input does not exist: %s",
                             input_name.c_str());
    }
    if (!context.setInputShape(input_name.c_str(), shape))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT input shape is outside the engine profile: %s", input_name.c_str());
    }
}

void TensorRTRuntimePlan::enqueue(nvinfer1::IExecutionContext                       &context,
                                  const std::vector<std::pair<std::string, void *>> &inputs,
                                  const std::vector<void *> &outputs, const cudaStream_t stream) const
{
    if (inputs.size() != input_names_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "TensorRT input binding count does not match engine");
    }
    if (outputs.size() != output_names_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "TensorRT output binding count does not match engine");
    }
    for (const auto &[name, pointer] : inputs)
    {
        if (!contains(input_names_, name) || pointer == nullptr || !context.setTensorAddress(name.c_str(), pointer))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to bind TensorRT input: %s", name.c_str());
        }
    }
    for (size_t index = 0; index < output_names_.size(); ++index)
    {
        if (!context.setTensorAddress(output_names_[index].c_str(), outputs[index]))
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Failed to bind TensorRT output: %s",
                                 output_names_[index].c_str());
        }
    }
    if (!context.enqueueV3(stream))
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "TensorRT enqueueV3 failed");
    }
}

const std::vector<std::string> &TensorRTRuntimePlan::inputNames() const noexcept
{
    return input_names_;
}

const std::vector<std::string> &TensorRTRuntimePlan::outputNames() const noexcept
{
    return output_names_;
}

nvinfer1::DataType TensorRTRuntimePlan::tensorDataType(const std::string &name) const
{
    return engine_->getTensorDataType(name.c_str());
}

nvinfer1::Dims TensorRTRuntimePlan::tensorShape(const nvinfer1::IExecutionContext &context,
                                                const std::string                 &name) const
{
    return context.getTensorShape(name.c_str());
}

int TensorRTRuntimePlan::fixedBatchSize() const noexcept
{
    return fixed_batch_size_;
}

} // namespace irt::engine::priv
