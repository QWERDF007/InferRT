#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/model/ModelRuntime.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace irt::engine {
namespace {

std::array<float, 3> readTriplet(const YAML::Node &node, const char *name, const std::array<float, 3> &fallback)
{
    if (!node)
    {
        return fallback;
    }
    if (!node.IsSequence() || node.size() != fallback.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must contain exactly three values", name);
    }

    std::array<float, 3> values{};
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = node[index].as<float>();
    }
    return values;
}

const YAML::Node requireMap(const YAML::Node &root, const char *name)
{
    const auto node = root[name];
    if (!node || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Missing mapping: %s", name);
    }
    return node;
}

PreprocessBackend readPreprocessBackend(const YAML::Node &node)
{
    if (!node)
    {
        return PreprocessBackend::CPU;
    }

    const auto backend = node.as<std::string>();
    if (backend == "cpu")
    {
        return PreprocessBackend::CPU;
    }
    if (backend == "cuda")
    {
        return PreprocessBackend::CUDA;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess.backend must be cpu or cuda, got: %s",
                         backend.c_str());
}

QueuePolicy readQueuePolicy(const YAML::Node &node)
{
    if (!node)
    {
        return QueuePolicy::Reject;
    }

    const auto policy = node.as<std::string>();
    if (policy == "reject")
    {
        return QueuePolicy::Reject;
    }
    if (policy == "block")
    {
        return QueuePolicy::Block;
    }
    if (policy == "drop_oldest")
    {
        return QueuePolicy::DropOldest;
    }
    if (policy == "drop_newest")
    {
        return QueuePolicy::DropNewest;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "runtime.queue_policy must be reject, block, drop_oldest, or drop_newest, got: %s",
                         policy.c_str());
}

StaticBatchPolicy readStaticBatchPolicy(const YAML::Node &node)
{
    if (!node)
    {
        return StaticBatchPolicy::Pad;
    }

    const auto policy = node.as<std::string>();
    if (policy == "reject")
    {
        return StaticBatchPolicy::Reject;
    }
    if (policy == "pad")
    {
        return StaticBatchPolicy::Pad;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "runtime.static_batch_policy must be reject or pad, got: %s", policy.c_str());
}

} // namespace

EngineConfig EngineConfig::load(const std::filesystem::path &path)
{
    if (path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Engine config path is empty");
    }

    const YAML::Node root = YAML::LoadFile(path.string());
    if (!root.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Engine config root must be a mapping");
    }

    EngineConfig config;
    const auto   model = requireMap(root, "model");
    const auto   input = requireMap(root, "input");

    if (!model["name"] || !model["engine"])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "model.name and model.engine are required");
    }
    config.model_name  = model["name"].as<std::string>();
    config.engine_file = model["engine"].as<std::string>();
    if (model["output_tensors"])
    {
        if (!model["output_tensors"].IsSequence())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "model.output_tensors must be a sequence");
        }
        config.output_tensor_names = model["output_tensors"].as<std::vector<std::string>>();
    }
    if (model["feature_tensors"])
    {
        if (!model["feature_tensors"].IsSequence())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "model.feature_tensors must be a sequence");
        }
        config.feature_tensor_names = model["feature_tensors"].as<std::vector<std::string>>();
    }
    config.feature_only = model["feature_only"].as<bool>(config.feature_only);

    if (model["runtime"])
    {
        const auto runtime = irt::model::ModelRuntime::parse(model["runtime"].as<std::string>());
        if (!runtime.isGpu() || runtime.backend() != irt::model::ModelRuntime::Backend::TensorRT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "engine supports only a TensorRT GPU runtime");
        }
        config.device_id = runtime.deviceId();
    }

    config.input_width  = input["width"].as<int>();
    config.input_height = input["height"].as<int>();
    if (input["channels"])
    {
        config.input_channels = input["channels"].as<int>();
    }
    config.mean   = readTriplet(input["mean"], "input.mean", config.mean);
    config.stddev = readTriplet(input["std"], "input.std", config.stddev);

    if (root["preprocess"])
    {
        const auto preprocess = root["preprocess"];
        if (!preprocess.IsMap())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess must be a mapping");
        }
        config.preprocess_backend = readPreprocessBackend(preprocess["backend"]);
        config.letterbox          = preprocess["letterbox"].as<bool>(config.letterbox);
        config.source_width       = preprocess["source_width"].as<int>(config.source_width);
        config.source_height      = preprocess["source_height"].as<int>(config.source_height);
    }

    const auto batch      = requireMap(root, "batch");
    config.min_batch_size = batch["min"].as<int>(1);
    config.opt_batch_size = batch["opt"].as<int>(config.min_batch_size);
    config.max_batch_size = batch["max"].as<int>(config.opt_batch_size);
    config.max_wait       = std::chrono::microseconds(batch["max_wait_us"].as<int64_t>(0));

    if (root["runtime"])
    {
        const auto runtime             = root["runtime"];
        config.execution_slots         = runtime["execution_slots"].as<size_t>(config.execution_slots);
        config.queue_capacity          = runtime["queue_capacity"].as<size_t>(config.queue_capacity);
        config.queue_policy            = readQueuePolicy(runtime["queue_policy"]);
        config.cpu_preprocess_workers  = runtime["cpu_preprocess_workers"].as<size_t>(config.cpu_preprocess_workers);
        config.cpu_postprocess_workers = runtime["cpu_postprocess_workers"].as<size_t>(config.cpu_postprocess_workers);
        config.pinned_input_tickets    = runtime["pinned_input_tickets"].as<size_t>(config.pinned_input_tickets);
        config.pinned_output_tickets   = runtime["pinned_output_tickets"].as<size_t>(config.pinned_output_tickets);
        config.pinned_memory_limit_bytes
            = runtime["pinned_memory_limit_bytes"].as<size_t>(config.pinned_memory_limit_bytes);
        config.device_memory_limit_bytes
            = runtime["device_memory_limit_bytes"].as<size_t>(config.device_memory_limit_bytes);
        config.fault_history_capacity = runtime["fault_history_capacity"].as<size_t>(config.fault_history_capacity);
        if (runtime["device_ids"])
        {
            if (!runtime["device_ids"].IsSequence())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "runtime.device_ids must be a sequence");
            }
            config.device_ids = runtime["device_ids"].as<std::vector<int>>();
        }
        config.static_batch_policy = readStaticBatchPolicy(runtime["static_batch_policy"]);
        if (runtime["preferred_batch_sizes"])
        {
            if (!runtime["preferred_batch_sizes"].IsSequence())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "runtime.preferred_batch_sizes must be a sequence");
            }
            config.preferred_batch_sizes = runtime["preferred_batch_sizes"].as<std::vector<int>>();
        }
    }

    if (config.engine_file.is_relative())
    {
        config.engine_file = path.parent_path() / config.engine_file;
    }
    config.validate();
    return config;
}

void EngineConfig::validate() const
{
    if (model_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "model_name must not be empty");
    }
    if (engine_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "engine_file must not be empty");
    }
    for (const auto &name : output_tensor_names)
    {
        if (name.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "model.output_tensors must not contain empty names");
        }
    }
    for (const auto &name : feature_tensor_names)
    {
        if (name.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "model.feature_tensors must not contain empty names");
        }
    }
    if (feature_only && (feature_tensor_names.empty() || output_tensor_names.size() != feature_tensor_names.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "feature-only engine requires matching model.output_tensors and model.feature_tensors");
    }
    if (!feature_only && !feature_tensor_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "model.feature_tensors requires model.feature_only=true");
    }
    if (device_id < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "device_id must be non-negative");
    }
    std::vector<int> configured_devices = device_ids;
    if (configured_devices.empty())
    {
        configured_devices.push_back(device_id);
    }
    for (const int id : configured_devices)
    {
        if (id < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "device_ids must contain non-negative ids");
        }
        if (std::count(configured_devices.begin(), configured_devices.end(), id) != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "device_ids must not contain duplicates");
        }
    }
    if (input_width <= 0 || input_height <= 0 || input_channels != 3)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "input width/height must be positive and channels must be 3");
    }
    if (min_batch_size <= 0 || min_batch_size > opt_batch_size || opt_batch_size > max_batch_size)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid batch range: min=%d opt=%d max=%d",
                             min_batch_size, opt_batch_size, max_batch_size);
    }
    if (max_wait.count() < 0 || execution_slots == 0 || queue_capacity == 0 || cpu_preprocess_workers == 0
        || cpu_postprocess_workers == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid runtime capacity configuration");
    }
    switch (queue_policy)
    {
    case QueuePolicy::Reject:
    case QueuePolicy::Block:
    case QueuePolicy::DropOldest:
    case QueuePolicy::DropNewest:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid queue policy");
    }
    switch (static_batch_policy)
    {
    case StaticBatchPolicy::Reject:
    case StaticBatchPolicy::Pad:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid static batch policy");
    }
    for (const int batch_size : preferred_batch_sizes)
    {
        if (batch_size <= 0 || batch_size > max_batch_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "preferred batch size must be in [1, max_batch_size]");
        }
    }
    for (const float value : stddev)
    {
        if (value == 0.0F)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "input.std must not contain zero");
        }
    }
    if ((source_width == 0) != (source_height == 0) || source_width < 0 || source_height < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "preprocess.source_width and preprocess.source_height must both be positive or omitted");
    }
}

} // namespace irt::engine
