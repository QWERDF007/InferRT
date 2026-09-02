#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/model/ModelRuntime.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace irt::engine {
namespace {

bool isPresent(const YAML::Node &node)
{
    if (!node.IsDefined())
    {
        return false;
    }
    return node.Type() != YAML::NodeType::Null && node.Type() != YAML::NodeType::Undefined;
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

YAML::Node firstPresent(std::initializer_list<YAML::Node> nodes)
{
    for (const auto &node : nodes)
    {
        if (isPresent(node))
        {
            return node;
        }
    }
    return {};
}

YAML::Node optionalMap(const YAML::Node &node)
{
    if (isPresent(node) && node.IsMap())
    {
        return node;
    }
    return YAML::Node(YAML::NodeType::Map);
}

std::vector<float> readFloatVector(const YAML::Node &node, const char *name, const std::vector<float> &fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }
    if (!node.IsSequence() || node.size() != fallback.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must contain exactly %zu values", name,
                             fallback.size());
    }

    std::vector<float> values(fallback.size());
    for (size_t index = 0; index < values.size(); ++index)
    {
        values[index] = node[index].as<float>();
    }
    return values;
}

const YAML::Node requireMap(const YAML::Node &root, const char *name)
{
    const auto node = root[name];
    if (!isPresent(node) || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Missing mapping: %s", name);
    }
    return node;
}

PreprocessBackend readPreprocessBackend(const YAML::Node &node)
{
    if (!isPresent(node))
    {
        return PreprocessBackend::CPU;
    }

    const auto backend = lower(node.as<std::string>());
    if (backend == "cpu")
    {
        return PreprocessBackend::CPU;
    }
    if (backend == "cuda" || backend == "gpu")
    {
        return PreprocessBackend::CUDA;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess.backend must be cpu or cuda, got: %s",
                         backend.c_str());
}

QueuePolicy readQueuePolicy(const YAML::Node &node)
{
    if (!isPresent(node))
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
    if (!isPresent(node))
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

irt::ColorFormat readColorFormat(const YAML::Node &node, const char *name, const irt::ColorFormat fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "bgr")
    {
        return irt::ColorFormat::BGR;
    }
    if (value == "rgb")
    {
        return irt::ColorFormat::RGB;
    }
    if (value == "gray" || value == "grey" || value == "gray8" || value == "grey8")
    {
        return irt::ColorFormat::GRAY;
    }
    if (value == "bgra")
    {
        return irt::ColorFormat::BGRA;
    }
    if (value == "rgba")
    {
        return irt::ColorFormat::RGBA;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "%s must be bgr, rgb, gray, bgra, or rgba, got: %s", name, value.c_str());
}

irt::Interpolation readInterpolation(const YAML::Node &node, const irt::Interpolation fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "nearest" || value == "nearest_neighbor")
    {
        return irt::Interpolation::Nearest;
    }
    if (value == "linear" || value == "bilinear")
    {
        return irt::Interpolation::Linear;
    }
    if (value == "cubic" || value == "bicubic")
    {
        return irt::Interpolation::Cubic;
    }
    if (value == "area")
    {
        return irt::Interpolation::Area;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "preprocess.interpolation must be nearest, linear, cubic, or area, got: %s",
                         value.c_str());
}

irt::PaddingMode readPaddingMode(const YAML::Node &node, const irt::PaddingMode fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "resize" || value == "direct" || value == "direct_resize")
    {
        return irt::PaddingMode::DirectResize;
    }
    if (value == "letterbox" || value == "letter_box")
    {
        return irt::PaddingMode::Letterbox;
    }
    if (value == "center_crop" || value == "centercrop" || value == "crop")
    {
        return irt::PaddingMode::CenterCrop;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "preprocess.padding_mode must be direct_resize, letterbox, or center_crop, got: %s",
                         value.c_str());
}

irt::PaddingAlignment readPaddingAlignment(const YAML::Node &node, const irt::PaddingAlignment fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "center" || value == "centre")
    {
        return irt::PaddingAlignment::Center;
    }
    if (value == "top_left" || value == "topleft" || value == "top-left")
    {
        return irt::PaddingAlignment::TopLeft;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "preprocess.padding_alignment must be center or top_left, got: %s", value.c_str());
}

irt::TensorLayout readOutputLayout(const YAML::Node &node, const irt::TensorLayout fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "nchw")
    {
        return irt::TensorLayout::NCHW;
    }
    if (value == "chw")
    {
        return irt::TensorLayout::CHW;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "preprocess.output_layout must be nchw or chw, got: %s", value.c_str());
}

irt::TensorDataType readOutputDtype(const YAML::Node &node, const irt::TensorDataType fallback)
{
    if (!isPresent(node))
    {
        return fallback;
    }

    const auto value = lower(node.as<std::string>());
    if (value == "float32" || value == "f32" || value == "fp32")
    {
        return irt::TensorDataType::F32;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "preprocess.output_dtype must be float32/f32, got: %s", value.c_str());
}

template<typename T>
void readScalarIfPresent(const YAML::Node &node, const char *name, T &target)
{
    if (isPresent(node))
    {
        try
        {
            target = node.as<T>();
        }
        catch (const YAML::Exception &error)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s has an invalid value: %s", name,
                                 error.what());
        }
    }
}

} // namespace

const irt::PreprocessSpec &EngineConfig::preprocessSpec() const noexcept
{
    return preprocess;
}

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

    try
    {
        config.preprocess.input_width  = input["width"].as<int>();
        config.preprocess.input_height = input["height"].as<int>();
    }
    catch (const YAML::Exception &error)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "input.width and input.height are required: %s",
                             error.what());
    }

    const auto preprocess_node = root["preprocess"];
    if (isPresent(preprocess_node) && !preprocess_node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess must be a mapping");
    }
    if (isPresent(preprocess_node) && isPresent(preprocess_node["normalize"])
        && !preprocess_node["normalize"].IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess.normalize must be a mapping");
    }
    if (isPresent(preprocess_node) && isPresent(preprocess_node["output"])
        && !preprocess_node["output"].IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "preprocess.output must be a mapping");
    }
    const auto preprocess_map = optionalMap(preprocess_node);
    const auto normalize_candidate = preprocess_map["normalize"];
    const auto output_candidate = preprocess_map["output"];
    const auto normalize_node = optionalMap(normalize_candidate);
    const auto output_node = optionalMap(output_candidate);

    const auto channels_node = firstPresent({preprocess_map["input_channels"], preprocess_map["channels"],
                                              input["channels"]});
    readScalarIfPresent(channels_node, "input.channels", config.preprocess.input_channels);

    const auto source_channels_node
        = firstPresent({preprocess_map["source_channels"], input["source_channels"]});
    readScalarIfPresent(source_channels_node, "preprocess.source_channels", config.preprocess.source_channels);

    const auto src_color_node = firstPresent({preprocess_map["src_color"], preprocess_map["source_color"],
                                               input["src_color"], input["source_color"]});
    const auto dst_color_node = firstPresent({preprocess_map["dst_color"], preprocess_map["destination_color"],
                                               input["dst_color"], input["destination_color"]});
    const bool has_explicit_colors = static_cast<bool>(src_color_node) || static_cast<bool>(dst_color_node);
    config.preprocess.src_color = readColorFormat(src_color_node, "preprocess.src_color", config.preprocess.src_color);
    config.preprocess.dst_color = readColorFormat(dst_color_node, "preprocess.dst_color", config.preprocess.dst_color);
    if (!has_explicit_colors)
    {
        if (config.preprocess.input_channels == 1)
        {
            config.preprocess.src_color = irt::ColorFormat::GRAY;
            config.preprocess.dst_color = irt::ColorFormat::GRAY;
        }
        else if (config.preprocess.input_channels == 4)
        {
            config.preprocess.src_color = irt::ColorFormat::BGRA;
            config.preprocess.dst_color = irt::ColorFormat::RGBA;
        }
    }

    const auto mean_node = firstPresent({preprocess_map["mean"], normalize_node["mean"], input["mean"]});
    const auto stddev_node = firstPresent({preprocess_map["stddev"], preprocess_map["std"],
                                           normalize_node["stddev"], normalize_node["std"], input["stddev"],
                                           input["std"]});
    if (config.preprocess.mean.size() != static_cast<size_t>(config.preprocess.input_channels))
    {
        config.preprocess.mean.assign(static_cast<size_t>(config.preprocess.input_channels), 0.0F);
    }
    if (config.preprocess.stddev.size() != static_cast<size_t>(config.preprocess.input_channels))
    {
        config.preprocess.stddev.assign(static_cast<size_t>(config.preprocess.input_channels), 1.0F);
    }
    if (!mean_node)
    {
        if (config.preprocess.input_channels == 3 && config.preprocess.dst_color == irt::ColorFormat::RGB)
        {
            config.preprocess.mean = {0.485F, 0.456F, 0.406F};
        }
    }
    if (!stddev_node)
    {
        if (config.preprocess.input_channels == 3 && config.preprocess.dst_color == irt::ColorFormat::RGB)
        {
            config.preprocess.stddev = {0.229F, 0.224F, 0.225F};
        }
    }
    config.preprocess.mean = readFloatVector(mean_node, "preprocess.mean", config.preprocess.mean);
    config.preprocess.stddev = readFloatVector(stddev_node, "preprocess.stddev", config.preprocess.stddev);

    config.preprocess.backend = readPreprocessBackend(preprocess_map["backend"]);
    config.preprocess.interpolation = readInterpolation(preprocess_map["interpolation"],
                                                        config.preprocess.interpolation);
    const auto padding_node = firstPresent({preprocess_map["padding_mode"], preprocess_map["padding"]});
    const bool has_padding_mode = static_cast<bool>(padding_node);
    if (has_padding_mode)
    {
        config.preprocess.padding_mode = readPaddingMode(padding_node, config.preprocess.padding_mode);
    }
    if (preprocess_map["letterbox"])
    {
        const bool letterbox = preprocess_map["letterbox"].as<bool>();
        const auto requested = letterbox ? irt::PaddingMode::Letterbox : irt::PaddingMode::DirectResize;
        if (has_padding_mode && config.preprocess.padding_mode != requested)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "preprocess.letterbox conflicts with preprocess.padding_mode");
        }
        config.preprocess.padding_mode = requested;
    }
    config.preprocess.padding_alignment = readPaddingAlignment(
        firstPresent({preprocess_map["padding_alignment"], preprocess_map["alignment"]}),
        config.preprocess.padding_alignment);
    readScalarIfPresent(firstPresent({preprocess_map["source_width"], input["source_width"]}),
                        "preprocess.source_width", config.preprocess.source_width);
    readScalarIfPresent(firstPresent({preprocess_map["source_height"], input["source_height"]}),
                        "preprocess.source_height", config.preprocess.source_height);
    readScalarIfPresent(firstPresent({preprocess_map["pad_value"], input["pad_value"]}),
                        "preprocess.pad_value", config.preprocess.pad_value);
    readScalarIfPresent(firstPresent({preprocess_map["pad_after_normalize"],
                                      preprocess_map["padding_after_normalize"]}),
                        "preprocess.pad_after_normalize", config.preprocess.pad_after_normalize);
    readScalarIfPresent(firstPresent({preprocess_map["scale"], normalize_node["scale"], input["scale"]}),
                        "preprocess.scale", config.preprocess.scale);
    config.preprocess.output_layout = readOutputLayout(
        firstPresent({preprocess_map["output_layout"], output_node["layout"], input["output_layout"]}),
        config.preprocess.output_layout);
    config.preprocess.output_dtype = readOutputDtype(
        firstPresent({preprocess_map["output_dtype"], output_node["dtype"], input["output_dtype"]}),
        config.preprocess.output_dtype);

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
    const auto &preprocess_spec = preprocessSpec();
    preprocess_spec.validate();

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
    if (preprocess_spec.input_width <= 0 || preprocess_spec.input_height <= 0 || preprocess_spec.input_channels <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "input width/height/channels must be positive");
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
}

} // namespace irt::engine
