#include <inferrt/model/ModelRuntime.hpp>

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <string>

namespace irt::model {
namespace {

std::string trimAndLower(std::string_view value)
{
    size_t begin = 0;
    size_t end   = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin])))
    {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        --end;
    }

    std::string result(value.substr(begin, end - begin));
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return result;
}

[[noreturn]] void invalidSpecification(const std::string &specification)
{
    throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid model runtime specification: %s",
                         specification.c_str());
}

bool parseBackendToken(std::string_view token, ModelRuntime::Backend &backend)
{
    if (token == "tensorrt" || token == "trt")
    {
        backend = ModelRuntime::Backend::TensorRT;
        return true;
    }
    if (token == "onnxruntime" || token == "onnx" || token == "ort")
    {
        backend = ModelRuntime::Backend::ONNXRuntime;
        return true;
    }
    if (token == "openvino" || token == "ov")
    {
        backend = ModelRuntime::Backend::OpenVINO;
        return true;
    }
    return false;
}

ModelRuntime::Device parseDeviceToken(std::string_view token, int &device_id)
{
    if (token == "cpu")
    {
        device_id = 0;
        return ModelRuntime::Device::CPU;
    }

    if (token == "gpu" || token == "cuda")
    {
        device_id = 0;
        return ModelRuntime::Device::GPU;
    }

    std::string_view prefix;
    if (token.starts_with("gpu:"))
    {
        prefix = "gpu:";
    }
    else if (token.starts_with("cuda:"))
    {
        prefix = "cuda:";
    }
    else if (!token.empty() && std::isdigit(static_cast<unsigned char>(token.front())))
    {
        prefix = {};
    }
    else
    {
        invalidSpecification(std::string(token));
    }

    const auto id_text = token.substr(prefix.size());
    if (id_text.empty() || id_text.front() == '-')
    {
        invalidSpecification(std::string(token));
    }

    const auto *first = id_text.data();
    const auto *last  = id_text.data() + id_text.size();
    const auto  result = std::from_chars(first, last, device_id);
    if (result.ec != std::errc{} || result.ptr != last || device_id < 0)
    {
        invalidSpecification(std::string(token));
    }
    return ModelRuntime::Device::GPU;
}

ModelRuntime::Backend defaultBackend(ModelRuntime::Device device) noexcept
{
    return device == ModelRuntime::Device::GPU ? ModelRuntime::Backend::TensorRT
                                                : ModelRuntime::Backend::ONNXRuntime;
}

ModelRuntime defaultRuntime(ModelRuntime::Backend backend)
{
    switch (backend)
    {
    case ModelRuntime::Backend::TensorRT:
        return {backend, ModelRuntime::Device::GPU, 0};
    case ModelRuntime::Backend::ONNXRuntime:
    case ModelRuntime::Backend::OpenVINO:
        return {backend, ModelRuntime::Device::CPU, 0};
    }
    invalidSpecification("unknown backend");
}

} // namespace

ModelRuntime::ModelRuntime(Backend backend, Device device, int device_id)
    : backend_(backend)
    , device_(device)
    , device_id_(device == Device::CPU ? 0 : device_id)
{
    validate();
}

ModelRuntime::ModelRuntime(std::string specification)
    : ModelRuntime(parse(specification))
{
}

ModelRuntime ModelRuntime::parse(std::string_view specification)
{
    const auto text = trimAndLower(specification);
    if (text.empty())
    {
        invalidSpecification(std::string(specification));
    }

    Backend backend{};
    const auto separator = text.find(':');
    if (separator == std::string::npos)
    {
        if (parseBackendToken(text, backend))
        {
            return defaultRuntime(backend);
        }

        int device_id{0};
        const auto device = parseDeviceToken(text, device_id);
        return {defaultBackend(device), device, device_id};
    }

    const auto prefix = std::string_view(text).substr(0, separator);
    if (!parseBackendToken(prefix, backend))
    {
        int device_id{0};
        const auto device = parseDeviceToken(text, device_id);
        return {defaultBackend(device), device, device_id};
    }

    const auto device_spec = std::string_view(text).substr(separator + 1);
    int        device_id{0};
    const auto device = parseDeviceToken(device_spec, device_id);
    return {backend, device, device_id};
}

void ModelRuntime::validate() const
{
    if (device_id_ < 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Model runtime device id must be non-negative, got %d",
                             device_id_);
    }

    switch (backend_)
    {
    case Backend::TensorRT:
    case Backend::ONNXRuntime:
    case Backend::OpenVINO:
        break;
    default:
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unsupported model runtime backend");
    }

    switch (device_)
    {
    case Device::CPU:
    case Device::GPU:
        break;
    default:
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unsupported model runtime device");
    }

    if (backend_ == Backend::TensorRT && device_ == Device::CPU)
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "TensorRT backend requires a GPU model runtime");
    }
}

std::string ModelRuntime::deviceName() const
{
    return device_ == Device::CPU ? "cpu" : "gpu:" + std::to_string(device_id_);
}

std::string ModelRuntime::toString() const
{
    const auto target = device_ == Device::CPU ? std::string{"cpu"} : std::to_string(device_id_);
    return std::string(backendName(backend_)) + ":" + target;
}

const char *ModelRuntime::backendName(Backend backend) noexcept
{
    switch (backend)
    {
    case Backend::TensorRT:
        return "tensorrt";
    case Backend::ONNXRuntime:
        return "onnxruntime";
    case Backend::OpenVINO:
        return "openvino";
    }
    return "unknown";
}

} // namespace irt::model
