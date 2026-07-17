#pragma once

#include <inferrt/model/Export.h>

#include <string>
#include <string_view>

namespace irt::model {

/**
 * @brief 模型运行目标。
 *
 * 一个运行目标同时描述推理后端和执行设备，避免在配置中分别维护后端、CPU/GPU 类型以及 GPU 编号。
 * 完整字符串格式为 ``<backend>:<device-id>``（GPU）或 ``<backend>:cpu``，例如 ``tensorrt:0``、
 * ``onnxruntime:1`` 和 ``openvino:cpu``。也支持设备简写：``cpu``、``gpu:0``、``cuda:0``；裸 GPU
 * 简写使用 TensorRT 作为默认后端，裸 CPU 简写使用 ONNX Runtime 作为默认后端。
 */
class INFERRT_MODEL_API ModelRuntime
{
public:
    enum class Backend
    {
        TensorRT,
        ONNXRuntime,
        OpenVINO,
    };

    enum class Device
    {
        CPU,
        GPU,
    };

    /** @brief 默认运行目标：TensorRT GPU 0。 */
    ModelRuntime() noexcept = default;

    /**
     * @brief 以结构化字段构造运行目标。
     * @param backend 推理后端。
     * @param device CPU 或 GPU。
     * @param device_id GPU 编号；CPU 会规范化为 0。
     */
    ModelRuntime(Backend backend, Device device, int device_id = 0);

    /**
     * @brief 从字符串构造运行目标。
     * @param specification 完整运行目标或设备简写。
     */
    explicit ModelRuntime(std::string specification);

    /**
     * @brief 解析运行目标字符串。
     * @param specification 完整运行目标或设备简写。
     * @return 解析后的运行目标。
     * @throws irt::Exception 字符串格式非法或设备编号无效时抛出。
     */
    static ModelRuntime parse(std::string_view specification);

    /**
     * @brief 校验后端与设备组合是否可用。
     * @throws irt::Exception TensorRT CPU 或非法设备编号等组合不受支持时抛出。
     */
    void validate() const;

    Backend backend() const noexcept
    {
        return backend_;
    }

    Device device() const noexcept
    {
        return device_;
    }

    int deviceId() const noexcept
    {
        return device_id_;
    }

    bool isCpu() const noexcept
    {
        return device_ == Device::CPU;
    }

    bool isGpu() const noexcept
    {
        return device_ == Device::GPU;
    }

    /** @brief 返回规范化设备字符串：``cpu`` 或 ``gpu:<id>``。 */
    std::string deviceName() const;

    /** @brief 返回规范化完整字符串：GPU 为 ``<backend>:<device-id>``，CPU 为 ``<backend>:cpu``。 */
    std::string toString() const;

    /** @brief 返回规范化后端名称。 */
    static const char *backendName(Backend backend) noexcept;

    bool operator==(const ModelRuntime &) const noexcept = default;

private:
    Backend backend_{Backend::TensorRT};
    Device  device_{Device::GPU};
    int     device_id_{0};
};

} // namespace irt::model
