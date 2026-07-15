/**
 * @file Device.hpp
 *
 * @brief CPU/GPU 设备信息工具函数声明。
 */

#pragma once

#include <inferrt/util/Export.h>

#include <cstdint>
#include <string>
#include <vector>

namespace irt::util {

/**
 * @brief 获取 NVML 可见的全部 GPU 设备名称。
 *
 * @return GPU 名称列表；NVML 不可用或没有可见 GPU 时返回空列表。
 */
INFERRT_UTIL_API std::vector<std::string> getGPUDeviceNames();

/**
 * @brief 获取 CPU 设备名称。
 *
 * @return cpuinfo 报告的 CPU 名称；cpuinfo 初始化失败时返回空字符串。
 */
INFERRT_UTIL_API std::string getCPUDeviceName();

/**
 * @brief 获取指定 GPU 的总显存大小。
 *
 * @param device_index GPU 在 NVML 中的设备索引。
 * @return 总显存大小，单位为字节；设备不存在或 NVML 查询失败时返回 0。
 */
INFERRT_UTIL_API uint64_t getGPUDeviceMemory(uint32_t device_index);

} // namespace irt::util
