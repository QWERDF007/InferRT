#include <cpuinfo.h>
#include <inferrt/util/Device.hpp>

#if INFERRT_HAS_CUDA
#include <nvml.h>
#endif

#include <mutex>

namespace irt::util {

namespace {

#if INFERRT_HAS_CUDA
class NvmlSession
{
public:
    NvmlSession()
        : status_(nvmlInit_v2())
    {
    }

    ~NvmlSession()
    {
        if (status_ == NVML_SUCCESS)
        {
            nvmlShutdown();
        }
    }

    NvmlSession(const NvmlSession &)            = delete;
    NvmlSession &operator=(const NvmlSession &) = delete;

    bool available() const
    {
        return status_ == NVML_SUCCESS;
    }

private:
    nvmlReturn_t status_;
};

NvmlSession &nvmlSession()
{
    static NvmlSession session;
    return session;
}

bool getNvmlDeviceCount(unsigned int &count)
{
    if (!nvmlSession().available())
    {
        return false;
    }

    return nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS;
}
#endif

bool cpuInfoInitialized()
{
    static std::once_flag flag;
    static bool            initialized = false;

    std::call_once(flag, [] { initialized = cpuinfo_initialize(); });
    return initialized;
}

} // namespace

std::vector<std::string> getGPUDeviceNames()
{
#if INFERRT_HAS_CUDA
    unsigned int count = 0;
    if (!getNvmlDeviceCount(count))
    {
        return {};
    }

    std::vector<std::string> names(count);
    for (unsigned int index = 0; index < count; ++index)
    {
        nvmlDevice_t device = nullptr;
        if (nvmlDeviceGetHandleByIndex_v2(index, &device) != NVML_SUCCESS)
        {
            continue;
        }

        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS)
        {
            names[index] = name;
        }
    }

    return names;
#else
    return {};
#endif
}

std::string getCPUDeviceName()
{
    if (!cpuInfoInitialized() || cpuinfo_get_packages_count() == 0)
    {
        return {};
    }

    const cpuinfo_package *package = cpuinfo_get_package(0);
    if (package == nullptr)
    {
        return {};
    }

    return package->name;
}

uint64_t getGPUDeviceMemory(uint32_t device_index)
{
#if INFERRT_HAS_CUDA
    unsigned int count = 0;
    if (!getNvmlDeviceCount(count) || device_index >= count)
    {
        return 0;
    }

    nvmlDevice_t device = nullptr;
    if (nvmlDeviceGetHandleByIndex_v2(device_index, &device) != NVML_SUCCESS)
    {
        return 0;
    }

    nvmlMemory_t memory{};
    if (nvmlDeviceGetMemoryInfo(device, &memory) != NVML_SUCCESS)
    {
        return 0;
    }

    return static_cast<uint64_t>(memory.total);
#else
    (void)device_index;
    return 0;
#endif
}

} // namespace irt::util
