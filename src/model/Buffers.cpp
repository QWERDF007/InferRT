#include <cuda_runtime_api.h>
#include <inferrt/model/Buffers.hpp>

#include <cstdlib>


namespace irt::model {

bool DeviceAllocator::operator()(void **ptr, size_t num_bytes) const noexcept
{
    if (num_bytes == 0)
    {
        *ptr = nullptr;
        return true;
    }
    return cudaMalloc(ptr, num_bytes) == cudaSuccess;
}

void DeviceFree::operator()(void *ptr) const noexcept
{
    cudaFree(ptr);
}

bool HostAllocator::operator()(void **ptr, size_t num_bytes) const noexcept
{
    if (num_bytes == 0)
    {
        *ptr = nullptr;
        return true;
    }
    *ptr = std::malloc(num_bytes);
    return *ptr != nullptr;
}

void HostFree::operator()(void *ptr) const noexcept
{
    std::free(ptr);
}

bool PinnedHostAllocator::operator()(void **ptr, size_t num_bytes) const noexcept
{
    if (num_bytes == 0)
    {
        *ptr = nullptr;
        return true;
    }
    return cudaHostAlloc(ptr, num_bytes, cudaHostAllocDefault) == cudaSuccess;
}

void PinnedHostFree::operator()(void *ptr) const noexcept
{
    cudaFreeHost(ptr);
}

} // namespace irt::model
