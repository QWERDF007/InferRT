/**
 * @file BenchmarkNMS.cpp
 * @brief NMS workspace reuse and latency benchmark.
 *
 * The benchmark keeps all input/output buffers alive across iterations and
 * emits workspace allocation counters together with percentile latency
 * counters.  Use Google Benchmark's ``--benchmark_out`` to archive JSON.
 */

#include "../BenchmarkCVCudaCommon.hpp"

#include <benchmark/benchmark.h>
#include <inferrt/cvcuda/OpNMS.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace bench = irt::cvcuda::bench;

double percentile(std::vector<double> values, const double quantile)
{
    if (values.empty())
    {
        return 0.0;
    }
    const auto position = quantile * static_cast<double>(values.size() - 1);
    const auto lower    = static_cast<size_t>(std::floor(position));
    const auto upper    = static_cast<size_t>(std::ceil(position));
    std::sort(values.begin(), values.end());
    if (lower == upper)
    {
        return values[lower];
    }
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

void BM_NMS(benchmark::State &state)
{
    const int num_boxes = static_cast<int>(state.range(0));
    if (num_boxes <= 0)
    {
        state.SkipWithError("NMS requires a positive box count");
        return;
    }

    int device_id = 0;
    if (!bench::CheckStep(state, cudaGetDevice(&device_id) == cudaSuccess, "NMS device query failed"))
    {
        return;
    }
    cudaDeviceProp device_properties{};
    if (!bench::CheckStep(state, cudaGetDeviceProperties(&device_properties, device_id) == cudaSuccess,
                          "NMS device properties query failed"))
    {
        return;
    }
    int driver_version = 0;
    int runtime_version = 0;
    if (!bench::CheckStep(state, cudaDriverGetVersion(&driver_version) == cudaSuccess,
                          "NMS CUDA driver version query failed")
        || !bench::CheckStep(state, cudaRuntimeGetVersion(&runtime_version) == cudaSuccess,
                             "NMS CUDA runtime version query failed"))
    {
        return;
    }

    std::vector<float> boxes(static_cast<size_t>(num_boxes) * 4U);
    std::vector<float> scores(static_cast<size_t>(num_boxes));
    for (int index = 0; index < num_boxes; ++index)
    {
        const float offset = static_cast<float>(index % 64) * 3.0F;
        boxes[static_cast<size_t>(index) * 4U + 0U] = offset;
        boxes[static_cast<size_t>(index) * 4U + 1U] = offset;
        boxes[static_cast<size_t>(index) * 4U + 2U] = offset + 10.0F;
        boxes[static_cast<size_t>(index) * 4U + 3U] = offset + 10.0F;
        scores[static_cast<size_t>(index)] = 1.0F - static_cast<float>(index) / static_cast<float>(num_boxes + 1);
    }

    bench::DeviceBuffer d_boxes;
    bench::DeviceBuffer d_scores;
    bench::DeviceBuffer d_keep;
    bench::DeviceBuffer d_keep_count;
    if (!bench::CheckStep(state, d_boxes.Allocate(boxes.size() * sizeof(float)), "NMS input allocation failed")
        || !bench::CheckStep(state, d_scores.Allocate(scores.size() * sizeof(float)), "NMS score allocation failed")
        || !bench::CheckStep(state, d_keep.Allocate(scores.size() * sizeof(int64_t)), "NMS output allocation failed")
        || !bench::CheckStep(state, d_keep_count.Allocate(sizeof(int)), "NMS count allocation failed")
        || !bench::CheckStep(state, d_boxes.CopyFromHost(boxes.data(), boxes.size() * sizeof(float)),
                             "NMS box upload failed")
        || !bench::CheckStep(state, d_scores.CopyFromHost(scores.data(), scores.size() * sizeof(float)),
                             "NMS score upload failed"))
    {
        return;
    }

    bench::CudaStream stream;
    if (!bench::CheckStep(state, stream.Create(), "NMS stream creation failed"))
    {
        return;
    }
    bench::CudaEventTimer timer;
    if (!bench::CheckStep(state, timer.Create(), "NMS timer creation failed"))
    {
        return;
    }

    irt::cvcuda::NMS op;
    const auto run_once = [&]() {
        return op(d_boxes.As<float>(), d_scores.As<float>(), d_keep.As<int64_t>(), d_keep_count.As<int>(), num_boxes,
                  0.5F, stream.Get());
    };
    if (!bench::CheckStatus(state, run_once(), "inferrt_cvcuda::nms warm-up")
        || !bench::CheckStep(state, cudaStreamSynchronize(stream.Get()) == cudaSuccess, "NMS warm-up failed"))
    {
        return;
    }
    const auto warmup_stats = op.workspaceStats();

    size_t free_before = 0;
    size_t total_memory = 0;
    (void)cudaMemGetInfo(&free_before, &total_memory);
    size_t min_free = free_before;
    std::vector<double> samples_us;
    samples_us.reserve(128);

    for (auto _ : state)
    {
        if (!bench::CheckStep(state, timer.RecordStart(stream.Get()), "NMS start event failed")
            || !bench::CheckStatus(state, run_once(), "inferrt_cvcuda::nms") )
        {
            break;
        }
        float elapsed_ms = 0.0F;
        if (!bench::CheckStep(state, timer.RecordStopAndElapsed(stream.Get(), elapsed_ms), "NMS stop event failed"))
        {
            break;
        }
        samples_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
        state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
        size_t free_memory = 0;
        size_t total = 0;
        if (cudaMemGetInfo(&free_memory, &total) == cudaSuccess)
        {
            min_free = std::min(min_free, free_memory);
        }
        benchmark::DoNotOptimize(d_keep_count.As<int>());
    }

    const auto stats = op.workspaceStats();
    state.counters["workspace_allocations"] = static_cast<double>(stats.allocation_count);
    state.counters["workspace_releases"] = static_cast<double>(stats.release_count);
    state.counters["steady_allocations"]
        = static_cast<double>(stats.allocation_count - warmup_stats.allocation_count);
    state.counters["workspace_bytes"] = static_cast<double>(stats.workspace_bytes);
    state.counters["peak_device_bytes"]
        = static_cast<double>(free_before > min_free ? free_before - min_free : stats.workspace_bytes);
    state.counters["p50_us"] = percentile(samples_us, 0.50);
    state.counters["p95_us"] = percentile(samples_us, 0.95);
    state.counters["p99_us"] = percentile(samples_us, 0.99);
    state.counters["box_count"] = static_cast<double>(num_boxes);
    state.counters["device_id"] = static_cast<double>(device_id);
    state.counters["gpu_global_memory_bytes"] = static_cast<double>(device_properties.totalGlobalMem);
    state.counters["gpu_multiprocessors"] = static_cast<double>(device_properties.multiProcessorCount);
    state.counters["gpu_compute_major"] = static_cast<double>(device_properties.major);
    state.counters["gpu_compute_minor"] = static_cast<double>(device_properties.minor);
    state.counters["cuda_driver_version"] = static_cast<double>(driver_version);
    state.counters["cuda_runtime_version"] = static_cast<double>(runtime_version);
    state.counters["warmup_iterations"] = 1.0;
    state.counters["measured_iterations"] = static_cast<double>(samples_us.size());
    state.counters["percentile_sample_count"] = static_cast<double>(samples_us.size());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_boxes));
    state.SetLabel(std::string("NMS workspace reuse; input=xyxy/f32; gpu=") + device_properties.name);
}

BENCHMARK(BM_NMS)->Args({64})->Args({160})->Args({512})->UseManualTime();

} // namespace
