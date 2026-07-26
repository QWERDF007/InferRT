#include "priv/TensorRTRuntimePlan.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/InferenceEngine.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::engine {
namespace {

TensorDesc tensor(const TensorDataType data_type, const TensorLayout layout, const MemoryKind memory_kind,
                  const int width, const int height, const int channels)
{
    return {data_type, layout, memory_kind, width, height, channels};
}

std::shared_ptr<const PipelinePlan> makeLegacyPipeline(const EngineConfig &config)
{
    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    const auto model_input = tensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, config.input_width,
                                    config.input_height, config.input_channels);
    builder.addTensor("model_input", model_input).setModelInput("model_input");

    if (config.preprocess_backend == PreprocessBackend::CPU)
    {
        builder.addTensor("host_input", tensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST,
                                               config.input_width, config.input_height, config.input_channels));
        builder.addNode("cpu.image_to_tensor",
                        {
                            {},
                            {"host_input"},
                            CpuImageToTensorOptions{config.mean, config.stddev, config.letterbox}
        });
        builder.addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}});
        return builder.build(std::move(registry));
    }

    if (config.source_width <= 0 || config.source_height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Legacy CUDA preprocessing requires source_width and source_height");
    }
    auto source = tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, config.source_width,
                         config.source_height, config.input_channels);
    builder.addTensor("host_image", source);
    source.memory_kind = MemoryKind::DEVICE;
    builder.addTensor("device_image", source);
    builder.addNode("cpu.copy_image", {{}, {"host_image"}, {}});
    builder.addNode("cuda.upload", {{"host_image"}, {"device_image"}, {}});

    if (config.letterbox)
    {
        for (size_t index = 0; index < config.mean.size(); ++index)
        {
            if (config.mean[index] != 0.0F || config.stddev[index] != 1.0F)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Legacy CUDA letterbox requires identity normalization");
            }
        }
        builder.addNode("cvcuda.letterbox", {{"device_image"}, {"model_input"}, {}});
    }
    else
    {
        const auto resized = tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE, config.input_width,
                                    config.input_height, config.input_channels);
        builder.addTensor("resized_bgr", resized).addTensor("rgb", resized);
        builder.addNode("cvcuda.resize", {{"device_image"}, {"resized_bgr"}, ResizeOptions{}});
        builder.addNode("cvcuda.cvt_color", {{"resized_bgr"}, {"rgb"}, CvtColorOptions{}});
        builder.addNode("cvcuda.normalize", {
                                                {"rgb"},
                                                {"model_input"},
                                                NormalizeOptions{config.mean, config.stddev}
        });
    }
    return builder.build(std::move(registry));
}

std::exception_ptr cancelledError()
{
    return std::make_exception_ptr(
        irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference request was cancelled"));
}

std::exception_ptr timeoutError()
{
    return std::make_exception_ptr(irt::Exception(irt::Status::ERROR_NOT_READY, "Inference request deadline expired"));
}

std::exception_ptr droppedError()
{
    return std::make_exception_ptr(irt::Exception(irt::Status::ERROR_NOT_READY, "Inference request was dropped"));
}

size_t alignDeviceOffset(const size_t value)
{
    constexpr size_t alignment = 256;
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t elapsedMicroseconds(const std::chrono::steady_clock::time_point begin,
                             const std::chrono::steady_clock::time_point end)
{
    if (begin == std::chrono::steady_clock::time_point{} || end < begin)
    {
        return 0;
    }
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
}

void recordLatency(StageLatencySnapshot &snapshot, const uint64_t microseconds)
{
    ++snapshot.samples;
    snapshot.total_microseconds += microseconds;
    snapshot.max_microseconds = std::max(snapshot.max_microseconds, microseconds);

    size_t   bucket = 0;
    uint64_t limit  = 1;
    while (bucket + 1 < snapshot.histogram.size() && microseconds > limit)
    {
        limit *= 10;
        ++bucket;
    }
    ++snapshot.histogram[bucket];
}

bool isFatalCudaError(const cudaError_t error)
{
    return error == cudaErrorContextIsDestroyed || error == cudaErrorIllegalAddress || error == cudaErrorLaunchFailure;
}

} // namespace

class InferenceEngine::Impl
{
public:
    struct SubmittedRequest
    {
        uint64_t                     id{0};
        std::future<InferenceResult> future;
    };

    Impl(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline)
        : config_(std::move(config))
        , pipeline_(std::move(pipeline))
    {
        if (!pipeline_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "InferenceEngine requires a PipelinePlan");
        }
    }

    ~Impl()
    {
        shutdown();
    }

    void             start();
    SubmittedRequest submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options);
    InferenceResult  infer(const cv::Mat &image, std::chrono::milliseconds timeout);
    bool             cancel(uint64_t request_id);
    void             shutdown();

    [[nodiscard]] size_t                   pendingRequests() const;
    [[nodiscard]] EngineMetricsSnapshot    metrics() const;
    [[nodiscard]] EngineState              state() const noexcept;
    [[nodiscard]] const EngineConfig      &config() const noexcept;
    [[nodiscard]] std::vector<FaultRecord> faults() const;

private:
    enum class FailureKind
    {
        Failed,
        Cancelled,
        TimedOut,
        Dropped,
    };

    struct Request
    {
        uint64_t                              id{0};
        cv::Mat                               image;
        std::chrono::steady_clock::time_point submitted;
        std::chrono::steady_clock::time_point deadline{std::chrono::steady_clock::time_point::max()};
        std::promise<InferenceResult>         promise;
        std::atomic_bool                      cancelled{false};
        std::atomic_bool                      finished{false};
        int                                   priority{0};
        std::string                           compatibility_key{"default"};
        TensorInputMap                        extra_inputs;
        std::string                           source_id{"default"};
        uint64_t                              source_sequence{0};
        bool                                  preserve_source_order{true};
    };

    struct InputTicket
    {
        std::unordered_map<std::string, irt::model::PinnedHostBuffer> buffers;
        TensorViewMap                                                 tensors;
    };

    struct OutputTicket
    {
        std::vector<irt::model::PinnedHostBuffer>                     model_outputs;
        std::unordered_map<std::string, irt::model::PinnedHostBuffer> result_outputs;
    };

    using RequestPtr = std::shared_ptr<Request>;

    struct PendingCompletion
    {
        RequestPtr         request;
        bool               success{false};
        InferenceResult    result;
        std::exception_ptr error;
    };

    struct BatchState
    {
        std::vector<RequestPtr>               requests;
        int                                   actual_batch{0};
        int                                   execution_batch{0};
        InputTicket                          *input_ticket{nullptr};
        OutputTicket                         *output_ticket{nullptr};
        std::vector<size_t>                   output_elements_per_request;
        std::vector<std::string>              output_names;
        int                                   device_id{-1};
        std::chrono::steady_clock::time_point scheduled;
        std::chrono::steady_clock::time_point cpu_preprocess_started;
        std::chrono::steady_clock::time_point cpu_preprocess_finished;
        std::chrono::steady_clock::time_point gpu_submitted;
        std::chrono::steady_clock::time_point gpu_finished;
        std::chrono::steady_clock::time_point cpu_postprocess_started;
        std::chrono::steady_clock::time_point cpu_postprocess_finished;
        std::atomic_bool                      finalized{false};
    };

    using BatchPtr = std::shared_ptr<BatchState>;

    struct Slot
    {
        explicit Slot(const int device)
            : device_id(device)
        {
        }

        ~Slot()
        {
            cudaSetDevice(device_id);
            context.reset();
            if (device_arena != nullptr && compute_stream != nullptr)
            {
                cudaFreeAsync(device_arena, compute_stream);
                cudaStreamSynchronize(compute_stream);
            }
            if (h2d_done != nullptr)
            {
                cudaEventDestroy(h2d_done);
            }
            if (compute_done != nullptr)
            {
                cudaEventDestroy(compute_done);
            }
            if (d2h_done != nullptr)
            {
                cudaEventDestroy(d2h_done);
            }
            if (h2d_stream != nullptr)
            {
                cudaStreamDestroy(h2d_stream);
            }
            if (compute_stream != nullptr)
            {
                cudaStreamDestroy(compute_stream);
            }
            if (d2h_stream != nullptr)
            {
                cudaStreamDestroy(d2h_stream);
            }
        }

        int                                              device_id;
        std::shared_ptr<priv::TensorRTRuntimePlan>       runtime_plan;
        std::unique_ptr<nvinfer1::IExecutionContext>     context;
        std::vector<std::pair<std::string, std::string>> model_inputs;
        std::vector<std::string>                         output_names;
        cudaStream_t                                     h2d_stream{nullptr};
        cudaStream_t                                     compute_stream{nullptr};
        cudaStream_t                                     d2h_stream{nullptr};
        cudaEvent_t                                      h2d_done{nullptr};
        cudaEvent_t                                      compute_done{nullptr};
        cudaEvent_t                                      d2h_done{nullptr};
        void                                            *device_arena{nullptr};
        size_t                                           device_arena_bytes{0};
        int                                              capacity_batch{0};
        std::unordered_map<std::string, TensorView>      device_tensors;
        std::vector<void *>                              device_outputs;
        std::vector<size_t>                              output_capacity_bytes;
        std::vector<std::unique_ptr<IOperator>>          operators;
        std::mutex                                       mutex;
        BatchPtr                                         active_batch;
        bool                                             input_released{false};
    };

    void                        initializeSlot(Slot &slot);
    void                        initializeTickets();
    [[nodiscard]] InputTicket  *acquireInputTicket();
    [[nodiscard]] OutputTicket *acquireOutputTicket();
    void                        releaseInputTicket(InputTicket *ticket);
    void                        releaseOutputTicket(OutputTicket *ticket);

    void schedulerLoop();
    void prepareLoop(std::vector<std::unique_ptr<IOperator>> operators);
    void gpuDispatcherLoop();
    void completionPollerLoop();
    void postprocessLoop(std::vector<std::unique_ptr<IOperator>> operators);

    void executeStage(const std::vector<std::unique_ptr<IOperator>> &operators, PipelineStage stage,
                      const BatchState &batch, int device_id, cudaStream_t stream, TensorViewMap &tensors,
                      ResultMap &results) const;
    void dispatchGpu(Slot &slot, const BatchPtr &batch);
    void processPostprocess(const std::vector<std::unique_ptr<IOperator>> &operators, const BatchPtr &batch);

    [[nodiscard]] bool prepareBatch(const BatchPtr &batch);
    [[nodiscard]] bool removeInvalidBeforeSubmit(BatchState &batch);
    void               finishBatch(const BatchPtr &batch);
    void               recordBatchMetrics(const BatchState &batch);

    void                returnSlot(Slot &slot);
    [[nodiscard]] Slot *acquireIdleSlot();
    void                synchronizeSlot(Slot &slot) const noexcept;

    void                     pruneQueuedRequestsLocked(std::vector<std::pair<RequestPtr, FailureKind>> &finished);
    [[nodiscard]] RequestPtr chooseSchedulerSeedLocked() const;
    [[nodiscard]] size_t     compatibleRequestCountLocked(const Request &seed) const;
    [[nodiscard]] std::chrono::steady_clock::time_point batchDeadlineLocked(const Request &seed) const;
    [[nodiscard]] size_t                                preferredBatchSize(size_t compatible_count) const;
    [[nodiscard]] std::vector<RequestPtr>               takeCompatibleBatchLocked(const Request &seed, size_t count);

    void               completeSuccess(const RequestPtr &request, InferenceResult result);
    void               completeFailure(const RequestPtr &request, std::exception_ptr error, FailureKind kind);
    void               failBatch(const BatchPtr &batch, std::exception_ptr error);
    void               failRequests(const std::vector<std::pair<RequestPtr, FailureKind>> &requests);
    void               transitionFailed();
    [[nodiscard]] bool isFailed() const;
    void               recordFault(FaultStage stage, int device_id, const RequestPtr &request, std::exception_ptr error,
                                   bool fatal = false);
    static void        fulfillCompletions(std::vector<PendingCompletion> completions);
    void collectOrderedCompletionsLocked(std::vector<PendingCompletion> &ready, const std::string &source_id);

    EngineConfig                                                        config_;
    std::shared_ptr<const PipelinePlan>                                 pipeline_;
    std::unordered_map<int, std::shared_ptr<priv::TensorRTRuntimePlan>> runtime_plans_;
    int                                                                 fixed_batch_size_{0};
    size_t                                                              max_inflight_batches_{0};

    mutable std::mutex                                                     mutex_;
    std::condition_variable                                                request_cv_;
    std::condition_variable                                                space_cv_;
    std::condition_variable                                                idle_cv_;
    std::condition_variable                                                batch_space_cv_;
    std::deque<RequestPtr>                                                 requests_;
    std::unordered_map<uint64_t, std::weak_ptr<Request>>                   active_requests_;
    std::deque<Slot *>                                                     idle_slots_;
    std::vector<std::unique_ptr<Slot>>                                     slots_;
    std::thread                                                            scheduler_;
    std::thread                                                            gpu_dispatcher_;
    std::thread                                                            completion_poller_;
    std::vector<std::thread>                                               prepare_workers_;
    std::vector<std::thread>                                               postprocess_workers_;
    uint64_t                                                               next_request_id_{1};
    uint64_t                                                               accepted_requests_{0};
    uint64_t                                                               completed_requests_{0};
    uint64_t                                                               failed_requests_{0};
    uint64_t                                                               cancelled_requests_{0};
    uint64_t                                                               timed_out_requests_{0};
    uint64_t                                                               rejected_requests_{0};
    uint64_t                                                               dropped_requests_{0};
    uint64_t                                                               completed_batches_{0};
    size_t                                                                 queued_high_watermark_{0};
    size_t                                                                 inflight_batches_{0};
    size_t                                                                 device_arena_bytes_{0};
    std::unordered_map<int, size_t>                                        device_arena_bytes_by_device_;
    size_t                                                                 pinned_memory_capacity_bytes_{0};
    EngineState                                                            state_{EngineState::Created};
    StageLatencySnapshot                                                   queue_latency_;
    StageLatencySnapshot                                                   cpu_preprocess_latency_;
    StageLatencySnapshot                                                   gpu_latency_;
    StageLatencySnapshot                                                   cpu_postprocess_latency_;
    std::unordered_map<std::string, uint64_t>                              next_source_submit_;
    std::unordered_map<std::string, uint64_t>                              next_source_deliver_;
    std::unordered_map<std::string, std::map<uint64_t, PendingCompletion>> pending_source_completions_;
    std::unordered_map<std::string, std::set<uint64_t>>                    unordered_delivered_sequences_;

    mutable std::mutex      fault_mutex_;
    std::deque<FaultRecord> faults_;

    std::mutex              prepare_mutex_;
    std::condition_variable prepare_cv_;
    std::deque<BatchPtr>    prepare_queue_;
    bool                    prepare_closed_{false};
    std::atomic_size_t      prepare_queue_high_watermark_{0};

    std::mutex              gpu_mutex_;
    std::condition_variable gpu_cv_;
    std::deque<BatchPtr>    gpu_queue_;
    bool                    gpu_closed_{false};
    std::atomic_size_t      gpu_queue_high_watermark_{0};

    std::mutex              postprocess_mutex_;
    std::condition_variable postprocess_cv_;
    std::deque<BatchPtr>    postprocess_queue_;
    bool                    postprocess_closed_{false};
    std::atomic_size_t      postprocess_queue_high_watermark_{0};

    std::mutex              completion_mutex_;
    std::condition_variable completion_cv_;
    bool                    dispatcher_closed_{false};

    std::vector<std::unique_ptr<InputTicket>>  input_tickets_;
    std::vector<std::unique_ptr<OutputTicket>> output_tickets_;
    mutable std::mutex                         input_pool_mutex_;
    std::condition_variable                    input_pool_cv_;
    std::deque<InputTicket *>                  free_input_tickets_;
    size_t                                     input_tickets_in_use_{0};
    size_t                                     input_tickets_high_watermark_{0};
    mutable std::mutex                         output_pool_mutex_;
    std::condition_variable                    output_pool_cv_;
    std::deque<OutputTicket *>                 free_output_tickets_;
    size_t                                     output_tickets_in_use_{0};
    size_t                                     output_tickets_high_watermark_{0};
    std::atomic_bool                           abort_pools_{false};
};

void InferenceEngine::Impl::initializeSlot(Slot &slot)
{
    irt::model::setCudaDevice(slot.device_id);
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.h2d_stream, cudaStreamNonBlocking), "cudaStreamCreate(h2d)");
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.compute_stream, cudaStreamNonBlocking),
                          "cudaStreamCreate(compute)");
    irt::model::checkCuda(cudaStreamCreateWithFlags(&slot.d2h_stream, cudaStreamNonBlocking), "cudaStreamCreate(d2h)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.h2d_done, cudaEventDisableTiming), "cudaEventCreate(h2d)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.compute_done, cudaEventDisableTiming),
                          "cudaEventCreate(compute)");
    irt::model::checkCuda(cudaEventCreateWithFlags(&slot.d2h_done, cudaEventDisableTiming), "cudaEventCreate(d2h)");

    if (!slot.runtime_plan)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ExecutionSlot has no TensorRT runtime plan");
    }
    slot.context                = slot.runtime_plan->createSession();
    slot.output_names           = slot.runtime_plan->outputNames();
    slot.capacity_batch         = fixed_batch_size_ > 0 ? fixed_batch_size_ : config_.max_batch_size;
    const auto &runtime_inputs  = slot.runtime_plan->inputNames();
    const auto &pipeline_inputs = pipeline_->modelInputs();
    if (pipeline_inputs.size() != runtime_inputs.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Pipeline binds %zu model inputs but TensorRT engine has %zu", pipeline_inputs.size(),
                             runtime_inputs.size());
    }
    std::unordered_map<std::string, bool> bound_runtime_inputs;
    for (const auto &binding : pipeline_inputs)
    {
        const auto pipeline_input = pipeline_->tensors().find(binding.tensor_name);
        if (pipeline_input == pipeline_->tensors().end())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pipeline model input is not allocated: %s",
                                 binding.tensor_name.c_str());
        }
        const auto &input_desc = pipeline_input->second;
        if (input_desc.data_type != TensorDataType::F32 || input_desc.layout != TensorLayout::NCHW
            || input_desc.memory_kind != MemoryKind::DEVICE)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input must be a device float32 NCHW tensor: %s",
                                 binding.tensor_name.c_str());
        }
        std::string engine_name = binding.engine_tensor_name;
        if (engine_name.empty())
        {
            if (runtime_inputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Multi-input TensorRT engines require explicit PipelineBuilder::bindModelInput()");
            }
            engine_name = runtime_inputs.front();
        }
        if (std::find(runtime_inputs.begin(), runtime_inputs.end(), engine_name) == runtime_inputs.end()
            || !bound_runtime_inputs.emplace(engine_name, true).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid or duplicate TensorRT input binding: %s",
                                 engine_name.c_str());
        }
        if (binding.tensor_name == pipeline_->modelInput()
            && (input_desc.width != config_.input_width || input_desc.height != config_.input_height
                || input_desc.channels != config_.input_channels))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Primary pipeline model input does not match EngineConfig image input");
        }
        slot.runtime_plan->setInputShape(
            *slot.context, engine_name,
            nvinfer1::Dims4{slot.capacity_batch, input_desc.channels, input_desc.height, input_desc.width});
        slot.model_inputs.emplace_back(binding.tensor_name, std::move(engine_name));
    }

    struct ArenaEntry
    {
        std::string name;
        TensorDesc  desc;
        size_t      bytes{0};
        size_t      first_use{0};
        size_t      last_use{0};
        size_t      offset{0};
    };

    struct ArenaBlock
    {
        size_t offset{0};
        size_t bytes{0};
        size_t last_use{0};
    };

    const size_t                                               terminal_use = pipeline_->nodes().size() + 2;
    std::unordered_map<std::string, std::pair<size_t, size_t>> lifetimes;
    for (const auto &[name, desc] : pipeline_->tensors())
    {
        if (desc.memory_kind == MemoryKind::DEVICE)
        {
            lifetimes.emplace(name, std::make_pair(terminal_use, 0U));
        }
    }
    for (size_t node_index = 0; node_index < pipeline_->nodes().size(); ++node_index)
    {
        const auto note_use = [&](const std::string &name)
        {
            const auto found = lifetimes.find(name);
            if (found == lifetimes.end())
            {
                return;
            }
            found->second.first  = std::min(found->second.first, node_index);
            found->second.second = std::max(found->second.second, node_index);
        };
        for (const auto &name : pipeline_->nodes()[node_index].config.inputs)
        {
            note_use(name);
        }
        for (const auto &name : pipeline_->nodes()[node_index].config.outputs)
        {
            note_use(name);
        }
    }
    // TensorRT consumes the model input between CUDA preprocess and CUDA postprocess; D2H consumes explicit results.
    for (const auto &[tensor_name, _] : slot.model_inputs)
    {
        lifetimes.at(tensor_name).second = terminal_use - 1;
    }
    for (const auto &binding : pipeline_->results())
    {
        lifetimes.at(binding.tensor_name).second = terminal_use;
    }

    std::vector<ArenaEntry> entries;
    for (const auto &[name, desc] : pipeline_->tensors())
    {
        if (desc.memory_kind != MemoryKind::DEVICE)
        {
            continue;
        }
        const auto lifetime = lifetimes.at(name);
        entries.push_back({name, desc, static_cast<size_t>(slot.capacity_batch) * desc.bytesPerRequest(),
                           lifetime.first == terminal_use ? 0U : lifetime.first,
                           std::max(lifetime.first == terminal_use ? 0U : lifetime.first, lifetime.second), 0});
    }
    std::sort(entries.begin(), entries.end(),
              [](const ArenaEntry &lhs, const ArenaEntry &rhs)
              { return lhs.first_use == rhs.first_use ? lhs.name < rhs.name : lhs.first_use < rhs.first_use; });

    size_t                  arena_bytes = 0;
    std::vector<ArenaBlock> blocks;
    for (auto &entry : entries)
    {
        auto selected = blocks.end();
        for (auto block = blocks.begin(); block != blocks.end(); ++block)
        {
            if (block->last_use < entry.first_use && block->bytes >= entry.bytes
                && (selected == blocks.end() || block->bytes < selected->bytes))
            {
                selected = block;
            }
        }
        if (selected == blocks.end())
        {
            arena_bytes  = alignDeviceOffset(arena_bytes);
            entry.offset = arena_bytes;
            blocks.push_back({entry.offset, entry.bytes, entry.last_use});
            arena_bytes += entry.bytes;
        }
        else
        {
            entry.offset       = selected->offset;
            selected->last_use = entry.last_use;
        }
    }

    slot.output_capacity_bytes.reserve(slot.output_names.size());
    for (const auto &name : slot.output_names)
    {
        const auto   shape = slot.runtime_plan->tensorShape(*slot.context, name);
        const size_t bytes = irt::model::elementCount(shape) * sizeof(float);
        arena_bytes        = alignDeviceOffset(arena_bytes);
        slot.output_capacity_bytes.push_back(bytes);
        slot.device_outputs.push_back(nullptr);
        entries.push_back({
            "model." + name,
            {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE,
                     static_cast<int>(bytes / sizeof(float) / static_cast<size_t>(slot.capacity_batch)), 1, 1},
            bytes,
            terminal_use - 1,
            terminal_use,
            arena_bytes
        });
        arena_bytes += bytes;
    }
    if (arena_bytes == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "ExecutionSlot has no device buffers");
    }

    const size_t existing_device_bytes = device_arena_bytes_by_device_[slot.device_id];
    if (config_.device_memory_limit_bytes != 0
        && existing_device_bytes + arena_bytes > config_.device_memory_limit_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                             "Configured device memory pool limit (%zu bytes) is insufficient for execution arenas",
                             config_.device_memory_limit_bytes);
    }
    irt::model::checkCuda(cudaMallocAsync(&slot.device_arena, arena_bytes, slot.compute_stream),
                          "cudaMallocAsync(DeviceArena)");
    irt::model::checkCuda(cudaStreamSynchronize(slot.compute_stream), "cudaStreamSynchronize(DeviceArena)");
    slot.device_arena_bytes = arena_bytes;
    device_arena_bytes_ += arena_bytes;
    device_arena_bytes_by_device_[slot.device_id] += arena_bytes;
    auto *base = static_cast<std::byte *>(slot.device_arena);
    for (const auto &entry : entries)
    {
        void *pointer = base + entry.offset;
        if (entry.name.starts_with("model."))
        {
            const auto output_index = static_cast<size_t>(
                std::distance(slot.output_names.begin(), std::find(slot.output_names.begin(), slot.output_names.end(),
                                                                   entry.name.substr(std::string("model.").size()))));
            slot.device_outputs[output_index] = pointer;
        }
        else
        {
            slot.device_tensors.emplace(entry.name, TensorView{pointer, entry.desc, entry.desc.bytesPerRequest()});
        }
    }
    slot.operators = pipeline_->createOperators();
}

void InferenceEngine::Impl::initializeTickets()
{
    const size_t input_count  = config_.pinned_input_tickets == 0
                                  ? config_.execution_slots + config_.cpu_preprocess_workers
                                  : config_.pinned_input_tickets;
    const size_t output_count = config_.pinned_output_tickets == 0
                                  ? config_.execution_slots + config_.cpu_postprocess_workers
                                  : config_.pinned_output_tickets;

    size_t input_ticket_bytes  = 0;
    size_t output_ticket_bytes = 0;
    for (const auto &[_, desc] : pipeline_->tensors())
    {
        if (desc.memory_kind == MemoryKind::HOST)
        {
            input_ticket_bytes += static_cast<size_t>(config_.max_batch_size) * desc.bytesPerRequest();
        }
    }
    const auto &reference_slot = *slots_.front();
    for (const size_t bytes : reference_slot.output_capacity_bytes)
    {
        output_ticket_bytes += bytes;
    }
    for (const auto &binding : pipeline_->results())
    {
        output_ticket_bytes += static_cast<size_t>(config_.max_batch_size)
                             * pipeline_->tensors().at(binding.tensor_name).bytesPerRequest();
    }
    if (input_ticket_bytes > std::numeric_limits<size_t>::max() / input_count
        || output_ticket_bytes > std::numeric_limits<size_t>::max() / output_count)
    {
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY, "Pinned ticket pool capacity overflows size_t");
    }
    pinned_memory_capacity_bytes_ = input_ticket_bytes * input_count + output_ticket_bytes * output_count;
    if (config_.pinned_memory_limit_bytes != 0 && pinned_memory_capacity_bytes_ > config_.pinned_memory_limit_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                             "Configured pinned memory pool limit (%zu bytes) is insufficient for %zu bytes",
                             config_.pinned_memory_limit_bytes, pinned_memory_capacity_bytes_);
    }

    input_tickets_.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index)
    {
        auto ticket = std::make_unique<InputTicket>();
        for (const auto &[name, desc] : pipeline_->tensors())
        {
            if (desc.memory_kind != MemoryKind::HOST)
            {
                continue;
            }
            auto [buffer, _]   = ticket->buffers.try_emplace(name, nvinfer1::DataType::kUINT8);
            const size_t bytes = static_cast<size_t>(config_.max_batch_size) * desc.bytesPerRequest();
            buffer->second.resize(bytes, nvinfer1::DataType::kUINT8);
            ticket->tensors.emplace(name, TensorView{buffer->second.data(), desc, desc.bytesPerRequest()});
        }
        free_input_tickets_.push_back(ticket.get());
        input_tickets_.push_back(std::move(ticket));
    }

    output_tickets_.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index)
    {
        auto ticket = std::make_unique<OutputTicket>();
        ticket->model_outputs.resize(reference_slot.output_capacity_bytes.size());
        for (size_t output_index = 0; output_index < reference_slot.output_capacity_bytes.size(); ++output_index)
        {
            ticket->model_outputs[output_index].resize(reference_slot.output_capacity_bytes[output_index],
                                                       nvinfer1::DataType::kUINT8);
        }
        for (const auto &binding : pipeline_->results())
        {
            const auto &desc = pipeline_->tensors().at(binding.tensor_name);
            auto [buffer, _] = ticket->result_outputs.try_emplace(binding.result_name, nvinfer1::DataType::kUINT8);
            buffer->second.resize(static_cast<size_t>(config_.max_batch_size) * desc.bytesPerRequest(),
                                  nvinfer1::DataType::kUINT8);
        }
        free_output_tickets_.push_back(ticket.get());
        output_tickets_.push_back(std::move(ticket));
    }
}

InferenceEngine::Impl::InputTicket *InferenceEngine::Impl::acquireInputTicket()
{
    std::unique_lock lock(input_pool_mutex_);
    input_pool_cv_.wait(lock, [this] { return !free_input_tickets_.empty() || abort_pools_.load(); });
    if (free_input_tickets_.empty())
    {
        return nullptr;
    }
    auto *ticket = free_input_tickets_.front();
    free_input_tickets_.pop_front();
    ++input_tickets_in_use_;
    input_tickets_high_watermark_ = std::max(input_tickets_high_watermark_, input_tickets_in_use_);
    return ticket;
}

InferenceEngine::Impl::OutputTicket *InferenceEngine::Impl::acquireOutputTicket()
{
    std::unique_lock lock(output_pool_mutex_);
    output_pool_cv_.wait(lock, [this] { return !free_output_tickets_.empty() || abort_pools_.load(); });
    if (free_output_tickets_.empty())
    {
        return nullptr;
    }
    auto *ticket = free_output_tickets_.front();
    free_output_tickets_.pop_front();
    ++output_tickets_in_use_;
    output_tickets_high_watermark_ = std::max(output_tickets_high_watermark_, output_tickets_in_use_);
    return ticket;
}

void InferenceEngine::Impl::releaseInputTicket(InputTicket *ticket)
{
    if (ticket == nullptr)
    {
        return;
    }
    {
        std::lock_guard lock(input_pool_mutex_);
        free_input_tickets_.push_back(ticket);
        if (input_tickets_in_use_ > 0)
        {
            --input_tickets_in_use_;
        }
    }
    input_pool_cv_.notify_one();
}

void InferenceEngine::Impl::releaseOutputTicket(OutputTicket *ticket)
{
    if (ticket == nullptr)
    {
        return;
    }
    {
        std::lock_guard lock(output_pool_mutex_);
        free_output_tickets_.push_back(ticket);
        if (output_tickets_in_use_ > 0)
        {
            --output_tickets_in_use_;
        }
    }
    output_pool_cv_.notify_one();
}

void InferenceEngine::Impl::start()
{
    std::lock_guard lock(mutex_);
    if (state_ == EngineState::Running || state_ == EngineState::Starting || state_ == EngineState::Draining)
    {
        return;
    }
    if (state_ == EngineState::Failed || state_ == EngineState::Stopped)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                             "A stopped or failed InferenceEngine must be recreated");
    }
    config_.validate();
    state_ = EngineState::Starting;

    try
    {
        abort_pools_        = false;
        device_arena_bytes_ = 0;
        device_arena_bytes_by_device_.clear();
        pinned_memory_capacity_bytes_ = 0;
        std::vector<int> device_ids   = config_.device_ids;
        if (device_ids.empty())
        {
            device_ids.push_back(config_.device_id);
        }
        for (const int device_id : device_ids)
        {
            if (!runtime_plans_.contains(device_id))
            {
                runtime_plans_.emplace(device_id, std::make_shared<priv::TensorRTRuntimePlan>(config_, device_id));
            }
        }
        fixed_batch_size_ = runtime_plans_.at(device_ids.front())->fixedBatchSize();
        for (const int device_id : device_ids)
        {
            if (runtime_plans_.at(device_id)->fixedBatchSize() != fixed_batch_size_)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "TensorRT runtime plans disagree on fixed batch size");
            }
        }
        if (fixed_batch_size_ > 0 && config_.max_batch_size > fixed_batch_size_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured max batch %d exceeds fixed TensorRT batch %d", config_.max_batch_size,
                                 fixed_batch_size_);
        }

        slots_.reserve(config_.execution_slots);
        for (size_t index = 0; index < config_.execution_slots; ++index)
        {
            const int device_id = device_ids[index % device_ids.size()];
            auto      slot      = std::make_unique<Slot>(device_id);
            slot->runtime_plan  = runtime_plans_.at(device_id);
            initializeSlot(*slot);
            idle_slots_.push_back(slot.get());
            slots_.push_back(std::move(slot));
        }
        initializeTickets();
        max_inflight_batches_
            = config_.execution_slots + config_.cpu_preprocess_workers + config_.cpu_postprocess_workers;

        state_ = EngineState::Running;
        for (size_t index = 0; index < config_.cpu_preprocess_workers; ++index)
        {
            prepare_workers_.emplace_back([this, operators = pipeline_->createOperators()]() mutable
                                          { prepareLoop(std::move(operators)); });
        }
        for (size_t index = 0; index < config_.cpu_postprocess_workers; ++index)
        {
            postprocess_workers_.emplace_back([this, operators = pipeline_->createOperators()]() mutable
                                              { postprocessLoop(std::move(operators)); });
        }
        gpu_dispatcher_    = std::thread([this] { gpuDispatcherLoop(); });
        completion_poller_ = std::thread([this] { completionPollerLoop(); });
        scheduler_         = std::thread([this] { schedulerLoop(); });
    }
    catch (...)
    {
        recordFault(FaultStage::Start, -1, nullptr, std::current_exception(), true);
        abort_pools_ = true;
        slots_.clear();
        idle_slots_.clear();
        input_tickets_.clear();
        output_tickets_.clear();
        free_input_tickets_.clear();
        free_output_tickets_.clear();
        runtime_plans_.clear();
        device_arena_bytes_by_device_.clear();
        state_ = EngineState::Failed;
        throw;
    }
}

InferenceEngine::Impl::SubmittedRequest InferenceEngine::Impl::submit(const cv::Mat &image, TensorInputMap inputs,
                                                                      RequestOptions options)
{
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (options.deadline.count() < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Request deadline must not be negative");
    }

    auto request                   = std::make_shared<Request>();
    request->image                 = image.clone();
    request->extra_inputs          = std::move(inputs);
    request->submitted             = std::chrono::steady_clock::now();
    request->priority              = options.priority;
    request->source_id             = options.source_id.empty() ? "default" : std::move(options.source_id);
    request->preserve_source_order = options.preserve_source_order;
    if (!options.compatibility_key.empty())
    {
        request->compatibility_key = std::move(options.compatibility_key);
    }
    if (options.deadline.count() > 0)
    {
        request->deadline = request->submitted + options.deadline;
    }
    auto       future = request->promise.get_future();
    RequestPtr dropped;
    bool       drop_newest = false;

    {
        std::unique_lock lock(mutex_);
        if (state_ != EngineState::Running)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference engine is not accepting requests");
        }
        request->id              = next_request_id_++;
        request->source_sequence = next_source_submit_[request->source_id]++;
        next_source_deliver_.try_emplace(request->source_id, 0);

        while (requests_.size() >= config_.queue_capacity)
        {
            switch (config_.queue_policy)
            {
            case QueuePolicy::Reject:
                ++rejected_requests_;
                throw irt::Exception(irt::Status::ERROR_NOT_READY, "Inference request queue is full");
            case QueuePolicy::Block:
                space_cv_.wait(lock, [this]
                               { return requests_.size() < config_.queue_capacity || state_ != EngineState::Running; });
                if (state_ != EngineState::Running)
                {
                    throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                         "Inference engine is not accepting requests");
                }
                break;
            case QueuePolicy::DropOldest:
                dropped = std::move(requests_.front());
                requests_.pop_front();
                break;
            case QueuePolicy::DropNewest:
                drop_newest = true;
                break;
            }
            if (dropped || drop_newest)
            {
                break;
            }
        }

        if (!drop_newest)
        {
            active_requests_.emplace(request->id, request);
            requests_.push_back(request);
            ++accepted_requests_;
            queued_high_watermark_ = std::max(queued_high_watermark_, requests_.size());
        }
    }

    if (dropped)
    {
        completeFailure(dropped, droppedError(), FailureKind::Dropped);
    }
    if (drop_newest)
    {
        completeFailure(request, droppedError(), FailureKind::Dropped);
    }
    else
    {
        request_cv_.notify_one();
    }
    return {request->id, std::move(future)};
}

InferenceResult InferenceEngine::Impl::infer(const cv::Mat &image, const std::chrono::milliseconds timeout)
{
    auto submitted = submit(image, {}, {});
    auto future    = std::move(submitted.future);
    if (future.wait_for(timeout) != std::future_status::ready)
    {
        cancel(submitted.id);
        throw irt::Exception(irt::Status::ERROR_NOT_READY, "Inference request timed out");
    }
    return future.get();
}

bool InferenceEngine::Impl::cancel(const uint64_t request_id)
{
    RequestPtr request;
    {
        std::lock_guard lock(mutex_);
        const auto      found = active_requests_.find(request_id);
        if (found == active_requests_.end())
        {
            return false;
        }
        request = found->second.lock();
    }
    if (!request || request->finished || request->cancelled.exchange(true))
    {
        return false;
    }
    request_cv_.notify_one();
    return true;
}

void InferenceEngine::Impl::pruneQueuedRequestsLocked(std::vector<std::pair<RequestPtr, FailureKind>> &finished)
{
    const auto now = std::chrono::steady_clock::now();
    for (auto it = requests_.begin(); it != requests_.end();)
    {
        if ((*it)->cancelled.load())
        {
            finished.emplace_back(std::move(*it), FailureKind::Cancelled);
            it = requests_.erase(it);
        }
        else if ((*it)->deadline <= now)
        {
            finished.emplace_back(std::move(*it), FailureKind::TimedOut);
            it = requests_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

InferenceEngine::Impl::RequestPtr InferenceEngine::Impl::chooseSchedulerSeedLocked() const
{
    RequestPtr selected;
    for (const auto &request : requests_)
    {
        if (!selected || request->priority > selected->priority
            || (request->priority == selected->priority && request->submitted < selected->submitted))
        {
            selected = request;
        }
    }
    return selected;
}

size_t InferenceEngine::Impl::compatibleRequestCountLocked(const Request &seed) const
{
    return static_cast<size_t>(std::count_if(requests_.begin(), requests_.end(),
                                             [&seed](const RequestPtr &request) {
                                                 return request->priority == seed.priority
                                                     && request->compatibility_key == seed.compatibility_key;
                                             }));
}

std::chrono::steady_clock::time_point InferenceEngine::Impl::batchDeadlineLocked(const Request &seed) const
{
    auto deadline = seed.submitted + config_.max_wait;
    for (const auto &request : requests_)
    {
        if (request->priority == seed.priority && request->compatibility_key == seed.compatibility_key)
        {
            deadline = std::min(deadline, request->submitted + config_.max_wait);
            deadline = std::min(deadline, request->deadline);
        }
    }
    return deadline;
}

size_t InferenceEngine::Impl::preferredBatchSize(const size_t compatible_count) const
{
    const size_t maximum = std::min(compatible_count, static_cast<size_t>(config_.max_batch_size));
    if (config_.preferred_batch_sizes.empty())
    {
        return maximum;
    }
    size_t preferred = 0;
    for (const int size : config_.preferred_batch_sizes)
    {
        if (size <= static_cast<int>(maximum))
        {
            preferred = std::max(preferred, static_cast<size_t>(size));
        }
    }
    return preferred == 0 ? maximum : preferred;
}

std::vector<InferenceEngine::Impl::RequestPtr> InferenceEngine::Impl::takeCompatibleBatchLocked(const Request &seed,
                                                                                                const size_t   count)
{
    std::vector<RequestPtr> batch;
    batch.reserve(count);
    for (auto it = requests_.begin(); it != requests_.end() && batch.size() < count;)
    {
        if ((*it)->priority == seed.priority && (*it)->compatibility_key == seed.compatibility_key)
        {
            batch.push_back(std::move(*it));
            it = requests_.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return batch;
}

void InferenceEngine::Impl::schedulerLoop()
{
    for (;;)
    {
        std::vector<std::pair<RequestPtr, FailureKind>> finished;
        BatchPtr                                        batch;
        bool                                            notify_space = false;

        {
            std::unique_lock lock(mutex_);
            for (;;)
            {
                pruneQueuedRequestsLocked(finished);
                if (!finished.empty())
                {
                    notify_space = true;
                    break;
                }
                if (state_ == EngineState::Failed)
                {
                    for (auto &request : requests_)
                    {
                        finished.emplace_back(std::move(request), FailureKind::Failed);
                    }
                    requests_.clear();
                    notify_space = true;
                    break;
                }
                if (requests_.empty())
                {
                    if (state_ != EngineState::Running)
                    {
                        break;
                    }
                    request_cv_.wait(lock);
                    continue;
                }
                if (inflight_batches_ >= max_inflight_batches_)
                {
                    batch_space_cv_.wait(lock);
                    continue;
                }

                const auto   seed             = chooseSchedulerSeedLocked();
                const auto   compatible_count = compatibleRequestCountLocked(*seed);
                const size_t target           = config_.preferred_batch_sizes.empty()
                                                  ? static_cast<size_t>(config_.max_batch_size)
                                                  : static_cast<size_t>(*std::max_element(config_.preferred_batch_sizes.begin(),
                                                                                          config_.preferred_batch_sizes.end()));
                const auto   deadline         = batchDeadlineLocked(*seed);
                const bool   draining         = state_ == EngineState::Draining;
                if (!draining && compatible_count < target && std::chrono::steady_clock::now() < deadline)
                {
                    request_cv_.wait_until(lock, deadline);
                    continue;
                }

                const size_t count = draining ? std::min(compatible_count, static_cast<size_t>(config_.max_batch_size))
                                              : preferredBatchSize(compatible_count);
                batch              = std::make_shared<BatchState>();
                batch->requests    = takeCompatibleBatchLocked(*seed, count);
                batch->scheduled   = std::chrono::steady_clock::now();
                ++inflight_batches_;
                notify_space = true;
                break;
            }
        }

        if (notify_space)
        {
            space_cv_.notify_all();
        }
        if (!finished.empty())
        {
            failRequests(finished);
            if (!batch)
            {
                continue;
            }
        }
        if (!batch)
        {
            return;
        }
        {
            std::lock_guard lock(prepare_mutex_);
            prepare_queue_.push_back(std::move(batch));
            prepare_queue_high_watermark_ = std::max(prepare_queue_high_watermark_.load(), prepare_queue_.size());
        }
        prepare_cv_.notify_one();
    }
}

bool InferenceEngine::Impl::removeInvalidBeforeSubmit(BatchState &batch)
{
    const auto now           = std::chrono::steady_clock::now();
    auto       first_invalid = std::remove_if(batch.requests.begin(), batch.requests.end(),
                                              [&](const RequestPtr &request)
                                              {
                                            if (request->cancelled.load())
                                            {
                                                completeFailure(request, cancelledError(), FailureKind::Cancelled);
                                                return true;
                                            }
                                            if (request->deadline <= now)
                                            {
                                                completeFailure(request, timeoutError(), FailureKind::TimedOut);
                                                return true;
                                            }
                                            return false;
                                        });
    batch.requests.erase(first_invalid, batch.requests.end());
    return !batch.requests.empty();
}

bool InferenceEngine::Impl::prepareBatch(const BatchPtr &batch)
{
    if (!removeInvalidBeforeSubmit(*batch))
    {
        finishBatch(batch);
        return false;
    }
    batch->actual_batch = static_cast<int>(batch->requests.size());
    if (batch->actual_batch < config_.min_batch_size)
    {
        failBatch(batch, std::make_exception_ptr(irt::Exception(
                             irt::Status::ERROR_NOT_READY, "Not enough requests to satisfy the minimum batch size")));
        finishBatch(batch);
        return false;
    }
    if (fixed_batch_size_ > 0)
    {
        if (batch->actual_batch != fixed_batch_size_ && config_.static_batch_policy == StaticBatchPolicy::Reject)
        {
            failBatch(batch, std::make_exception_ptr(irt::Exception(irt::Status::ERROR_NOT_READY,
                                                                    "Fixed TensorRT batch requires padding")));
            finishBatch(batch);
            return false;
        }
        batch->execution_batch = fixed_batch_size_;
    }
    else
    {
        batch->execution_batch = batch->actual_batch;
    }
    return true;
}

void InferenceEngine::Impl::prepareLoop(std::vector<std::unique_ptr<IOperator>> operators)
{
    for (;;)
    {
        BatchPtr batch;
        {
            std::unique_lock lock(prepare_mutex_);
            prepare_cv_.wait(lock, [this] { return !prepare_queue_.empty() || prepare_closed_; });
            if (prepare_queue_.empty() && prepare_closed_)
            {
                return;
            }
            batch = std::move(prepare_queue_.front());
            prepare_queue_.pop_front();
        }
        if (isFailed())
        {
            failBatch(batch, std::make_exception_ptr(
                                 irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference engine failed")));
            finishBatch(batch);
            continue;
        }
        if (!prepareBatch(batch))
        {
            continue;
        }

        try
        {
            auto *ticket = acquireInputTicket();
            if (ticket == nullptr)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Pinned input pool is stopping");
            }
            batch->input_ticket           = ticket;
            batch->cpu_preprocess_started = std::chrono::steady_clock::now();
            ResultMap ignored;
            executeStage(operators, PipelineStage::CPU_PREPROCESS, *batch, config_.device_id, nullptr, ticket->tensors,
                         ignored);
            batch->cpu_preprocess_finished = std::chrono::steady_clock::now();
            {
                std::lock_guard lock(gpu_mutex_);
                gpu_queue_.push_back(std::move(batch));
                gpu_queue_high_watermark_ = std::max(gpu_queue_high_watermark_.load(), gpu_queue_.size());
            }
            gpu_cv_.notify_one();
        }
        catch (...)
        {
            const auto error = std::current_exception();
            recordFault(FaultStage::CpuPreprocess, -1, batch->requests.empty() ? nullptr : batch->requests.front(),
                        error);
            releaseInputTicket(batch->input_ticket);
            batch->input_ticket = nullptr;
            failBatch(batch, error);
            finishBatch(batch);
        }
    }
}

void InferenceEngine::Impl::executeStage(const std::vector<std::unique_ptr<IOperator>> &operators,
                                         const PipelineStage stage, const BatchState &batch, const int device_id,
                                         const cudaStream_t stream, TensorViewMap &tensors, ResultMap &results) const
{
    for (size_t node_index = 0; node_index < pipeline_->nodes().size(); ++node_index)
    {
        if (pipeline_->nodes()[node_index].contract.stage != stage)
        {
            continue;
        }
        for (int request_index = 0; request_index < batch.actual_batch; ++request_index)
        {
            const auto     &request = batch.requests[static_cast<size_t>(request_index)];
            OperatorContext context{device_id, batch.actual_batch,    request_index, stream, &request->image, tensors,
                                    results,   &request->extra_inputs};
            operators[node_index]->execute(context);
        }
    }
}

InferenceEngine::Impl::Slot *InferenceEngine::Impl::acquireIdleSlot()
{
    std::unique_lock lock(mutex_);
    idle_cv_.wait(lock, [this] { return !idle_slots_.empty() || state_ == EngineState::Failed; });
    if (idle_slots_.empty())
    {
        return nullptr;
    }
    auto *slot = idle_slots_.front();
    idle_slots_.pop_front();
    return slot;
}

void InferenceEngine::Impl::gpuDispatcherLoop()
{
    for (;;)
    {
        BatchPtr batch;
        {
            std::unique_lock lock(gpu_mutex_);
            gpu_cv_.wait(lock, [this] { return !gpu_queue_.empty() || gpu_closed_; });
            if (gpu_queue_.empty() && gpu_closed_)
            {
                break;
            }
            batch = std::move(gpu_queue_.front());
            gpu_queue_.pop_front();
        }
        if (isFailed())
        {
            releaseInputTicket(batch->input_ticket);
            batch->input_ticket = nullptr;
            failBatch(batch, std::make_exception_ptr(
                                 irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference engine failed")));
            finishBatch(batch);
            continue;
        }

        auto *output_ticket = acquireOutputTicket();
        if (output_ticket == nullptr)
        {
            releaseInputTicket(batch->input_ticket);
            batch->input_ticket = nullptr;
            failBatch(batch, std::make_exception_ptr(irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                                                    "Pinned output pool is stopping")));
            finishBatch(batch);
            continue;
        }
        batch->output_ticket = output_ticket;

        auto *slot = acquireIdleSlot();
        if (slot == nullptr)
        {
            releaseInputTicket(batch->input_ticket);
            releaseOutputTicket(batch->output_ticket);
            batch->input_ticket  = nullptr;
            batch->output_ticket = nullptr;
            failBatch(batch, std::make_exception_ptr(
                                 irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference engine failed")));
            finishBatch(batch);
            continue;
        }
        try
        {
            dispatchGpu(*slot, batch);
            completion_cv_.notify_one();
        }
        catch (...)
        {
            const auto error = std::current_exception();
            recordFault(FaultStage::GpuDispatch, slot->device_id,
                        batch->requests.empty() ? nullptr : batch->requests.front(), error, isFailed());
            synchronizeSlot(*slot);
            releaseInputTicket(batch->input_ticket);
            releaseOutputTicket(batch->output_ticket);
            batch->input_ticket  = nullptr;
            batch->output_ticket = nullptr;
            failBatch(batch, error);
            finishBatch(batch);
            returnSlot(*slot);
        }
    }
    {
        std::lock_guard lock(completion_mutex_);
        dispatcher_closed_ = true;
    }
    completion_cv_.notify_one();
}

void InferenceEngine::Impl::dispatchGpu(Slot &slot, const BatchPtr &batch)
{
    TensorViewMap tensors = slot.device_tensors;
    for (const auto &[name, view] : batch->input_ticket->tensors)
    {
        tensors.emplace(name, view);
    }
    ResultMap ignored;
    irt::model::setCudaDevice(slot.device_id);
    executeStage(slot.operators, PipelineStage::H2D, *batch, slot.device_id, slot.h2d_stream, tensors, ignored);

    std::vector<std::pair<std::string, void *>> runtime_inputs;
    runtime_inputs.reserve(slot.model_inputs.size());
    for (const auto &[pipeline_name, runtime_name] : slot.model_inputs)
    {
        const auto model_input = tensors.find(pipeline_name);
        if (model_input == tensors.end())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pipeline model input is unavailable on the device: %s",
                                 pipeline_name.c_str());
        }
        if (batch->execution_batch > batch->actual_batch)
        {
            auto *tail = static_cast<uint8_t *>(model_input->second.data)
                       + static_cast<size_t>(batch->actual_batch) * model_input->second.bytes_per_request;
            const size_t bytes = static_cast<size_t>(batch->execution_batch - batch->actual_batch)
                               * model_input->second.bytes_per_request;
            irt::model::checkCuda(cudaMemsetAsync(tail, 0, bytes, slot.h2d_stream),
                                  "cudaMemsetAsync(static batch padding)");
        }
        runtime_inputs.emplace_back(runtime_name, model_input->second.data);
    }
    irt::model::checkCuda(cudaEventRecord(slot.h2d_done, slot.h2d_stream), "cudaEventRecord(H2D)");
    irt::model::checkCuda(cudaStreamWaitEvent(slot.compute_stream, slot.h2d_done, 0), "cudaStreamWaitEvent(H2D)");
    executeStage(slot.operators, PipelineStage::CUDA_PREPROCESS, *batch, slot.device_id, slot.compute_stream, tensors,
                 ignored);

    for (const auto &[pipeline_name, runtime_name] : slot.model_inputs)
    {
        const auto &desc = pipeline_->tensors().at(pipeline_name);
        slot.runtime_plan->setInputShape(
            *slot.context, runtime_name,
            nvinfer1::Dims4{batch->execution_batch, desc.channels, desc.height, desc.width});
    }
    slot.runtime_plan->enqueue(*slot.context, runtime_inputs, slot.device_outputs, slot.compute_stream);

    batch->output_elements_per_request.clear();
    batch->output_elements_per_request.reserve(slot.output_names.size());
    for (size_t index = 0; index < slot.output_names.size(); ++index)
    {
        const size_t elements
            = irt::model::elementCount(slot.runtime_plan->tensorShape(*slot.context, slot.output_names[index]));
        if (elements % static_cast<size_t>(batch->execution_batch) != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Unexpected output shape for tensor: %s",
                                 slot.output_names[index].c_str());
        }
        const size_t per_request = elements / static_cast<size_t>(batch->execution_batch);
        const size_t bytes       = per_request * sizeof(float);
        if (static_cast<size_t>(batch->execution_batch) * bytes > slot.output_capacity_bytes[index])
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Output exceeds DeviceArena capacity: %s",
                                 slot.output_names[index].c_str());
        }
        tensors.insert_or_assign(
            "model." + slot.output_names[index],
            TensorView{
                slot.device_outputs[index],
                {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, static_cast<int>(per_request), 1, 1},
                bytes
        });
        batch->output_elements_per_request.push_back(per_request);
    }
    executeStage(slot.operators, PipelineStage::CUDA_POSTPROCESS, *batch, slot.device_id, slot.compute_stream, tensors,
                 ignored);
    irt::model::checkCuda(cudaEventRecord(slot.compute_done, slot.compute_stream), "cudaEventRecord(compute)");
    irt::model::checkCuda(cudaStreamWaitEvent(slot.d2h_stream, slot.compute_done, 0), "cudaStreamWaitEvent(compute)");

    for (size_t index = 0; index < slot.output_names.size(); ++index)
    {
        const size_t bytes
            = static_cast<size_t>(batch->actual_batch) * batch->output_elements_per_request[index] * sizeof(float);
        auto &destination = batch->output_ticket->model_outputs[index];
        if (bytes > destination.capacity())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pinned output ticket is too small: %s",
                                 slot.output_names[index].c_str());
        }
        destination.resize(bytes, nvinfer1::DataType::kUINT8);
        irt::model::checkCuda(cudaMemcpyAsync(destination.data(), slot.device_outputs[index], bytes,
                                              cudaMemcpyDeviceToHost, slot.d2h_stream),
                              "cudaMemcpyAsync(model D2H)");
    }
    for (const auto &binding : pipeline_->results())
    {
        const auto  &source      = tensors.at(binding.tensor_name);
        const size_t bytes       = static_cast<size_t>(batch->actual_batch) * source.bytes_per_request;
        auto        &destination = batch->output_ticket->result_outputs.at(binding.result_name);
        if (bytes > destination.capacity())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pinned result ticket is too small: %s",
                                 binding.result_name.c_str());
        }
        destination.resize(bytes, nvinfer1::DataType::kUINT8);
        irt::model::checkCuda(
            cudaMemcpyAsync(destination.data(), source.data, bytes, cudaMemcpyDeviceToHost, slot.d2h_stream),
            "cudaMemcpyAsync(pipeline result D2H)");
    }
    irt::model::checkCuda(cudaEventRecord(slot.d2h_done, slot.d2h_stream), "cudaEventRecord(D2H)");

    batch->gpu_submitted = std::chrono::steady_clock::now();
    batch->device_id     = slot.device_id;
    batch->output_names  = slot.output_names;
    std::lock_guard lock(slot.mutex);
    slot.active_batch   = batch;
    slot.input_released = false;
}

void InferenceEngine::Impl::completionPollerLoop()
{
    for (;;)
    {
        bool active = false;
        for (const auto &slot_owner : slots_)
        {
            auto    &slot = *slot_owner;
            BatchPtr batch;
            bool     input_released = false;
            {
                std::lock_guard lock(slot.mutex);
                batch          = slot.active_batch;
                input_released = slot.input_released;
            }
            if (!batch)
            {
                continue;
            }
            irt::model::setCudaDevice(slot.device_id);
            active = true;

            if (!input_released)
            {
                const auto status = cudaEventQuery(slot.h2d_done);
                if (status == cudaSuccess)
                {
                    releaseInputTicket(batch->input_ticket);
                    batch->input_ticket = nullptr;
                    std::lock_guard lock(slot.mutex);
                    slot.input_released = true;
                }
                else if (status != cudaErrorNotReady)
                {
                    if (isFatalCudaError(status))
                    {
                        transitionFailed();
                    }
                    synchronizeSlot(slot);
                    releaseInputTicket(batch->input_ticket);
                    releaseOutputTicket(batch->output_ticket);
                    batch->input_ticket  = nullptr;
                    batch->output_ticket = nullptr;
                    failBatch(batch, std::make_exception_ptr(
                                         irt::Exception(irt::Status::ERROR_INTERNAL, "H2D completion query failed")));
                    recordFault(FaultStage::CompletionPoll, slot.device_id,
                                batch->requests.empty() ? nullptr : batch->requests.front(),
                                std::make_exception_ptr(
                                    irt::Exception(irt::Status::ERROR_INTERNAL, "H2D completion query failed")),
                                isFatalCudaError(status));
                    {
                        std::lock_guard lock(slot.mutex);
                        slot.active_batch.reset();
                    }
                    finishBatch(batch);
                    returnSlot(slot);
                    continue;
                }
            }

            const auto status = cudaEventQuery(slot.d2h_done);
            if (status == cudaSuccess)
            {
                batch->gpu_finished = std::chrono::steady_clock::now();
                if (batch->input_ticket != nullptr)
                {
                    releaseInputTicket(batch->input_ticket);
                    batch->input_ticket = nullptr;
                }
                {
                    std::lock_guard lock(slot.mutex);
                    slot.active_batch.reset();
                }
                returnSlot(slot);
                {
                    std::lock_guard lock(postprocess_mutex_);
                    postprocess_queue_.push_back(std::move(batch));
                    postprocess_queue_high_watermark_
                        = std::max(postprocess_queue_high_watermark_.load(), postprocess_queue_.size());
                }
                postprocess_cv_.notify_one();
            }
            else if (status != cudaErrorNotReady)
            {
                if (isFatalCudaError(status))
                {
                    transitionFailed();
                }
                synchronizeSlot(slot);
                releaseInputTicket(batch->input_ticket);
                releaseOutputTicket(batch->output_ticket);
                batch->input_ticket  = nullptr;
                batch->output_ticket = nullptr;
                failBatch(batch, std::make_exception_ptr(
                                     irt::Exception(irt::Status::ERROR_INTERNAL, "D2H completion query failed")));
                recordFault(
                    FaultStage::CompletionPoll, slot.device_id,
                    batch->requests.empty() ? nullptr : batch->requests.front(),
                    std::make_exception_ptr(irt::Exception(irt::Status::ERROR_INTERNAL, "D2H completion query failed")),
                    isFatalCudaError(status));
                {
                    std::lock_guard lock(slot.mutex);
                    slot.active_batch.reset();
                }
                finishBatch(batch);
                returnSlot(slot);
            }
        }

        bool dispatcher_closed = false;
        {
            std::lock_guard lock(completion_mutex_);
            dispatcher_closed = dispatcher_closed_;
        }
        if (dispatcher_closed && !active)
        {
            return;
        }
        std::unique_lock lock(completion_mutex_);
        completion_cv_.wait_for(lock, std::chrono::milliseconds(1));
    }
}

void InferenceEngine::Impl::processPostprocess(const std::vector<std::unique_ptr<IOperator>> &operators,
                                               const BatchPtr                                &batch)
{
    TensorViewMap tensors;
    for (size_t index = 0; index < batch->output_names.size(); ++index)
    {
        const size_t bytes = batch->output_elements_per_request[index] * sizeof(float);
        tensors.emplace("model." + batch->output_names[index],
                        TensorView{
                            batch->output_ticket->model_outputs[index].data(),
                            {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST,
                                                                          static_cast<int>(batch->output_elements_per_request[index]), 1, 1},
                            bytes
        });
    }
    for (const auto &binding : pipeline_->results())
    {
        auto desc        = pipeline_->tensors().at(binding.tensor_name);
        desc.memory_kind = MemoryKind::HOST;
        tensors.emplace("result." + binding.result_name,
                        TensorView{batch->output_ticket->result_outputs.at(binding.result_name).data(), desc,
                                   desc.bytesPerRequest()});
    }

    for (int batch_index = 0; batch_index < batch->actual_batch; ++batch_index)
    {
        const auto &request = batch->requests[static_cast<size_t>(batch_index)];
        if (request->cancelled.load())
        {
            completeFailure(request, cancelledError(), FailureKind::Cancelled);
            continue;
        }
        if (request->deadline <= std::chrono::steady_clock::now())
        {
            completeFailure(request, timeoutError(), FailureKind::TimedOut);
            continue;
        }

        InferenceResult result;
        result.request_id      = request->id;
        result.source_id       = request->source_id;
        result.source_sequence = request->source_sequence;
        for (size_t output_index = 0; output_index < batch->output_names.size(); ++output_index)
        {
            const size_t elements = batch->output_elements_per_request[output_index];
            const auto  *source   = static_cast<const float *>(batch->output_ticket->model_outputs[output_index].data())
                               + static_cast<size_t>(batch_index) * elements;
            result.outputs.emplace(batch->output_names[output_index], std::vector<float>(source, source + elements));
        }
        for (const auto &binding : pipeline_->results())
        {
            const auto &view = tensors.at("result." + binding.result_name);
            if (view.desc.data_type != TensorDataType::F32)
            {
                continue;
            }
            const size_t elements = view.bytes_per_request / sizeof(float);
            const auto  *source   = static_cast<const float *>(view.dataForRequest(batch_index));
            result.outputs.emplace(binding.result_name, std::vector<float>(source, source + elements));
        }

        for (size_t node_index = 0; node_index < pipeline_->nodes().size(); ++node_index)
        {
            if (pipeline_->nodes()[node_index].contract.stage != PipelineStage::CPU_POSTPROCESS)
            {
                continue;
            }
            OperatorContext context{batch->device_id, batch->actual_batch,   batch_index,
                                    nullptr,          &request->image,       tensors,
                                    result.outputs,   &request->extra_inputs};
            operators[node_index]->execute(context);
        }
        completeSuccess(request, std::move(result));
    }
}

void InferenceEngine::Impl::postprocessLoop(std::vector<std::unique_ptr<IOperator>> operators)
{
    for (;;)
    {
        BatchPtr batch;
        {
            std::unique_lock lock(postprocess_mutex_);
            postprocess_cv_.wait(lock, [this] { return !postprocess_queue_.empty() || postprocess_closed_; });
            if (postprocess_queue_.empty() && postprocess_closed_)
            {
                return;
            }
            batch = std::move(postprocess_queue_.front());
            postprocess_queue_.pop_front();
        }
        try
        {
            batch->cpu_postprocess_started = std::chrono::steady_clock::now();
            processPostprocess(operators, batch);
            batch->cpu_postprocess_finished = std::chrono::steady_clock::now();
        }
        catch (...)
        {
            const auto error = std::current_exception();
            recordFault(FaultStage::CpuPostprocess, batch->device_id,
                        batch->requests.empty() ? nullptr : batch->requests.front(), error);
            failBatch(batch, error);
            batch->cpu_postprocess_finished = std::chrono::steady_clock::now();
        }
        releaseOutputTicket(batch->output_ticket);
        batch->output_ticket = nullptr;
        recordBatchMetrics(*batch);
        finishBatch(batch);
    }
}

void InferenceEngine::Impl::completeSuccess(const RequestPtr &request, InferenceResult result)
{
    if (!request || request->finished.exchange(true))
    {
        return;
    }
    result.request_id      = request->id;
    result.source_id       = request->source_id;
    result.source_sequence = request->source_sequence;
    std::vector<PendingCompletion> ready;
    {
        std::lock_guard lock(mutex_);
        active_requests_.erase(request->id);
        ++completed_requests_;
        PendingCompletion completion{request, true, std::move(result), nullptr};
        if (request->preserve_source_order)
        {
            pending_source_completions_[request->source_id].emplace(request->source_sequence, std::move(completion));
        }
        else
        {
            ready.push_back(std::move(completion));
            unordered_delivered_sequences_[request->source_id].insert(request->source_sequence);
        }
        collectOrderedCompletionsLocked(ready, request->source_id);
    }
    fulfillCompletions(std::move(ready));
}

void InferenceEngine::Impl::completeFailure(const RequestPtr &request, const std::exception_ptr error,
                                            const FailureKind kind)
{
    if (!request || request->finished.exchange(true))
    {
        return;
    }
    std::vector<PendingCompletion> ready;
    {
        std::lock_guard lock(mutex_);
        active_requests_.erase(request->id);
        switch (kind)
        {
        case FailureKind::Failed:
            ++failed_requests_;
            break;
        case FailureKind::Cancelled:
            ++cancelled_requests_;
            break;
        case FailureKind::TimedOut:
            ++timed_out_requests_;
            break;
        case FailureKind::Dropped:
            ++dropped_requests_;
            break;
        }
        PendingCompletion completion{request, false, {}, error};
        if (request->preserve_source_order)
        {
            pending_source_completions_[request->source_id].emplace(request->source_sequence, std::move(completion));
        }
        else
        {
            ready.push_back(std::move(completion));
            unordered_delivered_sequences_[request->source_id].insert(request->source_sequence);
        }
        collectOrderedCompletionsLocked(ready, request->source_id);
    }
    fulfillCompletions(std::move(ready));
}

void InferenceEngine::Impl::collectOrderedCompletionsLocked(std::vector<PendingCompletion> &ready,
                                                            const std::string              &source_id)
{
    auto &next_sequence = next_source_deliver_[source_id];
    auto &unordered     = unordered_delivered_sequences_[source_id];
    auto &pending       = pending_source_completions_[source_id];
    for (;;)
    {
        const auto unordered_it = unordered.find(next_sequence);
        if (unordered_it != unordered.end())
        {
            unordered.erase(unordered_it);
            ++next_sequence;
            continue;
        }
        const auto pending_it = pending.find(next_sequence);
        if (pending_it == pending.end())
        {
            break;
        }
        ready.push_back(std::move(pending_it->second));
        pending.erase(pending_it);
        ++next_sequence;
    }
}

void InferenceEngine::Impl::fulfillCompletions(std::vector<PendingCompletion> completions)
{
    for (auto &completion : completions)
    {
        try
        {
            if (completion.success)
            {
                completion.request->promise.set_value(std::move(completion.result));
            }
            else
            {
                completion.request->promise.set_exception(completion.error);
            }
        }
        catch (const std::future_error &)
        {
        }
    }
}

void InferenceEngine::Impl::failBatch(const BatchPtr &batch, const std::exception_ptr error)
{
    if (!batch)
    {
        return;
    }
    for (const auto &request : batch->requests)
    {
        const bool cancelled = request->cancelled.load();
        completeFailure(request, cancelled ? cancelledError() : error,
                        cancelled ? FailureKind::Cancelled : FailureKind::Failed);
    }
}

void InferenceEngine::Impl::failRequests(const std::vector<std::pair<RequestPtr, FailureKind>> &requests)
{
    for (const auto &[request, kind] : requests)
    {
        switch (kind)
        {
        case FailureKind::Cancelled:
        {
            const auto error = cancelledError();
            recordFault(FaultStage::Scheduler, -1, request, error);
            completeFailure(request, error, kind);
            break;
        }
        case FailureKind::TimedOut:
        {
            const auto error = timeoutError();
            recordFault(FaultStage::Scheduler, -1, request, error);
            completeFailure(request, error, kind);
            break;
        }
        case FailureKind::Dropped:
        {
            const auto error = droppedError();
            recordFault(FaultStage::Scheduler, -1, request, error);
            completeFailure(request, error, kind);
            break;
        }
        case FailureKind::Failed:
        {
            const auto error = std::make_exception_ptr(
                irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Inference engine failed"));
            recordFault(FaultStage::Scheduler, -1, request, error, true);
            completeFailure(request, error, kind);
            break;
        }
        }
    }
}

void InferenceEngine::Impl::finishBatch(const BatchPtr &batch)
{
    if (!batch || batch->finalized.exchange(true))
    {
        return;
    }
    {
        std::lock_guard lock(mutex_);
        if (inflight_batches_ > 0)
        {
            --inflight_batches_;
        }
        ++completed_batches_;
    }
    batch_space_cv_.notify_one();
}

void InferenceEngine::Impl::recordBatchMetrics(const BatchState &batch)
{
    std::chrono::steady_clock::time_point earliest = batch.scheduled;
    for (const auto &request : batch.requests)
    {
        earliest = std::min(earliest, request->submitted);
    }
    std::lock_guard lock(mutex_);
    recordLatency(queue_latency_, elapsedMicroseconds(earliest, batch.cpu_preprocess_started));
    recordLatency(cpu_preprocess_latency_,
                  elapsedMicroseconds(batch.cpu_preprocess_started, batch.cpu_preprocess_finished));
    recordLatency(gpu_latency_, elapsedMicroseconds(batch.gpu_submitted, batch.gpu_finished));
    recordLatency(cpu_postprocess_latency_,
                  elapsedMicroseconds(batch.cpu_postprocess_started, batch.cpu_postprocess_finished));
}

void InferenceEngine::Impl::returnSlot(Slot &slot)
{
    {
        std::lock_guard lock(mutex_);
        idle_slots_.push_back(&slot);
    }
    idle_cv_.notify_one();
}

void InferenceEngine::Impl::synchronizeSlot(Slot &slot) const noexcept
{
    cudaSetDevice(slot.device_id);
    if (slot.h2d_stream != nullptr)
    {
        cudaStreamSynchronize(slot.h2d_stream);
    }
    if (slot.compute_stream != nullptr)
    {
        cudaStreamSynchronize(slot.compute_stream);
    }
    if (slot.d2h_stream != nullptr)
    {
        cudaStreamSynchronize(slot.d2h_stream);
    }
}

void InferenceEngine::Impl::transitionFailed()
{
    {
        std::lock_guard lock(mutex_);
        if (state_ == EngineState::Running || state_ == EngineState::Draining)
        {
            state_ = EngineState::Failed;
        }
    }
    abort_pools_ = true;
    request_cv_.notify_all();
    space_cv_.notify_all();
    idle_cv_.notify_all();
    input_pool_cv_.notify_all();
    output_pool_cv_.notify_all();
}

void InferenceEngine::Impl::recordFault(const FaultStage stage, const int device_id, const RequestPtr &request,
                                        const std::exception_ptr error, const bool fatal)
{
    if (config_.fault_history_capacity == 0)
    {
        return;
    }
    FaultRecord record;
    record.timestamp       = std::chrono::system_clock::now();
    record.request_id      = request ? request->id : 0;
    record.source_id       = request ? request->source_id : std::string{};
    record.source_sequence = request ? request->source_sequence : 0;
    record.device_id       = device_id;
    record.stage           = stage;
    record.fatal           = fatal;
    record.status          = irt::Status::ERROR_INTERNAL;
    record.message         = "Unknown engine failure";
    try
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }
    catch (const irt::Exception &exception)
    {
        record.status  = exception.code();
        record.message = exception.what();
    }
    catch (const std::bad_alloc &exception)
    {
        record.status  = irt::Status::ERROR_OUT_OF_MEMORY;
        record.message = exception.what();
    }
    catch (const std::exception &exception)
    {
        record.message = exception.what();
    }
    catch (...)
    {
    }

    std::lock_guard lock(fault_mutex_);
    while (faults_.size() >= config_.fault_history_capacity)
    {
        faults_.pop_front();
    }
    faults_.push_back(std::move(record));
}

bool InferenceEngine::Impl::isFailed() const
{
    std::lock_guard lock(mutex_);
    return state_ == EngineState::Failed;
}

void InferenceEngine::Impl::shutdown()
{
    {
        std::lock_guard lock(mutex_);
        if (state_ == EngineState::Created || state_ == EngineState::Stopped)
        {
            return;
        }
        if (state_ == EngineState::Running)
        {
            state_ = EngineState::Draining;
        }
    }
    request_cv_.notify_all();
    space_cv_.notify_all();
    batch_space_cv_.notify_all();

    if (scheduler_.joinable())
    {
        scheduler_.join();
    }
    {
        std::lock_guard lock(prepare_mutex_);
        prepare_closed_ = true;
    }
    prepare_cv_.notify_all();
    for (auto &worker : prepare_workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    prepare_workers_.clear();

    {
        std::lock_guard lock(gpu_mutex_);
        gpu_closed_ = true;
    }
    gpu_cv_.notify_all();
    if (gpu_dispatcher_.joinable())
    {
        gpu_dispatcher_.join();
    }
    completion_cv_.notify_all();
    if (completion_poller_.joinable())
    {
        completion_poller_.join();
    }

    {
        std::lock_guard lock(postprocess_mutex_);
        postprocess_closed_ = true;
    }
    postprocess_cv_.notify_all();
    for (auto &worker : postprocess_workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    postprocess_workers_.clear();

    abort_pools_ = true;
    input_pool_cv_.notify_all();
    output_pool_cv_.notify_all();
    {
        std::lock_guard lock(mutex_);
        if (state_ != EngineState::Failed)
        {
            state_ = EngineState::Stopped;
        }
        idle_slots_.clear();
        slots_.clear();
        input_tickets_.clear();
        output_tickets_.clear();
        free_input_tickets_.clear();
        free_output_tickets_.clear();
        runtime_plans_.clear();
        device_arena_bytes_by_device_.clear();
    }
}

size_t InferenceEngine::Impl::pendingRequests() const
{
    std::lock_guard lock(mutex_);
    return requests_.size();
}

EngineMetricsSnapshot InferenceEngine::Impl::metrics() const
{
    std::lock_guard state_lock(mutex_);
    size_t          input_in_use          = 0;
    size_t          input_high_watermark  = 0;
    size_t          output_in_use         = 0;
    size_t          output_high_watermark = 0;
    {
        std::lock_guard input_lock(input_pool_mutex_);
        input_in_use         = input_tickets_in_use_;
        input_high_watermark = input_tickets_high_watermark_;
    }
    {
        std::lock_guard output_lock(output_pool_mutex_);
        output_in_use         = output_tickets_in_use_;
        output_high_watermark = output_tickets_high_watermark_;
    }
    return {accepted_requests_,
            completed_requests_,
            failed_requests_,
            cancelled_requests_,
            timed_out_requests_,
            rejected_requests_,
            dropped_requests_,
            requests_.size(),
            queued_high_watermark_,
            inflight_batches_,
            completed_batches_,
            prepare_queue_high_watermark_.load(),
            gpu_queue_high_watermark_.load(),
            postprocess_queue_high_watermark_.load(),
            input_in_use,
            input_high_watermark,
            output_in_use,
            output_high_watermark,
            pinned_memory_capacity_bytes_,
            device_arena_bytes_,
            queue_latency_,
            cpu_preprocess_latency_,
            gpu_latency_,
            cpu_postprocess_latency_};
}

std::vector<FaultRecord> InferenceEngine::Impl::faults() const
{
    std::lock_guard lock(fault_mutex_);
    return {faults_.begin(), faults_.end()};
}

EngineState InferenceEngine::Impl::state() const noexcept
{
    std::lock_guard lock(mutex_);
    return state_;
}

const EngineConfig &InferenceEngine::Impl::config() const noexcept
{
    return config_;
}

RequestHandle::RequestHandle(const uint64_t request_id, std::future<InferenceResult> future)
    : request_id_(request_id)
    , future_(std::move(future))
{
}

uint64_t RequestHandle::id() const noexcept
{
    return request_id_;
}

bool RequestHandle::valid() const noexcept
{
    return request_id_ != 0 && future_.valid();
}

std::future<InferenceResult> RequestHandle::takeFuture() &&
{
    return std::move(future_);
}

InferenceEngine::InferenceEngine(EngineConfig config)
    : InferenceEngine(config, makeLegacyPipeline(config))
{
}

InferenceEngine::InferenceEngine(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(pipeline)))
{
}

InferenceEngine::~InferenceEngine() = default;

std::unique_ptr<InferenceEngine> InferenceEngine::create(EngineConfig config)
{
    return std::make_unique<InferenceEngine>(std::move(config));
}

std::unique_ptr<InferenceEngine> InferenceEngine::create(EngineConfig                        config,
                                                         std::shared_ptr<const PipelinePlan> pipeline)
{
    return std::make_unique<InferenceEngine>(std::move(config), std::move(pipeline));
}

void InferenceEngine::start()
{
    impl_->start();
}

std::future<InferenceResult> InferenceEngine::submit(const cv::Mat &image)
{
    return std::move(impl_->submit(image, {}, {}).future);
}

std::future<InferenceResult> InferenceEngine::submit(const cv::Mat &image, TensorInputMap inputs)
{
    return std::move(impl_->submit(image, std::move(inputs), {}).future);
}

RequestHandle InferenceEngine::submit(const cv::Mat &image, RequestOptions options)
{
    auto submitted = impl_->submit(image, {}, std::move(options));
    return {submitted.id, std::move(submitted.future)};
}

RequestHandle InferenceEngine::submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options)
{
    auto submitted = impl_->submit(image, std::move(inputs), std::move(options));
    return {submitted.id, std::move(submitted.future)};
}

InferenceResult InferenceEngine::infer(const cv::Mat &image, const std::chrono::milliseconds timeout)
{
    return impl_->infer(image, timeout);
}

bool InferenceEngine::cancel(const uint64_t request_id)
{
    return impl_->cancel(request_id);
}

void InferenceEngine::shutdown()
{
    impl_->shutdown();
}

size_t InferenceEngine::pendingRequests() const
{
    return impl_->pendingRequests();
}

EngineMetricsSnapshot InferenceEngine::metrics() const
{
    return impl_->metrics();
}

std::vector<FaultRecord> InferenceEngine::faults() const
{
    return impl_->faults();
}

EngineState InferenceEngine::state() const noexcept
{
    return impl_->state();
}

const EngineConfig &InferenceEngine::config() const noexcept
{
    return impl_->config();
}

} // namespace irt::engine
