#include "EngineExecution.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <chrono>

namespace irt::engine::priv {

void executePipelineStage(const std::vector<std::unique_ptr<IOperator>> &operators,
                          const std::vector<PipelinePlan::Node>         &nodes,
                          const PipelineStage stage, const BatchState &batch, const int device_id,
                          const cudaStream_t stream, TensorViewMap &tensors, ResultMap &results)
{
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
    {
        if (nodes[node_index].contract.stage != stage)
        {
            continue;
        }
        for (int request_index = 0; request_index < batch.actual_batch; ++request_index)
        {
            const auto     &request = batch.requests[static_cast<size_t>(request_index)];
            auto &operator_instance = operators[node_index];
            OperatorContext context{device_id,
                                    batch.actual_batch,
                                    request_index,
                                    stream,
                                    &request->image,
                                    tensors,
                                    results,
                                    &request->extra_inputs,
                                    operator_instance->scratchData(),
                                    operator_instance->scratchBytes()};
            operator_instance->executeWithContract(context, nodes[node_index].contract);
        }
    }
}

void dispatchGpuBatch(Slot &slot, const BatchPtr &batch, const PipelinePlan &pipeline)
{
    TensorViewMap tensors = slot.device_tensors;
    for (const auto &[name, view] : batch->input_ticket->tensors)
    {
        tensors.emplace(name, view);
    }
    ResultMap ignored;
    irt::model::setCudaDevice(slot.device_id);
    executePipelineStage(slot.operators, pipeline.nodes(), PipelineStage::H2D, *batch, slot.device_id, slot.h2d_stream,
                         tensors, ignored);

    std::vector<irt::BufferView> runtime_buffers;
    runtime_buffers.reserve(slot.model_inputs.size() + slot.device_outputs.size());
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
            auto *tail = model_input->second.dataForRequest(batch->actual_batch);
            const size_t bytes = irt::checkedSizeMul(
                static_cast<size_t>(batch->execution_batch - batch->actual_batch),
                model_input->second.bytes_per_request, "Static batch padding bytes");
            if (tail == nullptr)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "Model input padding exceeds device tensor capacity: %s", pipeline_name.c_str());
            }
            irt::model::checkCuda(cudaMemsetAsync(tail, 0, bytes, slot.h2d_stream),
                                  "cudaMemsetAsync(static batch padding)");
        }
        auto runtime_input    = model_input->second;
        runtime_input.tensor_name = runtime_name;
        runtime_buffers.push_back(std::move(runtime_input));
    }
    irt::model::checkCuda(cudaEventRecord(slot.h2d_done, slot.h2d_stream), "cudaEventRecord(H2D)");
    irt::model::checkCuda(cudaStreamWaitEvent(slot.compute_stream, slot.h2d_done, 0), "cudaStreamWaitEvent(H2D)");
    executePipelineStage(slot.operators, pipeline.nodes(), PipelineStage::CUDA_PREPROCESS, *batch, slot.device_id,
                         slot.compute_stream, tensors, ignored);

    const auto runtime_inputs = slot.runtime_plan->inputs();
    for (const auto &[pipeline_name, runtime_name] : slot.model_inputs)
    {
        const auto &desc = pipeline.tensors().at(pipeline_name);
        const auto runtime_input = std::find_if(
            runtime_inputs.begin(), runtime_inputs.end(),
            [&runtime_name](const irt::TensorInfo &info) { return info.name == runtime_name; });
        if (runtime_input == runtime_inputs.end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input is not declared by runtime plan: %s", runtime_name.c_str());
        }
        slot.context->setTensorShape(
            runtime_name, irt::Shape{batch->execution_batch, desc.channels(), desc.height(), desc.width()});
    }
    for (const auto &output : slot.device_outputs)
    {
        runtime_buffers.push_back(output);
    }
    slot.runtime_plan->executeSession(*slot.context, runtime_buffers,
                                      irt::ExecuteOptions{reinterpret_cast<std::uintptr_t>(slot.compute_stream), true});

    batch->output_elements_per_request.clear();
    batch->output_elements_per_request.reserve(slot.output_infos.size());
    for (size_t index = 0; index < slot.output_infos.size(); ++index)
    {
        const auto &output_name = slot.output_infos[index].name;
        const size_t elements  = slot.context->tensorShape(output_name).elementCount();
        if (elements % static_cast<size_t>(batch->execution_batch) != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Unexpected output shape for tensor: %s",
                                 output_name.c_str());
        }
        const size_t per_request = elements / static_cast<size_t>(batch->execution_batch);
        const size_t bytes = irt::checkedSizeMul(per_request, sizeof(float), "Model output bytes per request");
        if (irt::checkedSizeMul(static_cast<size_t>(batch->execution_batch), bytes, "Model output capacity")
            > slot.output_capacity_bytes[index])
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Output exceeds DeviceArena capacity: %s",
                                 output_name.c_str());
        }
        tensors.insert_or_assign(
            "model." + output_name,
            TensorView{slot.device_outputs[index].data,
                       {TensorDataType::F32, TensorLayout::Opaque, MemoryKind::DEVICE,
                        irt::Shape{static_cast<int64_t>(per_request)}},
                       bytes,
                       batch->execution_batch,
                       slot.output_capacity_bytes[index],
                       output_name});
        batch->output_elements_per_request.push_back(per_request);
    }
    executePipelineStage(slot.operators, pipeline.nodes(), PipelineStage::CUDA_POSTPROCESS, *batch, slot.device_id,
                         slot.compute_stream, tensors, ignored);
    irt::model::checkCuda(cudaEventRecord(slot.compute_done, slot.compute_stream), "cudaEventRecord(compute)");
    irt::model::checkCuda(cudaStreamWaitEvent(slot.d2h_stream, slot.compute_done, 0), "cudaStreamWaitEvent(compute)");

    for (size_t index = 0; index < slot.output_infos.size(); ++index)
    {
        const size_t bytes = irt::checkedSizeMul(
            irt::checkedSizeMul(static_cast<size_t>(batch->actual_batch),
                                batch->output_elements_per_request[index], "Model output batch elements"),
            sizeof(float), "Model output host bytes");
        auto &destination = batch->output_ticket->model_outputs[index];
        if (bytes > destination.capacity())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pinned output ticket is too small: %s",
                                 slot.output_infos[index].name.c_str());
        }
        destination.resize(bytes, irt::TensorDataType::U8);
        irt::model::checkCuda(cudaMemcpyAsync(destination.data(), slot.device_outputs[index].data, bytes,
                                              cudaMemcpyDeviceToHost, slot.d2h_stream),
                              "cudaMemcpyAsync(model D2H)");
    }
    for (const auto &binding : pipeline.results())
    {
        const auto  &source      = tensors.at(binding.tensor_name);
        const size_t bytes = irt::checkedSizeMul(static_cast<size_t>(batch->actual_batch), source.bytes_per_request,
                                                 "Pipeline result host bytes");
        auto        &destination = batch->output_ticket->result_outputs.at(binding.result_name);
        if (bytes > destination.capacity())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pinned result ticket is too small: %s",
                                 binding.result_name.c_str());
        }
        destination.resize(bytes, irt::TensorDataType::U8);
        irt::model::checkCuda(
            cudaMemcpyAsync(destination.data(), source.data, bytes, cudaMemcpyDeviceToHost, slot.d2h_stream),
            "cudaMemcpyAsync(pipeline result D2H)");
    }
    irt::model::checkCuda(cudaEventRecord(slot.d2h_done, slot.d2h_stream), "cudaEventRecord(D2H)");

    batch->gpu_submitted = std::chrono::steady_clock::now();
    batch->device_id     = slot.device_id;
    batch->output_names.clear();
    batch->output_names.reserve(slot.output_infos.size());
    for (const auto &output_info : slot.output_infos)
    {
        batch->output_names.push_back(output_info.name);
    }
    std::lock_guard lock(slot.mutex);
    slot.active_batch   = batch;
    slot.input_released = false;
}

void processPostprocessBatch(const std::vector<std::unique_ptr<IOperator>>                                 &operators,
                             const BatchPtr                                                                &batch,
                             const PipelinePlan                                                            &pipeline,
                             const std::function<void(const RequestPtr &, InferenceResult)>                &on_success,
                             const std::function<void(const RequestPtr &, std::exception_ptr, FailureKind)> &on_failure)
{
    TensorViewMap tensors;
    for (size_t index = 0; index < batch->output_names.size(); ++index)
    {
        const size_t bytes = irt::checkedSizeMul(batch->output_elements_per_request[index], sizeof(float),
                                                 "Postprocess model output bytes");
        tensors.emplace("model." + batch->output_names[index],
                        TensorView{
                            batch->output_ticket->model_outputs[index].data(),
                            {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST,
                             static_cast<int>(batch->output_elements_per_request[index]), 1, 1},
                            bytes,
                            batch->actual_batch,
                            batch->output_ticket->model_outputs[index].capacity()
        });
    }
    for (const auto &binding : pipeline.results())
    {
        auto desc        = pipeline.tensors().at(binding.tensor_name);
        desc.memory_kind = MemoryKind::HOST;
        tensors.emplace("result." + binding.result_name,
                        TensorView{batch->output_ticket->result_outputs.at(binding.result_name).data(), desc,
                                   desc.bytesPerRequest(), batch->actual_batch,
                                   batch->output_ticket->result_outputs.at(binding.result_name).capacity()});
    }

    for (int batch_index = 0; batch_index < batch->actual_batch; ++batch_index)
    {
        const auto &request = batch->requests[static_cast<size_t>(batch_index)];
        if (request->cancelled.load())
        {
            on_failure(request, cancelledError(), FailureKind::Cancelled);
            continue;
        }
        if (request->deadline <= std::chrono::steady_clock::now())
        {
            on_failure(request, timeoutError(), FailureKind::TimedOut);
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
                               + irt::checkedSizeMul(static_cast<size_t>(batch_index), elements,
                                                     "Postprocess model output offset");
            auto [it, inserted] = result.outputs.emplace(batch->output_names[output_index],
                                                         std::vector<float>(source, source + elements));
            if (!inserted)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Duplicate model output tensor name: %s", batch->output_names[output_index].c_str());
            }
        }
        for (const auto &binding : pipeline.results())
        {
            const auto &view = tensors.at("result." + binding.result_name);
            if (view.desc.data_type != TensorDataType::F32)
            {
                continue;
            }
            const size_t elements = view.bytes_per_request / sizeof(float);
            const auto  *source   = static_cast<const float *>(view.dataForRequest(batch_index));
            if (source == nullptr)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "Failed to obtain tensor data for result: %s", binding.result_name.c_str());
            }
            auto [it, inserted] = result.outputs.emplace(binding.result_name,
                                                         std::vector<float>(source, source + elements));
            if (!inserted)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Pipeline result name '%s' conflicts with model output tensor name",
                                     binding.result_name.c_str());
            }
        }

        for (size_t node_index = 0; node_index < pipeline.nodes().size(); ++node_index)
        {
            if (pipeline.nodes()[node_index].contract.stage != PipelineStage::CPU_POSTPROCESS)
            {
                continue;
            }
            auto &operator_instance = operators[node_index];
            OperatorContext context{batch->device_id,
                                    batch->actual_batch,
                                    batch_index,
                                    nullptr,
                                    &request->image,
                                    tensors,
                                    result.outputs,
                                    &request->extra_inputs,
                                    operator_instance->scratchData(),
                                    operator_instance->scratchBytes()};
            operator_instance->executeWithContract(context, pipeline.nodes()[node_index].contract);
        }
        on_success(request, std::move(result));
    }
}

} // namespace irt::engine::priv
