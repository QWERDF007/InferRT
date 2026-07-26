/**
 * @file TensorRTRuntimePlan.hpp
 * @brief 供多个 ExecutionSlot 共享的只读 TensorRT engine。
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/model/Logging.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace irt::engine::priv {

/**
 * @brief 在一个 CUDA device 上反序列化 TensorRT engine，并为该 device 的 Slot 创建独立 execution context。
 *
 * TensorRT 的 ICudaEngine 为同 device Slot 只读共享对象；多 GPU 使用彼此独立的 RuntimePlan。
 * IExecutionContext 保持 Slot 私有，因而形状和 tensor address 不会在并发 batch 之间相互覆盖。
 */
class TensorRTRuntimePlan final
{
public:
    TensorRTRuntimePlan(const EngineConfig &config, int device_id);

    [[nodiscard]] std::unique_ptr<nvinfer1::IExecutionContext> createSession() const;
    void setInputShape(nvinfer1::IExecutionContext &context, const std::string &input_name,
                       const nvinfer1::Dims4 &shape) const;
    void enqueue(nvinfer1::IExecutionContext &context, const std::vector<std::pair<std::string, void *>> &inputs,
                 const std::vector<void *> &outputs, cudaStream_t stream) const;

    [[nodiscard]] const std::vector<std::string> &inputNames() const noexcept;
    [[nodiscard]] const std::vector<std::string> &outputNames() const noexcept;
    [[nodiscard]] nvinfer1::DataType              tensorDataType(const std::string &name) const;
    [[nodiscard]] nvinfer1::Dims tensorShape(const nvinfer1::IExecutionContext &context, const std::string &name) const;
    /** 正数表示 TensorRT engine 固定 batch；零表示 batch 维动态。 */
    [[nodiscard]] int            fixedBatchSize() const noexcept;

private:
    irt::model::Logger                     logger_;
    std::unique_ptr<nvinfer1::IRuntime>    runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::vector<std::string>               input_names_;
    std::vector<std::string>               output_names_;
    int                                    fixed_batch_size_{0};
};

} // namespace irt::engine::priv
