/**
 * @file TensorRTRuntimePlan.hpp
 * @brief 供多个 ExecutionSlot 共享的只读 TensorRT engine。
 */

#pragma once

#include "EngineRuntimePlan.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace irt::engine::priv {

/**
 * @brief 在一个 CUDA device 上反序列化 TensorRT engine，并为该 device 的 Slot 创建独立 execution context。
 *
 * 后端运行时为同 device Slot 只读共享对象；每个 Slot 通过独立 session
 * 保存形状和地址绑定状态，因而并发 batch 之间不会相互覆盖。
 */
class TensorRTRuntimePlan final : public irt::IExecutionPlan
{
public:
    TensorRTRuntimePlan(const EngineConfig &config, int device_id,
                        const irt::engine::PipelinePlan *pipeline = nullptr);
    ~TensorRTRuntimePlan() override;

    [[nodiscard]] std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override;
    void executeSession(irt::ITensorRuntimeSession &session, std::span<const irt::BufferView> buffers,
                        irt::ExecuteOptions options = {}) const override;

    [[nodiscard]] std::vector<irt::TensorInfo> inputs() const override;
    [[nodiscard]] std::vector<irt::TensorInfo> outputs() const override;
    [[nodiscard]] irt::ExecutionCapabilities capabilities() const noexcept override;

private:
    std::unique_ptr<irt::IExecutionPlan> backend_;
    std::vector<irt::TensorInfo>                        inputs_;
    std::vector<irt::TensorInfo>                        outputs_;
    int                                                 fixed_batch_size_{0};
    bool                                                supports_feature_outputs_{false};
};

} // namespace irt::engine::priv
