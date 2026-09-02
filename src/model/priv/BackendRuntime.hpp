#pragma once

#include <inferrt/model/BackendRuntime.hpp>

#ifndef INFERRT_BUILD_ONNX
#define INFERRT_BUILD_ONNX 0
#endif

#ifndef INFERRT_BUILD_OPENVINO
#define INFERRT_BUILD_OPENVINO 0
#endif

namespace irt::model {

/**
 * @brief Replace the process-wide backend factory used by subsequently created models.
 *
 * Passing nullptr restores the production factory.  The hook is intentionally kept in
 * the private model boundary so fault-injection and embedding code can exercise the
 * same transactional path without changing production backend selection.
 */
INFERRT_MODEL_API void SetBackendRuntimeFactoryOverride(
    std::unique_ptr<IBackendRuntime> (*factory)(ModelRuntime::Backend)) noexcept;
}

namespace irt::model::priv {

/**
 * @brief 创建 ONNX Runtime 后端实例。
 * @return ONNX Runtime 运行时；未启用编译选项时抛出异常。
 */
#if INFERRT_BUILD_ONNX
std::unique_ptr<irt::model::IBackendRuntime> CreateONNXRuntimeBackend();
#endif

/**
 * @brief 创建 OpenVINO 后端实例。
 * @return OpenVINO 运行时；未启用编译选项时抛出异常。
 */
#if INFERRT_BUILD_OPENVINO
std::unique_ptr<irt::model::IBackendRuntime> CreateOpenVINOBackend();
#endif

} // namespace irt::model::priv
