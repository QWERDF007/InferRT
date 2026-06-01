#pragma once

#include <NvInfer.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace irt::model {

// ============================================================================
// 权重辅助函数
// ============================================================================

/**
 * @brief 判断权重表中是否存在指定 key。
 */
inline bool hasWeight(const WeightsMap &weights_map, const std::string &key)
{
    return weights_map.find(key) != weights_map.end();
}

/**
 * @brief 读取权重并按需校验元素数量。
 * @param tag 模型名称标签，用于错误消息。
 */
inline const nvinfer1::Weights &requireWeight(const WeightsMap &weights_map, const std::string &key, const char *tag,
                                              int64_t expected_count = -1)
{
    const auto it = weights_map.find(key);
    if (it == weights_map.end())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Missing %s weight: %s", tag, key.c_str());
    }

    if (expected_count >= 0 && it->second.count != expected_count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Unexpected %s weight element count for %s: got %lld, expected %lld", tag, key.c_str(),
                             static_cast<long long>(it->second.count), static_cast<long long>(expected_count));
    }
    return it->second;
}

// ============================================================================
// 权重生命周期管理
// ============================================================================

namespace detail {

inline std::vector<std::vector<float>> &weightArena()
{
    static std::vector<std::vector<float>> arena;
    return arena;
}

inline std::vector<std::vector<int64_t>> &int64WeightArena()
{
    static std::vector<std::vector<int64_t>> arena;
    return arena;
}

} // namespace detail

/**
 * @brief 构造在建网期间保持有效的标量权重。
 */
inline nvinfer1::Weights ownedScalarWeight(float value)
{
    auto &arena = detail::weightArena();
    arena.push_back({value});
    return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, arena.back().data(), 1};
}

/**
 * @brief 构造在建网期间保持有效的 float 数组权重。
 */
inline nvinfer1::Weights ownedFloatVector(std::vector<float> values)
{
    auto &arena = detail::weightArena();
    arena.push_back(std::move(values));
    return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, arena.back().data(),
                             static_cast<int64_t>(arena.back().size())};
}

/**
 * @brief 构造在建网期间保持有效的 int64 shape 常量权重。
 */
inline nvinfer1::Weights ownedInt64Vector(std::vector<int64_t> values)
{
    auto &arena = detail::int64WeightArena();
    arena.push_back(std::move(values));
    return nvinfer1::Weights{nvinfer1::DataType::kINT64, arena.back().data(),
                             static_cast<int64_t>(arena.back().size())};
}

/**
 * @brief 空权重，用于无 bias 的卷积等层。
 */
inline nvinfer1::Weights emptyWeights()
{
    return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
}

} // namespace irt::model
