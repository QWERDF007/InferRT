#pragma once

#include <inferrt/features/DinoRegionSearch.hpp>
#include <string>

namespace irt::features::priv {

/** @brief 无需创建骨干或读取权重的模型与视图预处理描述。 */
struct DinoBackboneMetadata
{
    int patch_size{0};
    int channels{0};
    int encoder_edge{0};
    std::string preprocess{};
};

/** @brief 从注册模型规格与配置派生骨干元数据；未知模型抛出异常。 */
INFERRT_FEATURES_API DinoBackboneMetadata dinoDescribeBackbone(const DinoRegionSearchConfig &config);

} // namespace irt::features::priv
