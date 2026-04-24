#pragma once

#include <inferrt/model/Utils.hpp>

namespace irt::model {

/**
 * @brief 将 PyTorch BatchNorm2d 层转换为 TensorRT IScaleLayer
 *
 * BatchNorm2d 的推理公式为:
 *   y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * 可重写为 Scale 层的逐通道仿射变换:
 *   y = scale * x + shift = (gamma / sqrt(var + eps)) * x + (beta - mean * gamma / sqrt(var + eps))
 * 其中:
 *   scale = gamma / sqrt(var + eps)
 *   shift = beta - mean * gamma / sqrt(var + eps)
 *
 * @param network    TensorRT 网络定义
 * @param weights_map 模型权重映射表
 * @param input      输入张量
 * @param lname      BatchNorm 层名前缀
 * @param eps        防止除零的小常数
 * @return IScaleLayer 指针
 */
nvinfer1::IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                      nvinfer1::ITensor &input, std::string lname, float eps)
{
    // 从权重表中读取 BatchNorm 的四个参数
    float *gamma = (float *)weights_map.at(lname + ".weight").values;       // 缩放因子 (γ)
    float *beta  = (float *)weights_map.at(lname + ".bias").values;         // 偏移量 (β)
    float *mean  = (float *)weights_map.at(lname + ".running_mean").values; // 运行均值 (μ)
    float *var   = (float *)weights_map.at(lname + ".running_var").values;  // 运行方差 (var)

    // 通道数
    const int64_t len = weights_map.at(lname + ".running_var").count;
    // std::cout << "len " << len << std::endl;

    // 计算 scale: γ / sqrt(var + eps)
    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    // 计算 shift: β - mean * γ / sqrt(var + eps)
    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    // power 全部为 1.0，即 y = scale * x^1 + shift
    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    // weights_map[lname + ".scale"] = scale;
    // weights_map[lname + ".shift"] = shift;
    // weights_map[lname + ".power"] = power;

    // 添加逐通道 Scale 层: y = shift + scale * (x ^ power)
    nvinfer1::IScaleLayer *scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

} // namespace irt::model