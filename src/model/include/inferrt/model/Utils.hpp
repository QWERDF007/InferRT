#pragma once

#include <NvInfer.h>
#include <inferrt/model/Export.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief 模型权重映射表。
 *
 * key 通常与 PyTorch state_dict 中的参数名称一致，value 为 TensorRT 权重对象。
 */
using WeightsMap = std::map<std::string, nvinfer1::Weights>;

/**
 * @brief 从文本格式 .wts 文件加载权重。
 * @param file 权重文件路径。
 * @return 解析后的权重映射表。
 */
INFERRT_MODEL_API WeightsMap loadWeights(const std::string &file);

/**
 * @brief 读取 ImageNet 1000 类标签文件。
 * @param label_file 标签文件路径。
 * @return 长度为 1000 的标签列表。
 */
INFERRT_MODEL_API std::vector<std::string> readImagenetLabels(const std::string &label_file);

class INFERRT_MODEL_API ImageNetUtil
{
public:
    /**
     * @brief 默认的 ImageNet 示例输入图片路径。
     */
    static const std::filesystem::path kDefaultImagePath;

    /**
     * @brief 默认的 ImageNet 1000 类标签文件路径。
     */
    static const std::filesystem::path kDefaultLabelPath;

    /**
     * @brief 对 BGR 输入图像执行 ImageNet 分类预处理。
     *
     * 处理流程包括 BGR 转 RGB、缩放到目标尺寸、归一化到 [0, 1]，
     * 并按 ImageNet 均值与方差做标准化。
     *
     * @param bgr_image 输入的 BGR 图像。
     * @param target_size 目标尺寸，默认为 224x224。
     * @return 预处理后的 `CV_32FC3` 图像。
     */
    static cv::Mat preprocess(const cv::Mat &bgr_image, cv::Size target_size = cv::Size(224, 224));

    /**
     * @brief 将 `CV_32FC3` 图像转换为 CHW 排列的连续 float 张量。
     * @param image 输入图像，要求类型为 `CV_32FC3`。
     * @return 按 CHW 排列的 float 数据。
     */
    static std::vector<float> imageToTensorCHW(const cv::Mat &image);
};

} // namespace irt::model
