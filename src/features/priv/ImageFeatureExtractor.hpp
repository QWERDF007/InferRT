#pragma once

/**
 * @file ImageFeatureExtractor.hpp
 * @brief 图像级检索与 ROI 检索共用的模型特征抽取器。
 */

#include "FeatureSearchCommon.hpp"
#include "ModelShapeAdapter.hpp"

#include <NvInfer.h>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace irt::features::priv {

/**
 * @brief 原始图像尺寸。
 */
struct ImageSize
{
    int width{0};  ///< 原始图像宽度。
    int height{0}; ///< 原始图像高度。
};

/**
 * @brief 一批模型特征图输出。
 */
struct FeatureTensorBatch
{
    std::vector<float>     data;           ///< 扁平化 NCHW 特征图数据。
    nvinfer1::Dims         dims{};         ///< 输出张量形状，通常为 ``NCHW``。
    std::vector<ImageSize> original_sizes; ///< 每张输入图像的原始尺寸。
};

/**
 * @brief 基于 InferRT 模型的图像特征抽取器。
 *
 * 该类负责模型加载、ImageNet 预处理、批量前向和特征归一化。图像检索直接使用
 * 展平特征；ROI 检索使用未归一化的特征图后再做 ROIAlign。
 */
class ImageFeatureExtractor
{
public:
    /**
     * @brief 创建特征抽取器并构建或加载模型。
     * @param model_name 内置模型名称。
     * @param feature_name 中间特征张量名称。
     * @param weights_file 权重、engine 或图模型文件路径。
     * @param config 检索配置。
     */
    ImageFeatureExtractor(std::string model_name, std::string feature_name, const std::filesystem::path &weights_file,
                          ImageSearchConfig config);

    /**
     * @brief 析构抽取器，先释放模型再释放 CUDA 缓冲区。
     */
    ~ImageFeatureExtractor();

    ImageFeatureExtractor(const ImageFeatureExtractor &)            = delete;
    ImageFeatureExtractor &operator=(const ImageFeatureExtractor &) = delete;

    /**
     * @brief 获取单张图像展平后的输出特征维度。
     */
    int featureDim() const noexcept;

    /** @brief 返回加载模型后解析出的有效预处理规格。 */
    const irt::PreprocessSpec &preprocessSpec() const noexcept;

    /**
     * @brief 获取模型输入宽度。
     */
    int inputWidth() const noexcept;

    /**
     * @brief 获取模型输入高度。
     */
    int inputHeight() const noexcept;

    /**
     * @brief 获取模型允许的最大推理 batch。
     */
    size_t maxBatchSize() const noexcept;

    /**
     * @brief 获取构建时解析到的特征张量形状。
     */
    nvinfer1::Dims featureTensorShape() const noexcept;

    /**
     * @brief 提取单张图像的归一化展平特征。
     * @param image_path 图像路径。
     * @return 长度为 ``featureDim()`` 的特征向量。
     */
    std::vector<float> extract(const std::filesystem::path &image_path);

    /**
     * @brief 从连续图像区间批量提取归一化展平特征。
     * @param image_paths 图像路径数组。
     * @param begin 起始图像下标。
     * @param count 图像数量。
     * @return 扁平化 ``count x featureDim()`` 特征。
     */
    std::vector<float> extractBatch(const std::vector<std::filesystem::path> &image_paths, size_t begin, size_t count);

    /**
     * @brief 从任意下标列表批量提取归一化展平特征。
     * @param image_paths 完整图像路径数组。
     * @param indices 待抽取图像在 ``image_paths`` 中的下标。
     * @return 扁平化 ``indices.size() x featureDim()`` 特征。
     */
    std::vector<float> extractBatch(const std::vector<std::filesystem::path> &image_paths,
                                    const std::vector<size_t>                &indices);

    /**
     * @brief 提取未归一化的原始特征张量。
     *
     * ROI 检索需要在空间特征图上执行 ROIAlign，因此使用该接口获取 NCHW 特征图。
     * 调用方负责后续池化、展平和归一化。
     *
     * @param image_paths 图像路径数组。
     * @param begin 起始图像下标。
     * @param count 图像数量，必须不超过 ``maxBatchSize()``。
     * @return 特征张量批次。
     */
    FeatureTensorBatch extractFeatureTensorBatch(const std::vector<std::filesystem::path> &image_paths, size_t begin,
                                                 size_t count);

private:
    /**
     * @brief 预处理后的一批模型输入。
     */
    struct PreprocessedBatch
    {
        std::vector<float>     input_data;  ///< NCHW 输入张量。
        std::vector<ImageSize> image_sizes; ///< 原始图像尺寸。
    };

    nvinfer1::Dims    resolveInputShape(nvinfer1::Dims input_shape);
    void              resolvePreprocessSpec();
    PreprocessedBatch preprocessBatch(const std::vector<std::filesystem::path> &image_paths, size_t begin,
                                      size_t count);
    nvinfer1::Dims    setRuntimeBatchSize(size_t batch_size);

    std::string model_name_;   ///< 模型名称。
    std::string feature_name_; ///< 输出特征张量名称。

    ImageSearchConfig config_{}; ///< 检索配置。
    irt::PreprocessSpec preprocess_spec_{}; ///< 模型输入解析后的唯一预处理规格。

    std::unique_ptr<irt::model::IModel> model_; ///< 推理模型。

    std::string input_name_;  ///< 输入张量名称。
    std::string output_name_; ///< 输出张量名称。

    nvinfer1::Dims     input_shape_{}; ///< 最大 batch 对应的输入形状。
    nvinfer1::Dims     output_dims_{}; ///< 最大 batch 对应的输出形状。
    irt::TensorDataType output_type_{irt::TensorDataType::F32}; ///< 输出张量数据类型。

    int input_height_{224}; ///< 模型输入高度。
    int input_width_{224};  ///< 模型输入宽度。

    size_t feature_dim_{0};               ///< 单样本展平特征维度。
    size_t max_batch_size_{1};            ///< 最大推理 batch。
    size_t input_elements_per_sample_{0}; ///< 单样本输入元素数。

    irt::model::DeviceBuffer device_input_;  ///< TensorRT 设备输入缓冲区。
    irt::model::DeviceBuffer device_output_; ///< TensorRT 设备输出缓冲区。

    // GPU preprocessing scratch buffers.  They grow to the largest image
    // seen and are reused by subsequent batches.
    irt::model::DeviceBuffer preprocess_source_;
    irt::model::DeviceBuffer preprocess_converted_;
    irt::model::DeviceBuffer preprocess_resized_;
    irt::model::DeviceBuffer preprocess_cropped_;
};

} // namespace irt::features::priv
