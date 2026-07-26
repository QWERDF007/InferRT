#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief YOLOv5 检测模型的结构缩放参数。
 */
struct YOLOv5Spec
{
    const char *display_name; ///< 模型显示名称。
    float       depth;        ///< 深度缩放系数，对应 Ultralytics YAML 中的 depth。
    float       width;        ///< 宽度缩放系数，对应 Ultralytics YAML 中的 width。
};

/**
 * @brief YOLOv8 检测模型的结构缩放参数。
 */
struct YOLOv8Spec
{
    const char *display_name; ///< 模型显示名称。
    float       depth;        ///< 深度缩放系数。
    float       width;        ///< 宽度缩放系数。
    int         max_channels; ///< 宽度缩放后的最大通道数。
};

/**
 * @brief YOLO 检测模型公共基类。
 *
 * 该类统一处理检测模型默认配置：COCO 80 类、1x3x640x640 输入以及 3 个检测输出。
 * 具体 YOLOv5/YOLOv8 派生类只需要实现对应的 TensorRT 网络结构。
 */
class YOLOModelBase : public priv::IModelImpl
{
public:
    /**
     * @brief YOLO 网络仅将 batch 维作为动态维，空间尺寸仍由配置固定。
     */
    bool supportsDynamicBatch() const noexcept override
    {
        return true;
    }

protected:
    /**
     * @brief 将默认 ImageNet 分类配置规整为 YOLO 检测配置。
     * @param config 待规整的模型配置。
     */
    void normalizeModelConfig(IModelConfig &config) const override;

    /**
     * @brief 校验 YOLO 检测网络支持的输入/输出配置。
     */
    void validateDetectionConfig() const;
};

/**
 * @brief Ultralytics YOLOv5 检测网络实现。
 */
class YOLOv5Detector : public YOLOModelBase
{
public:
    /**
     * @brief 构造 YOLOv5 检测模型。
     * @param spec 结构缩放参数。
     */
    explicit YOLOv5Detector(YOLOv5Spec spec)
        : spec_(spec)
    {
    }

    std::string name() const noexcept override
    {
        return spec_.display_name;
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    const YOLOv5Spec &spec() const noexcept
    {
        return spec_;
    }

private:
    YOLOv5Spec spec_;
};

/**
 * @brief Ultralytics YOLOv8 检测网络实现。
 */
class YOLOv8Detector : public YOLOModelBase
{
public:
    /**
     * @brief 构造 YOLOv8 检测模型。
     * @param spec 结构缩放参数。
     */
    explicit YOLOv8Detector(YOLOv8Spec spec)
        : spec_(spec)
    {
    }

    std::string name() const noexcept override
    {
        return spec_.display_name;
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    const YOLOv8Spec &spec() const noexcept
    {
        return spec_;
    }

private:
    YOLOv8Spec spec_;
};

/**
 * @brief Ultralytics YOLOv8 实例分割网络实现。
 */
class YOLOv8Segmenter : public YOLOv8Detector
{
public:
    /**
     * @brief 构造 YOLOv8 segmentation 模型。
     * @param spec 结构缩放参数。
     */
    explicit YOLOv8Segmenter(YOLOv8Spec spec)
        : YOLOv8Detector(spec)
    {
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    void normalizeModelConfig(IModelConfig &config) const override;
};

class YOLOv5 : public YOLOv5Detector
{
public:
    YOLOv5()
        : YOLOv5Detector({"YOLOv5s", 0.33F, 0.50F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5";
    }
};

class YOLOv5n : public YOLOv5Detector
{
public:
    YOLOv5n()
        : YOLOv5Detector({"YOLOv5n", 0.33F, 0.25F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5n";
    }
};

class YOLOv5s : public YOLOv5Detector
{
public:
    YOLOv5s()
        : YOLOv5Detector({"YOLOv5s", 0.33F, 0.50F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5s";
    }
};

class YOLOv5m : public YOLOv5Detector
{
public:
    YOLOv5m()
        : YOLOv5Detector({"YOLOv5m", 0.67F, 0.75F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5m";
    }
};

class YOLOv5l : public YOLOv5Detector
{
public:
    YOLOv5l()
        : YOLOv5Detector({"YOLOv5l", 1.00F, 1.00F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5l";
    }
};

class YOLOv5x : public YOLOv5Detector
{
public:
    YOLOv5x()
        : YOLOv5Detector({"YOLOv5x", 1.33F, 1.25F})
    {
    }

    static const char *key() noexcept
    {
        return "yolov5x";
    }
};

class YOLOv8 : public YOLOv8Detector
{
public:
    YOLOv8()
        : YOLOv8Detector({"YOLOv8n", 0.33F, 0.25F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8";
    }
};

class YOLOv8n : public YOLOv8Detector
{
public:
    YOLOv8n()
        : YOLOv8Detector({"YOLOv8n", 0.33F, 0.25F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8n";
    }
};

class YOLOv8s : public YOLOv8Detector
{
public:
    YOLOv8s()
        : YOLOv8Detector({"YOLOv8s", 0.33F, 0.50F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8s";
    }
};

class YOLOv8m : public YOLOv8Detector
{
public:
    YOLOv8m()
        : YOLOv8Detector({"YOLOv8m", 0.67F, 0.75F, 576})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8m";
    }
};

class YOLOv8l : public YOLOv8Detector
{
public:
    YOLOv8l()
        : YOLOv8Detector({"YOLOv8l", 1.00F, 1.00F, 512})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8l";
    }
};

class YOLOv8x : public YOLOv8Detector
{
public:
    YOLOv8x()
        : YOLOv8Detector({"YOLOv8x", 1.00F, 1.25F, 640})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8x";
    }
};

class YOLOv8Seg : public YOLOv8Segmenter
{
public:
    YOLOv8Seg()
        : YOLOv8Segmenter({"YOLOv8nSeg", 0.33F, 0.25F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8_seg";
    }
};

class YOLOv8nSeg : public YOLOv8Segmenter
{
public:
    YOLOv8nSeg()
        : YOLOv8Segmenter({"YOLOv8nSeg", 0.33F, 0.25F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8n_seg";
    }
};

class YOLOv8sSeg : public YOLOv8Segmenter
{
public:
    YOLOv8sSeg()
        : YOLOv8Segmenter({"YOLOv8sSeg", 0.33F, 0.50F, 1024})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8s_seg";
    }
};

class YOLOv8mSeg : public YOLOv8Segmenter
{
public:
    YOLOv8mSeg()
        : YOLOv8Segmenter({"YOLOv8mSeg", 0.67F, 0.75F, 576})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8m_seg";
    }
};

class YOLOv8lSeg : public YOLOv8Segmenter
{
public:
    YOLOv8lSeg()
        : YOLOv8Segmenter({"YOLOv8lSeg", 1.00F, 1.00F, 512})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8l_seg";
    }
};

class YOLOv8xSeg : public YOLOv8Segmenter
{
public:
    YOLOv8xSeg()
        : YOLOv8Segmenter({"YOLOv8xSeg", 1.00F, 1.25F, 640})
    {
    }

    static const char *key() noexcept
    {
        return "yolov8x_seg";
    }
};

#define INFERRT_YOLO_ALIAS_CLASS(CLASS_NAME, BASE_CLASS, KEY_LITERAL) \
    class CLASS_NAME : public BASE_CLASS                              \
    {                                                                 \
    public:                                                           \
        static const char *key() noexcept                             \
        {                                                             \
            return KEY_LITERAL;                                       \
        }                                                             \
    }

INFERRT_YOLO_ALIAS_CLASS(YOLOv8SegHyphenAlias, YOLOv8Seg, "yolov8-seg");
INFERRT_YOLO_ALIAS_CLASS(YOLOv8nSegHyphenAlias, YOLOv8nSeg, "yolov8n-seg");
INFERRT_YOLO_ALIAS_CLASS(YOLOv8sSegHyphenAlias, YOLOv8sSeg, "yolov8s-seg");
INFERRT_YOLO_ALIAS_CLASS(YOLOv8mSegHyphenAlias, YOLOv8mSeg, "yolov8m-seg");
INFERRT_YOLO_ALIAS_CLASS(YOLOv8lSegHyphenAlias, YOLOv8lSeg, "yolov8l-seg");
INFERRT_YOLO_ALIAS_CLASS(YOLOv8xSegHyphenAlias, YOLOv8xSeg, "yolov8x-seg");

#undef INFERRT_YOLO_ALIAS_CLASS

} // namespace irt::model
