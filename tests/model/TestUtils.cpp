#include "TestModelCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/Utils.hpp>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

namespace fs = std::filesystem;

using test::model::ExpectIrtExceptionCode;

namespace {

/**
 * @brief 临时测试文件辅助类
 *
 * 构造时写入文件，析构时自动删除，避免污染仓库目录。
 */
class TempTextFile
{
public:
    /**
     * @brief 创建带指定后缀和内容的临时文本文件。
     * @param suffix 文件后缀。
     * @param content 写入文件的文本内容。
     */
    explicit TempTextFile(const std::string &suffix, const std::string &content)
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_test_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + suffix);
        std::ofstream file(path_);
        file << content;
    }

    /**
     * @brief 析构时删除临时文件。
     */
    ~TempTextFile()
    {
        std::error_code ec;
        fs::remove(path_, ec);
    }

    /**
     * @brief 获取临时文件路径。
     * @return 临时文件路径。
     */
    const fs::path &path() const
    {
        return path_;
    }

    TempTextFile(const TempTextFile &)            = delete;
    TempTextFile &operator=(const TempTextFile &) = delete;

private:
    fs::path path_;
};

} // namespace

/**
 * @brief 单条权重记录应被正确解析
 */
TEST(LoadWeightsTest, SingleWeightEntryParsedCorrectly)
{
    TempTextFile tmp(".wts", "1\nconv.weight 2 3F800000 40000000\n");

    auto weights = irt::model::loadWeights(tmp.path().string());

    ASSERT_EQ(weights.size(), 1u);
    ASSERT_TRUE(weights.count("conv.weight") > 0);
    EXPECT_EQ(weights.at("conv.weight").count, 2);
    EXPECT_NE(weights.at("conv.weight").values, nullptr);

    // 验证十六进制值被正确读取为 uint32_t
    const auto *vals = static_cast<const uint32_t *>(weights.at("conv.weight").values);
    EXPECT_EQ(vals[0], 0x3F800000u);
    EXPECT_EQ(vals[1], 0x40000000u);
}

/**
 * @brief 多条权重记录应分别按名称和数量正确解析
 */
TEST(LoadWeightsTest, MultipleWeightEntriesParsedCorrectly)
{
    TempTextFile tmp(".wts",
                     "3\n"
                     "conv1.weight 1 3F800000\n"
                     "conv1.bias 1 40000000\n"
                     "conv2.weight 3 3F800000 FF8800FF 80000000\n");

    auto weights = irt::model::loadWeights(tmp.path().string());

    ASSERT_EQ(weights.size(), 3u);
    EXPECT_EQ(weights.at("conv1.weight").count, 1);
    EXPECT_EQ(weights.at("conv1.bias").count, 1);
    EXPECT_EQ(weights.at("conv2.weight").count, 3);

    const auto *conv1w = static_cast<const uint32_t *>(weights.at("conv1.weight").values);
    const auto *conv1b = static_cast<const uint32_t *>(weights.at("conv1.bias").values);
    const auto *conv2w = static_cast<const uint32_t *>(weights.at("conv2.weight").values);

    EXPECT_EQ(conv1w[0], 0x3F800000u);
    EXPECT_EQ(conv1b[0], 0x40000000u);
    EXPECT_EQ(conv2w[0], 0x3F800000u);
    EXPECT_EQ(conv2w[1], 0xFF8800FFu);
    EXPECT_EQ(conv2w[2], 0x80000000u);
}

/**
 * @brief 元素数量为 0 的权重项在当前实现中是允许的
 */
TEST(LoadWeightsTest, ZeroCountWeightEntryIsValid)
{
    TempTextFile tmp(".wts", "1\nempty.weight 0\n");

    auto weights = irt::model::loadWeights(tmp.path().string());

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_EQ(weights.at("empty.weight").count, 0);
}

/**
 * @brief 不存在的权重文件应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(LoadWeightsTest, NonExistentFileThrowsInvalidArgument)
{
    ExpectIrtExceptionCode([&] { irt::model::loadWeights("/non/existent/path/weights.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 权重条目总数小于等于 0 时应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(LoadWeightsTest, NonPositiveEntryCountThrowsInvalidArgument)
{
    TempTextFile zero(".wts", "0\n");
    TempTextFile negative(".wts", "-1\n");

    EXPECT_THROW({ irt::model::loadWeights(zero.path().string()); }, irt::Exception);
    EXPECT_THROW({ irt::model::loadWeights(negative.path().string()); }, irt::Exception);
}

/**
 * @brief 空文件无法读出条目总数，应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(LoadWeightsTest, EmptyFileThrowsInvalidArgument)
{
    TempTextFile tmp(".wts", "");
    EXPECT_THROW({ irt::model::loadWeights(tmp.path().string()); }, irt::Exception);
}

/**
 * @brief readImagenetLabels 应能按索引解析标签内容
 */
TEST(ReadImagenetLabelsTest, ParsesIndexedLabelsCorrectly)
{
    TempTextFile tmp(".txt", "0: 'tench',\n1: 'goldfish',\n999: 'toilet tissue',\n");

    auto labels = irt::model::readImagenetLabels(tmp.path().string());

    ASSERT_EQ(labels.size(), 1000u);
    EXPECT_EQ(labels[0], "tench");
    EXPECT_EQ(labels[1], "goldfish");
    EXPECT_EQ(labels[999], "toilet tissue");
}

/**
 * @brief 超出 [0, 999] 范围的标签索引应被忽略
 */
TEST(ReadImagenetLabelsTest, IgnoresOutOfRangeIndices)
{
    TempTextFile tmp(".txt", "-1: 'invalid',\n0: 'tench',\n1000: 'invalid2',\n");

    auto labels = irt::model::readImagenetLabels(tmp.path().string());

    ASSERT_EQ(labels.size(), 1000u);
    EXPECT_EQ(labels[0], "tench");
}

/**
 * @brief 不符合 “index: label” 格式的行应被忽略
 */
TEST(ReadImagenetLabelsTest, IgnoresMalformedLines)
{
    TempTextFile tmp(".txt", "bad line\n0: 'tench',\n1 goldfish\n2: 'great white shark',\n");

    auto labels = irt::model::readImagenetLabels(tmp.path().string());

    ASSERT_EQ(labels.size(), 1000u);
    EXPECT_EQ(labels[0], "tench");
    EXPECT_EQ(labels[1], "");
    EXPECT_EQ(labels[2], "great white shark");
}

/**
 * @brief 不存在的标签文件应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(ReadImagenetLabelsTest, NonExistentFileThrowsInvalidArgument)
{
    ExpectIrtExceptionCode([&] { irt::model::readImagenetLabels("/non/existent/path/labels.txt"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief elementSize 应返回 TensorRT 常用数据类型的单元素字节数。
 */
TEST(ModelUtilTest, ElementSizeMatchesTensorRTTypes)
{
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kFLOAT), 4U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kHALF), 2U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kINT8), 1U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kUINT8), 1U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kINT32), 4U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kINT64), 8U);
    EXPECT_EQ(irt::model::elementSize(nvinfer1::DataType::kBOOL), 1U);
    EXPECT_EQ(irt::model::dataTypeSize(nvinfer1::DataType::kFLOAT), irt::model::elementSize(nvinfer1::DataType::kFLOAT));
}

/**
 * @brief elementSize 对未知 TensorRT 数据类型应抛出未实现异常，避免静默返回错误字节数。
 */
TEST(ModelUtilTest, ElementSizeRejectsUnknownDataType)
{
    ExpectIrtExceptionCode([&] { irt::model::elementSize(static_cast<nvinfer1::DataType>(999)); },
                           irt::Status::ERROR_NOT_IMPLEMENTED);
}

/**
 * @brief elementCount 应计算维度乘积，并拒绝未解析或非法维度。
 */
TEST(ModelUtilTest, ElementCountValidatesTensorDims)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1;
    dims.d[1] = 3;
    dims.d[2] = 224;
    dims.d[3] = 224;
    EXPECT_EQ(irt::model::elementCount(dims), 150528U);

    dims.d[2] = 0;
    EXPECT_THROW({ irt::model::elementCount(dims); }, irt::Exception);
}

/**
 * @brief elementCount 对标量维度应返回 1，对负维度应抛出非法参数异常。
 */
TEST(ModelUtilTest, ElementCountHandlesScalarAndRejectsNegativeDims)
{
    nvinfer1::Dims scalar{};
    scalar.nbDims = 0;
    EXPECT_EQ(irt::model::elementCount(scalar), 1U);

    nvinfer1::Dims invalid{};
    invalid.nbDims = 2;
    invalid.d[0] = 4;
    invalid.d[1] = -1;
    EXPECT_THROW({ irt::model::elementCount(invalid); }, irt::Exception);
}

/**
 * @brief dtype 与维度格式化工具应输出稳定文本，供日志和 manifest 复用。
 */
TEST(ModelUtilTest, FormatsDataTypeAndDims)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = 3;
    dims.d[1] = 224;
    dims.d[2] = 224;

    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kFLOAT), "float32");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kHALF), "float16");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kINT8), "int8");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kUINT8), "uint8");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kINT32), "int32");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kINT64), "int64");
    EXPECT_EQ(irt::model::dataTypeToString(nvinfer1::DataType::kBOOL), "bool");
    EXPECT_EQ(irt::model::dimsToCsv(dims), "3,224,224");
    EXPECT_EQ(irt::model::dimsToString(dims), "[3, 224, 224]");
}

/**
 * @brief dtype 与维度格式化工具应对未知类型和空维度输出稳定文本。
 */
TEST(ModelUtilTest, FormatsUnknownDataTypeAndEmptyDims)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 0;

    EXPECT_EQ(irt::model::dataTypeToString(static_cast<nvinfer1::DataType>(999)), "unknown");
    EXPECT_EQ(irt::model::dimsToCsv(dims), "");
    EXPECT_EQ(irt::model::dimsToString(dims), "[]");
}

/**
 * @brief checkCuda 应允许成功状态并将失败状态转换为 InferRT 异常。
 */
TEST(ModelUtilTest, CheckCudaConvertsErrorStatus)
{
    EXPECT_NO_THROW({ irt::model::checkCuda(cudaSuccess, "cudaSuccess"); });
    ExpectIrtExceptionCode([&] { irt::model::checkCuda(cudaErrorInvalidValue, "invalid"); },
                           irt::Status::ERROR_INTERNAL);
}

/**
 * @brief DeviceBuffer 默认构造时不持有设备内存。
 */
TEST(DeviceBufferTest, DefaultConstructsEmptyBuffer)
{
    const irt::model::DeviceBuffer buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_EQ(buffer.sizeBytes(), 0U);
}

/**
 * @brief DeviceBuffer 调整为 0 个元素时应保持为空，便于调用方统一处理动态尺寸。
 */
TEST(DeviceBufferTest, ResizeZeroKeepsBufferEmpty)
{
    irt::model::DeviceBuffer buffer;

    buffer.resize(0);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_EQ(buffer.sizeBytes(), 0U);
}

/**
 * @brief DeviceBuffer 移动空缓冲区后，源对象和目标对象都应处于有效空状态。
 */
TEST(DeviceBufferTest, MoveEmptyBufferKeepsValidState)
{
    irt::model::DeviceBuffer source;
    irt::model::DeviceBuffer target(std::move(source));

    EXPECT_TRUE(source.empty());
    EXPECT_TRUE(target.empty());
    EXPECT_EQ(source.size(), 0U);
    EXPECT_EQ(target.size(), 0U);

    irt::model::DeviceBuffer assigned;
    assigned = std::move(target);
    EXPECT_TRUE(target.empty());
    EXPECT_TRUE(assigned.empty());
}

/**
 * @brief DeviceBuffer 按非法 TensorRT 维度 resize 时，应在申请显存前抛出异常。
 */
TEST(DeviceBufferTest, ResizeRejectsInvalidDimsBeforeAllocation)
{
    irt::model::DeviceBuffer buffer;
    nvinfer1::Dims           dims{};
    dims.nbDims = 2;
    dims.d[0] = 4;
    dims.d[1] = 0;

    EXPECT_THROW({ buffer.resize(dims); }, irt::Exception);
    EXPECT_TRUE(buffer.empty());
}

/**
 * @brief HostBuffer 应按元素数量和数据类型计算字节数，并分配主机内存。
 */
TEST(HostBufferTest, AllocatesHostMemoryByElementCountAndType)
{
    irt::model::HostBuffer buffer(4, nvinfer1::DataType::kFLOAT);

    ASSERT_FALSE(buffer.empty());
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer.sizeBytes(), 4U * sizeof(float));
    EXPECT_EQ(buffer.nbBytes(), buffer.sizeBytes());
    EXPECT_GE(buffer.capacity(), buffer.size());

    auto *values = static_cast<float *>(buffer.data());
    values[0] = 1.0f;
    values[1] = 2.0f;
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
}

/**
 * @brief HostBuffer 使用仅指定类型的构造函数时不应分配内存，但应保留元素类型。
 */
TEST(HostBufferTest, TypeOnlyConstructorKeepsTypeWithoutAllocation)
{
    irt::model::HostBuffer buffer(nvinfer1::DataType::kHALF);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(buffer.dataType(), nvinfer1::DataType::kHALF);
    EXPECT_EQ(buffer.sizeBytes(), 0U);
}

/**
 * @brief HostBuffer 按 TensorRT 维度 resize 时应复用 elementCount 计算元素数量。
 */
TEST(HostBufferTest, ResizeByDimsComputesElementCount)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = 2;
    dims.d[1] = 3;
    dims.d[2] = 4;

    irt::model::HostBuffer buffer;
    buffer.resize(dims, nvinfer1::DataType::kINT32);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 24U);
    EXPECT_EQ(buffer.sizeBytes(), 24U * sizeof(int32_t));
    EXPECT_EQ(buffer.dataType(), nvinfer1::DataType::kINT32);
}

/**
 * @brief HostBuffer 切换数据类型时应重新按新元素大小计算逻辑字节数。
 */
TEST(HostBufferTest, ResizeWithDifferentTypeUpdatesByteSize)
{
    irt::model::HostBuffer buffer(4, nvinfer1::DataType::kINT8);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.sizeBytes(), 4U);

    buffer.resize(4, nvinfer1::DataType::kFLOAT);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.dataType(), nvinfer1::DataType::kFLOAT);
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer.sizeBytes(), 4U * sizeof(float));
}

/**
 * @brief HostBuffer resize 小于当前容量时应只更新逻辑大小，不释放已有内存。
 */
TEST(HostBufferTest, ResizeWithinCapacityKeepsAllocation)
{
    irt::model::HostBuffer buffer(8, nvinfer1::DataType::kINT32);
    void                  *original = buffer.data();

    buffer.resize(2);

    EXPECT_EQ(buffer.data(), original);
    EXPECT_EQ(buffer.size(), 2U);
    EXPECT_EQ(buffer.sizeBytes(), 2U * sizeof(int32_t));
    EXPECT_GE(buffer.capacity(), 8U);
}

/**
 * @brief HostBuffer 在 0 元素状态切换类型时应丢弃旧容量，避免按错误元素大小复用内存。
 */
TEST(HostBufferTest, ResizeZeroWithDifferentTypeDropsOldCapacity)
{
    irt::model::HostBuffer buffer(4, nvinfer1::DataType::kINT8);

    buffer.resize(0, nvinfer1::DataType::kFLOAT);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.capacity(), 0U);
    EXPECT_EQ(buffer.dataType(), nvinfer1::DataType::kFLOAT);

    buffer.resize(4);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer.sizeBytes(), 4U * sizeof(float));
    static_cast<float *>(buffer.data())[3] = 4.0F;
    EXPECT_FLOAT_EQ(static_cast<float *>(buffer.data())[3], 4.0F);
}

/**
 * @brief HostBuffer 移动后应转移内存所有权并清空源对象。
 */
TEST(HostBufferTest, MoveTransfersOwnership)
{
    irt::model::HostBuffer source(3, nvinfer1::DataType::kINT8);
    void                  *original = source.data();

    irt::model::HostBuffer target(std::move(source));

    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(target.data(), original);
    EXPECT_EQ(target.size(), 3U);
    EXPECT_EQ(target.sizeBytes(), 3U);
}

/**
 * @brief HostBuffer reset 应释放内存并清空容量，同时保留当前元素类型。
 */
TEST(HostBufferTest, ResetReleasesMemoryAndKeepsDataType)
{
    irt::model::HostBuffer buffer(4, nvinfer1::DataType::kINT32);
    ASSERT_FALSE(buffer.empty());

    buffer.reset();

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_EQ(buffer.capacity(), 0U);
    EXPECT_EQ(buffer.dataType(), nvinfer1::DataType::kINT32);
}

/**
 * @brief ImageNet 默认资源路径应保持稳定，供 samples 复用。
 */
TEST(ImageNetUtilTest, DefaultPathsMatchSampleAssets)
{
    EXPECT_EQ(irt::model::ImageNetUtil::kDefaultImagePath.generic_string(), "assets/pics/dog.jpg");
    EXPECT_EQ(irt::model::ImageNetUtil::kDefaultLabelPath.generic_string(),
              "assets/imagenet1000_clsidx_to_labels.txt");
}

/**
 * @brief preprocess 应拒绝空图像输入。
 */
TEST(ImageNetUtilTest, PreprocessRejectsEmptyImage)
{
    EXPECT_THROW({ irt::model::ImageNetUtil::preprocess(cv::Mat()); }, irt::Exception);
}

/**
 * @brief preprocess 应拒绝非法目标尺寸。
 */
TEST(ImageNetUtilTest, PreprocessRejectsInvalidTargetSize)
{
    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    EXPECT_THROW({ irt::model::ImageNetUtil::preprocess(image, cv::Size(0, 2)); }, irt::Exception);
    EXPECT_THROW({ irt::model::ImageNetUtil::preprocess(image, cv::Size(2, -1)); }, irt::Exception);
}

/**
 * @brief preprocess 应完成 BGR->RGB、缩放和 ImageNet 标准化。
 */
TEST(ImageNetUtilTest, PreprocessConvertsToRgbAndNormalizes)
{
    cv::Mat image(1, 1, CV_8UC3);
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30); // BGR

    const cv::Mat processed = irt::model::ImageNetUtil::preprocess(image, cv::Size(1, 1));

    ASSERT_EQ(processed.rows, 1);
    ASSERT_EQ(processed.cols, 1);
    ASSERT_EQ(processed.type(), CV_32FC3);

    const cv::Vec3f pixel = processed.at<cv::Vec3f>(0, 0);
    const float r = (30.0f / 255.0f - 0.485f) / 0.229f;
    const float g = (20.0f / 255.0f - 0.456f) / 0.224f;
    const float b = (10.0f / 255.0f - 0.406f) / 0.225f;

    EXPECT_NEAR(pixel[0], r, 1e-5f);
    EXPECT_NEAR(pixel[1], g, 1e-5f);
    EXPECT_NEAR(pixel[2], b, 1e-5f);
}

/**
 * @brief imageToTensorCHW 应拒绝空图像和非 CV_32FC3 输入。
 */
TEST(ImageNetUtilTest, ImageToTensorChwRejectsInvalidInput)
{
    EXPECT_THROW({ irt::model::ImageNetUtil::imageToTensorCHW(cv::Mat()); }, irt::Exception);

    cv::Mat wrong_type(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    EXPECT_THROW({ irt::model::ImageNetUtil::imageToTensorCHW(wrong_type); }, irt::Exception);
}

/**
 * @brief imageToTensorCHW 应按通道优先顺序输出连续数据。
 */
TEST(ImageNetUtilTest, ImageToTensorChwProducesChannelMajorLayout)
{
    cv::Mat image(2, 2, CV_32FC3);
    image.at<cv::Vec3f>(0, 0) = cv::Vec3f(1.0f, 2.0f, 3.0f);
    image.at<cv::Vec3f>(0, 1) = cv::Vec3f(4.0f, 5.0f, 6.0f);
    image.at<cv::Vec3f>(1, 0) = cv::Vec3f(7.0f, 8.0f, 9.0f);
    image.at<cv::Vec3f>(1, 1) = cv::Vec3f(10.0f, 11.0f, 12.0f);

    const std::vector<float> tensor = irt::model::ImageNetUtil::imageToTensorCHW(image);

    const std::vector<float> expected = {
        1.0f, 4.0f, 7.0f, 10.0f,
        2.0f, 5.0f, 8.0f, 11.0f,
        3.0f, 6.0f, 9.0f, 12.0f,
    };
    EXPECT_EQ(tensor, expected);
}
