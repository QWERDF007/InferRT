#include "TestModelCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/Utils.hpp>

#include "priv/TRTUtils.hpp"
#include "priv/Weights.hpp"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST(LoadWeightsTest, TruncatedWeightDataThrowsInvalidArgument)
{
    TempTextFile tmp(".wts", "2\nfirst.weight 2 3F800000\nsecond.weight 1\n");

    ExpectIrtExceptionCode([&] { irt::model::loadWeights(tmp.path().string()); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(LoadWeightsTest, DuplicateWeightNamesThrowInvalidArgument)
{
    TempTextFile tmp(".wts", "2\nshared.weight 1 3F800000\nshared.weight 1 40000000\n");

    ExpectIrtExceptionCode([&] { irt::model::loadWeights(tmp.path().string()); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
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
 * @brief elementSize 应返回核心数据类型的单元素字节数。
 */
TEST(ModelUtilTest, ElementSizeMatchesCoreTypes)
{
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::F32), 4U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::F16), 2U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::I8), 1U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::U8), 1U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::I32), 4U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::I64), 8U);
    EXPECT_EQ(irt::model::elementSize(irt::TensorDataType::Bool), 1U);
    EXPECT_EQ(irt::model::dataTypeSize(irt::TensorDataType::F32), irt::model::elementSize(irt::TensorDataType::F32));
}

/**
 * @brief elementSize 对未知核心数据类型应抛出未实现异常，避免静默返回错误字节数。
 */
TEST(ModelUtilTest, ElementSizeRejectsUnknownDataType)
{
    ExpectIrtExceptionCode([&] { irt::model::elementSize(static_cast<irt::TensorDataType>(999)); },
                           irt::Status::ERROR_NOT_IMPLEMENTED);
}

/**
 * @brief elementCount 应计算维度乘积，并拒绝未解析或非法维度。
 */
TEST(ModelUtilTest, ElementCountValidatesTensorDims)
{
    irt::Shape dims{1, 3, 224, 224};
    EXPECT_EQ(irt::model::elementCount(dims), 150528U);

    dims[2] = 0;
    EXPECT_THROW({ irt::model::elementCount(dims); }, irt::Exception);
}

/**
 * @brief 空形状和负维度均应沿用 core Shape 的非法参数契约。
 */
TEST(ModelUtilTest, RejectsEmptyAndNegativeDims)
{
    irt::Shape scalar{};
    ExpectIrtExceptionCode([&] { (void)irt::model::elementCount(scalar); }, irt::Status::ERROR_INVALID_ARGUMENT);

    irt::Shape invalid{4, -1};
    ExpectIrtExceptionCode([&] { (void)irt::model::elementCount(invalid); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief dtype 与维度格式化工具应输出稳定文本，供日志和 manifest 复用。
 */
TEST(ModelUtilTest, FormatsDataTypeAndDims)
{
    irt::Shape dims{3, 224, 224};

    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::F32), "float32");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::F16), "float16");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::I8), "int8");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::U8), "uint8");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::I32), "int32");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::I64), "int64");
    EXPECT_EQ(irt::model::dataTypeToString(irt::TensorDataType::Bool), "bool");
    EXPECT_EQ(irt::model::dimsToCsv(dims), "3,224,224");
    EXPECT_EQ(irt::model::dimsToString(dims), "[3,224,224]");
}

/**
 * @brief dtype 与维度格式化工具应对未知类型和空维度输出稳定文本。
 */
TEST(ModelUtilTest, FormatsUnknownDataTypeAndEmptyDims)
{
    irt::Shape dims{};

    EXPECT_EQ(irt::model::dataTypeToString(static_cast<irt::TensorDataType>(999)), "unknown");
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
 * @brief DeviceBuffer 按非法核心形状 resize 时，应在申请显存前抛出异常。
 */
TEST(DeviceBufferTest, ResizeRejectsInvalidDimsBeforeAllocation)
{
    irt::model::DeviceBuffer buffer;
    irt::Shape dims{4, 0};

    EXPECT_THROW({ buffer.resize(dims); }, irt::Exception);
    EXPECT_TRUE(buffer.empty());
}

TEST(ModelUtilTest, CheckedWeightProductValidatesDimensions)
{
    EXPECT_EQ(irt::model::checkedWeightProduct({2, 3, 4}, "test weight"), 24);

    ExpectIrtExceptionCode([&] { (void)irt::model::checkedWeightProduct({2, -1}, "test weight"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
    ExpectIrtExceptionCode(
        [&] {
            (void)irt::model::checkedWeightProduct({std::numeric_limits<int64_t>::max(), 2}, "test weight");
        },
        irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(HostBufferTest, ResizeRejectsElementToByteOverflowBeforeAllocation)
{
    irt::model::HostBuffer buffer;

    EXPECT_THROW({ buffer.resize(std::numeric_limits<size_t>::max(), irt::TensorDataType::F32); }, irt::Exception);
    EXPECT_TRUE(buffer.empty());
}

/**
 * @brief HostBuffer 应按元素数量和数据类型计算字节数，并分配主机内存。
 */
TEST(HostBufferTest, AllocatesHostMemoryByElementCountAndType)
{
    irt::model::HostBuffer buffer(4, irt::TensorDataType::F32);

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
    irt::model::HostBuffer buffer(irt::TensorDataType::F16);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(buffer.dataType(), irt::TensorDataType::F16);
    EXPECT_EQ(buffer.sizeBytes(), 0U);
}

/**
 * @brief HostBuffer 按核心形状 resize 时应复用 elementCount 计算元素数量。
 */
TEST(HostBufferTest, ResizeByDimsComputesElementCount)
{
    irt::Shape dims{2, 3, 4};

    irt::model::HostBuffer buffer;
    buffer.resize(dims, irt::TensorDataType::I32);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 24U);
    EXPECT_EQ(buffer.sizeBytes(), 24U * sizeof(int32_t));
    EXPECT_EQ(buffer.dataType(), irt::TensorDataType::I32);
}

/**
 * @brief HostBuffer 切换数据类型时应重新按新元素大小计算逻辑字节数。
 */
TEST(HostBufferTest, ResizeWithDifferentTypeUpdatesByteSize)
{
    irt::model::HostBuffer buffer(4, irt::TensorDataType::I8);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.sizeBytes(), 4U);

    buffer.resize(4, irt::TensorDataType::F32);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.dataType(), irt::TensorDataType::F32);
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer.sizeBytes(), 4U * sizeof(float));
}

/**
 * @brief HostBuffer resize 小于当前容量时应只更新逻辑大小，不释放已有内存。
 */
TEST(HostBufferTest, ResizeWithinCapacityKeepsAllocation)
{
    irt::model::HostBuffer buffer(8, irt::TensorDataType::I32);
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
    irt::model::HostBuffer buffer(4, irt::TensorDataType::I8);

    buffer.resize(0, irt::TensorDataType::F32);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.capacity(), 0U);
    EXPECT_EQ(buffer.dataType(), irt::TensorDataType::F32);

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
    irt::model::HostBuffer source(3, irt::TensorDataType::I8);
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
    irt::model::HostBuffer buffer(4, irt::TensorDataType::I32);
    ASSERT_FALSE(buffer.empty());

    buffer.reset();

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
    EXPECT_EQ(buffer.capacity(), 0U);
    EXPECT_EQ(buffer.dataType(), irt::TensorDataType::I32);
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

TEST(ImageNetUtilTest, PreprocessSpecProvidesLetterboxGeometryAndNormalization)
{
    irt::PreprocessSpec spec;
    spec.input_width     = 4;
    spec.input_height    = 4;
    spec.input_channels  = 3;
    spec.source_channels = 3;
    spec.padding_mode    = irt::PaddingMode::Letterbox;
    spec.pad_value       = 10.0F;
    spec.scale            = 1.0F;
    spec.mean            = {0.0F, 0.0F, 0.0F};
    spec.stddev          = {1.0F, 1.0F, 1.0F};

    cv::Mat image(2, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(image, spec);

    EXPECT_EQ(result.geometry.original_width, 4);
    EXPECT_EQ(result.geometry.original_height, 2);
    EXPECT_EQ(result.geometry.resized_width, 4);
    EXPECT_EQ(result.geometry.resized_height, 2);
    EXPECT_EQ(result.geometry.pad_left, 0);
    EXPECT_EQ(result.geometry.pad_top, 1);
    ASSERT_EQ(result.image.type(), CV_32FC3);

    const cv::Vec3f padded = result.image.at<cv::Vec3f>(0, 0);
    EXPECT_FLOAT_EQ(padded[0], 10.0F);
    EXPECT_FLOAT_EQ(padded[1], 10.0F);
    EXPECT_FLOAT_EQ(padded[2], 10.0F);

    const cv::Vec3f content = result.image.at<cv::Vec3f>(1, 0);
    EXPECT_FLOAT_EQ(content[0], 3.0F);
    EXPECT_FLOAT_EQ(content[1], 2.0F);
    EXPECT_FLOAT_EQ(content[2], 1.0F);
}

TEST(ImageNetUtilTest, PreprocessSpecCanPadAfterNormalization)
{
    irt::PreprocessSpec spec;
    spec.input_width          = 4;
    spec.input_height         = 4;
    spec.input_channels       = 3;
    spec.source_channels      = 3;
    spec.padding_mode         = irt::PaddingMode::Letterbox;
    spec.pad_value             = 10.0F;
    spec.pad_after_normalize   = true;
    spec.scale                 = 1.0F;
    spec.mean                  = {0.0F, 0.0F, 0.0F};
    spec.stddev                = {1.0F, 1.0F, 1.0F};

    const cv::Mat image(2, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(image, spec);

    ASSERT_EQ(result.image.type(), CV_32FC3);
    EXPECT_EQ(result.geometry.pad_top, 1);
    EXPECT_EQ(result.image.at<cv::Vec3f>(0, 0), cv::Vec3f(0.0F, 0.0F, 0.0F));
    EXPECT_EQ(result.image.at<cv::Vec3f>(3, 3), cv::Vec3f(0.0F, 0.0F, 0.0F));
    EXPECT_EQ(result.image.at<cv::Vec3f>(1, 0), cv::Vec3f(3.0F, 2.0F, 1.0F));
}

TEST(ImageNetUtilTest, PreprocessSpecSupportsTopLeftLetterboxAlignment)
{
    irt::PreprocessSpec spec;
    spec.input_width       = 4;
    spec.input_height      = 4;
    spec.input_channels    = 3;
    spec.source_channels   = 3;
    spec.padding_mode      = irt::PaddingMode::Letterbox;
    spec.padding_alignment = irt::PaddingAlignment::TopLeft;
    spec.pad_after_normalize = true;
    spec.scale              = 1.0F;
    spec.mean               = {0.0F, 0.0F, 0.0F};
    spec.stddev             = {1.0F, 1.0F, 1.0F};

    const cv::Mat image(2, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(image, spec);

    EXPECT_EQ(result.geometry.pad_left, 0);
    EXPECT_EQ(result.geometry.pad_top, 0);
    EXPECT_EQ(result.geometry.resized_width, 4);
    EXPECT_EQ(result.geometry.resized_height, 2);
    EXPECT_EQ(result.image.at<cv::Vec3f>(0, 0), cv::Vec3f(3.0F, 2.0F, 1.0F));
    EXPECT_EQ(result.image.at<cv::Vec3f>(3, 3), cv::Vec3f(0.0F, 0.0F, 0.0F));
}

TEST(ImageNetUtilTest, PreprocessSpecSupportsBgraToRgb)
{
    irt::PreprocessSpec spec;
    spec.input_width     = 1;
    spec.input_height    = 1;
    spec.input_channels  = 3;
    spec.source_channels = 4;
    spec.src_color       = irt::ColorFormat::BGRA;
    spec.dst_color       = irt::ColorFormat::RGB;
    spec.scale            = 1.0F;
    spec.mean             = {0.0F, 0.0F, 0.0F};
    spec.stddev           = {1.0F, 1.0F, 1.0F};

    cv::Mat image(1, 1, CV_8UC4);
    image.at<cv::Vec4b>(0, 0) = cv::Vec4b(1, 2, 3, 255);
    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(image, spec);

    ASSERT_EQ(result.image.type(), CV_32FC3);
    EXPECT_EQ(result.image.at<cv::Vec3f>(0, 0), cv::Vec3f(3.0F, 2.0F, 1.0F));
}

TEST(ImageNetUtilTest, PreprocessSpecSupportsGrayAndNonContinuousInput)
{
    irt::PreprocessSpec spec;
    spec.input_width     = 2;
    spec.input_height    = 2;
    spec.input_channels  = 1;
    spec.source_channels = 1;
    spec.src_color       = irt::ColorFormat::GRAY;
    spec.dst_color       = irt::ColorFormat::GRAY;
    spec.scale            = 1.0F;
    spec.mean            = {1.0F};
    spec.stddev          = {2.0F};

    cv::Mat backing(3, 3, CV_8UC1, cv::Scalar(5));
    const cv::Mat roi = backing(cv::Rect(1, 1, 2, 2));
    ASSERT_FALSE(roi.isContinuous());
    const auto result = irt::model::ImageNetUtil::preprocessWithGeometry(roi, spec);
    ASSERT_EQ(result.image.type(), CV_32FC1);
    EXPECT_FLOAT_EQ(result.image.at<float>(0, 0), 2.0F);

    const auto tensor = irt::model::ImageNetUtil::imageToTensorCHW(result.image);
    EXPECT_EQ(tensor, std::vector<float>({2.0F, 2.0F, 2.0F, 2.0F}));
}

TEST(ImageNetUtilTest, PreprocessSpecRejectsColorChannelMismatch)
{
    irt::PreprocessSpec spec;
    spec.input_width     = 2;
    spec.input_height    = 2;
    spec.input_channels  = 3;
    spec.source_channels = 4;
    spec.src_color       = irt::ColorFormat::BGRA;
    spec.dst_color       = irt::ColorFormat::RGB;
    spec.mean            = {0.0F, 0.0F, 0.0F};
    spec.stddev          = {1.0F, 1.0F, 1.0F};

    cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_THROW(irt::model::ImageNetUtil::preprocess(bgr, spec), irt::Exception);
}
