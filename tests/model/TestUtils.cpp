#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/Utils.hpp>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

/**
 * @brief 临时测试文件辅助类
 *
 * 构造时写入文件，析构时自动删除，避免污染仓库目录。
 */
class TempTextFile
{
public:
    explicit TempTextFile(const std::string &suffix, const std::string &content)
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_test_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + suffix);
        std::ofstream file(path_);
        file << content;
    }

    ~TempTextFile()
    {
        std::error_code ec;
        fs::remove(path_, ec);
    }

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
    EXPECT_THROW({ irt::model::loadWeights("/non/existent/path/weights.wts"); }, irt::Exception);

    try
    {
        irt::model::loadWeights("/non/existent/path/weights.wts");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
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
    EXPECT_THROW({ irt::model::readImagenetLabels("/non/existent/path/labels.txt"); }, irt::Exception);

    try
    {
        irt::model::readImagenetLabels("/non/existent/path/labels.txt");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}
