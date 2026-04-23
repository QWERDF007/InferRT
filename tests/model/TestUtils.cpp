
#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <cstdio>
#include <fstream>
#include <string>

namespace t = ::testing;

namespace {

/**
 * @brief 创建临时权重文件的辅助类
 * 
 * 在构造时创建一个 .wts 格式的临时文件，在析构时删除该文件。
 * 文件格式：
 *   count
 *   name count hex_value1 hex_value2 ...
 *   ...
 */
class TempWeightsFile
{
public:
    explicit TempWeightsFile(const std::string &content)
    {
        path_ += ".wts";
        std::ofstream file(path_);
        file << content;
        file.close();
    }

    ~TempWeightsFile()
    {
        std::remove(path_.c_str());
    }

    const std::string &path() const
    {
        return path_;
    }

    // 禁止拷贝
    TempWeightsFile(const TempWeightsFile &)            = delete;
    TempWeightsFile &operator=(const TempWeightsFile &) = delete;

private:
    std::string path_{""};
};

} // anonymous namespace

// ============================================================================
// loadWeights 正常输入测试
// ============================================================================

/**
 * @brief 验证 loadWeights 可正确读取单条权重条目，包括名称、数量和十六进制值
 */
TEST(LoadWeightsTest, SingleWeightEntryParsedCorrectly)
{
    // 1 个权重条目，名称 "conv.weight"，2 个 float 值
    // float 值以 uint32_t 十六进制表示：0x3F800000 = 1.0f, 0x40000000 = 2.0f
    std::string content = "1\nconv.weight 2 3F800000 40000000\n";
    TempWeightsFile tmp(content);

    auto weights = irt::model::loadWeights(tmp.path());

    EXPECT_EQ(weights.size(), 1u);
    EXPECT_TRUE(weights.count("conv.weight") > 0);
    EXPECT_EQ(weights.at("conv.weight").count, 2);
    EXPECT_NE(weights.at("conv.weight").values, nullptr);

    // 验证十六进制值被正确读取为 uint32_t
    const auto *vals = static_cast<const uint32_t *>(weights.at("conv.weight").values);
    EXPECT_EQ(vals[0], 0x3F800000u);
    EXPECT_EQ(vals[1], 0x40000000u);
}

/**
 * @brief 验证 loadWeights 可正确读取多条权重条目，每条的数量和十六进制值均正确
 */
TEST(LoadWeightsTest, MultipleWeightEntriesParsedCorrectly)
{
    std::string content = "3\n"
                          "conv1.weight 1 3F800000\n"
                          "conv1.bias 1 40000000\n"
                          "conv2.weight 3 3F800000 FF8800FF 80000000\n";
    TempWeightsFile tmp(content);

    auto weights = irt::model::loadWeights(tmp.path());

    EXPECT_EQ(weights.size(), 3u);
    EXPECT_TRUE(weights.count("conv1.weight") > 0);
    EXPECT_TRUE(weights.count("conv1.bias") > 0);
    EXPECT_TRUE(weights.count("conv2.weight") > 0);

    EXPECT_EQ(weights.at("conv1.weight").count, 1);
    EXPECT_EQ(weights.at("conv1.bias").count, 1);
    EXPECT_EQ(weights.at("conv2.weight").count, 3);

    // 验证十六进制值被正确读取为 uint32_t
    const auto *vals = static_cast<const uint32_t *>(weights.at("conv1.weight").values);
    EXPECT_EQ(vals[0], 0x3F800000u);
    
    vals = static_cast<const uint32_t *>(weights.at("conv1.bias").values);
    EXPECT_EQ(vals[0], 0x40000000u);
    
    vals = static_cast<const uint32_t *>(weights.at("conv2.weight").values);
    EXPECT_EQ(vals[0], 0x3F800000u);
    EXPECT_EQ(vals[1], 0xFF8800FFu);
    EXPECT_EQ(vals[2], 0x80000000u);
}

/**
 * @brief 验证 loadWeights 可处理元素数量为 0 的权重条目
 */
TEST(LoadWeightsTest, ZeroCountWeightEntryIsValid)
{
    // 权重值为 0 个元素
    std::string content = "1\nempty.weight 0\n";
    TempWeightsFile tmp(content);

    auto weights = irt::model::loadWeights(tmp.path());

    EXPECT_EQ(weights.size(), 1u);
    EXPECT_EQ(weights.at("empty.weight").count, 0);
}

// ============================================================================
// loadWeights 异常输入测试
// ============================================================================

/**
 * @brief 验证 loadWeights 对不存在的文件抛出 irt::Exception
 */
TEST(LoadWeightsTest, NonExistentFileThrowsException)
{
    EXPECT_THROW({ irt::model::loadWeights("/non/existent/path/weights.wts"); }, irt::Exception);
}

/**
 * @brief 验证 loadWeights 对条目数为 0 的文件抛出 irt::Exception
 */
TEST(LoadWeightsTest, ZeroEntryCountThrowsException)
{
    // count = 0，应抛出异常
    std::string content = "0\n";
    TempWeightsFile tmp(content);

    EXPECT_THROW({ irt::model::loadWeights(tmp.path()); }, irt::Exception);
}

/**
 * @brief 验证 loadWeights 对负数条目数的文件抛出 irt::Exception
 */
TEST(LoadWeightsTest, NegativeEntryCountThrowsException)
{
    // count = -1，应抛出异常
    std::string content = "-1\n";
    TempWeightsFile tmp(content);

    EXPECT_THROW({ irt::model::loadWeights(tmp.path()); }, irt::Exception);
}

/**
 * @brief 验证 loadWeights 对空文件抛出 irt::Exception
 */
TEST(LoadWeightsTest, EmptyFileThrowsException)
{
    // 空文件无法读取 count
    std::string content = "";
    TempWeightsFile tmp(content);

    EXPECT_THROW({ irt::model::loadWeights(tmp.path()); }, irt::Exception);
}
