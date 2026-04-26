#include <gtest/gtest.h>

#include <inferrt/model/Logging.hpp>

#include <sstream>
#include <string>

using namespace irt::model;

namespace {

/**
 * @brief 捕获指定输出流内容的辅助类
 *
 * 测试结束后会自动恢复原始 streambuf，避免影响其他测试。
 */
class StreamRedirect
{
public:
    explicit StreamRedirect(std::ostream &target)
        : target_(target)
        , saved_(target_.rdbuf(buffer_.rdbuf()))
    {
    }

    ~StreamRedirect()
    {
        target_.rdbuf(saved_);
    }

    std::string captured() const
    {
        return buffer_.str();
    }

    StreamRedirect(const StreamRedirect &)            = delete;
    StreamRedirect &operator=(const StreamRedirect &) = delete;

private:
    std::ostream &    target_;
    std::streambuf *  saved_;
    std::ostringstream buffer_;
};

/**
 * @brief 同时捕获 stdout / stderr，便于验证日志路由行为
 */
struct CapturedStdStreams
{
    StreamRedirect cout_redirect{std::cout};
    StreamRedirect cerr_redirect{std::cerr};
};

} // namespace

/**
 * @brief Logger 默认可报告级别应为 WARNING
 */
TEST(LoggerTest, DefaultSeverityIsWarning)
{
    Logger logger("TestLogger");
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kWARNING);
}

/**
 * @brief 构造时应支持显式指定可报告级别
 */
TEST(LoggerTest, CustomSeverityInConstructor)
{
    Logger logger("TestLogger", Severity::kVERBOSE);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kVERBOSE);
}

/**
 * @brief setReportableSeverity 应能动态更新可报告级别
 */
TEST(LoggerTest, SetReportableSeverityTakesEffect)
{
    Logger logger("TestLogger");
    logger.setReportableSeverity(Severity::kERROR);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kERROR);
}

/**
 * @brief getTRTLogger 返回的引用应指向 Logger 自身
 */
TEST(LoggerTest, GetTRTLoggerReturnsSelf)
{
    Logger logger("TestLogger");
    auto  &trt_logger = logger.getTRTLogger();
    EXPECT_EQ(&trt_logger, &logger);
}

/**
 * @brief 当 shouldLog=false 时，缓冲区同步不应输出任何内容
 */
TEST(LogStreamConsumerBufferTest, SuppressesOutputWhenShouldLogFalse)
{
    std::ostringstream oss;
    LogStreamConsumerBuffer buffer(oss, "[I] ", false);

    buffer.sputn("hidden message", 14);
    buffer.pubsync();

    EXPECT_TRUE(oss.str().empty());
}

/**
 * @brief 动态切换 shouldLog 后，后续写入应按新状态决定是否输出
 */
TEST(LogStreamConsumerBufferTest, SetShouldLogTogglesOutput)
{
    std::ostringstream oss;
    LogStreamConsumerBuffer buffer(oss, "[I] ", false);

    buffer.sputn("hidden", 6);
    buffer.pubsync();
    EXPECT_TRUE(oss.str().empty());

    buffer.setShouldLog(true);
    buffer.sputn("visible", 7);
    buffer.pubsync();

    EXPECT_NE(oss.str().find("[I] "), std::string::npos);
    EXPECT_NE(oss.str().find("visible"), std::string::npos);
}

/**
 * @brief INFO 日志在可报告级别为 INFO 时应输出到 stdout
 */
TEST(LogStreamConsumerTest, InfoSeverityOutputsToStdout)
{
    CapturedStdStreams captured;
    LogStreamConsumer consumer(Severity::kINFO, Severity::kINFO);
    consumer << "info message" << std::endl;

    EXPECT_NE(captured.cout_redirect.captured().find("[I] "), std::string::npos);
    EXPECT_NE(captured.cout_redirect.captured().find("info message"), std::string::npos);
    EXPECT_TRUE(captured.cerr_redirect.captured().empty());
}

/**
 * @brief ERROR 日志在可报告级别为 ERROR 时应输出到 stderr
 */
TEST(LogStreamConsumerTest, ErrorSeverityOutputsToStderr)
{
    CapturedStdStreams captured;
    LogStreamConsumer consumer(Severity::kERROR, Severity::kERROR);
    consumer << "error message" << std::endl;

    EXPECT_NE(captured.cerr_redirect.captured().find("[E] "), std::string::npos);
    EXPECT_NE(captured.cerr_redirect.captured().find("error message"), std::string::npos);
    EXPECT_TRUE(captured.cout_redirect.captured().empty());
}

/**
 * @brief 当日志级别低于当前可报告级别时，日志应被抑制
 */
TEST(LogStreamConsumerTest, LowerSeverityMessageIsSuppressed)
{
    CapturedStdStreams captured;
    LogStreamConsumer consumer(Severity::kWARNING, Severity::kVERBOSE);
    consumer << "suppressed message" << std::endl;

    EXPECT_EQ(captured.cout_redirect.captured().find("suppressed message"), std::string::npos);
    EXPECT_EQ(captured.cerr_redirect.captured().find("suppressed message"), std::string::npos);
}

/**
 * @brief 调整可报告级别后，之前被抑制的日志级别应可以重新输出
 */
TEST(LogStreamConsumerTest, DynamicReportableSeverityChangeEnablesOutput)
{
    CapturedStdStreams captured;
    LogStreamConsumer consumer(Severity::kERROR, Severity::kINFO);

    consumer << "hidden first" << std::endl;
    EXPECT_EQ(captured.cout_redirect.captured().find("hidden first"), std::string::npos);

    consumer.setReportableSeverity(Severity::kVERBOSE);
    consumer << "visible later" << std::endl;
    EXPECT_NE(captured.cout_redirect.captured().find("visible later"), std::string::npos);
}

/**
 * @brief LOG_VERBOSE 宏应带有 [V] 前缀并输出到 stdout
 */
TEST(LogMacroTest, VerboseMacroOutputsPrefixAndMessage)
{
    CapturedStdStreams captured;
    Logger logger("TestLogger", Severity::kVERBOSE);
    LOG_VERBOSE(logger) << "verbose message" << std::endl;

    EXPECT_NE(captured.cout_redirect.captured().find("[V] "), std::string::npos);
    EXPECT_NE(captured.cout_redirect.captured().find("verbose message"), std::string::npos);
}

/**
 * @brief LOG_INFO 宏在 ERROR 可报告级别下应被抑制
 */
TEST(LogMacroTest, InfoMacroSuppressedWhenReportableIsError)
{
    CapturedStdStreams captured;
    Logger logger("TestLogger", Severity::kERROR);
    LOG_INFO(logger) << "suppressed info" << std::endl;

    EXPECT_EQ(captured.cout_redirect.captured().find("suppressed info"), std::string::npos);
    EXPECT_EQ(captured.cerr_redirect.captured().find("suppressed info"), std::string::npos);
}

/**
 * @brief LOG_WARN 宏应输出到 stderr，并带有 [W] 前缀
 */
TEST(LogMacroTest, WarnMacroOutputsPrefixAndMessage)
{
    CapturedStdStreams captured;
    Logger logger("TestLogger", Severity::kWARNING);
    LOG_WARN(logger) << "warning message" << std::endl;

    EXPECT_NE(captured.cerr_redirect.captured().find("[W] "), std::string::npos);
    EXPECT_NE(captured.cerr_redirect.captured().find("warning message"), std::string::npos);
    EXPECT_TRUE(captured.cout_redirect.captured().empty());
}

/**
 * @brief LOG_ERROR 与 LOG_FATAL 宏都应落到 stderr，并使用正确前缀
 */
TEST(LogMacroTest, ErrorAndFatalMacrosUseExpectedPrefixes)
{
    CapturedStdStreams captured;
    Logger logger("TestLogger", Severity::kVERBOSE);

    LOG_ERROR(logger) << "error message" << std::endl;
    LOG_FATAL(logger) << "fatal message" << std::endl;

    const std::string output = captured.cerr_redirect.captured();
    EXPECT_NE(output.find("[E] "), std::string::npos);
    EXPECT_NE(output.find("error message"), std::string::npos);
    EXPECT_NE(output.find("[F] "), std::string::npos);
    EXPECT_NE(output.find("fatal message"), std::string::npos);
    EXPECT_TRUE(captured.cout_redirect.captured().empty());
}

/**
 * @brief Logger::log 输出中应包含 Logger 名称，便于定位来源
 */
TEST(LoggerOutputTest, LoggerNameAppearsInOutput)
{
    CapturedStdStreams captured;
    Logger logger("MyModel", Severity::kVERBOSE);
    logger.log(Severity::kINFO, "test message");

    EXPECT_NE(captured.cout_redirect.captured().find("[MyModel]"), std::string::npos);
    EXPECT_NE(captured.cout_redirect.captured().find("test message"), std::string::npos);
}
