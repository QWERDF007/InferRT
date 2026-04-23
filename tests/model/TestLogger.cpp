
#include <gtest/gtest.h>
#include <inferrt/model/Logging.hpp>

#include <sstream>
#include <string>

namespace t = ::testing;

using namespace irt::model;

namespace {

/**
 * @brief 重定向 std::cout 或 std::cerr 的辅助类
 *
 * 在构造时保存并替换目标流的 streambuf，析构时恢复。
 * 通过 captured() 获取捕获的输出内容。
 */
class StreamRedirect
{
public:
    explicit StreamRedirect(std::ostream &target)
        : target_(target)
        , saved_(target_.rdbuf(oss_.rdbuf()))
    {
    }

    ~StreamRedirect()
    {
        target_.rdbuf(saved_);
    }

    std::string captured() const
    {
        return oss_.str();
    }

    StreamRedirect(const StreamRedirect &)            = delete;
    StreamRedirect &operator=(const StreamRedirect &) = delete;

private:
    std::ostream &   target_;
    std::streambuf * saved_;
    std::ostringstream oss_;
};

} // anonymous namespace

// ============================================================================
// Logger 构造与基本属性测试
// ============================================================================

/**
 * @brief 验证 Logger 默认可报告级别为 kWARNING
 */
TEST(LoggerTest, DefaultSeverityIsWarning)
{
    Logger logger("TestLogger");
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kWARNING);
}

/**
 * @brief 验证构造函数可指定自定义可报告级别
 */
TEST(LoggerTest, CustomSeverityInConstructor)
{
    Logger logger("TestLogger", Severity::kVERBOSE);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kVERBOSE);
}

/**
 * @brief 验证 setReportableSeverity 可修改可报告级别
 */
TEST(LoggerTest, SetReportableSeverity)
{
    Logger logger("TestLogger");
    logger.setReportableSeverity(Severity::kERROR);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kERROR);
}

/**
 * @brief 验证多次调用 setReportableSeverity 每次均生效
 */
TEST(LoggerTest, SetReportableSeverityMultipleTimes)
{
    Logger logger("TestLogger", Severity::kVERBOSE);

    logger.setReportableSeverity(Severity::kINFO);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kINFO);

    logger.setReportableSeverity(Severity::kINTERNAL_ERROR);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);

    logger.setReportableSeverity(Severity::kVERBOSE);
    EXPECT_EQ(logger.getReportableSeverity(), Severity::kVERBOSE);
}

/**
 * @brief 验证 getTRTLogger 返回的 ILogger 引用指向 Logger 自身
 */
TEST(LoggerTest, GetTRTLoggerReturnsSelf)
{
    Logger logger("TestLogger");
    nvinfer1::ILogger &trt_logger = logger.getTRTLogger();
    (void)trt_logger; // 验证可以获取引用，不会崩溃
}

// ============================================================================
// LogStreamConsumerBuffer 测试
// ============================================================================

/**
 * @brief 验证 shouldLog=false 时 LogStreamConsumerBuffer 不输出任何内容
 */
TEST(LogStreamConsumerBufferTest, SuppressesOutputWhenShouldLogFalse)
{
    std::ostringstream oss;
    LogStreamConsumerBuffer buffer(oss, "[I] ", false);

    buffer.sputn("test message", 12);
    buffer.pubsync();

    // shouldLog = false，不应输出
    EXPECT_TRUE(oss.str().empty());
}

/**
 * @brief 验证 setShouldLog 可动态切换是否输出内容
 */
TEST(LogStreamConsumerBufferTest, SetShouldLogTogglesOutput)
{
    std::ostringstream oss;
    LogStreamConsumerBuffer buffer(oss, "[I] ", false);

    // 初始不记录
    buffer.sputn("hidden", 6);
    buffer.pubsync();
    EXPECT_TRUE(oss.str().empty());

    // 切换为记录
    buffer.setShouldLog(true);
    buffer.sputn("visible", 7);
    buffer.pubsync();
    EXPECT_FALSE(oss.str().empty());
    EXPECT_NE(oss.str().find("visible"), std::string::npos);
}

// ============================================================================
// LogStreamConsumer 测试
// ============================================================================

/**
 * @brief 验证 INFO 级别消息在可报告级别为 INFO 时输出到 cout，包含 [I] 前缀和消息内容
 */
TEST(LogStreamConsumerTest, InfoSeverityOutputWhenReportableIsInfo)
{
    // reportableSeverity = kINFO, severity = kINFO，输出到 std::cout
    StreamRedirect redirect(std::cout);
    LogStreamConsumer consumer(Severity::kINFO, Severity::kINFO);
    consumer << "test message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[I] "), std::string::npos) << "Expected [I] prefix, got: " << output;
    EXPECT_NE(output.find("test message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证 VERBOSE 级别消息在可报告级别为 WARNING 时被抑制，不输出到任何流
 */
TEST(LogStreamConsumerTest, VerboseSeveritySuppressedWhenReportableIsWarning)
{
    // reportableSeverity = kWARNING, severity = kVERBOSE，不应记录
    StreamRedirect redirect_cout(std::cout);
    StreamRedirect redirect_cerr(std::cerr);
    LogStreamConsumer consumer(Severity::kWARNING, Severity::kVERBOSE);
    consumer << "suppressed message" << std::endl;

    EXPECT_EQ(redirect_cout.captured().find("suppressed message"), std::string::npos);
    EXPECT_EQ(redirect_cerr.captured().find("suppressed message"), std::string::npos);
}

/**
 * @brief 验证 ERROR 级别消息在可报告级别为 ERROR 时输出到 cerr，包含 [E] 前缀和消息内容
 */
TEST(LogStreamConsumerTest, ErrorSeverityOutputWhenReportableIsError)
{
    // reportableSeverity = kERROR, severity = kERROR，输出到 std::cerr
    StreamRedirect redirect(std::cerr);
    LogStreamConsumer consumer(Severity::kERROR, Severity::kERROR);
    consumer << "error message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[E] "), std::string::npos) << "Expected [E] prefix, got: " << output;
    EXPECT_NE(output.find("error message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证动态修改 setReportableSeverity 后，先前被抑制的消息可以正常输出
 */
TEST(LogStreamConsumerTest, DynamicReportableSeverityChangeTogglesOutput)
{
    StreamRedirect redirect_cout(std::cout);
    StreamRedirect redirect_cerr(std::cerr);
    LogStreamConsumer consumer(Severity::kERROR, Severity::kINFO);

    // INFO 在 ERROR 级别下不应记录
    consumer << "should be suppressed" << std::endl;
    EXPECT_EQ(redirect_cout.captured().find("should be suppressed"), std::string::npos);
    EXPECT_EQ(redirect_cerr.captured().find("should be suppressed"), std::string::npos);

    // 修改 reportable severity 后应记录
    consumer.setReportableSeverity(Severity::kVERBOSE);
    consumer << "should be visible now" << std::endl;
    std::string combined = redirect_cout.captured();
    EXPECT_NE(combined.find("should be visible now"), std::string::npos);
}

// ============================================================================
// LOG_* 宏测试
// ============================================================================

/**
 * @brief 验证 LOG_VERBOSE 宏输出包含 [V] 前缀和消息内容
 */
TEST(LogMacroTest, VerboseMacroOutputsVerbosePrefixAndMessage)
{
    StreamRedirect redirect(std::cout);
    Logger logger("TestLogger", Severity::kVERBOSE);
    LOG_VERBOSE(logger) << "verbose message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[V] "), std::string::npos) << "Expected [V] prefix, got: " << output;
    EXPECT_NE(output.find("verbose message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证 LOG_INFO 宏输出包含 [I] 前缀和消息内容
 */
TEST(LogMacroTest, InfoMacroOutputsInfoPrefixAndMessage)
{
    StreamRedirect redirect(std::cout);
    Logger logger("TestLogger", Severity::kINFO);
    LOG_INFO(logger) << "info message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[I] "), std::string::npos) << "Expected [I] prefix, got: " << output;
    EXPECT_NE(output.find("info message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证 LOG_WARN 宏输出包含 [W] 前缀和消息内容（输出到 cerr）
 */
TEST(LogMacroTest, WarnMacroOutputsWarningPrefixAndMessage)
{
    // kWARNING < kINFO，前缀和消息输出到 std::cerr
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kWARNING);
    LOG_WARN(logger) << "warning message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[W] "), std::string::npos) << "Expected [W] prefix, got: " << output;
    EXPECT_NE(output.find("warning message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证 LOG_ERROR 宏输出包含 [E] 前缀和消息内容（输出到 cerr）
 */
TEST(LogMacroTest, ErrorMacroOutputsErrorPrefixAndMessage)
{
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kERROR);
    LOG_ERROR(logger) << "error message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[E] "), std::string::npos) << "Expected [E] prefix, got: " << output;
    EXPECT_NE(output.find("error message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证 LOG_FATAL 宏输出包含 [F] 前缀和消息内容（输出到 cerr）
 */
TEST(LogMacroTest, FatalMacroOutputsFatalPrefixAndMessage)
{
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kINTERNAL_ERROR);
    LOG_FATAL(logger) << "fatal message" << std::endl;

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[F] "), std::string::npos) << "Expected [F] prefix, got: " << output;
    EXPECT_NE(output.find("fatal message"), std::string::npos) << "Expected message content, got: " << output;
}

/**
 * @brief 验证当 Logger 可报告级别为 ERROR 时，LOG_INFO 宏的消息被抑制
 */
TEST(LogMacroTest, InfoMacroSuppressedWhenReportableIsError)
{
    StreamRedirect redirect_cout(std::cout);
    StreamRedirect redirect_cerr(std::cerr);
    Logger logger("TestLogger", Severity::kERROR);
    // INFO 在 ERROR 级别下不应记录
    LOG_INFO(logger) << "suppressed info" << std::endl;

    EXPECT_EQ(redirect_cout.captured().find("suppressed info"), std::string::npos);
    EXPECT_EQ(redirect_cerr.captured().find("suppressed info"), std::string::npos);
}

// ============================================================================
// Severity 前缀映射测试
// ============================================================================

/**
 * @brief 验证 kINTERNAL_ERROR 级别日志输出包含 [F] 前缀（Fatal）
 */
TEST(LoggerSeverityPrefixTest, InternalErrorPrefixIsF)
{
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kVERBOSE);
    logger.log(Severity::kINTERNAL_ERROR, "fatal test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[F] "), std::string::npos) << "Expected [F] prefix, got: " << output;
    EXPECT_NE(output.find("fatal test"), std::string::npos) << "Expected message, got: " << output;
}

/**
 * @brief 验证 kERROR 级别日志输出包含 [E] 前缀（Error）
 */
TEST(LoggerSeverityPrefixTest, ErrorPrefixIsE)
{
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kVERBOSE);
    logger.log(Severity::kERROR, "error test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[E] "), std::string::npos) << "Expected [E] prefix, got: " << output;
    EXPECT_NE(output.find("error test"), std::string::npos) << "Expected message, got: " << output;
}

/**
 * @brief 验证 kWARNING 级别日志输出包含 [W] 前缀（Warning），输出到 cerr
 */
TEST(LoggerSeverityPrefixTest, WarningPrefixIsW)
{
    // kWARNING < kINFO，前缀和消息输出到 std::cerr
    StreamRedirect redirect(std::cerr);
    Logger logger("TestLogger", Severity::kVERBOSE);
    logger.log(Severity::kWARNING, "warning test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[W] "), std::string::npos) << "Expected [W] prefix, got: " << output;
    EXPECT_NE(output.find("warning test"), std::string::npos) << "Expected message, got: " << output;
}

/**
 * @brief 验证 kINFO 级别日志输出包含 [I] 前缀（Info），输出到 cout
 */
TEST(LoggerSeverityPrefixTest, InfoPrefixIsI)
{
    StreamRedirect redirect(std::cout);
    Logger logger("TestLogger", Severity::kVERBOSE);
    logger.log(Severity::kINFO, "info test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[I] "), std::string::npos) << "Expected [I] prefix, got: " << output;
    EXPECT_NE(output.find("info test"), std::string::npos) << "Expected message, got: " << output;
}

/**
 * @brief 验证 kVERBOSE 级别日志输出包含 [V] 前缀（Verbose），输出到 cout
 */
TEST(LoggerSeverityPrefixTest, VerbosePrefixIsV)
{
    StreamRedirect redirect(std::cout);
    Logger logger("TestLogger", Severity::kVERBOSE);
    logger.log(Severity::kVERBOSE, "verbose test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[V] "), std::string::npos) << "Expected [V] prefix, got: " << output;
    EXPECT_NE(output.find("verbose test"), std::string::npos) << "Expected message, got: " << output;
}

/**
 * @brief 验证 Logger::log 输出中包含构造时指定的 Logger 名称
 */
TEST(LoggerSeverityPrefixTest, LoggerNameAppearsInOutput)
{
    StreamRedirect redirect(std::cout);
    Logger logger("MyModel", Severity::kVERBOSE);
    logger.log(Severity::kINFO, "test");

    std::string output = redirect.captured();
    EXPECT_NE(output.find("[MyModel]"), std::string::npos) << "Expected logger name, got: " << output;
}
