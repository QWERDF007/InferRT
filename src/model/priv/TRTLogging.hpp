#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace irt::model {

using Severity = nvinfer1::ILogger::Severity;

/**
 * @brief 日志流消费者缓冲区。
 *
 * 该类继承 std::stringbuf，在缓冲区同步或析构时将内容写入目标输出流，
 * 并按需添加时间戳和严重性前缀。
 */
class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    /**
     * @brief 构造日志缓冲区。
     * @param stream 目标输出流。
     * @param prefix 日志前缀，例如 `[I] ` 或 `[E] `。
     * @param shouldLog 是否输出当前日志。
     */
    LogStreamConsumerBuffer(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mOutput(stream)
        , mPrefix(prefix)
        , mShouldLog(shouldLog)
    {
    }

    /**
     * @brief 移动构造日志缓冲区。
     * @param other 被移动的缓冲区对象。
     */
    LogStreamConsumerBuffer(LogStreamConsumerBuffer &&other)
        : mOutput(other.mOutput)
        , mPrefix(other.mPrefix)
        , mShouldLog(other.mShouldLog)
    {
    }

    /**
     * @brief 析构日志缓冲区。
     *
     * 若缓冲区中仍有未输出内容，则在析构前写入目标流。
     */
    ~LogStreamConsumerBuffer()
    {
        if (pbase() != pptr())
        {
            putOutput();
        }
    }

    /**
     * @brief 同步流缓冲区。
     * @return 成功时返回 0。
     */
    virtual int sync()
    {
        putOutput();
        return 0;
    }

    /**
     * @brief 将缓冲区内容输出到目标流。
     *
     * 当 mShouldLog 为 true 时，会在日志前添加时间戳和严重性前缀，随后清空缓冲区。
     */
    void putOutput()
    {
        if (mShouldLog)
        {
            // 添加时间戳前缀
            std::time_t timestamp{};
            std::time(&timestamp);
            tm tm_local{};
#if defined(_WIN32)
            localtime_s(&tm_local, &timestamp);
#else
            localtime_r(&timestamp, &tm_local);
#endif
            mOutput << "[";
            mOutput << std::setw(4) << std::setfill('0') << 1900 + tm_local.tm_year;
            mOutput << std::setw(2) << std::setfill('0') << 1 + tm_local.tm_mon;
            mOutput << std::setw(2) << std::setfill('0') << tm_local.tm_mday << "-";
            mOutput << std::setw(2) << std::setfill('0') << tm_local.tm_hour << ":";
            mOutput << std::setw(2) << std::setfill('0') << tm_local.tm_min << ":";
            mOutput << std::setw(2) << std::setfill('0') << tm_local.tm_sec << "] ";
            // std::stringbuf::str() 获取缓冲区的字符串内容
            // 将带有适当前缀的缓冲区内容插入到流中
            mOutput << mPrefix << str();
            // 清空缓冲区
            str("");
            // 刷新流
            mOutput.flush();
        }
    }

    /**
     * @brief 设置当前缓冲区是否输出日志。
     * @param shouldLog 是否输出日志。
     */
    void setShouldLog(bool shouldLog)
    {
        mShouldLog = shouldLog;
    }

private:
    /// 目标输出流。
    std::ostream &mOutput;
    /// 日志严重性前缀。
    std::string   mPrefix;
    /// 是否输出当前日志。
    bool          mShouldLog;
};

/**
 * @brief 日志流消费者基类。
 *
 * 该基类保证 LogStreamConsumerBuffer 先于 std::ostream 初始化。
 */
class LogStreamConsumerBase
{
public:
    /**
     * @brief 构造日志流消费者基类。
     * @param stream 目标输出流。
     * @param prefix 日志前缀。
     * @param shouldLog 是否输出当前日志。
     */
    LogStreamConsumerBase(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog)
    {
    }

protected:
    /// 日志流缓冲区。
    LogStreamConsumerBuffer mBuffer;
};

/**
 * @class LogStreamConsumer
 * @brief 日志流消费者类
 * 
 * 便利对象，用于在记录消息时方便使用 C++ 流语法
 * 
 * 基类顺序是 LogStreamConsumerBase 然后是 std::ostream
 * 这是因为 LogStreamConsumerBase 类用于初始化 LogStreamConsumer 中的 LogStreamConsumerBuffer 成员字段
 * 然后将缓冲区的地址传递给 std::ostream
 * 这样做是为了防止将未初始化的缓冲区地址传递给 std::ostream
 * 请不要更改父类的顺序
 */
class LogStreamConsumer
    : protected LogStreamConsumerBase
    , public std::ostream
{
public:
    /**
     * @brief 构造日志流消费者。
     * @param reportableSeverity 当前允许输出的最低严重性级别。
     * @param severity 当前消息的严重性级别。
     */
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
        , std::ostream(&mBuffer)
        , mShouldLog(severity <= reportableSeverity)
        , mSeverity(severity)
    {
    }

    /**
     * @brief 移动构造日志流消费者。
     * @param other 被移动的日志流消费者。
     */
    LogStreamConsumer(LogStreamConsumer &&other)
        : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog)
        , std::ostream(&mBuffer)
        , mShouldLog(other.mShouldLog)
        , mSeverity(other.mSeverity)
    {
    }

    /**
     * @brief 设置可报告的最低严重性级别。
     * @param reportableSeverity 新的可报告严重性级别。
     */
    void setReportableSeverity(Severity reportableSeverity)
    {
        mShouldLog = mSeverity <= reportableSeverity;
        mBuffer.setShouldLog(mShouldLog);
    }

private:
    /**
     * @brief 根据严重性级别选择输出流。
     * @param severity 严重性级别。
     * @return INFO 及更低严重性输出到 std::cout，WARNING 及更高严重性输出到 std::cerr。
     */
    static std::ostream &severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    /**
     * @brief 根据严重性级别生成日志前缀。
     * @param severity 严重性级别。
     * @return 日志前缀字符串。
     */
    static std::string severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[F] "; // Fatal - 致命错误
        case Severity::kERROR:
            return "[E] "; // Error - 错误
        case Severity::kWARNING:
            return "[W] "; // Warning - 警告
        case Severity::kINFO:
            return "[I] "; // Info - 信息
        case Severity::kVERBOSE:
            return "[V] "; // Verbose - 详细信息
        default:
            assert(0);
            return "";
        }
    }

    /// 当前消息是否应输出。
    bool     mShouldLog;
    /// 当前消息的严重性级别。
    Severity mSeverity;
};

/**
 * @class Logger
 * @brief TensorRT 工具和示例的日志管理类
 * 
 * 该类为 TensorRT 工具和示例提供了一个通用的日志记录接口，支持记录两种类型的消息：
 * 
 * 1. 带有相关严重性级别的调试消息（info、warning、error 或 internal error/fatal）
 * 2. 测试通过/失败消息
 * 
 * 让所有示例使用此类进行日志记录而不是直接输出到 stdout/stderr 的优势在于
 * 控制示例输出的详细程度和格式的逻辑集中在一个位置
 * 
 * 未来，此类可以扩展以支持将测试结果转储到某种标准格式的文件中
 * （例如 JUnit XML），并提供额外的元数据（例如测试运行的持续时间）
 * 
 * TODO: 为了与现有示例向后兼容，此类直接继承自 nvinfer1::ILogger 接口
 * 这是有问题的，因为来自 TensorRT 库的消息和来自示例的消息之间没有清晰的分离
 * 
 * 未来（一旦所有示例都更新为使用 Logger::getTRTLogger() 访问 ILogger）
 * 我们可以重构该类以消除继承，而是将 nvinfer1::ILogger 实现作为 Logger 对象的成员
 */
class TRTLogger : public nvinfer1::ILogger
{
public:
    /**
     * @brief 构造日志对象。
     * @param name 日志来源名称。
     * @param severity 默认可报告严重性级别。
     */
    TRTLogger(const std::string &name, Severity severity = Severity::kWARNING)
        : mName(name)
        , mReportableSeverity(severity)
    {
    }

    /**
     * @brief 获取 TensorRT ILogger 接口引用。
     * @return 当前对象的 nvinfer1::ILogger 引用。
     */
    nvinfer1::ILogger &getTRTLogger()
    {
        return *this;
    }

    /**
     * @brief 实现 nvinfer1::ILogger::log()。
     * @param severity 消息严重性级别。
     * @param msg 消息内容。
     */
    void log(Severity severity, const char *msg) noexcept override
    {
        LogStreamConsumer(mReportableSeverity, severity) << "[" << mName << "] " << std::string(msg) << std::endl;
    }

    /**
     * @brief 设置可报告的最低严重性级别。
     * @param severity logger 将输出该级别及更高严重性的消息。
     */
    void setReportableSeverity(Severity severity)
    {
        mReportableSeverity = severity;
    }

    /**
     * @brief 获取可报告的最低严重性级别。
     * @return 当前可报告严重性级别。
     */
    Severity getReportableSeverity() const
    {
        return mReportableSeverity;
    }

private:
    /**
     * @brief 根据严重性级别生成日志前缀。
     * @param severity 严重性级别。
     * @return 日志前缀字符串。
     */
    static const char *severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[F] ";
        case Severity::kERROR:
            return "[E] ";
        case Severity::kWARNING:
            return "[W] ";
        case Severity::kINFO:
            return "[I] ";
        case Severity::kVERBOSE:
            return "[V] ";
        default:
            assert(0);
            return "";
        }
    }

    /**
     * @brief 从 argc 和 argv 生成命令行字符串。
     * @param argc 参数数量。
     * @param argv 参数数组。
     * @return 命令行字符串。
     */
    static std::string genCmdlineString(int argc, const char *const *argv)
    {
        std::stringstream ss;
        for (int i = 0; i < argc; i++)
        {
            if (i > 0)
            {
                ss << " ";
            }
            ss << argv[i];
        }
        return ss.str();
    }

    /// 可报告的最低严重性级别。
    Severity    mReportableSeverity;
    /// 日志来源名称。
    std::string mName;
};

namespace {

/**
 * @brief 生成一个可用于记录 kVERBOSE 严重性级别消息的 LogStreamConsumer 对象
 * 
 * 使用示例：
 *     LOG_VERBOSE(logger) << "hello world" << std::endl;
 * 
 * @param logger Logger 对象引用
 * @return LogStreamConsumer 对象
 */
inline LogStreamConsumer LOG_VERBOSE(const TRTLogger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
}

/**
 * @brief 生成一个可用于记录 kINFO 严重性级别消息的 LogStreamConsumer 对象
 * 
 * 使用示例：
 *     LOG_INFO(logger) << "hello world" << std::endl;
 * 
 * @param logger Logger 对象引用
 * @return LogStreamConsumer 对象
 */
inline LogStreamConsumer LOG_INFO(const TRTLogger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
}

/**
 * @brief 生成一个可用于记录 kWARNING 严重性级别消息的 LogStreamConsumer 对象
 * 
 * 使用示例：
 *     LOG_WARN(logger) << "hello world" << std::endl;
 * 
 * @param logger Logger 对象引用
 * @return LogStreamConsumer 对象
 */
inline LogStreamConsumer LOG_WARN(const TRTLogger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
}

/**
 * @brief 生成一个可用于记录 kERROR 严重性级别消息的 LogStreamConsumer 对象
 * 
 * 使用示例：
 *     LOG_ERROR(logger) << "hello world" << std::endl;
 * 
 * @param logger Logger 对象引用
 * @return LogStreamConsumer 对象
 */
inline LogStreamConsumer LOG_ERROR(const TRTLogger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
}

/**
 * @brief 生成一个可用于记录 kINTERNAL_ERROR（"致命"严重性）级别消息的 LogStreamConsumer 对象
 * 
 * 使用示例：
 *     LOG_FATAL(logger) << "hello world" << std::endl;
 * 
 * @param logger Logger 对象引用
 * @return LogStreamConsumer 对象
 */
inline LogStreamConsumer LOG_FATAL(const TRTLogger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
}

} // anonymous namespace

} // namespace irt::model
