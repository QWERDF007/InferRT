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
 * @class LogStreamConsumerBuffer
 * @brief 日志流消费者缓冲区类，继承自 std::stringbuf
 * 
 * 该类用于管理日志输出的缓冲区，支持添加前缀和时间戳
 */
class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    /**
     * @brief 构造函数
     * @param stream 输出流引用（如 std::cout 或 std::cerr）
     * @param prefix 日志前缀字符串（如 "[I] ", "[E] " 等）
     * @param shouldLog 是否应该记录日志的标志
     */
    LogStreamConsumerBuffer(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mOutput(stream)
        , mPrefix(prefix)
        , mShouldLog(shouldLog)
    {
    }

    /**
     * @brief 移动构造函数
     * @param other 要移动的源对象
     */
    LogStreamConsumerBuffer(LogStreamConsumerBuffer &&other)
        : mOutput(other.mOutput)
        , mPrefix(other.mPrefix)
        , mShouldLog(other.mShouldLog)
    {
    }

    /**
     * @brief 析构函数
     * 
     * 在对象销毁时，如果缓冲区中还有未输出的内容，则将其输出
     * std::streambuf::pbase() 返回指向输出序列缓冲部分开始位置的指针
     * std::streambuf::pptr() 返回指向输出序列当前位置的指针
     * 如果开始位置指针不等于当前位置指针，说明缓冲区中有内容需要输出
     */
    ~LogStreamConsumerBuffer()
    {
        if (pbase() != pptr())
        {
            putOutput();
        }
    }

    /**
     * @brief 同步流缓冲区
     * @return 成功返回 0
     * 
     * 同步操作包括：将缓冲区内容插入到流中，重置缓冲区，并刷新流
     */
    virtual int sync()
    {
        putOutput();
        return 0;
    }

    /**
     * @brief 将缓冲区内容输出到流
     * 
     * 如果 mShouldLog 为 true，则在日志前添加时间戳和前缀，然后输出缓冲区内容
     */
    void putOutput()
    {
        if (mShouldLog)
        {
            // 添加时间戳前缀
            std::time_t timestamp = std::time(nullptr);
            tm         *tm_local  = std::localtime(&timestamp);
            std::cout << "[";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
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
     * @brief 设置是否应该记录日志
     * @param shouldLog 是否记录日志的标志
     */
    void setShouldLog(bool shouldLog)
    {
        mShouldLog = shouldLog;
    }

private:
    std::ostream &mOutput;    // 输出流引用
    std::string   mPrefix;    // 日志前缀
    bool          mShouldLog; // 是否应该记录日志
};

/**
 * @class LogStreamConsumerBase
 * @brief 日志流消费者基类
 * 
 * 便利对象，用于在 LogStreamConsumer 中的 std::ostream 之前初始化 LogStreamConsumerBuffer
 */
class LogStreamConsumerBase
{
public:
    /**
     * @brief 构造函数
     * @param stream 输出流引用
     * @param prefix 日志前缀
     * @param shouldLog 是否应该记录日志
     */
    LogStreamConsumerBase(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog)
    {
    }

protected:
    LogStreamConsumerBuffer mBuffer; // 日志流缓冲区
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
     * @brief 构造函数，创建一个记录指定严重性级别消息的 LogStreamConsumer
     * @param reportableSeverity 可报告的严重性级别，决定消息是否足够严重以被记录
     * @param severity 当前消息的严重性级别
     */
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
        , std::ostream(&mBuffer) // 将流缓冲区与流关联
        , mShouldLog(severity <= reportableSeverity)
        , mSeverity(severity)
    {
    }

    /**
     * @brief 移动构造函数
     * @param other 要移动的源对象
     */
    LogStreamConsumer(LogStreamConsumer &&other)
        : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog)
        , std::ostream(&mBuffer) // 将流缓冲区与流关联
        , mShouldLog(other.mShouldLog)
        , mSeverity(other.mSeverity)
    {
    }

    /**
     * @brief 设置可报告的严重性级别
     * @param reportableSeverity 新的可报告严重性级别
     */
    void setReportableSeverity(Severity reportableSeverity)
    {
        mShouldLog = mSeverity <= reportableSeverity;
        mBuffer.setShouldLog(mShouldLog);
    }

private:
    /**
     * @brief 根据严重性级别返回相应的输出流
     * @param severity 严重性级别
     * @return INFO 及以上级别返回 std::cout，否则返回 std::cerr
     */
    static std::ostream &severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    /**
     * @brief 根据严重性级别返回相应的前缀字符串
     * @param severity 严重性级别
     * @return 前缀字符串（如 "[I] ", "[E] " 等）
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

    bool     mShouldLog; // 是否应该记录日志
    Severity mSeverity;  // 当前消息的严重性级别
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
class Logger : public nvinfer1::ILogger
{
public:
    /**
     * @brief 构造函数
     * @param severity 默认的严重性级别，默认为 WARNING
     */
    Logger(const std::string &name, Severity severity = Severity::kWARNING)
        : mName(name)
        , mReportableSeverity(severity)
    {
    }

    /**
     * @brief 获取与此 Logger 关联的 nvinfer::ILogger 的前向兼容方法
     * @return 与此 Logger 关联的 nvinfer1::ILogger
     * 
     * TODO: 一旦所有示例都更新为使用此方法向 TensorRT 注册 logger
     * 我们就可以消除 Logger 对 ILogger 的继承
     */
    nvinfer1::ILogger &getTRTLogger()
    {
        return *this;
    }

    /**
     * @brief nvinfer1::ILogger::log() 虚方法的实现
     * 
     * 注意：示例不应直接调用此函数；一旦我们消除对 nvinfer1::ILogger 的继承
     * 此函数最终将被移除
     * 
     * @param severity 消息的严重性级别
     * @param msg 消息内容
     */
    void log(Severity severity, const char *msg) noexcept override
    {
        LogStreamConsumer(mReportableSeverity, severity) << "[" << mName << "]: " << std::string(msg) << std::endl;
    }

    /**
     * @brief 控制日志输出详细程度的方法
     * @param severity logger 只会输出此级别或更高级别的消息
     */
    void setReportableSeverity(Severity severity)
    {
        mReportableSeverity = severity;
    }

    /**
     * @brief 获取可报告的严重性级别
     * @return 当前的可报告严重性级别
     */
    Severity getReportableSeverity() const
    {
        return mReportableSeverity;
    }

private:
    /**
     * @brief 根据严重性级别返回适当的日志消息前缀字符串
     * @param severity 严重性级别
     * @return 前缀字符串
     */
    static const char *severityPrefix(Severity severity)
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

    /**
     * @brief 从给定的 (argc, argv) 值生成命令行字符串
     * @param argc 参数数量
     * @param argv 参数数组
     * @return 命令行字符串
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

    Severity mReportableSeverity; // 可报告的严重性级别

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
inline LogStreamConsumer LOG_VERBOSE(const Logger &logger)
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
inline LogStreamConsumer LOG_INFO(const Logger &logger)
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
inline LogStreamConsumer LOG_WARN(const Logger &logger)
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
inline LogStreamConsumer LOG_ERROR(const Logger &logger)
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
inline LogStreamConsumer LOG_FATAL(const Logger &logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
}

} // anonymous namespace

} // namespace irt::model