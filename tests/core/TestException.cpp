#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Version.h>

#include <future>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

/**
 * @brief Public Exception 应保留状态码和格式化消息。
 */
TEST(ExceptionTest, StoresCodeAndFormattedMessage)
{
    irt::SetThreadError(nullptr);
    const irt::Exception ex(irt::Status::ERROR_INVALID_ARGUMENT, "value=%d", 7);

    EXPECT_EQ(ex.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    EXPECT_STREQ(ex.msg(), "value=7");
    EXPECT_NE(std::string(ex.what()).find("ERROR_INVALID_ARGUMENT: value=7"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief 验证 Exception 复制后原始对象销毁，副本的 msg() 和 what() 依然有效且一致（值语义测试，防止野指针）。
 */
TEST(ExceptionTest, CopyPreservesMessage)
{
    irt::SetThreadError(nullptr);
    irt::Exception copy_target(irt::Status::SUCCESS);

    {
        const irt::Exception source(irt::Status::ERROR_INVALID_ARGUMENT, "dangling check: %d", 42);
        copy_target = source;
        // source 在此作用域结束时被销毁
    }

    EXPECT_EQ(copy_target.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    EXPECT_STREQ(copy_target.msg(), "dangling check: 42");
    EXPECT_NE(std::string(copy_target.what()).find("ERROR_INVALID_ARGUMENT: dangling check: 42"), std::string::npos);
}

/**
 * @brief 验证 Exception 移动语义正常工作。
 */
TEST(ExceptionTest, MovePreservesMessage)
{
    irt::SetThreadError(nullptr);
    irt::Exception source(irt::Status::ERROR_DEVICE, "cuda failure on device %d", 0);
    const irt::Exception moved(std::move(source));

    EXPECT_EQ(moved.code(), irt::Status::ERROR_DEVICE);
    EXPECT_STREQ(moved.msg(), "cuda failure on device 0");
    EXPECT_NE(std::string(moved.what()).find("ERROR_DEVICE: cuda failure on device 0"), std::string::npos);
}

/**
 * @brief 验证公开 irt::Exception 能被 SetThreadError 正确识别且保留特定状态码（绝不降级为 INTERNAL）。
 */
TEST(ExceptionTest, MapsPublicExceptionCode)
{
    irt::SetThreadError(nullptr);
    irt::SetThreadError(std::make_exception_ptr(irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY, "buffer allocation failed")));

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::GetLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_OUT_OF_MEMORY);
    EXPECT_STREQ(msg, "buffer allocation failed");
}

/**
 * @brief 验证 SetThreadStatus 直写 TLS，不抛异常也不触发额外栈展开。
 */
TEST(ExceptionTest, SetThreadStatusDoesNotThrow)
{
    irt::SetThreadError(nullptr);
    EXPECT_NO_THROW({
        irt::SetThreadStatus(IRT_ERROR_INVALID_ARGUMENT, "direct write: %s", "ok");
    });

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::GetLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INVALID_ARGUMENT);
    EXPECT_STREQ(msg, "direct write: ok");
}

/**
 * @brief 验证超长消息在 SetThreadStatus 中安全截断并包含 null 结尾。
 */
TEST(ExceptionTest, MessageIsTruncatedAndTerminated)
{
    irt::SetThreadError(nullptr);
    const std::string long_str(IRT_MAX_STATUS_MESSAGE_LENGTH + 200, 'X');
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "%s", long_str.c_str());

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::GetLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INTERNAL);
    EXPECT_EQ(strlen(msg), static_cast<size_t>(IRT_MAX_STATUS_MESSAGE_LENGTH - 1));
    EXPECT_EQ(msg[IRT_MAX_STATUS_MESSAGE_LENGTH - 1], '\0');
}

/**
 * @brief 验证工作线程捕获的异常经跨线程保存 exception_ptr 后，在主线程仍能准确恢复错误上下文。
 */
TEST(ExceptionTest, ErrorContextSurvivesWorkerBoundary)
{
    std::exception_ptr captured_error;
    std::thread worker([&] {
        try
        {
            throw irt::Exception(irt::Status::NOT_READY, "async pipeline queue empty");
        }
        catch (...)
        {
            captured_error = std::current_exception();
        }
    });
    worker.join();

    ASSERT_NE(captured_error, nullptr);
    irt::SetThreadError(captured_error);

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::GetLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_NOT_READY);
    EXPECT_STREQ(msg, "async pipeline queue empty");
}

/**
 * @brief SetThreadError 应将 std::invalid_argument 转换为对应线程状态。
 */
TEST(ExceptionTest, SetThreadErrorMapsInvalidArgument)
{
    irt::SetThreadError(nullptr);
    irt::SetThreadError(std::make_exception_ptr(std::invalid_argument("bad arg")));

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::PeekAtLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::string(msg).find("bad arg"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief SetThreadError 传入空异常时应重置为成功状态。
 */
TEST(ExceptionTest, SetThreadErrorResetsStatusOnNull)
{
    irt::SetThreadError(nullptr);
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "old error");
    irt::SetThreadError(nullptr);

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::GetLastErrorMessage(msg, sizeof(msg)), IRT_SUCCESS);
}

/**
 * @brief ProtectCall 在成功路径下应返回成功状态。
 */
TEST(ExceptionTest, ProtectCallReturnsSuccessWhenNoExceptionThrown)
{
    irt::SetThreadError(nullptr);
    EXPECT_EQ(irt::ProtectCall([] {}), IRT_SUCCESS);
    irt::SetThreadError(nullptr);
}

/**
 * @brief ProtectCall 应将 InferRT 异常转换为状态码并写入线程错误信息。
 */
TEST(ExceptionTest, ProtectCallConvertsInferRtExceptionToThreadStatus)
{
    irt::SetThreadError(nullptr);
    const IRTStatus status = irt::ProtectCall(
        []
        {
            throw irt::Exception(irt::Status::NOT_READY, "later");
        });

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(status, IRT_ERROR_NOT_READY);
    EXPECT_EQ(irt::PeekAtLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_NOT_READY);
    EXPECT_NE(std::string(msg).find("later"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief ProtectCall 应将 std::bad_alloc 映射为内存不足错误。
 */
TEST(ExceptionTest, ProtectCallConvertsBadAllocToOutOfMemory)
{
    irt::SetThreadError(nullptr);
    const IRTStatus status = irt::ProtectCall(
        []
        {
            throw std::bad_alloc();
        });

    EXPECT_EQ(status, IRT_ERROR_OUT_OF_MEMORY);
    irt::SetThreadError(nullptr);
}

/**
 * @brief SetThreadError 应将普通 std::exception 转换为内部错误。
 */
TEST(ExceptionTest, SetThreadErrorMapsStdExceptionToInternal)
{
    irt::SetThreadError(nullptr);
    irt::SetThreadError(std::make_exception_ptr(std::runtime_error("runtime failure")));

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(irt::PeekAtLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INTERNAL);
    EXPECT_NE(std::string(msg).find("runtime failure"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief ProtectCall 应将 std::invalid_argument 映射为非法参数错误。
 */
TEST(ExceptionTest, ProtectCallConvertsInvalidArgumentToInvalidArgumentStatus)
{
    irt::SetThreadError(nullptr);
    const IRTStatus status = irt::ProtectCall(
        []
        {
            throw std::invalid_argument("bad input");
        });

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(status, IRT_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(irt::PeekAtLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::string(msg).find("bad input"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief ProtectCall 应将未知异常映射为内部错误并写入兜底消息。
 */
TEST(ExceptionTest, ProtectCallConvertsUnknownExceptionToInternal)
{
    irt::SetThreadError(nullptr);
    const IRTStatus status = irt::ProtectCall(
        []
        {
            throw 42;
        });

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    EXPECT_EQ(status, IRT_ERROR_INTERNAL);
    EXPECT_EQ(irt::PeekAtLastErrorMessage(msg, sizeof(msg)), IRT_ERROR_INTERNAL);
    EXPECT_NE(std::string(msg).find("Unexpected error"), std::string::npos);
    irt::SetThreadError(nullptr);
}

/**
 * @brief 版本相关接口应返回自洽的非空字符串。
 */
TEST(VersionTest, VersionStringsAreNonEmptyAndConsistent)
{
    const std::string version    = irt::GetVersionString();
    const std::string branch     = irt::GetBranchString();
    const std::string commit     = irt::GetCommitHashString();
    const std::string full       = irt::GetFullVersionString();
    const std::string build_time = irt::GetBuildTimeString();

    EXPECT_FALSE(version.empty());
    EXPECT_FALSE(branch.empty());
    EXPECT_FALSE(commit.empty());
    EXPECT_FALSE(full.empty());
    EXPECT_FALSE(build_time.empty());
    EXPECT_NE(full.find(version), std::string::npos);
    EXPECT_NE(full.find(branch), std::string::npos);
    EXPECT_NE(full.find(commit), std::string::npos);
}
