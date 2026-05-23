#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Version.h>

#include <new>
#include <stdexcept>
#include <string>

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
            throw irt::Exception(irt::Status::ERROR_NOT_READY, "later");
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
    const std::string version = irt::GetVersionString();
    const std::string branch = irt::GetBranchString();
    const std::string commit = irt::GetCommitHashString();
    const std::string full = irt::GetFullVersionString();
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
