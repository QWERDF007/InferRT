
#include <gtest/gtest.h>
#include <inferrt/core/Status.h>

#include <string>

namespace t = ::testing;

/**
 * @brief 参数化校验 C API 状态码到名称的映射关系。
 */
class StatusNameTest : public t::TestWithParam<std::tuple<IRTStatus, const char *>>
{
};

#define MAKE_STATUS_NAME(X) std::make_tuple(X, #X)

// clang-format off
#ifndef ENABLE_SANITIZER
INSTANTIATE_TEST_SUITE_P(AllStatuses, StatusNameTest,
                         t::Values(MAKE_STATUS_NAME(IRT_SUCCESS),
                                    MAKE_STATUS_NAME(IRT_ERROR_NOT_IMPLEMENTED),
                                    MAKE_STATUS_NAME(IRT_ERROR_INVALID_ARGUMENT),
                                    MAKE_STATUS_NAME(IRT_ERROR_INVALID_OPERATION),
                                    MAKE_STATUS_NAME(IRT_ERROR_DEVICE),
                                    MAKE_STATUS_NAME(IRT_ERROR_NOT_READY),
                                    MAKE_STATUS_NAME(IRT_ERROR_OUT_OF_MEMORY),
                                    MAKE_STATUS_NAME(IRT_ERROR_INTERNAL),
                                    MAKE_STATUS_NAME(IRT_ERROR_UNKNOWN),
                                    std::make_tuple(static_cast<IRTStatus>(255), "Unknown error")));
#else
INSTANTIATE_TEST_SUITE_P(AllStatuses, StatusNameTest,
                         t::Values(MAKE_STATUS_NAME(IRT_SUCCESS),
                                    MAKE_STATUS_NAME(IRT_ERROR_NOT_IMPLEMENTED),
                                    MAKE_STATUS_NAME(IRT_ERROR_INVALID_ARGUMENT),
                                    MAKE_STATUS_NAME(IRT_ERROR_INVALID_OPERATION),
                                    MAKE_STATUS_NAME(IRT_ERROR_DEVICE),
                                    MAKE_STATUS_NAME(IRT_ERROR_NOT_READY),
                                    MAKE_STATUS_NAME(IRT_ERROR_OUT_OF_MEMORY),
                                    MAKE_STATUS_NAME(IRT_ERROR_INTERNAL),
                                    MAKE_STATUS_NAME(IRT_ERROR_UNKNOWN)));
#endif
// clang-format on

/**
 * @brief StatusGetName 应为所有公开状态码返回稳定名称。
 */
TEST_P(StatusNameTest, GetName)
{
    IRTStatus   status = std::get<0>(GetParam());
    const char *gold   = std::get<1>(GetParam());

    EXPECT_STREQ(gold, irt::StatusGetName(status));
}

/**
 * @brief 主线程初始状态应为成功。
 */
TEST(StatusTest, MainThreadHasSuccessStatusByDefault)
{
    EXPECT_EQ(IRT_SUCCESS, irt::GetLastError());
    EXPECT_EQ(IRT_SUCCESS, irt::PeekAtLastError());
}

/**
 * @brief 成功状态的错误消息应为固定文本 success。
 */
TEST(StatusTest, GetLastStatusMsgSuccessHasCorrectMessage)
{
    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_SUCCESS, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

/**
 * @brief GetLastError 应读取并重置线程错误状态。
 */
TEST(StatusTest, GetLastStatusResetsErrorState)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "%s", "");
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::GetLastError());
    EXPECT_EQ(IRT_SUCCESS, irt::GetLastError());
}

/**
 * @brief PeekAtLastError 只读状态，不应重置线程错误状态。
 */
TEST(StatusTest, PeekLastStatusDoesntResetErrorState)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "%s", "");
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastError());
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastError());
}

/**
 * @brief GetLastErrorMessage 应返回当前错误消息并重置状态。
 */
TEST(StatusTest, GetLastStatusMsgErrorHasCorrectMessage)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "test message");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);

    ASSERT_EQ(IRT_SUCCESS, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

/**
 * @brief PeekAtLastErrorMessage 在成功状态下应返回 success 且不改变状态。
 */
TEST(StatusTest, PeekAtLastStatusMsgSuccessHasCorrectMessage)
{
    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_SUCCESS, irt::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

// TEST(StatusTest, function_success_doesnot_reset_status)
// {
//     ASSERT_EQ(IRT_ERROR_INVALID_ARGUMENT, irt::ImageCalcRequirements(640, 480, IRT_IMAGE_FORMAT_U8, 0, 0, nullptr));

//     NVCVImageRequirements reqs;
//     ASSERT_EQ(IRT_SUCCESS, irt::ImageCalcRequirements(640, 480, IRT_IMAGE_FORMAT_U8, 0, 0, &reqs));

//     EXPECT_EQ(IRT_ERROR_INVALID_ARGUMENT, irt::GetLastError());
// }

/**
 * @brief PeekAtLastErrorMessage 应可重复读取同一条错误消息。
 */
TEST(StatusTest, PeekAtLastStatusMsgErrorHasCorrectMessage)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "test message");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);

    msg[0] = '\0';
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);
}

/**
 * @brief SetThreadStatus 应支持 printf 风格可变参数格式化。
 */
TEST(StatusTest, SetThreadStatusVarArg)
{
    irt::SetThreadStatus(IRT_ERROR_DEVICE, "test message %d %c %s", 456, 'W', "Liliya");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message 456 W Liliya", msg);
}

/**
 * @brief SetThreadStatusVarArgList 应支持转发 va_list 参数。
 */
TEST(StatusTest, SetThreadStatusVarArgList)
{
    auto fn = [](const char *fmt, ...)
    {
        va_list va;
        va_start(va, fmt);

        irt::SetThreadStatusVarArgList(IRT_ERROR_DEVICE, fmt, va);
        va_end(va);
    };

    fn("test message %d %s %c", 321, "rod", 'l');

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message 321 rod l", msg);
}

/**
 * @brief SetThreadStatusVarArgList 在空格式串时仍应保留错误码。
 */
TEST(StatusTest, SetThreadStatusVarArgListWithNullFormat)
{
    va_list va{};
    irt::SetThreadStatusVarArgList(IRT_ERROR_DEVICE, nullptr, va);
    EXPECT_EQ(IRT_ERROR_DEVICE, irt::GetLastError());
}

/**
 * @brief SetThreadStatus 传入空消息时应写入空字符串消息。
 */
TEST(StatusTest, SetThreadStatusNullMessage)
{
    irt::SetThreadStatus(IRT_ERROR_DEVICE, nullptr);

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("", msg);
}

/**
 * @brief GetLastErrorMessage 允许空输出缓冲区，并且仍会重置线程状态。
 */
TEST(StatusTest, GetLastErrorMessageAllowsNullBufferAndResetsStatus)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "will reset");

    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::GetLastErrorMessage(nullptr, 0));
    EXPECT_EQ(IRT_SUCCESS, irt::PeekAtLastError());
}

/**
 * @brief PeekAtLastErrorMessage 对小缓冲区应安全截断消息且不重置状态。
 */
TEST(StatusTest, PeekAtLastErrorMessageTruncatesSmallBufferWithoutReset)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "abcdef");

    char msg[4] = {};
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_EQ(std::string(msg), "abc");
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastError());

    irt::GetLastError();
}

/**
 * @brief PeekAtLastErrorMessage 在 len 小于等于 0 时不应写入输出缓冲区。
 */
TEST(StatusTest, PeekAtLastErrorMessageIgnoresNonPositiveLength)
{
    irt::SetThreadStatus(IRT_ERROR_INTERNAL, "abcdef");

    char msg[] = "unchanged";
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::PeekAtLastErrorMessage(msg, 0));
    EXPECT_STREQ(msg, "unchanged");

    irt::GetLastError();
}
