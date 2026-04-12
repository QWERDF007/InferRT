
#include <gtest/gtest.h>
#include <inferrt/core/Status.h>

namespace t = ::testing;

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
                                    MAKE_STATUS_NAME(IRT_ERROR_INTERNAL)));
#endif
// clang-format on

TEST_P(StatusNameTest, get_name)
{
    IRTStatus   status = std::get<0>(GetParam());
    const char *gold   = std::get<1>(GetParam());

    EXPECT_STREQ(gold, irt::core::StatusGetName(status));
}

TEST(StatusTest, main_thread_has_success_status_by_default)
{
    EXPECT_EQ(IRT_SUCCESS, irt::core::GetLastError());
    EXPECT_EQ(IRT_SUCCESS, irt::core::PeekAtLastError());
}

TEST(StatusTest, get_last_status_msg_success_has_correct_message)
{
    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_SUCCESS, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

TEST(StatusTest, get_last_status_resets_error_state)
{
    irt::core::SetThreadStatus(IRT_ERROR_INTERNAL, "%s", "");
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::core::GetLastError());
    EXPECT_EQ(IRT_SUCCESS, irt::core::GetLastError());
}

TEST(StatusTest, peek_last_status_doesnt_reset_error_state)
{
    irt::core::SetThreadStatus(IRT_ERROR_INTERNAL, "%s", "");
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::core::PeekAtLastError());
    EXPECT_EQ(IRT_ERROR_INTERNAL, irt::core::PeekAtLastError());
}

TEST(StatusTest, get_last_status_msg_error_has_correct_message)
{
    irt::core::SetThreadStatus(IRT_ERROR_INTERNAL, "test message");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);

    ASSERT_EQ(IRT_SUCCESS, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

TEST(StatusTest, peek_at_last_status_msg_success_has_correct_message)
{
    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_SUCCESS, irt::core::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

// TEST(StatusTest, function_success_doesnot_reset_status)
// {
//     ASSERT_EQ(IRT_ERROR_INVALID_ARGUMENT, irt::core::ImageCalcRequirements(640, 480, IRT_IMAGE_FORMAT_U8, 0, 0, nullptr));

//     NVCVImageRequirements reqs;
//     ASSERT_EQ(IRT_SUCCESS, irt::core::ImageCalcRequirements(640, 480, IRT_IMAGE_FORMAT_U8, 0, 0, &reqs));

//     EXPECT_EQ(IRT_ERROR_INVALID_ARGUMENT, irt::core::GetLastError());
// }

TEST(StatusTest, peek_at_last_status_msg_error_has_correct_message)
{
    irt::core::SetThreadStatus(IRT_ERROR_INTERNAL, "test message");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::core::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);

    msg[0] = '\0';
    ASSERT_EQ(IRT_ERROR_INTERNAL, irt::core::PeekAtLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message", msg);
}

TEST(StatusTest, set_thread_status_var_arg)
{
    irt::core::SetThreadStatus(IRT_ERROR_DEVICE, "test message %d %c %s", 456, 'W', "Liliya");

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message 456 W Liliya", msg);
}

TEST(StatusTest, set_thread_status_var_arg_list)
{
    auto fn = [](const char *fmt, ...)
    {
        va_list va;
        va_start(va, fmt);

        irt::core::SetThreadStatusVarArgList(IRT_ERROR_DEVICE, fmt, va);
        va_end(va);
    };

    fn("test message %d %s %c", 321, "rod", 'l');

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test message 321 rod l", msg);
}

TEST(StatusTest, set_thread_status_var_arg_list_1)
{
    va_list va{};
    irt::core::SetThreadStatusVarArgList(IRT_ERROR_DEVICE, nullptr, va);
    EXPECT_EQ(IRT_ERROR_DEVICE, irt::core::GetLastError());
}

TEST(StatusTest, set_thread_status_null_message)
{
    irt::core::SetThreadStatus(IRT_ERROR_DEVICE, nullptr);

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(IRT_ERROR_DEVICE, irt::core::GetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("", msg);
}
