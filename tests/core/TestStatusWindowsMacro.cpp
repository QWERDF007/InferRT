// Simulate the Windows SDK macro being visible before a public InferRT header.
#pragma push_macro("ERROR_INVALID_OPERATION")
#pragma push_macro("ERROR_NOT_READY")
#pragma push_macro("ERROR_UNKNOWN")
#ifdef ERROR_INVALID_OPERATION
#    undef ERROR_INVALID_OPERATION
#endif
#ifdef ERROR_NOT_READY
#    undef ERROR_NOT_READY
#endif
#ifdef ERROR_UNKNOWN
#    undef ERROR_UNKNOWN
#endif
#define ERROR_INVALID_OPERATION 4317L
#define ERROR_NOT_READY 21L
#define ERROR_UNKNOWN 9999L

#include <inferrt/core/Status.hpp>

#pragma pop_macro("ERROR_INVALID_OPERATION")
#pragma pop_macro("ERROR_NOT_READY")
#pragma pop_macro("ERROR_UNKNOWN")

#include <gtest/gtest.h>

TEST(StatusWindowsMacroTest, PublicStatusHeaderCompilesWithWindowsErrorMacro)
{
    EXPECT_EQ(static_cast<IRTStatus>(irt::Status::INVALID_OPERATION), IRT_ERROR_INVALID_OPERATION);
    EXPECT_EQ(static_cast<IRTStatus>(irt::Status::NOT_READY), IRT_ERROR_NOT_READY);
    EXPECT_EQ(static_cast<IRTStatus>(irt::Status::UNKNOWN), IRT_ERROR_UNKNOWN);
}
