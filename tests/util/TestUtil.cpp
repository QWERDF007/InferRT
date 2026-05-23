#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/util/CheckError.hpp>
#include <inferrt/util/Path.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

/**
 * @brief 自动清理的临时目录，避免路径类测试污染系统临时目录。
 */
class TempDir
{
public:
    /**
     * @brief 创建唯一临时目录。
     */
    TempDir()
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_util_test_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        fs::create_directories(path_);
    }

    /**
     * @brief 析构时递归删除临时目录。
     */
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    /**
     * @brief 获取临时目录路径。
     * @return 临时目录路径。
     */
    const fs::path &path() const
    {
        return path_;
    }

    TempDir(const TempDir &)            = delete;
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_;
};

/**
 * @brief 在作用域内切换当前工作目录，并在析构时恢复。
 */
class ScopedCurrentPath
{
public:
    /**
     * @brief 保存当前目录并切换到目标目录。
     * @param target 目标工作目录。
     */
    explicit ScopedCurrentPath(const fs::path &target)
        : old_(fs::current_path())
    {
        fs::current_path(target);
    }

    /**
     * @brief 恢复构造前的工作目录。
     */
    ~ScopedCurrentPath()
    {
        std::error_code ec;
        fs::current_path(old_, ec);
    }

    ScopedCurrentPath(const ScopedCurrentPath &)            = delete;
    ScopedCurrentPath &operator=(const ScopedCurrentPath &) = delete;

private:
    fs::path old_;
};

} // namespace

/**
 * @brief findProjectRoot 应能从 source_file 所在目录向上找到项目根目录。
 */
TEST(PathUtilTest, FindsProjectRootFromSourceFileAncestor)
{
    TempDir temp;
    const fs::path root = temp.path() / "repo";
    const fs::path source_dir = root / "src" / "module";
    const fs::path bin_dir = root / "build" / "bin";

    fs::create_directories(source_dir);
    fs::create_directories(bin_dir);
    std::ofstream(root / "marker.txt") << "marker";

    const ScopedCurrentPath cwd_guard(temp.path());
    const fs::path found = irt::util::findProjectRoot((bin_dir / "app.exe").string().c_str(), {"marker.txt"},
                                                      (source_dir / "File.cpp").string().c_str());

    EXPECT_EQ(fs::weakly_canonical(found), fs::weakly_canonical(root));
}

/**
 * @brief findProjectRoot 应能从可执行文件路径向上找到包含全部 marker 的项目根目录。
 */
TEST(PathUtilTest, FindsProjectRootFromProgramNameAncestor)
{
    TempDir temp;
    const fs::path root = temp.path() / "repo";
    const fs::path bin_dir = root / "build" / "bin";

    fs::create_directories(bin_dir);
    fs::create_directories(root / "assets");
    std::ofstream(root / "CMakeLists.txt") << "cmake";
    std::ofstream(root / "assets" / "marker.txt") << "asset";

    const ScopedCurrentPath cwd_guard(temp.path());
    const fs::path found = irt::util::findProjectRoot((bin_dir / "app.exe").string().c_str(),
                                                      {"CMakeLists.txt", "assets/marker.txt"}, nullptr);

    EXPECT_EQ(fs::weakly_canonical(found), fs::weakly_canonical(root));
}

/**
 * @brief findProjectRoot 在找不到匹配根目录时应回退到当前工作目录。
 */
TEST(PathUtilTest, FallsBackToCurrentWorkingDirectoryWhenNoMarkerExists)
{
    TempDir temp;
    const fs::path cwd = temp.path() / "cwd";
    fs::create_directories(cwd);

    const ScopedCurrentPath cwd_guard(cwd);
    const fs::path found = irt::util::findProjectRoot(nullptr, {"missing.marker"}, nullptr);

    EXPECT_EQ(fs::weakly_canonical(found), fs::weakly_canonical(cwd));
}

/**
 * @brief GetCheckMessage 的无格式版本应返回空字符串。
 */
TEST(CheckErrorUtilTest, EmptyCheckMessageOverloadReturnsEmptyString)
{
    char buf[32] = {};
    EXPECT_STREQ(irt::util::detail::GetCheckMessage(buf, sizeof(buf)), "");
}

/**
 * @brief GetCheckMessage 的格式化版本应正确写入消息。
 */
TEST(CheckErrorUtilTest, FormattedCheckMessageOverloadFormatsText)
{
    char buf[64] = {};
    EXPECT_STREQ(irt::util::detail::GetCheckMessage(buf, sizeof(buf), "value=%d", 42), "value=42");
}

/**
 * @brief FormatErrorMessage 在没有调用语句和附加消息时也应输出错误名。
 */
TEST(CheckErrorUtilTest, FormatErrorMessageHandlesMinimalInput)
{
    const std::string text = irt::util::detail::FormatErrorMessage("cudaSuccess", "", "");

    EXPECT_NE(text.find("cudaSuccess"), std::string::npos);
    EXPECT_EQ(text.find("allocation failed"), std::string::npos);
}

/**
 * @brief FormatErrorMessage 应组合调用语句、错误名和附加消息。
 */
TEST(CheckErrorUtilTest, FormatErrorMessageCombinesParts)
{
    const std::string text = irt::util::detail::FormatErrorMessage("cudaErrorInvalidValue", "cudaMalloc(ptr, 4)",
                                                                   "allocation failed");
    EXPECT_NE(text.find("cudaMalloc(ptr, 4)"), std::string::npos);
    EXPECT_NE(text.find("cudaErrorInvalidValue"), std::string::npos);
    EXPECT_NE(text.find("allocation failed"), std::string::npos);
}

/**
 * @brief CUDA 错误码应映射为预期的 InferRT 状态码。
 */
TEST(CheckErrorUtilTest, TranslateCudaErrorMapsExpectedStatuses)
{
    EXPECT_EQ(irt::util::TranslateError(cudaErrorMemoryAllocation), IRT_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(irt::util::TranslateError(cudaErrorNotReady), IRT_ERROR_NOT_READY);
    EXPECT_EQ(irt::util::TranslateError(cudaErrorInvalidValue), IRT_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(irt::util::TranslateError(cudaErrorUnknown), IRT_ERROR_INTERNAL);
}

/**
 * @brief ToString 应返回 CUDA 错误名称，并在请求时提供描述。
 */
TEST(CheckErrorUtilTest, ToStringReturnsCudaErrorNameAndDescription)
{
    const char *description = nullptr;
    const char *name = irt::util::ToString(cudaErrorInvalidValue, &description);

    EXPECT_STRNE(name, "");
    ASSERT_NE(description, nullptr);
    EXPECT_STRNE(description, "");
}

/**
 * @brief IRT_CHECK_THROW 对成功状态不应抛异常。
 */
TEST(CheckErrorUtilTest, CheckThrowPassesOnSuccessStatus)
{
    EXPECT_NO_THROW({ IRT_CHECK_THROW(cudaSuccess); });
}

/**
 * @brief IRT_CHECK_THROW 对失败状态应抛出带格式化消息的异常。
 */
TEST(CheckErrorUtilTest, CheckThrowRaisesExceptionOnFailureStatus)
{
    try
    {
        IRT_CHECK_THROW(cudaErrorInvalidValue, "bad value=%d", 9);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::string(e.what()).find("bad value=9"), std::string::npos);
    }
}

/**
 * @brief IRT_CHECK_LOG 对失败状态应返回 false 并输出错误日志。
 */
TEST(CheckErrorUtilTest, CheckLogReturnsFalseAndPrintsMessageOnFailureStatus)
{
    testing::internal::CaptureStderr();
    const bool ok = IRT_CHECK_LOG(cudaErrorInvalidValue, "log value=%d", 5);
    const std::string stderr_text = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(ok);
    EXPECT_NE(stderr_text.find("log value=5"), std::string::npos);
}

/**
 * @brief IRT_CHECK_LOG 对成功状态应返回 true。
 */
TEST(CheckErrorUtilTest, CheckLogReturnsTrueOnSuccessStatus)
{
    testing::internal::CaptureStderr();
    EXPECT_TRUE((IRT_CHECK_LOG(cudaSuccess, "unused")));
    const std::string stderr_text = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(stderr_text.empty());
}
