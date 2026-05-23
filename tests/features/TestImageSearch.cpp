#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ImageSearch.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

/**
 * @brief 自动清理的临时目录，避免测试文件污染仓库。
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
              / fs::path("inferrt_image_search_test_"
                         + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
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
    const fs::path &path() const noexcept
    {
        return path_;
    }

    TempDir(const TempDir &)            = delete;
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_;
};

/**
 * @brief 写入一个简单测试文件。
 * @param path 文件路径。
 */
void writeFile(const fs::path &path)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << "x";
}

/**
 * @brief 断言调用抛出 InferRT 异常且错误码符合预期。
 * @tparam Fn 可调用对象类型。
 * @param fn 待执行调用。
 * @param expected_code 期望错误码。
 */
template <typename Fn>
void expectIrtExceptionCode(Fn &&fn, irt::Status expected_code)
{
    try
    {
        std::forward<Fn>(fn)();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), expected_code);
    }
}

} // namespace

/**
 * @brief 默认构造应使用约定模型和特征名，且初始状态不可搜索。
 */
TEST(ImageSearchTest, DefaultConstructsNotReadySearcher)
{
    const irt::features::ImageSearch search;

    EXPECT_EQ(search.modelName(), irt::features::ImageSearch::kDefaultModelName);
    EXPECT_EQ(search.featureName(), irt::features::ImageSearch::kDefaultFeatureName);
    EXPECT_FALSE(search.isReady());
    EXPECT_TRUE(search.indexPath().empty());
    EXPECT_TRUE(search.galleryImages().empty());
    EXPECT_EQ(search.featureDim(), 0);
}

/**
 * @brief ImageSearch 应识别常见图片扩展名，并忽略大小写。
 */
TEST(ImageSearchTest, IsImageFileAcceptsKnownExtensions)
{
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.jpg"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.JPEG"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.png"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.bmp"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.webp"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("dir.with.dot/a.jpg"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("a.txt"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("no_extension"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile(".jpg"));
}

/**
 * @brief 默认索引路径应将模型名和特征名消毒为安全文件名。
 */
TEST(ImageSearchTest, DefaultIndexPathSanitizesModelAndFeatureNames)
{
    const auto path = irt::features::ImageSearch::defaultIndexPath("gallery", "wide_resnet50_2", "layer/4.out");
    EXPECT_EQ(path.generic_string(), "gallery/wide_resnet50_2_layer_4_out.faiss");
}

/**
 * @brief collectGalleryImages 应递归收集图片文件，忽略非图片，并返回排序后的绝对路径。
 */
TEST(ImageSearchTest, CollectGalleryImagesReturnsSortedAbsoluteImages)
{
    TempDir temp;
    writeFile(temp.path() / "b.png");
    writeFile(temp.path() / "nested" / "c.webp");
    writeFile(temp.path() / "a.JPG");
    writeFile(temp.path() / "notes.txt");

    auto images = irt::features::ImageSearch::collectGalleryImages(temp.path());

    std::vector<fs::path> expected{
        fs::absolute(temp.path() / "a.JPG"),
        fs::absolute(temp.path() / "b.png"),
        fs::absolute(temp.path() / "nested" / "c.webp"),
    };
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(images, expected);
}

/**
 * @brief collectGalleryImages 应拒绝不存在的图库目录。
 */
TEST(ImageSearchTest, CollectGalleryImagesRejectsMissingDirectory)
{
    TempDir temp;
    expectIrtExceptionCode([&] { irt::features::ImageSearch::collectGalleryImages(temp.path() / "missing"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief collectGalleryImages 应拒绝没有任何图片文件的图库目录。
 */
TEST(ImageSearchTest, CollectGalleryImagesRejectsEmptyGallery)
{
    TempDir temp;
    writeFile(temp.path() / "notes.txt");

    expectIrtExceptionCode([&] { irt::features::ImageSearch::collectGalleryImages(temp.path()); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未注册模型名称应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsUnsupportedModel)
{
    expectIrtExceptionCode([&] { irt::features::ImageSearch search("not_a_model", "layer4"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 空特征名应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsEmptyFeatureName)
{
    expectIrtExceptionCode([&] { irt::features::ImageSearch search("resnet18", ""); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未 buildOrLoad 的检索器调用 search 时应返回未就绪错误。
 */
TEST(ImageSearchTest, SearchBeforeBuildOrLoadThrowsInvalidOperation)
{
    irt::features::ImageSearch search("resnet18", "layer4");

    expectIrtExceptionCode([&] { search.search("query.jpg"); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 移动构造后的检索器应保留配置字段，便于作为可移动资源返回。
 */
TEST(ImageSearchTest, MoveConstructedSearcherKeepsConfiguration)
{
    irt::features::ImageSearch source("resnet18", "layer4");
    irt::features::ImageSearch moved(std::move(source));

    EXPECT_EQ(moved.modelName(), "resnet18");
    EXPECT_EQ(moved.featureName(), "layer4");
    EXPECT_FALSE(moved.isReady());
}
