#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ImageSearch.hpp>

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
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("a.txt"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("no_extension"));
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
 * @brief 未注册模型名称应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsUnsupportedModel)
{
    EXPECT_THROW({ irt::features::ImageSearch("not_a_model", "layer4"); }, irt::Exception);

    try
    {
        irt::features::ImageSearch search("not_a_model", "layer4");
        static_cast<void>(search);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

/**
 * @brief 空特征名应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsEmptyFeatureName)
{
    EXPECT_THROW({ irt::features::ImageSearch("resnet18", ""); }, irt::Exception);

    try
    {
        irt::features::ImageSearch search("resnet18", "");
        static_cast<void>(search);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}
