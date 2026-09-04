/**
 * @file TestShapeTemplateMatcher.cpp
 * @brief ``ShapeTemplateMatcher`` API、匹配、NMS、模板变体和持久化测试。
 */

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include "ShapeTemplateMatcherFactory.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

template<irt::features::priv::ShapeTemplateMatcherImplementation Implementation>
class TestShapeTemplateMatcher final : public irt::features::IShapeTemplateMatcher
{
public:
    explicit TestShapeTemplateMatcher(irt::features::ShapeTemplateMatcherConfig config = {})
        : implementation_(irt::features::priv::createShapeTemplateMatcherForImplementation(Implementation,
                                                                                             std::move(config)))
    {
    }

    int addTemplate(const cv::Mat &image, const cv::Mat &object_mask = cv::Mat(),
                    irt::features::ShapeTemplateVariant variant = {}) override
    {
        return implementation_->addTemplate(image, object_mask, variant);
    }

    int addTemplateFile(const std::filesystem::path &image_file, const std::filesystem::path &mask_file = {},
                        irt::features::ShapeTemplateVariant variant = {}) override
    {
        return implementation_->addTemplateFile(image_file, mask_file, variant);
    }

    std::vector<int> addTemplateVariants(const cv::Mat &image, const cv::Mat &object_mask,
                                         const std::vector<irt::features::ShapeTemplateVariant> &variants) override
    {
        return implementation_->addTemplateVariants(image, object_mask, variants);
    }

    std::vector<std::vector<int>>
    addTemplateVariantsBatch(const std::vector<irt::features::ShapeTemplateTrainingInput> &inputs,
                             const std::vector<irt::features::ShapeTemplateVariant> &variants) override
    {
        return implementation_->addTemplateVariantsBatch(inputs, variants);
    }

    std::vector<irt::features::ShapeTemplateMatch>
    match(const cv::Mat &image, float threshold = -1.0f, const cv::Mat &search_mask = cv::Mat(),
          irt::features::ShapeTemplateMatchOptions options = {}) const override
    {
        return implementation_->match(image, threshold, search_mask, options);
    }

    std::vector<irt::features::ShapeTemplateMatch>
    matchFile(const std::filesystem::path &image_file, float threshold = -1.0f,
              const std::filesystem::path &mask_file = {},
              irt::features::ShapeTemplateMatchOptions options = {}) const override
    {
        return implementation_->matchFile(image_file, threshold, mask_file, options);
    }

    void clear() override { implementation_->clear(); }
    bool empty() const noexcept override { return implementation_->empty(); }
    int numTemplates() const noexcept override { return implementation_->numTemplates(); }
    const irt::features::ShapeTemplateInfo &getTemplate(int template_id) const override
    {
        return implementation_->getTemplate(template_id);
    }
    const irt::features::ShapeTemplateMatcherConfig &config() const noexcept override
    {
        return implementation_->config();
    }
    void save(const std::filesystem::path &template_file) const override { implementation_->save(template_file); }
    void load(const std::filesystem::path &template_file) override { implementation_->load(template_file); }

private:
    std::unique_ptr<irt::features::IShapeTemplateMatcher> implementation_;
};

using V0ShapeTemplateMatcher =
    TestShapeTemplateMatcher<irt::features::priv::ShapeTemplateMatcherImplementation::Scalar>;
using V1ShapeTemplateMatcher =
    TestShapeTemplateMatcher<irt::features::priv::ShapeTemplateMatcherImplementation::Avx2>;
using V2ShapeTemplateMatcher =
    TestShapeTemplateMatcher<irt::features::priv::ShapeTemplateMatcherImplementation::Avx512>;

/** @brief 当前 CPU 是否具备 v2 执行所需的 AVX512F/BW 指令集。 */
bool supportsAvx512() noexcept
{
    return cv::checkHardwareSupport(CV_CPU_AVX_512F) && cv::checkHardwareSupport(CV_CPU_AVX_512BW);
}

/**
 * @brief 自动清理的临时目录。
 */
class TempDir
{
public:
    /** @brief 创建唯一临时目录。 */
    TempDir()
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_shape_template_test_"
                         + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        fs::create_directories(path_);
    }

    /** @brief 析构时递归删除临时目录。 */
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    /** @brief 获取临时目录路径。 */
    const fs::path &path() const noexcept
    {
        return path_;
    }

    TempDir(const TempDir &)            = delete;
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_; ///< 临时目录路径。
};

/**
 * @brief 断言调用抛出 InferRT 异常且错误码符合预期。
 */
template<typename Fn>
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

/**
 * @brief 构造用于小尺寸合成图的快速测试配置。
 */
irt::features::ShapeTemplateMatcherConfig fastConfig()
{
    irt::features::ShapeTemplateMatcherConfig config;
    config.num_features         = 48;
    config.min_features         = 6;
    config.weak_threshold       = 10.0f;
    config.strong_threshold     = 20.0f;
    config.match_threshold      = 90.0f;
    config.nms_threshold        = 0.3f;
    config.max_label_difference = 0;
    return config;
}

/**
 * @brief 生成 L 形合成模板。
 */
cv::Mat makeLShape(int size = 48)
{
    cv::Mat image(size, size, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Rect(10, 10, 28, 7), cv::Scalar(255), cv::FILLED);
    cv::rectangle(image, cv::Rect(10, 10, 7, 28), cv::Scalar(255), cv::FILLED);
    return image;
}

/**
 * @brief 生成 T 形合成模板，用于不同形状的负例。
 */
cv::Mat makeTShape(int size = 48)
{
    cv::Mat image(size, size, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, cv::Rect(10, 10, 28, 7), cv::Scalar(255), cv::FILLED);
    cv::rectangle(image, cv::Rect(21, 10, 7, 28), cv::Scalar(255), cv::FILLED);
    return image;
}

/** @brief 生成高密度梯度图，用于覆盖大候选集训练路径。 */
cv::Mat makeDenseGradientImage(cv::Size size = cv::Size(151, 137))
{
    cv::Mat image(size, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        auto *row = image.ptr<unsigned char>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            row[x] = static_cast<unsigned char>((x * 29 + y * 17 + (x * y) % 251) & 0xff);
        }
    }
    return image;
}

/**
 * @brief 将目标图粘贴到黑色场景中。
 */
cv::Mat makeSceneWith(const cv::Mat &object, cv::Point top_left, cv::Size scene_size = cv::Size(112, 104))
{
    cv::Mat scene(scene_size, CV_8UC1, cv::Scalar(0));
    object.copyTo(scene(cv::Rect(top_left, object.size())));
    return scene;
}

/**
 * @brief 在匹配结果中查找指定左上角坐标的精确命中。
 */
const irt::features::ShapeTemplateMatch *findExactMatch(const std::vector<irt::features::ShapeTemplateMatch> &matches,
                                                        int expected_x, int expected_y)
{
    const auto it = std::find_if(matches.begin(), matches.end(),
                                 [&](const irt::features::ShapeTemplateMatch &match)
                                 { return match.x == expected_x && match.y == expected_y; });
    return it == matches.end() ? nullptr : &(*it);
}

} // namespace

/**
 * @brief 默认构造应为空模板库并保留默认配置。
 */
TEST(ShapeTemplateMatcherTest, DefaultConstructsEmptyMatcher)
{
    const V1ShapeTemplateMatcher matcher;

    EXPECT_TRUE(matcher.empty());
    EXPECT_EQ(matcher.numTemplates(), 0);
    EXPECT_EQ(matcher.config().num_features, irt::features::kDefaultShapeTemplateNumFeatures);
    EXPECT_EQ(matcher.config().min_features, irt::features::kDefaultShapeTemplateMinFeatures);
    EXPECT_FLOAT_EQ(matcher.config().match_threshold, irt::features::kDefaultShapeTemplateMatchThreshold);
}

/** @brief 公共工厂应自动选择可执行实现，内部工厂供 parity 测试固定算法。 */
TEST(ShapeTemplateMatcherFactoryTest, SelectsSupportedImplementation)
{
    auto automatic = irt::features::createShapeTemplateMatcher(fastConfig());
    auto scalar    = irt::features::priv::createShapeTemplateMatcherForImplementation(
        irt::features::priv::ShapeTemplateMatcherImplementation::Scalar, fastConfig());
    auto avx2 = irt::features::priv::createShapeTemplateMatcherForImplementation(
        irt::features::priv::ShapeTemplateMatcherImplementation::Avx2, fastConfig());

    ASSERT_NE(automatic, nullptr);
    ASSERT_NE(scalar, nullptr);
    ASSERT_NE(avx2, nullptr);
    EXPECT_TRUE(automatic->empty());
    EXPECT_TRUE(scalar->empty());
    EXPECT_TRUE(avx2->empty());
    EXPECT_STREQ(irt::features::priv::shapeTemplateMatcherImplementationName(
                     irt::features::priv::ShapeTemplateMatcherImplementation::Scalar),
                 "scalar");
    EXPECT_STREQ(irt::features::priv::shapeTemplateMatcherImplementationName(
                     irt::features::priv::ShapeTemplateMatcherImplementation::Avx2),
                 "avx2");

    if (supportsAvx512())
    {
        auto v2 = irt::features::priv::createShapeTemplateMatcherForImplementation(
            irt::features::priv::ShapeTemplateMatcherImplementation::Avx512, fastConfig());
        ASSERT_NE(v2, nullptr);
        EXPECT_TRUE(v2->empty());
    }
    else
    {
        expectIrtExceptionCode(
            []
            {
                (void)irt::features::priv::createShapeTemplateMatcherForImplementation(
                    irt::features::priv::ShapeTemplateMatcherImplementation::Avx512, fastConfig());
            },
            irt::Status::INVALID_OPERATION);
    }
}

/**
 * @brief 构造函数应拒绝非法配置。
 */
TEST(ShapeTemplateMatcherTest, ConstructorRejectsInvalidConfig)
{
    auto config          = fastConfig();
    config.num_features = 0;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    config              = fastConfig();
    config.min_features = config.num_features + 1;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    config                     = fastConfig();
    config.max_label_difference = 5;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    config                 = fastConfig();
    config.match_threshold = 101.0f;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    config                          = fastConfig();
    config.max_training_parallelism = -1;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    config           = fastConfig();
    config.scan_step = 0;
    expectIrtExceptionCode([&] { V1ShapeTemplateMatcher matcher(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/** @brief v1 近似梯度预处理默认关闭，并且不允许误用于 v0。 */
TEST(ShapeTemplateMatcherTest, ApproximatePreprocessingIsV1OnlyAndPersisted)
{
    auto config = fastConfig();
    EXPECT_FALSE(config.use_gaussian_gradient);
    EXPECT_FALSE(config.use_orientation_histogram);

    config.use_gaussian_gradient     = true;
    config.use_orientation_histogram = true;
    config.use_edge_nms              = true;
    config.use_edge_connectivity     = true;
    config.use_polarity_invariant    = true;
    config.use_spatial_spread        = true;
    config.reuse_base_features_for_variants = true;
    V1ShapeTemplateMatcher matcher(config);
    EXPECT_TRUE(matcher.config().use_gaussian_gradient);
    EXPECT_TRUE(matcher.config().use_orientation_histogram);

    TempDir temp;
    const auto file = temp.path() / "approximate_preprocessing.yaml";
    matcher.save(file);
    V1ShapeTemplateMatcher loaded;
    loaded.load(file);
    EXPECT_TRUE(loaded.config().use_gaussian_gradient);
    EXPECT_TRUE(loaded.config().use_orientation_histogram);
    EXPECT_TRUE(loaded.config().use_edge_nms);
    EXPECT_TRUE(loaded.config().use_edge_connectivity);
    EXPECT_TRUE(loaded.config().use_polarity_invariant);
    EXPECT_TRUE(loaded.config().use_spatial_spread);
    EXPECT_TRUE(loaded.config().reuse_base_features_for_variants);

    auto approx_v0_config = config;
    EXPECT_THROW({ V0ShapeTemplateMatcher rejected(approx_v0_config); }, irt::Exception);
}

/** @brief v1 基础特征复用应生成完整的旋转/尺度变体，并可立即匹配。 */
TEST(ShapeTemplateMatcherTrainingOptimizationTest, ReusedBaseFeaturesGenerateVariants)
{
    auto config = fastConfig();
    config.max_label_difference = 1;
    config.reuse_base_features_for_variants = true;
    config.max_training_parallelism = 2;

    const auto object = makeLShape(64);
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(-15.0f, 15.0f, 15.0f,
                                                                               0.9f, 1.1f, 0.1f);
    V1ShapeTemplateMatcher matcher(config);
    const auto ids = matcher.addTemplateVariants(object, cv::Mat(), variants);
    ASSERT_EQ(ids.size(), variants.size());
    ASSERT_EQ(matcher.numTemplates(), static_cast<int>(variants.size()));
    for (size_t index = 0; index < ids.size(); ++index)
    {
        const auto &templ = matcher.getTemplate(ids[index]);
        EXPECT_EQ(templ.angle_degrees, variants[index].angle_degrees);
        EXPECT_FLOAT_EQ(templ.scale, variants[index].scale);
        EXPECT_GE(static_cast<int>(templ.features.size()), config.min_features);
    }

    const auto scene = makeSceneWith(object, cv::Point(23, 19), cv::Size(128, 120));
    const auto matches = matcher.match(scene, 50.0f);
    EXPECT_FALSE(matches.empty());
}

/**
 * @brief 训练阶段应拒绝空图、错误掩膜尺寸和无梯度模板。
 */
TEST(ShapeTemplateMatcherTest, AddTemplateRejectsBadInputs)
{
    V1ShapeTemplateMatcher matcher(fastConfig());
    const auto                          object = makeLShape();

    expectIrtExceptionCode([&] { matcher.addTemplate(cv::Mat()); }, irt::Status::ERROR_INVALID_ARGUMENT);
    expectIrtExceptionCode([&] { matcher.addTemplate(object, cv::Mat(8, 8, CV_8UC1)); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
    expectIrtExceptionCode([&] { matcher.addTemplate(cv::Mat(48, 48, CV_8UC1, cv::Scalar(0))); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 训练单模板后应能读取全局模板 ID、数量和特征元数据。
 */
TEST(ShapeTemplateMatcherTest, AddTemplateExtractsMetadata)
{
    V1ShapeTemplateMatcher matcher(fastConfig());

    const int id = matcher.addTemplate(makeLShape());

    ASSERT_EQ(id, 0);
    EXPECT_FALSE(matcher.empty());
    EXPECT_EQ(matcher.numTemplates(), 1);

    const auto &templ = matcher.getTemplate(id);
    EXPECT_EQ(templ.template_id, 0);
    EXPECT_GT(templ.width, 0);
    EXPECT_GT(templ.height, 0);
    EXPECT_EQ(templ.template_width, makeLShape().cols);
    EXPECT_EQ(templ.template_height, makeLShape().rows);
    EXPECT_GE(static_cast<int>(templ.features.size()), matcher.config().min_features);
}

/**
 * @brief 未训练模板时调用匹配应返回非法状态。
 */
TEST(ShapeTemplateMatcherTest, MatchBeforeTrainingThrowsInvalidOperation)
{
    V1ShapeTemplateMatcher matcher(fastConfig());

    expectIrtExceptionCode([&] { matcher.match(makeLShape()); }, irt::Status::INVALID_OPERATION);
}

/**
 * @brief 匹配器应找到发生平移的同一形状。
 */
TEST(ShapeTemplateMatcherTest, MatchFindsTranslatedShape)
{
    V1ShapeTemplateMatcher matcher(fastConfig());
    const auto                          object = makeLShape();
    const int                           id     = matcher.addTemplate(object);
    const cv::Point                     paste_at(34, 42);
    const auto                          scene = makeSceneWith(object, paste_at);

    const auto matches = matcher.match(scene, 95.0f);

    ASSERT_FALSE(matches.empty());
    const int expected_x = paste_at.x;
    const int expected_y = paste_at.y;
    const auto *exact    = findExactMatch(matches, expected_x, expected_y);
    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(exact->template_id, id);
    EXPECT_NEAR(exact->similarity, 100.0f, 1.0e-4f);
}

/**
 * @brief 非 SIMD 对齐尺寸也应命中，用于覆盖训练和匹配路径的尾部处理。
 */
TEST(ShapeTemplateMatcherTest, MatchesNonAlignedImageSizes)
{
    V1ShapeTemplateMatcher matcher(fastConfig());
    const auto                          object = makeLShape(50);
    matcher.addTemplate(object);
    const cv::Point                     paste_at(31, 27);
    const auto                          scene = makeSceneWith(object, paste_at, cv::Size(119, 107));

    const auto matches = matcher.match(scene, 95.0f);

    ASSERT_FALSE(matches.empty());
    const auto *exact = findExactMatch(matches, paste_at.x, paste_at.y);
    ASSERT_NE(exact, nullptr);
    EXPECT_NEAR(exact->similarity, 100.0f, 1.0e-4f);
}

/**
 * @brief 重复模板命中应被 NMS 压制。
 */
TEST(ShapeTemplateMatcherTest, NmsSuppressesDuplicateTemplates)
{
    V1ShapeTemplateMatcher matcher(fastConfig());
    const auto                          object = makeLShape();
    matcher.addTemplate(object);
    matcher.addTemplate(object);

    const auto matches = matcher.match(makeSceneWith(object, cv::Point(28, 26)), 99.0f);

    ASSERT_EQ(matches.size(), 1U);
    EXPECT_EQ(matches[0].template_id, 0);
    EXPECT_NEAR(matches[0].similarity, 100.0f, 1.0e-4f);
}

/**
 * @brief 关闭 NMS 时仍应按 ``max_results`` 限制返回数量。
 */
TEST(ShapeTemplateMatcherTest, MaxResultsLimitsSortedOutputWhenNmsDisabled)
{
    auto config          = fastConfig();
    config.nms_threshold = -1.0f;
    config.max_results   = 1;

    V1ShapeTemplateMatcher matcher(config);
    const auto                          object = makeLShape();
    matcher.addTemplate(object);
    matcher.addTemplate(object);

    const auto matches = matcher.match(makeSceneWith(object, cv::Point(28, 26)), 99.0f);

    ASSERT_EQ(matches.size(), 1U);
    EXPECT_EQ(matches[0].template_id, 0);
}

/**
 * @brief 旋转模板变体应能匹配旋转后的目标。
 */
TEST(ShapeTemplateMatcherTest, VariantsDetectRotatedShape)
{
    auto config                   = fastConfig();
    config.match_threshold        = 80.0f;
    config.max_label_difference   = 1;
    V1ShapeTemplateMatcher matcher(config);

    const auto object   = makeLShape();
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(0.0f, 90.0f, 90.0f);
    const auto ids      = matcher.addTemplateVariants(object, cv::Mat(), variants);
    ASSERT_EQ(ids.size(), 2U);

    const auto rotated = irt::features::transformShapeTemplateImage(
        object, irt::features::ShapeTemplateVariant{90.0f, 1.0f});
    const auto matches = matcher.match(makeSceneWith(rotated, cv::Point(26, 24)), 85.0f);

    ASSERT_FALSE(matches.empty());
    EXPECT_NEAR(matches.front().angle_degrees, 90.0f, 1.0e-4f);
    EXPECT_NEAR(matches.front().similarity, 100.0f, 1.0e-4f);
}

/**
 * @brief 保存再加载模板后应保留模板元数据并产生一致匹配。
 */
TEST(ShapeTemplateMatcherTest, SaveLoadRoundTripPreservesMatches)
{
    TempDir temp;
    const auto template_file = temp.path() / "shape_templates.yaml";
    const auto object        = makeLShape();
    const auto scene         = makeSceneWith(object, cv::Point(30, 34));

    V1ShapeTemplateMatcher writer(fastConfig());
    writer.addTemplate(object, cv::Mat(), irt::features::ShapeTemplateVariant{15.0f, 1.25f});
    writer.save(template_file);

    std::ifstream input(template_file, std::ios::binary);
    ASSERT_TRUE(input.is_open());
    const std::string serialized{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_NE(serialized.find("version: 4"), std::string::npos);
    EXPECT_EQ(serialized.find("class_id:"), std::string::npos);
    const auto features_position = serialized.find("features:");
    ASSERT_NE(features_position, std::string::npos);
    EXPECT_NE(serialized.find("- [", features_position), std::string::npos);

    V1ShapeTemplateMatcher reader;
    reader.load(template_file);
    const auto matches = reader.match(scene, 95.0f);

    ASSERT_FALSE(matches.empty());
    EXPECT_EQ(reader.numTemplates(), 1);
    EXPECT_EQ(reader.getTemplate(0).features.size(), writer.getTemplate(0).features.size());
    EXPECT_NEAR(reader.getTemplate(0).angle_degrees, 15.0f, 1.0e-4f);
    EXPECT_NEAR(reader.getTemplate(0).scale, 1.25f, 1.0e-4f);
    const auto &expected_feature = writer.getTemplate(0).features.front();
    const auto &loaded_feature   = reader.getTemplate(0).features.front();
    EXPECT_EQ(loaded_feature.x, expected_feature.x);
    EXPECT_EQ(loaded_feature.y, expected_feature.y);
    EXPECT_EQ(loaded_feature.label, expected_feature.label);
    EXPECT_FLOAT_EQ(loaded_feature.angle_degrees, expected_feature.angle_degrees);
    EXPECT_NEAR(matches.front().similarity, 100.0f, 1.0e-4f);
}

/** @brief 标准 YAML v4 持久化格式是破坏性升级，不再接收旧 v3 模板文件。 */
TEST(ShapeTemplateMatcherTest, LegacyV3TemplateFilesAreRejectedAfterFormatUpgrade)
{
    TempDir temp;
    const auto legacy_file = temp.path() / "legacy_v3_templates.yaml";
    std::ofstream output(legacy_file);
    ASSERT_TRUE(output.is_open());
    output << "version: 3\n"
              "templates: []\n";
    ASSERT_TRUE(output.good());
    output.close();

    V1ShapeTemplateMatcher matcher;
    expectIrtExceptionCode([&] { matcher.load(legacy_file); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/** @brief v4 的每个特征必须严格包含四个按位置约定的字段。 */
TEST(ShapeTemplateMatcherTest, CompactFeatureRowsRequireExactlyFourValues)
{
    TempDir temp;
    const auto malformed_file = temp.path() / "malformed_compact_templates.yaml";
    std::ofstream output(malformed_file);
    ASSERT_TRUE(output.is_open());
    output << R"(version: 4
templates:
  - template_id: 0
    width: 8
    height: 8
    template_width: 8
    template_height: 8
    tl_x: 0
    tl_y: 0
    angle_degrees: 0
    scale: 1
    features:
      - [1, 2, 3]
)";
    ASSERT_TRUE(output.good());
    output.close();

    V1ShapeTemplateMatcher matcher;
    expectIrtExceptionCode([&] { matcher.load(malformed_file); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 文件 API 应支持从图片训练和从图片匹配。
 */
TEST(ShapeTemplateMatcherTest, FileApisTrainAndMatchImages)
{
    TempDir temp;
    const auto object_file = temp.path() / "object.png";
    const auto scene_file  = temp.path() / "scene.png";
    const auto object      = makeLShape();
    const auto scene       = makeSceneWith(object, cv::Point(36, 22));
    ASSERT_TRUE(cv::imwrite(object_file.string(), object));
    ASSERT_TRUE(cv::imwrite(scene_file.string(), scene));

    V1ShapeTemplateMatcher matcher(fastConfig());
    const int id = matcher.addTemplateFile(object_file);
    const auto matches = matcher.matchFile(scene_file, 95.0f);

    ASSERT_FALSE(matches.empty());
    EXPECT_EQ(matches.front().template_id, id);
    EXPECT_NEAR(matches.front().similarity, 100.0f, 1.0e-4f);
}

/**
 * @brief 角度/尺度变体生成应覆盖闭区间并校验非法范围。
 */
TEST(ShapeTemplateMatcherTest, MakeAngleScaleVariantsValidatesRanges)
{
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(0.0f, 90.0f, 45.0f, 1.0f,
                                                                                     1.5f, 0.5f);
    ASSERT_EQ(variants.size(), 6U);
    EXPECT_FLOAT_EQ(variants[0].angle_degrees, 0.0f);
    EXPECT_FLOAT_EQ(variants[1].angle_degrees, 45.0f);
    EXPECT_FLOAT_EQ(variants[2].angle_degrees, 90.0f);
    EXPECT_FLOAT_EQ(variants[3].scale, 1.5f);

    expectIrtExceptionCode(
        [&] { irt::features::makeShapeTemplateAngleScaleVariants(90.0f, 0.0f, 1.0f); },
        irt::Status::ERROR_INVALID_ARGUMENT);
    expectIrtExceptionCode(
        [&] { irt::features::makeShapeTemplateAngleScaleVariants(0.0f, 90.0f, 0.0f); },
        irt::Status::ERROR_INVALID_ARGUMENT);
    expectIrtExceptionCode(
        [&] { irt::features::makeShapeTemplateAngleScaleVariants(0.0f, 90.0f, 1.0f, 0.0f, 1.0f); },
        irt::Status::ERROR_INVALID_ARGUMENT);
}

/** @brief AVX2 与标量实现应在训练、掩膜匹配及 YAML 加载后给出完全相同的结果。 */
TEST(ShapeTemplateMatcherParityTest, Avx2AndScalarProduceIdenticalTemplatesAndMatches)
{
    auto config = fastConfig();
    config.max_label_difference = 1;
    config.match_threshold = 75.0f;
    config.scan_step = 1;

    const auto object = makeLShape(50);
    const auto alternate = makeTShape(50);
    cv::Mat scene(137, 151, CV_8UC1, cv::Scalar(0));
    object.copyTo(scene(cv::Rect(37, 41, object.cols, object.rows)));
    cv::Mat search_mask(scene.size(), CV_8UC1, cv::Scalar(255));
    cv::rectangle(search_mask, cv::Rect(0, 0, 12, scene.rows), cv::Scalar(0), cv::FILLED);

    V1ShapeTemplateMatcher avx2(config);
    V0ShapeTemplateMatcher scalar(config);
    ASSERT_EQ(avx2.addTemplate(object), scalar.addTemplate(object));
    ASSERT_EQ(avx2.addTemplate(alternate), scalar.addTemplate(alternate));
    ASSERT_EQ(avx2.numTemplates(), scalar.numTemplates());

    for (int id = 0; id < avx2.numTemplates(); ++id)
    {
        const auto &a = avx2.getTemplate(id);
        const auto &b = scalar.getTemplate(id);
        ASSERT_EQ(a.features.size(), b.features.size());
        EXPECT_EQ(a.width, b.width);
        EXPECT_EQ(a.height, b.height);
        for (size_t i = 0; i < a.features.size(); ++i)
        {
            EXPECT_EQ(a.features[i].x, b.features[i].x);
            EXPECT_EQ(a.features[i].y, b.features[i].y);
            EXPECT_EQ(a.features[i].label, b.features[i].label);
            EXPECT_FLOAT_EQ(a.features[i].angle_degrees, b.features[i].angle_degrees);
        }
    }

    const auto avx2_matches = avx2.match(scene, 75.0f, search_mask);
    const auto scalar_matches = scalar.match(scene, 75.0f, search_mask);
    ASSERT_EQ(avx2_matches.size(), scalar_matches.size());
    for (size_t i = 0; i < avx2_matches.size(); ++i)
    {
        const auto &a = avx2_matches[i];
        const auto &b = scalar_matches[i];
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.width, b.width);
        EXPECT_EQ(a.height, b.height);
        EXPECT_FLOAT_EQ(a.similarity, b.similarity);
        EXPECT_EQ(a.template_id, b.template_id);
        EXPECT_FLOAT_EQ(a.angle_degrees, b.angle_degrees);
        EXPECT_FLOAT_EQ(a.scale, b.scale);
    }

    TempDir temp;
    const auto template_file = temp.path() / "scalar_templates.yaml";
    scalar.save(template_file);
    V1ShapeTemplateMatcher loaded;
    loaded.load(template_file);
    const auto loaded_matches = loaded.match(scene, 75.0f, search_mask);
    ASSERT_EQ(avx2_matches.size(), loaded_matches.size());
    for (size_t i = 0; i < avx2_matches.size(); ++i)
        EXPECT_FLOAT_EQ(avx2_matches[i].similarity, loaded_matches[i].similarity);
}
namespace {

void expectIdenticalMatches(const std::vector<irt::features::ShapeTemplateMatch> &expected,
                            const std::vector<irt::features::ShapeTemplateMatch> &actual)
{
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x);
        EXPECT_EQ(expected[i].y, actual[i].y);
        EXPECT_EQ(expected[i].width, actual[i].width);
        EXPECT_EQ(expected[i].height, actual[i].height);
        EXPECT_FLOAT_EQ(expected[i].similarity, actual[i].similarity);
        EXPECT_EQ(expected[i].template_id, actual[i].template_id);
        EXPECT_FLOAT_EQ(expected[i].angle_degrees, actual[i].angle_degrees);
        EXPECT_FLOAT_EQ(expected[i].scale, actual[i].scale);
    }
}

void expectIdenticalTemplate(const irt::features::ShapeTemplateInfo &expected,
                             const irt::features::ShapeTemplateInfo &actual)
{
    EXPECT_EQ(expected.template_id, actual.template_id);
    EXPECT_EQ(expected.width, actual.width);
    EXPECT_EQ(expected.height, actual.height);
    EXPECT_EQ(expected.template_width, actual.template_width);
    EXPECT_EQ(expected.template_height, actual.template_height);
    EXPECT_EQ(expected.tl_x, actual.tl_x);
    EXPECT_EQ(expected.tl_y, actual.tl_y);
    EXPECT_FLOAT_EQ(expected.angle_degrees, actual.angle_degrees);
    EXPECT_FLOAT_EQ(expected.scale, actual.scale);
    ASSERT_EQ(expected.features.size(), actual.features.size());
    for (size_t index = 0; index < expected.features.size(); ++index)
    {
        const auto &a = expected.features[index];
        const auto &b = actual.features[index];
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.label, b.label);
        EXPECT_FLOAT_EQ(a.angle_degrees, b.angle_degrees);
    }
}

} // namespace

/** @brief 有 AVX512F/BW 的机器上，v2 训练和匹配必须与 v1 完全一致。 */
TEST(ShapeTemplateMatcherParityTest, Avx512AndAvx2ProduceIdenticalTemplatesAndMatches)
{
    if (!supportsAvx512())
    {
        GTEST_SKIP() << "AVX512F/BW is not available on this CPU";
    }

    auto config = fastConfig();
    config.max_label_difference = 1;
    config.match_threshold = 70.0f;
    config.max_training_parallelism = 1;
    const auto object = makeLShape(56);
    const auto alternate = makeTShape(56);
    const auto scene = makeSceneWith(object, cv::Point(37, 29), cv::Size(151, 137));
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(-10.0f, 10.0f, 10.0f,
                                                                               0.9f, 1.1f, 0.1f);

    V1ShapeTemplateMatcher avx2(config);
    V2ShapeTemplateMatcher avx512(config);
    EXPECT_EQ(avx2.addTemplateVariants(object, cv::Mat(), variants),
              avx512.addTemplateVariants(object, cv::Mat(), variants));
    ASSERT_EQ(avx2.numTemplates(), avx512.numTemplates());
    for (int template_id = 0; template_id < avx2.numTemplates(); ++template_id)
    {
        expectIdenticalTemplate(avx2.getTemplate(template_id), avx512.getTemplate(template_id));
    }

    avx2.addTemplate(alternate);
    avx512.addTemplate(alternate);
    expectIdenticalMatches(avx2.match(scene, 70.0f), avx512.match(scene, 70.0f));
}

/** @brief 全非零搜索掩膜应与无掩膜的全图精确结果完全相同。 */
TEST(ShapeTemplateMatcherParityTest, FullSearchMaskMatchesUnmaskedResults)
{
    auto config = fastConfig();
    config.max_label_difference = 1;
    config.match_threshold = 75.0f;

    const auto object = makeLShape(50);
    const auto alternate = makeTShape(50);
    const auto scene = makeSceneWith(object, cv::Point(31, 27), cv::Size(151, 137));
    cv::Mat full_search_mask(scene.size(), CV_8UC1, cv::Scalar(255));

    V1ShapeTemplateMatcher avx2(config);
    V0ShapeTemplateMatcher scalar(config);
    avx2.addTemplate(object);
    avx2.addTemplate(alternate);
    scalar.addTemplate(object);
    scalar.addTemplate(alternate);

    const auto unmasked_matches = avx2.match(scene);
    const auto full_mask_matches = avx2.match(scene, -1.0f, full_search_mask);
    const auto scalar_matches = scalar.match(scene, -1.0f, full_search_mask);
    expectIdenticalMatches(unmasked_matches, full_mask_matches);
    expectIdenticalMatches(unmasked_matches, scalar_matches);
}

/** @brief v0 串行训练与 v1 串行/并行训练必须生成逐字段一致的变体模板。 */
TEST(ShapeTemplateMatcherTrainingParallelTest, V1ParallelVariantTrainingPreservesTemplatesAndMatches)
{
    auto v0_config = fastConfig();
    v0_config.max_label_difference = 1;
    v0_config.match_threshold = 70.0f;
    v0_config.max_parallelism = 1;
    v0_config.max_training_parallelism = 3;
    auto v1_serial_config = v0_config;
    v1_serial_config.max_training_parallelism = 1;
    auto v1_parallel_config = v0_config;
    v1_parallel_config.max_training_parallelism = 3;
    auto v1_auto_config = v0_config;
    v1_auto_config.max_training_parallelism = 0;

    const auto object = makeLShape(56);
    const auto scene = makeSceneWith(object, cv::Point(37, 29), cv::Size(151, 137));
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(-10.0f, 10.0f, 10.0f,
                                                                               0.9f, 1.1f, 0.1f);

    V0ShapeTemplateMatcher v0(v0_config);
    V1ShapeTemplateMatcher v1_serial(v1_serial_config);
    V1ShapeTemplateMatcher v1_parallel(v1_parallel_config);
    V1ShapeTemplateMatcher v1_auto(v1_auto_config);
    const auto v0_ids = v0.addTemplateVariants(object, cv::Mat(), variants);
    const auto serial_ids = v1_serial.addTemplateVariants(object, cv::Mat(), variants);
    const auto parallel_ids = v1_parallel.addTemplateVariants(object, cv::Mat(), variants);
    const auto auto_ids = v1_auto.addTemplateVariants(object, cv::Mat(), variants);
    EXPECT_EQ(v0_ids, serial_ids);
    EXPECT_EQ(serial_ids, parallel_ids);
    EXPECT_EQ(parallel_ids, auto_ids);

    ASSERT_EQ(v0.numTemplates(), static_cast<int>(variants.size()));
    ASSERT_EQ(v0.numTemplates(), v1_serial.numTemplates());
    ASSERT_EQ(v0.numTemplates(), v1_parallel.numTemplates());
    ASSERT_EQ(v0.numTemplates(), v1_auto.numTemplates());
    for (int template_id = 0; template_id < v0.numTemplates(); ++template_id)
    {
        expectIdenticalTemplate(v0.getTemplate(template_id), v1_serial.getTemplate(template_id));
        expectIdenticalTemplate(v0.getTemplate(template_id), v1_parallel.getTemplate(template_id));
        expectIdenticalTemplate(v0.getTemplate(template_id), v1_auto.getTemplate(template_id));
    }

    const auto v0_matches = v0.match(scene, 70.0f);
    expectIdenticalMatches(v0_matches, v1_serial.match(scene, 70.0f));
    expectIdenticalMatches(v0_matches, v1_parallel.match(scene, 70.0f));
    expectIdenticalMatches(v0_matches, v1_auto.match(scene, 70.0f));
}

/** @brief 多输入全局训练队列必须保持 v0/v1 的模板顺序、内容和匹配结果。 */
TEST(ShapeTemplateMatcherTrainingParallelTest, BatchTrainingPreservesSequentialMultiInputResults)
{
    auto config = fastConfig();
    config.max_label_difference = 1;
    config.match_threshold = 70.0f;
    config.max_parallelism = 1;
    config.max_training_parallelism = 3;

    const auto object = makeLShape(56);
    const auto alternate = makeTShape(56);
    const auto scene = makeSceneWith(object, cv::Point(37, 29), cv::Size(151, 137));
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(-10.0f, 10.0f, 10.0f,
                                                                               0.9f, 1.1f, 0.1f);
    const std::vector<irt::features::ShapeTemplateTrainingInput> inputs{
        {object, cv::Mat()},
        {alternate, cv::Mat()},
    };

    V0ShapeTemplateMatcher v0_sequential(config);
    V0ShapeTemplateMatcher v0_batch(config);
    V1ShapeTemplateMatcher v1_sequential(config);
    V1ShapeTemplateMatcher v1_batch(config);

    const auto v0_first = v0_sequential.addTemplateVariants(object, cv::Mat(), variants);
    const auto v0_second = v0_sequential.addTemplateVariants(alternate, cv::Mat(), variants);
    const auto v1_first = v1_sequential.addTemplateVariants(object, cv::Mat(), variants);
    const auto v1_second = v1_sequential.addTemplateVariants(alternate, cv::Mat(), variants);
    const auto v0_batch_ids = v0_batch.addTemplateVariantsBatch(inputs, variants);
    const auto v1_batch_ids = v1_batch.addTemplateVariantsBatch(inputs, variants);

    ASSERT_EQ(v0_batch_ids.size(), inputs.size());
    ASSERT_EQ(v1_batch_ids.size(), inputs.size());
    EXPECT_EQ(v0_batch_ids[0], v0_first);
    EXPECT_EQ(v0_batch_ids[1], v0_second);
    EXPECT_EQ(v1_batch_ids[0], v1_first);
    EXPECT_EQ(v1_batch_ids[1], v1_second);
    EXPECT_EQ(v0_batch_ids, v1_batch_ids);

    ASSERT_EQ(v0_sequential.numTemplates(), static_cast<int>(inputs.size() * variants.size()));
    ASSERT_EQ(v0_batch.numTemplates(), v0_sequential.numTemplates());
    ASSERT_EQ(v1_sequential.numTemplates(), v0_sequential.numTemplates());
    ASSERT_EQ(v1_batch.numTemplates(), v0_sequential.numTemplates());
    for (int template_id = 0; template_id < v0_sequential.numTemplates(); ++template_id)
    {
        const auto &expected = v0_sequential.getTemplate(template_id);
        expectIdenticalTemplate(expected, v0_batch.getTemplate(template_id));
        expectIdenticalTemplate(expected, v1_sequential.getTemplate(template_id));
        expectIdenticalTemplate(expected, v1_batch.getTemplate(template_id));
    }

    const auto expected_matches = v0_sequential.match(scene, 70.0f);
    expectIdenticalMatches(expected_matches, v0_batch.match(scene, 70.0f));
    expectIdenticalMatches(expected_matches, v1_sequential.match(scene, 70.0f));
    expectIdenticalMatches(expected_matches, v1_batch.match(scene, 70.0f));
}

/** @brief v1 复用候选缓冲的高密度候选训练应与 v0 逐字段一致。 */
TEST(ShapeTemplateMatcherTrainingOptimizationTest, ReusedCandidateBufferPreservesDenseVariants)
{
    auto config                      = fastConfig();
    config.num_features              = 96;
    config.min_features              = 32;
    config.weak_threshold            = 4.0f;
    config.strong_threshold          = 8.0f;
    config.min_feature_distance      = 10.5f;
    config.max_training_parallelism = 3;

    const auto image = makeDenseGradientImage();
    const auto variants = irt::features::makeShapeTemplateAngleScaleVariants(-15.0f, 15.0f, 15.0f,
                                                                               0.9f, 1.1f, 0.1f);

    V0ShapeTemplateMatcher v0(config);
    V1ShapeTemplateMatcher v1(config);
    EXPECT_EQ(v0.addTemplateVariants(image, cv::Mat(), variants),
              v1.addTemplateVariants(image, cv::Mat(), variants));
    ASSERT_EQ(v0.numTemplates(), static_cast<int>(variants.size()));
    ASSERT_EQ(v1.numTemplates(), v0.numTemplates());
    for (int template_id = 0; template_id < v0.numTemplates(); ++template_id)
    {
        expectIdenticalTemplate(v0.getTemplate(template_id), v1.getTemplate(template_id));
    }
}

/** @brief 默认运行时选项必须保持精确路径；非默认模板步长只能作为显式近似策略生效。 */
TEST(ShapeTemplateMatcherTest, MatchOptionsDefaultIsExactAndTemplateStrideIsExplicit)
{
    auto config = fastConfig();
    config.max_label_difference = 1;
    config.match_threshold = 90.0f;

    const auto object = makeLShape(50);
    const auto scene = makeSceneWith(object, cv::Point(31, 27), cv::Size(151, 137));
    V1ShapeTemplateMatcher matcher(config);
    matcher.addTemplate(makeTShape(50));
    matcher.addTemplate(object);

    const auto default_matches = matcher.match(scene, 90.0f);
    irt::features::ShapeTemplateMatchOptions exact_options;
    expectIdenticalMatches(default_matches, matcher.match(scene, 90.0f, cv::Mat(), exact_options));
    exact_options.scan_step = 1;
    expectIdenticalMatches(default_matches, matcher.match(scene, 90.0f, cv::Mat(), exact_options));

    const auto exact_target = std::find_if(default_matches.begin(), default_matches.end(),
                                           [](const auto &match) { return match.template_id == 1; });
    ASSERT_NE(exact_target, default_matches.end());

    irt::features::ShapeTemplateMatchOptions approximate_options;
    approximate_options.template_stride = 2;
    const auto approximate_matches = matcher.match(scene, 90.0f, cv::Mat(), approximate_options);
    EXPECT_EQ(std::find_if(approximate_matches.begin(), approximate_matches.end(),
                           [](const auto &match) { return match.template_id == 1; }),
              approximate_matches.end());

    approximate_options = {};
    approximate_options.template_stride = 0;
    expectIrtExceptionCode([&]
                           { (void)matcher.match(scene, 90.0f, cv::Mat(), approximate_options); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
    approximate_options = {};
    approximate_options.scan_step = -1;
    expectIrtExceptionCode([&]
                           { (void)matcher.match(scene, 90.0f, cv::Mat(), approximate_options); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/** @brief 不同扫描参数、输入尺寸和方向容差下，AVX2 与标量路径必须逐结果一致。 */
TEST(ShapeTemplateMatcherParityTest, Avx2AndScalarRemainIdenticalAcrossParameters)
{
    // ``2 * 48 = 96`` 的上界之外还覆盖 ``8 * 48 = 384``，验证 v1 的 uint8 和
    // uint16 累加路径都与固定的 v0 标量基线逐字段一致。
    const std::array<int, 3> label_differences{0, 1, 4};
    const std::array<int, 4> scan_steps{1, 2, 3, 4};
    const std::array<cv::Size, 2> scene_sizes{cv::Size(119, 107), cv::Size(151, 137)};

    for (const int label_difference : label_differences)
    {
        for (const int scan_step : scan_steps)
        {
            for (const auto scene_size : scene_sizes)
            {
                auto config = fastConfig();
                config.max_label_difference = label_difference;
                config.scan_step = scan_step;
                config.match_threshold = 70.0f;
                const auto object = makeLShape(50);
                const auto alternate = makeTShape(50);
                const auto scene = makeSceneWith(object, cv::Point(31, 27), scene_size);

                V1ShapeTemplateMatcher avx2(config);
                V0ShapeTemplateMatcher scalar(config);
                avx2.addTemplate(object);
                avx2.addTemplate(alternate);
                scalar.addTemplate(object);
                scalar.addTemplate(alternate);
                expectIdenticalMatches(avx2.match(scene), scalar.match(scene));
            }
        }
    }
}

/** @brief 多模板串行与自动并行扫描必须返回逐字段一致的稳定结果。 */
TEST(ShapeTemplateMatcherParallelTest, SerialAndAutomaticParallelismProduceIdenticalMatches)
{
    auto serial_config = fastConfig();
    serial_config.max_label_difference = 1;
    serial_config.match_threshold = 70.0f;
    serial_config.max_parallelism = 1;
    auto parallel_config = serial_config;
    parallel_config.max_parallelism = 0;

    const auto object = makeLShape(50);
    const auto alternate = makeTShape(50);
    const auto scene = makeSceneWith(object, cv::Point(37, 29), cv::Size(151, 137));
    V1ShapeTemplateMatcher serial(serial_config);
    V1ShapeTemplateMatcher parallel(parallel_config);
    for (const auto &image : std::array<cv::Mat, 4>{object, object, alternate, alternate})
    {
        serial.addTemplate(image);
        parallel.addTemplate(image);
    }
    expectIdenticalMatches(serial.match(scene), parallel.match(scene));
}
