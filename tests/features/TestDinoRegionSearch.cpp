/**
 * @file TestDinoRegionSearch.cpp
 * @brief DINO 区域检索的 CPU 单元测试：契约、几何、视图、描述子、量化与融合。
 *
 * 覆盖配置、坐标、视图覆盖、描述保真和双路候选行为。
 */

#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/features/DinoRegionSearch.hpp>
#include <yaml-cpp/yaml.h>

#include "dino/DinoContracts.hpp"
#include "dino/DinoFeatureCache.hpp"
#include "dino/DinoIngest.hpp"
#include "dino/DinoSearchCache.hpp"
#include "dino/DinoIndexStore.hpp"
#include "dino/DinoDescriptors.hpp"
#include "dino/DinoFusion.hpp"
#include "dino/DinoGeometry.hpp"
#include "dino/DinoPaths.hpp"
#include "dino/DinoSimilarity.hpp"
#include "dino/DinoTopK.hpp"
#include "dino/DinoTypes.hpp"
#include "dino/DinoTime.hpp"
#include "dino/DinoViews.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using irt::features::priv::DinoAffine;
using irt::features::priv::DinoCandidate;
using irt::features::priv::DinoFeatureGrid;
using irt::features::priv::DinoPoint;
using irt::features::priv::DinoRect;
using irt::features::priv::DinoRoi;
using irt::features::priv::DinoViewPlan;
using irt::features::priv::DinoViewPlanner;
using irt::features::priv::DinoDeadline;
using irt::features::priv::DinoFeatureGridCache;

using irt::features::priv::DinoCanonicalImage;
using irt::features::priv::DinoImageCache;

/** @brief 构造一个可预测特征的网格：每个 patch 的通道 0 编码为量化后的空间位置。 */
DinoFeatureGrid makeGrid(const int grid_height, const int grid_width, const int channels, const DinoViewPlan &plan)
{
    DinoFeatureGrid grid;
    grid.plan       = plan;
    grid.channels   = channels;
    grid.valid_area.assign(static_cast<size_t>(grid_height) * static_cast<size_t>(grid_width), 1.0F);
    grid.tokens.assign(static_cast<size_t>(grid_height) * static_cast<size_t>(grid_width) * static_cast<size_t>(channels),
                       0.0F);
    for (int row = 0; row < grid_height; ++row)
    {
        for (int col = 0; col < grid_width; ++col)
        {
            const auto index = static_cast<size_t>(row) * static_cast<size_t>(grid_width) + static_cast<size_t>(col);
            auto      *token = grid.tokens.data() + index * static_cast<size_t>(channels);
            token[0]         = static_cast<float>(row);
            token[1]         = static_cast<float>(col);
            token[2]         = 1.0F;
            irt::features::priv::dinoNormalizeVector(token, static_cast<size_t>(channels));
        }
    }
    return grid;
}

DinoViewPlan makePlan(const int image_width, const int image_height, const int patch = 16, const int edge = 512)
{
    DinoViewPlanner planner(patch, edge, 0.25, {512, 1024}, {128.0, 256.0, 448.0});
    return planner.makePlan(DinoRect{0.0, 0.0, static_cast<double>(image_width), static_cast<double>(image_height)},
                            image_width, image_height, true, -1);
}

} // namespace

// ---------------------------------------------------------------- 契约与配置

TEST(DinoRegionSearchContract, BboxAndPolygonAreMutuallyExclusive)
{
    const auto box = irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\ntop_k: 3\n");
    EXPECT_TRUE(box.roi.has_bbox);
    EXPECT_FALSE(box.roi.has_polygon);
    EXPECT_FLOAT_EQ(box.roi.bbox.x1, 8.0F);
    const auto polygon = irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\npolygon: [[1, 2], [8, 2], [8, 9]]\n");
    EXPECT_TRUE(polygon.roi.has_polygon);
    EXPECT_FALSE(polygon.roi.has_bbox);
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml("query_path: a.png\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\npolygon: [[1, 2], [8, 2], [8, 9]]\n"), irt::Exception);
}

TEST(DinoRegionSearchContract, InvalidRoiAndSearchSizesAreRejected)
{
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\ntop_k: -1\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, .nan, 9]\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoConfigFromYaml(
        "model: {name: dinov3_vits16, weights_path: unused.wts}\nsearch: {block_descriptors: 0}\n"), irt::Exception);
}

TEST(DinoRegionSearchContract, MalformedProfileValuesUseInvalidArgument)
{
    EXPECT_THROW((void)irt::features::dinoConfigFromYaml(
                     "model: {name: dinov3_vits16, weights_path: unused.wts, encoder_edge: invalid}\n"),
                 irt::Exception);
}

TEST(DinoRegionSearchContract, UnsupportedProfileEnumsAreRejected)
{
    const std::string base = "model: {name: dinov3_vits16, weights_path: unused.wts}\n";
    EXPECT_THROW((void)irt::features::dinoConfigFromYaml(base + "mode: unsupported\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoConfigFromYaml(
                     "model: {name: dinov3_vits16, weights_path: unused.wts, precision: int8}\n"),
                 irt::Exception);
    EXPECT_THROW((void)irt::features::dinoConfigFromYaml(
                     base + "quantization: {format: unsupported}\n"),
                 irt::Exception);
}
TEST(DinoRegionSearchContract, RequestDeadlineAndThresholdUseLightweightContract)
{
    const auto config = irt::features::dinoConfigFromYaml(
        "model: {name: dinov3_vits16, weights_path: unused.wts}\n"
        "decision: {threshold: 0.73}\n"
        "deadline_ms: 30000\n");
    EXPECT_TRUE(config.enable_decision_threshold);
    EXPECT_DOUBLE_EQ(config.decision_threshold, 0.73);

    const auto request = irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\ndeadline_ms: 1\n");
    EXPECT_EQ(request.deadline_ms, 1);
}
TEST(DinoRegionSearchContract, ResponseIncludesCandidatesTimingsAndDiagnostics)
{
    irt::features::DinoSearchResponse response;
    response.request_id = "request-1";
    response.status = irt::features::DinoSearchStatus::Completed;
    response.decision = irt::features::DinoSearchDecision::RankedOnly;
    response.completed_candidates = 2;
    response.total_candidates = 3;
    response.region_candidates.push_back({std::filesystem::path("gallery/a.jpg"),
                                          irt::features::DinoSearchRect{1.0F, 2.0F, 3.0F, 4.0F}});
    response.timings.wall_ms = 12.5;
    response.diagnostics.scanned_region_descriptors = 7;
    response.diagnostics.similarity_backend = "cpu";
    const auto node = YAML::Load(irt::features::dinoSearchResponseToYaml(response));

    EXPECT_EQ(node["status"].as<std::string>(), "completed");
    EXPECT_EQ(node["decision"].as<std::string>(), "ranked_only");
    EXPECT_EQ(node["region_candidates"].size(), 1U);
    EXPECT_EQ(node["region_candidates"][0]["source_path"].as<std::string>(), "gallery/a.jpg");
    EXPECT_DOUBLE_EQ(node["timings"]["wall_ms"].as<double>(), 12.5);
    EXPECT_EQ(node["diagnostics"]["scanned_region_descriptors"].as<size_t>(), 7U);
    EXPECT_EQ(node["diagnostics"]["similarity_backend"].as<std::string>(), "cpu");
}

TEST(DinoRegionSearchContract, RequestDeadlineMustBePositive)
{
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\ndeadline_ms: 0\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "query_path: a.png\nbbox: [1, 2, 8, 9]\ndeadline_ms: invalid\n"), irt::Exception);
    EXPECT_THROW((void)irt::features::dinoSearchRequestFromYaml(
        "request_id: [not-a-string]\nquery_path: a.png\nbbox: [1, 2, 8, 9]\n"), irt::Exception);
}

TEST(DinoRegionSearchPaths, WindowsCaseAliasesShareImageIdentity)
{
#ifdef _WIN32
    const auto root = std::filesystem::path(testing::TempDir()) / "dino_case_identity";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto original = root / "Image.JPG";
    std::ofstream(original.string()).close();
    const auto first = irt::features::priv::DinoImageLoader::statIdentity(original);
    const auto alias = irt::features::priv::DinoImageLoader::statIdentity(root / "image.jpg");
    EXPECT_EQ(first.image_id, alias.image_id);
    std::filesystem::remove_all(root);
#else
    GTEST_SKIP() << "Windows path identity behavior";
#endif
}

TEST(DinoRegionSearchIngest, LargeJpegPreservesDecodedDimensionsAndEdgePixels)
{
    const auto root = std::filesystem::path(testing::TempDir()) / "inferrt_dino_large_jpeg";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto path = root / "large.jpg";

    constexpr int width = 3060;
    constexpr int height = 4520;
    cv::Mat source(height, width, CV_8UC3, cv::Scalar(23, 87, 191));
    // Keep a substantial marker away from the border while still exercising the final decoded pixels.
    source(cv::Rect(width - 24, height - 24, 16, 16)).setTo(cv::Scalar(41, 129, 213));
    ASSERT_TRUE(cv::imwrite(path.string(), source, {cv::IMWRITE_JPEG_QUALITY, 100}));

    const auto loaded = irt::features::priv::DinoImageLoader::load(path);
    EXPECT_EQ(loaded.record.width, width);
    EXPECT_EQ(loaded.record.height, height);
    ASSERT_EQ(loaded.image.cols, width);
    ASSERT_EQ(loaded.image.rows, height);

    const cv::Vec3b edge_pixel = loaded.image.at<cv::Vec3b>(height - 16, width - 16);
    EXPECT_NEAR(edge_pixel[0], 41, 8);
    EXPECT_NEAR(edge_pixel[1], 129, 8);
    EXPECT_NEAR(edge_pixel[2], 213, 8);

    std::filesystem::remove_all(root);
}



TEST(DinoRegionSearchIndex, ViewsRetainDistinctDescriptorsAfterReopening)
{
    for (const bool quantized : {false, true})
    {
        const auto root = std::filesystem::path(testing::TempDir())
            / (quantized ? "inferrt_dino_views_int8" : "inferrt_dino_views_fp32");
        std::filesystem::remove_all(root);
        {
            irt::features::priv::DinoIndexWriter writer(root, 3, quantized);
            irt::features::priv::DinoImageIdentity image;
            image.image_id = "test-image";
            image.source_path = "test-image.png";
            image.width = image.height = 64;
            writer.addImage(image);
            for (int view = 0; view < 2; ++view)
            {
                const auto id = writer.addView(0, makePlan(64, 64));
                for (int row = 0; row <= view; ++row)
                {
                    irt::features::priv::DinoRegionDescriptor region;
                    region.grid_row = row;
                    region.grid_height = region.grid_width = 1;
                    region.valid_fraction = 1.0F;
                    irt::features::priv::DinoLocalLeaf local;
                    local.grid_row = local.rep_row = row;
                    local.grid_height = local.grid_width = local.member_count = 1;
                    const std::vector<float> vector = view == 0 ? std::vector<float>{1, 0, 0}
                        : std::vector<float>{0, 1, 0};
                    writer.addRegion(id, region, vector);
                    writer.addLocal(id, local, vector);
                }
            }
            writer.finish();
        }
        {
            irt::features::priv::DinoIndexReader reader(root);
            for (size_t view = 0; view < 2; ++view)
            {
                const auto region = reader.regionRange(view);
                const auto local = reader.localRange(view);
                ASSERT_EQ(region.count, view + 1);
                ASSERT_EQ(local.count, view + 1);
                std::vector<float> values;
                reader.readRegionVectors(region.begin, region.count, values);
                for (size_t row = 0; row < region.count; ++row)
                {
                    EXPECT_NEAR(values[row * 3 + view], 1.0F, 1e-5F);
                    EXPECT_EQ(reader.regionMetaAt(region.begin + row).view_id, static_cast<int>(view));
                }
                reader.readLocalVectors(local.begin, local.count, values);
                for (size_t row = 0; row < local.count; ++row)
                    EXPECT_NEAR(values[row * 3 + view], 1.0F, 1e-5F);
            }
        }
        std::filesystem::remove_all(root);
    }
}

// ---------------------------------------------------------------- 几何与坐标

TEST(DinoRegionSearchGeometry, RectAndPolygonAreasMatchAnalyticValues)
{
    const DinoRect rect{1.0, 2.0, 5.0, 7.0};
    EXPECT_NEAR(rect.area(), 20.0, 1e-9);
    // [1,5]x[2,7] 与 [4,9]x[6,9] 的交集是 [4,5]x[6,7] = 1
    EXPECT_NEAR(irt::features::priv::dinoRectIntersectionArea(rect, DinoRect{4.0, 6.0, 9.0, 9.0}), 1.0, 1e-9);

    const std::vector<DinoPoint> square{{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0}};
    EXPECT_NEAR(irt::features::priv::dinoPolygonArea(square), 16.0, 1e-9);
    // 与半平面裁剪后的解析面积一致：矩形 [0,2]x[0,2] 覆盖四分之一。
    EXPECT_NEAR(irt::features::priv::dinoPolygonRectArea(square, DinoRect{0.0, 0.0, 2.0, 2.0}), 4.0, 1e-9);

    std::string message;
    const std::vector<DinoPoint> bowtie{{0.0, 0.0}, {4.0, 4.0}, {4.0, 0.0}, {0.0, 4.0}};
    EXPECT_FALSE(irt::features::priv::dinoValidatePolygon(bowtie, message));
    EXPECT_FALSE(message.empty());
    EXPECT_TRUE(irt::features::priv::dinoValidatePolygon(square, message));
}

TEST(DinoRegionSearchGeometry, RoiValidationRejectsDegenerateAndOutOfRangeInputs)
{
    irt::features::DinoSearchRoi roi;
    roi.has_bbox = true;
    roi.bbox     = irt::features::DinoSearchRect{0.0F, 0.0F, 0.0F, 10.0F};
    EXPECT_THROW((void)irt::features::priv::dinoToInternalRoi(roi, 100, 100), irt::Exception);

    roi.bbox = irt::features::DinoSearchRect{0.0F, 0.0F, 300.0F, 300.0F};
    EXPECT_THROW((void)irt::features::priv::dinoToInternalRoi(roi, 100, 100), irt::Exception);

    roi.bbox = irt::features::DinoSearchRect{-1e-5F, 0.0F, 90.0F, 90.0F};
    const auto inner = irt::features::priv::dinoToInternalRoi(roi, 100, 100);
    EXPECT_NEAR(inner.bbox.x0, 0.0, 1e-9);
}

TEST(DinoRegionSearchGeometry, CoordinateRoundTripStaysWithinTolerance)
{
    const auto plan = makePlan(1000, 700);
    // TC005：原图四角与半像素 bbox 往返误差 ≤ 1e-3 px。
    const std::vector<DinoRect> samples{
        {0.0, 0.0, 1.0, 1.0},
        {999.0, 699.0, 1000.0, 700.0},
        {10.5, 20.25, 310.75, 420.5},
    };
    for (const auto &rect : samples)
    {
        const auto input     = plan.canonical_to_input.apply(rect);
        const auto restored  = plan.input_to_canonical.apply(input);
        EXPECT_NEAR(restored.x0, rect.x0, 1e-3);
        EXPECT_NEAR(restored.y0, rect.y0, 1e-3);
        EXPECT_NEAR(restored.x1, rect.x1, 1e-3);
        EXPECT_NEAR(restored.y1, rect.y1, 1e-3);
    }

    // 单个 patch 的 canonical 覆盖必须与采样间隔一致。
    const auto patch = plan.patchRect(0, 1);
    EXPECT_NEAR(patch.x1 - patch.x0, plan.sourcePxPerPatchX(), 1e-3);
    EXPECT_NEAR(patch.x1 - patch.x0, static_cast<double>(plan.patch_size) * plan.canonicalPerInputPxX(), 1e-3);
}

TEST(DinoRegionSearchGeometry, PatchWeightsFollowAreaNotCentres)
{
    const auto plan = makePlan(512, 512);
    DinoRoi    roi;
    roi.is_polygon = false;
    roi.bbox       = DinoRect{16.0, 0.0, 32.0, 512.0}; // 正好覆盖左起第 2 个 patch 列

    const auto weights = irt::features::priv::dinoPatchRoiWeights(plan, roi);
    EXPECT_NEAR(weights[1], 1.0F, 1e-6F);
    EXPECT_NEAR(weights[0], 0.0F, 1e-6F);
    EXPECT_NEAR(weights[2], 0.0F, 1e-6F);

    // 半个 patch 的覆盖必须得到 0.5，而不是按中心点判定的 0 或 1。
    roi.bbox        = DinoRect{0.0, 0.0, 8.0, 16.0};
    const auto half = irt::features::priv::dinoPatchRoiWeights(plan, roi);
    EXPECT_NEAR(half[0], 0.5F, 1e-6F);
    EXPECT_NEAR(half[1], 0.0F, 1e-6F);
}

// ---------------------------------------------------------------- 视图规划

TEST(DinoRegionSearchViews, GalleryViewsCoverEveryRealPixel)
{
    const int   width  = 1001;
    const int   height = 513; // TC011：非整除、非 2 次幂边长
    DinoViewPlanner planner(16, 512, 0.25, {512, 1024, 2048}, {128.0, 256.0, 448.0});
    const auto  plans = planner.planGalleryViews(width, height);
    ASSERT_FALSE(plans.empty());

    std::vector<int> covered(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    for (const auto &plan : plans)
    {
        const auto rect = DinoRect::intersect(plan.source_rect, DinoRect{0.0, 0.0, static_cast<double>(width),
                                                                        static_cast<double>(height)});
        for (int y = static_cast<int>(rect.y0); y < static_cast<int>(rect.y1); ++y)
        {
            for (int x = static_cast<int>(rect.x0); x < static_cast<int>(rect.x1); ++x)
            {
                ++covered[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
            }
        }
    }
    EXPECT_EQ(std::count(covered.begin(), covered.end(), 0), 0);

    // 整图视图必须是第一个，且同一源区域不重复登记。
    EXPECT_TRUE(plans.front().is_full_view);
    std::vector<std::string> keys;
    for (const auto &plan : plans)
    {
        keys.push_back(std::to_string(static_cast<long long>(plan.source_rect.x0)) + ":"
                       + std::to_string(static_cast<long long>(plan.source_rect.y0)) + ":"
                       + std::to_string(static_cast<long long>(plan.source_rect.x1)) + ":"
                       + std::to_string(static_cast<long long>(plan.source_rect.y1)));
    }
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(std::adjacent_find(keys.begin(), keys.end()), keys.end());
}

TEST(DinoRegionSearchViews, QueryViewsPlaceRoiAtDistinctTargetLengths)
{
    DinoViewPlanner planner(16, 512, 0.25, {512}, {128.0, 256.0, 448.0});
    DinoRoi         roi;
    roi.is_polygon = false;
    roi.bbox       = DinoRect{200.0, 150.0, 480.0, 430.0};

    const auto plans = planner.planQueryViews(roi, 1000, 800);
    ASSERT_GE(plans.size(), 2U);
    for (const auto &plan : plans)
    {
        // ROI 必须完整落在上下文视图内，padding 不参与描述。
        const auto roi_in_input = plan.canonical_to_input.apply(roi.bbox);
        EXPECT_GE(roi_in_input.x0, -1e-6);
        EXPECT_GE(roi_in_input.y0, -1e-6);
        EXPECT_LE(roi_in_input.x1, plan.input_width + 1e-6);
        EXPECT_LE(roi_in_input.y1, plan.input_height + 1e-6);
    }
    EXPECT_TRUE(plans[0].source_rect.width() > plans.back().source_rect.width());
}

TEST(DinoRegionSearchViews, PaddingPatchesAreNotValidArea)
{
    // 上下文方框超出原图时，越界部分不能计入有效面积。
    DinoViewPlanner planner(16, 512, 0.25, {512}, {64.0});
    DinoRoi         roi;
    roi.is_polygon = false;
    roi.bbox       = DinoRect{5.0, 5.0, 200.0, 200.0};
    const auto plans = planner.planQueryViews(roi, 400, 300);
    ASSERT_FALSE(plans.empty());

    const auto area = irt::features::priv::dinoPatchValidArea(plans.front());
    const auto sum  = std::accumulate(area.begin(), area.end(), 0.0F);
    EXPECT_GT(sum, 0.0F);
    EXPECT_LT(sum, static_cast<float>(plans.front().patchCount()));
}

// ---------------------------------------------------------------- 描述与压缩

TEST(DinoRegionSearchDescriptors, PooledWindowMatchesAreaWeightedReference)
{
    const auto plan = makePlan(512, 512);
    auto       grid = makeGrid(plan.grid_height, plan.grid_width, 4, plan);
    // 让部分 patch 无效，验证池化只计入有效面积。
    grid.valid_area[0] = 0.0F;
    grid.valid_area[1] = 0.5F;

    irt::features::priv::DinoRegionWindowConfig window;
    window.ratios             = {0.5};
    window.stride_ratio       = 0.5;
    window.min_valid_fraction = 0.0;

    const auto descriptors = irt::features::priv::dinoEnumerateRegionWindows(grid, window, 0);
    ASSERT_FALSE(descriptors.empty());
    const auto &descriptor = descriptors.front();
    const auto  pooled     = irt::features::priv::dinoPoolRegionWindow(grid, descriptor);

    // 参考实现：逐 patch 面积加权求和后归一化。
    std::vector<double> reference(static_cast<size_t>(grid.channels), 0.0);
    double              weight_sum = 0.0;
    for (int row = descriptor.grid_row; row < descriptor.grid_row + descriptor.grid_height; ++row)
    {
        for (int col = descriptor.grid_col; col < descriptor.grid_col + descriptor.grid_width; ++col)
        {
            const auto   index  = static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width) + static_cast<size_t>(col);
            const float  weight = grid.valid_area[index];
            const float *token  = grid.token(row, col);
            for (int channel = 0; channel < grid.channels; ++channel)
            {
                reference[static_cast<size_t>(channel)] += static_cast<double>(token[channel]) * weight;
            }
            weight_sum += weight;
        }
    }
    double norm = 0.0;
    for (auto &value : reference)
    {
        value /= weight_sum;
        norm += value * value;
    }
    norm = std::sqrt(norm);
    ASSERT_EQ(pooled.size(), reference.size());
    for (size_t channel = 0; channel < pooled.size(); ++channel)
    {
        EXPECT_NEAR(pooled[channel], reference[channel] / norm, 1e-5);
    }
}

TEST(DinoRegionSearchDescriptors, MergeRespectsRadiusAndLeafLimitAndDropsNothing)
{
    const auto plan = makePlan(512, 512);
    const auto grid = makeGrid(plan.grid_height, plan.grid_width, 4, plan);

    irt::features::priv::DinoDescriptorBuildConfig config;
    config.window.ratios             = {1.0};
    config.window.stride_ratio       = 0.5;
    config.window.min_valid_fraction = 0.0;
    config.merge_enabled             = true;
    config.merge_epsilon             = 0.10;
    config.max_leaf_side_patches     = 4;

    const auto descriptors = irt::features::priv::dinoBuildViewDescriptors(0, grid, config);
    EXPECT_EQ(descriptors.stats.original_patch_count, static_cast<uint64_t>(grid.plan.patchCount()));
    EXPECT_EQ(descriptors.stats.valid_patch_count, static_cast<uint64_t>(grid.plan.patchCount()));

    // 每个叶节点必须覆盖连续矩形，且覆盖的 patch 总数等于有效 patch 数（不丢证据）。
    std::vector<int> coverage(static_cast<size_t>(grid.plan.patchCount()), 0);
    for (const auto &leaf : descriptors.local_meta)
    {
        EXPECT_GE(leaf.grid_height, 1);
        EXPECT_GE(leaf.grid_width, 1);
        EXPECT_LE(leaf.grid_height, config.max_leaf_side_patches);
        EXPECT_LE(leaf.grid_width, config.max_leaf_side_patches);
        for (int row = leaf.grid_row; row < leaf.grid_row + leaf.grid_height; ++row)
        {
            for (int col = leaf.grid_col; col < leaf.grid_col + leaf.grid_width; ++col)
            {
                ++coverage[static_cast<size_t>(row) * static_cast<size_t>(plan.grid_width) + static_cast<size_t>(col)];
            }
        }
    }
    EXPECT_EQ(std::count(coverage.begin(), coverage.end(), 0), 0);
    EXPECT_EQ(std::count_if(coverage.begin(), coverage.end(), [](const int value) { return value > 1; }), 0);
}

TEST(DinoRegionSearchDescriptors, MergeNeverLosesAnOutlierFeature)
{
    // TC016：在同质区域中插入一个远离特征，该特征不能被合并吞掉。
    const auto   plan = makePlan(512, 512);
    auto         grid = makeGrid(plan.grid_height, plan.grid_width, 4, plan);
    const size_t outlier_index = 5U * static_cast<size_t>(plan.grid_width) + 5U;
    auto        *outlier = grid.tokens.data() + outlier_index * static_cast<size_t>(grid.channels);
    outlier[0]           = 0.0F;
    outlier[1]           = 1.0F;
    irt::features::priv::dinoNormalizeVector(outlier, static_cast<size_t>(grid.channels));

    irt::features::priv::DinoDescriptorBuildConfig config;
    config.window.ratios             = {1.0};
    config.window.min_valid_fraction = 0.0;
    config.merge_enabled             = true;
    config.merge_epsilon             = 0.10;
    config.max_leaf_side_patches     = 4;

    const auto descriptors = irt::features::priv::dinoBuildViewDescriptors(0, grid, config);

    bool found = false;
    for (const auto &leaf : descriptors.local_meta)
    {
        EXPECT_LE(leaf.radius, static_cast<float>(config.merge_epsilon) + 1e-6F);
        if (leaf.member_count == 1 && leaf.grid_row == 5 && leaf.grid_col == 5)
        {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "outlier patch must stay as its own leaf";
}

TEST(DinoRegionSearchDescriptors, UnmergedBaselineKeepsEveryValidPatch)
{
    const auto plan = makePlan(512, 512);
    auto       grid = makeGrid(plan.grid_height, plan.grid_width, 4, plan);
    grid.valid_area[3] = 0.0F;

    irt::features::priv::DinoDescriptorBuildConfig config;
    config.window.ratios             = {1.0};
    config.window.min_valid_fraction = 0.0;
    config.merge_enabled             = false;

    const auto descriptors = irt::features::priv::dinoBuildViewDescriptors(0, grid, config);
    EXPECT_EQ(descriptors.local_meta.size(), static_cast<size_t>(grid.plan.patchCount() - 1));
    for (const auto &leaf : descriptors.local_meta)
    {
        EXPECT_EQ(leaf.member_count, 1);
        EXPECT_EQ(leaf.radius, 0.0F);
    }
}

TEST(DinoRegionSearchDescriptors, Int8RoundTripHonoursTheErrorBound)
{
    std::vector<float> values{0.3F, -0.9F, 0.15F, 0.0F, 0.25F, -0.05F};
    ASSERT_TRUE(irt::features::priv::dinoNormalizeVector(values.data(), values.size()));

    const auto quantized = irt::features::priv::dinoQuantizeInt8(values.data(), values.size());
    EXPECT_GT(quantized.scale, 0.0F);
    ASSERT_EQ(quantized.codes.size(), values.size());

    std::vector<float> restored(values.size());
    irt::features::priv::dinoDequantizeInt8(quantized, restored.data());
    float norm = 0.0F;
    for (const auto value : restored)
    {
        norm += value * value;
    }
    EXPECT_NEAR(norm, 1.0F, 1e-4F);

    const auto delta = irt::features::priv::dinoQuantizationDelta(values.data(), quantized);
    EXPECT_LT(delta, 0.02F);

    // TC020：单条特征的点积误差不超过 radius + delta。
    const float        radius = 0.05F;
    std::vector<float> query(values.size());
    query[0] = 1.0F;
    float dot_original = 0.0F;
    float dot_restored = 0.0F;
    for (size_t index = 0; index < values.size(); ++index)
    {
        dot_original += query[index] * values[index];
        dot_restored += query[index] * restored[index];
    }
    EXPECT_LE(std::abs(dot_original - dot_restored), radius + delta + 1e-5F);
}

TEST(DinoRegionSearchDescriptors, Int8RejectsDegenerateInput)
{
    std::vector<float> zeros(8, 0.0F);
    EXPECT_THROW((void)irt::features::priv::dinoQuantizeInt8(zeros.data(), zeros.size()), irt::Exception);
}

// ---------------------------------------------------------------- 扫描与融合

TEST(DinoRegionSearchPaths, NonAsciiPathSurvivesTheUtf8ContractRoundTrip)
{
    // 契约路径统一 UTF-8，Windows 文件路径通过公共转换入口还原。
    const std::string utf8 = "D:/数据集/擦花20180830164545对照样本.jpg";
    const auto        path = irt::features::priv::dinoPathFromUtf8(utf8);

    EXPECT_EQ(irt::features::priv::dinoPathToUtf8(path), utf8);
    EXPECT_EQ(irt::features::priv::dinoPathToUtf8(path.filename()), utf8.substr(utf8.rfind('/') + 1));

    const auto request = irt::features::dinoSearchRequestFromYaml(
        "query_path: " + utf8 + "\nbbox: [0, 0, 10, 10]\n");
    EXPECT_EQ(irt::features::priv::dinoPathToUtf8(request.query_path), utf8);
}

TEST(DinoRegionSearchScan, SimilarityReductionMatchesThePerTokenMaximum)
{
    // 参考定义：每个查询 token 在所有描述子上的最大值；描述子数与 token 数都取
    // 非 4 的倍数，覆盖向量内核的描述子展开尾部与查询 token 尾部。
    const int    dimension     = 24;
    const size_t descriptor_count = 11U;
    const int    token_count   = 3;

    std::vector<float>      descriptors(descriptor_count * dimension);
    std::vector<float>      tokens(token_count * dimension);
    for (size_t index = 0; index < descriptors.size(); ++index)
    {
        descriptors[index] = static_cast<float>(std::sin(static_cast<double>(index) * 0.31));
    }
    for (size_t index = 0; index < tokens.size(); ++index)
    {
        tokens[index] = static_cast<float>(std::cos(static_cast<double>(index) * 0.17));
    }

    // token 0/2 共享同一个归约槽位，token 1 独占一槽，验证按槽位取最大而不是按 token。
    const std::vector<int> slot_of_token{0, 5, 0};
    std::vector<float>     maxima(6, -std::numeric_limits<float>::infinity());
    irt::features::priv::dinoAccumulateSimilarityMaxima(descriptors.data(), descriptor_count, tokens.data(),
                                                       token_count, dimension, slot_of_token.data(), maxima.data());

    std::vector<float> reference(6, -std::numeric_limits<float>::infinity());
    for (int token = 0; token < token_count; ++token)
    {
        for (size_t descriptor = 0; descriptor < descriptor_count; ++descriptor)
        {
            float score = 0.0F;
            for (int channel = 0; channel < dimension; ++channel)
            {
                score += descriptors[descriptor * dimension + channel] * tokens[token * dimension + channel];
            }
            const int slot = slot_of_token[static_cast<size_t>(token)];
            reference[static_cast<size_t>(slot)] = std::max(reference[static_cast<size_t>(slot)], score);
        }
    }

    for (size_t slot = 0; slot < reference.size(); ++slot)
    {
        if (reference[slot] == -std::numeric_limits<float>::infinity())
        {
            EXPECT_EQ(maxima[slot], reference[slot]) << "slot " << slot;
            continue;
        }
        EXPECT_NEAR(maxima[slot], reference[slot], 1e-5F) << "slot " << slot;
    }
}

TEST(DinoRegionSearchScan, CompactReductionMatchesAcrossBackends)
{
    // T13 契约：紧凑块（INT8 码 + 还原因子）在 CPU 与 CUDA 归约后给出与
    // FP32 参考相同的逐 token 最大值；各实现只允许改变浮点累加顺序。
    const int    dimension        = 384;
    const size_t descriptor_count = 5000U;
    const int    token_count      = 5;

    std::vector<int8_t> codes(descriptor_count * dimension);
    std::vector<float> factors(descriptor_count);
    std::vector<float> tokens(token_count * dimension);
    for (size_t index = 0; index < codes.size(); ++index)
    {
        codes[index] = static_cast<int8_t>((index * 37U + 11U) % 251U - 125U);
    }
    for (size_t index = 0; index < factors.size(); ++index)
    {
        factors[index] = 0.001F + 0.002F * static_cast<float>(index % 7U);
    }
    for (size_t index = 0; index < tokens.size(); ++index)
    {
        tokens[index] = static_cast<float>(std::cos(static_cast<double>(index) * 0.13));
    }

    const std::vector<int> identity_slots{0, 1, 2, 3, 4};
    std::vector<float>      restored(descriptor_count * dimension);
    for (size_t descriptor = 0; descriptor < descriptor_count; ++descriptor)
    {
        for (int channel = 0; channel < dimension; ++channel)
        {
            restored[descriptor * dimension + channel]
                = static_cast<float>(codes[descriptor * dimension + channel]) * factors[descriptor];
        }
    }
    std::vector<float> reference(token_count, -std::numeric_limits<float>::infinity());
    irt::features::priv::dinoAccumulateSimilarityMaxima(restored.data(), descriptor_count, tokens.data(),
                                                        token_count, dimension, identity_slots.data(),
                                                        reference.data());

    irt::features::priv::DinoCompactBlock block{};
    block.codes     = codes.data();
    block.factors   = factors.data();
    block.count     = descriptor_count;
    block.dimension = static_cast<size_t>(dimension);

    const auto runReducer = [&](const irt::features::priv::DinoSimilarityBackend backend)
    {
        std::vector<float> maxima(token_count, -std::numeric_limits<float>::infinity());
        irt::features::priv::DinoSimilarityReducer reducer(backend, dimension);
        const auto reduce_chunk = [&](const irt::features::priv::DinoCompactBlock &chunk)
        {
            const size_t view_offset = 0U;
            const size_t view_count  = chunk.count;
            std::vector<float> chunk_max(token_count, -std::numeric_limits<float>::infinity());
            reducer.reduceViewGroup(chunk, &view_offset, &view_count, 1U, tokens.data(), token_count,
                                    chunk_max.data());
            for (int token = 0; token < token_count; ++token)
            {
                maxima[static_cast<size_t>(token)]
                    = std::max(maxima[static_cast<size_t>(token)], chunk_max[static_cast<size_t>(token)]);
            }
        };

        // 分两块归约，覆盖 block 边界与多次调用的最大值合并。
        irt::features::priv::DinoCompactBlock first  = block;
        irt::features::priv::DinoCompactBlock second = block;
        first.count       = 3100U;
        second.codes     = block.codes + first.count * block.dimension;
        second.factors   = block.factors + first.count;
        second.count      = descriptor_count - first.count;
        reduce_chunk(first);
        reduce_chunk(second);
        return maxima;
    };

    const auto cpu_maxima = runReducer(irt::features::priv::DinoSimilarityBackend::Cpu);
    for (size_t token = 0; token < reference.size(); ++token)
    {
        EXPECT_NEAR(cpu_maxima[token], reference[token], 1e-4F) << "cpu token " << token;
    }

    if (irt::features::priv::dinoCudaSimilarityAvailable())
    {
        const auto cuda_maxima = runReducer(irt::features::priv::DinoSimilarityBackend::Cuda);
        for (size_t token = 0; token < reference.size(); ++token)
        {
            EXPECT_NEAR(cuda_maxima[token], reference[token], 1e-4F) << "cuda token " << token;
        }
    }
    else
    {
        GTEST_SKIP() << "No CUDA device available for the similarity kernel";
    }
}

TEST(DinoRegionSearchScan, CompactRegionScoresMatchTheRestoredDotProducts)
{
    const int    dimension           = 31;
    const size_t descriptor_count    = 37U;
    const int    query_view_count    = 3;
    std::vector<int8_t> codes(descriptor_count * dimension);
    std::vector<float> factors(descriptor_count);
    std::vector<float> roi_vectors(static_cast<size_t>(query_view_count) * dimension);
    for (size_t index = 0; index < codes.size(); ++index)
    {
        codes[index] = static_cast<int8_t>((index * 19U + 7U) % 127U - 63U);
    }
    for (size_t index = 0; index < factors.size(); ++index)
    {
        factors[index] = 0.002F + 0.001F * static_cast<float>(index % 5U);
    }
    for (size_t index = 0; index < roi_vectors.size(); ++index)
    {
        roi_vectors[index] = static_cast<float>(std::sin(static_cast<double>(index) * 0.21));
    }

    std::vector<float> reference(descriptor_count * static_cast<size_t>(query_view_count));
    for (size_t descriptor = 0; descriptor < descriptor_count; ++descriptor)
    {
        for (int query_view = 0; query_view < query_view_count; ++query_view)
        {
            float score = 0.0F;
            for (int channel = 0; channel < dimension; ++channel)
            {
                score += static_cast<float>(codes[descriptor * dimension + channel]) * factors[descriptor]
                       * roi_vectors[static_cast<size_t>(query_view) * dimension + channel];
            }
            reference[descriptor * static_cast<size_t>(query_view_count) + query_view] = score;
        }
    }

    irt::features::priv::DinoCompactBlock block{codes.data(), factors.data(), descriptor_count,
                                                static_cast<size_t>(dimension)};
    const auto runReducer = [&](const irt::features::priv::DinoSimilarityBackend backend)
    {
        std::vector<float> actual(reference.size(), 0.0F);
        irt::features::priv::DinoSimilarityReducer reducer(backend, dimension);
        reducer.reduceRegionScores(block, roi_vectors.data(), query_view_count, actual.data());
        return actual;
    };

    const auto cpu_scores = runReducer(irt::features::priv::DinoSimilarityBackend::Cpu);
    for (size_t index = 0; index < reference.size(); ++index)
    {
        EXPECT_NEAR(cpu_scores[index], reference[index], 1e-5F) << "cpu score " << index;
    }
    if (irt::features::priv::dinoCudaSimilarityAvailable())
    {
        const auto cuda_scores = runReducer(irt::features::priv::DinoSimilarityBackend::Cuda);
        for (size_t index = 0; index < reference.size(); ++index)
        {
            EXPECT_NEAR(cuda_scores[index], reference[index], 1e-4F) << "cuda score " << index;
        }
    }
}

TEST(DinoRegionSearchScan, CompactReductionHonoursMultipleViewRanges)
{
    const int    dimension        = 17;
    const int    token_count      = 3;
    const size_t descriptor_count = 9U;
    std::vector<int8_t> codes(descriptor_count * dimension);
    std::vector<float> factors(descriptor_count, 0.01F);
    std::vector<float> tokens(static_cast<size_t>(token_count) * dimension);
    for (size_t index = 0; index < codes.size(); ++index)
    {
        codes[index] = static_cast<int8_t>((index * 23U + 5U) % 127U - 63U);
    }
    for (size_t index = 0; index < tokens.size(); ++index)
    {
        tokens[index] = static_cast<float>(std::cos(static_cast<double>(index) * 0.17));
    }
    const std::vector<size_t> offsets{0U, 4U, 7U};
    const std::vector<size_t> counts{4U, 3U, 2U};
    std::vector<float> reference(counts.size() * static_cast<size_t>(token_count),
                                 -std::numeric_limits<float>::infinity());
    for (size_t view = 0; view < counts.size(); ++view)
    {
        for (size_t descriptor = offsets[view]; descriptor < offsets[view] + counts[view]; ++descriptor)
        {
            for (int token = 0; token < token_count; ++token)
            {
                float score = 0.0F;
                for (int channel = 0; channel < dimension; ++channel)
                {
                    score += static_cast<float>(codes[descriptor * dimension + channel]) * factors[descriptor]
                           * tokens[static_cast<size_t>(token) * dimension + channel];
                }
                auto &best = reference[view * static_cast<size_t>(token_count) + static_cast<size_t>(token)];
                best = std::max(best, score);
            }
        }
    }

    irt::features::priv::DinoCompactBlock block{codes.data(), factors.data(), descriptor_count,
                                                static_cast<size_t>(dimension)};
    const auto runReducer = [&](const irt::features::priv::DinoSimilarityBackend backend)
    {
        std::vector<float> actual(reference.size(), -std::numeric_limits<float>::infinity());
        irt::features::priv::DinoSimilarityReducer reducer(backend, dimension);
        reducer.reduceViewGroup(block, offsets.data(), counts.data(), counts.size(), tokens.data(), token_count,
                                actual.data());
        return actual;
    };
    const auto cpu_actual = runReducer(irt::features::priv::DinoSimilarityBackend::Cpu);
    for (size_t index = 0; index < reference.size(); ++index)
    {
        EXPECT_NEAR(cpu_actual[index], reference[index], 1e-5F) << "cpu result " << index;
    }
    if (irt::features::priv::dinoCudaSimilarityAvailable())
    {
        const auto cuda_actual = runReducer(irt::features::priv::DinoSimilarityBackend::Cuda);
        for (size_t index = 0; index < reference.size(); ++index)
        {
            EXPECT_NEAR(cuda_actual[index], reference[index], 1e-4F) << "cuda result " << index;
        }
    }
}

TEST(DinoRegionSearchFusion, TopKIsIndependentOfBlockSize)
{
    std::vector<float> scores(1000);
    for (size_t index = 0; index < scores.size(); ++index)
    {
        scores[index] = static_cast<float>(std::sin(static_cast<double>(index) * 0.37));
    }

    const auto collect = [&](const size_t block)
    {
        irt::features::priv::DinoTopK top(20);
        for (size_t begin = 0; begin < scores.size(); begin += block)
        {
            const size_t count = std::min(block, scores.size() - begin);
            for (size_t index = 0; index < count; ++index)
            {
                top.push(scores[begin + index], begin + index);
            }
        }
        return top.sorted();
    };

    const auto reference = collect(scores.size());
    const std::vector<size_t> blocks{1U, 7U, 64U, 333U};
    for (const auto block : blocks)
    {
        const auto other = collect(block);
        ASSERT_EQ(reference.size(), other.size());
        for (size_t index = 0; index < reference.size(); ++index)
        {
            EXPECT_EQ(reference[index].second, other[index].second);
            EXPECT_NEAR(reference[index].first, other[index].first, 1e-6F);
        }
    }
}

TEST(DinoRegionSearchFusion, QuotasBackfillAndDuplicateMergingAreCounted)
{
    auto make = [](const int view_id, const double x0, const double y0, const double size, const float score)
    {
        DinoCandidate candidate;
        candidate.image_id  = "image";
        candidate.view_id   = view_id;
        candidate.from_region = true;
        candidate.region_score = score;
        candidate.score        = score;
        candidate.source_bbox  = DinoRect{x0, y0, x0 + size, y0 + size};
        return candidate;
    };

    std::vector<DinoCandidate> region;
    for (int index = 0; index < 30; ++index)
    {
        region.push_back(make(index, index * 100.0, 0.0, 50.0, 1.0F - 0.01F * static_cast<float>(index)));
    }
    std::vector<DinoCandidate> local;
    for (int index = 0; index < 30; ++index)
    {
        local.push_back(make(100 + index, index * 100.0, 0.0, 50.0, 0.9F - 0.01F * static_cast<float>(index)));
    }

    irt::features::DinoRegionSearchConfig config;
    config.coarse_k = 10;

    const auto fused = irt::features::priv::dinoFuseCandidates(region, local, config);
    EXPECT_LE(fused.candidates.size(), config.coarse_k);
    EXPECT_GT(fused.region_taken, 0U);
    EXPECT_GT(fused.local_taken, 0U);
    EXPECT_EQ(fused.candidates.size(), fused.region_taken + fused.local_taken);

    // 同一图片的重叠候选必须合并，而不是按 image_id 整张删除。
    std::vector<DinoCandidate> duplicated{make(0, 0.0, 0.0, 100.0, 0.8F), make(1, 1.0, 1.0, 100.0, 0.7F)};
    const auto merged = irt::features::priv::dinoFuseCandidates(duplicated, {}, config);
    EXPECT_EQ(merged.candidates.size(), 1U);
    EXPECT_EQ(merged.duplicates_merged, 1U);
    EXPECT_NEAR(merged.candidates.front().source_bbox.x0, 0.0, 1e-9);

    // 分开的位置不能被合并。
    std::vector<DinoCandidate> separate{make(0, 0.0, 0.0, 100.0, 0.8F), make(1, 500.0, 500.0, 100.0, 0.7F)};
    const auto kept = irt::features::priv::dinoFuseCandidates(separate, {}, config);
    EXPECT_EQ(kept.candidates.size(), 2U);
}

TEST(DinoRegionSearchFusion, NmsSuppressesOnlyWithinTheSameImage)
{
    auto make = [](const std::string &image_id, const double x0, const float score)
    {
        irt::features::priv::DinoMatchResult result;
        result.image_id = image_id;
        result.bbox     = DinoRect{x0, 0.0, x0 + 100.0, 100.0};
        result.score    = score;
        return result;
    };

    std::vector<irt::features::priv::DinoMatchResult> results{make("a", 0.0, 0.9F), make("a", 5.0, 0.8F),
                                                              make("b", 5.0, 0.7F), make("a", 500.0, 0.6F)};
    irt::features::priv::dinoNmsWithinImages(results, 0.5);
    ASSERT_EQ(results.size(), 3U);
    EXPECT_FLOAT_EQ(results.front().score, 0.9F);
    EXPECT_EQ(std::count_if(results.begin(), results.end(),
                            [](const irt::features::priv::DinoMatchResult &result)
                            { return result.image_id == "b"; }),

              1);
}
TEST(DinoResourceTest, FeatureCacheUsesByteBudgetAndContentScopedKey)
{
    DinoFeatureGrid grid;
    grid.channels = 1;
    grid.tokens   = {1.0F};
    grid.valid_area = {1.0F};

    DinoFeatureGridCache cache(2U * sizeof(float));
    const auto key_a = irt::features::priv::dinoFeatureCacheKey("image-a", "extractor-a", DinoRect{0.0, 0.0, 1.0, 1.0});
    const auto key_b = irt::features::priv::dinoFeatureCacheKey("image-b", "extractor-a", DinoRect{0.0, 0.0, 1.0, 1.0});
    const auto key_c = irt::features::priv::dinoFeatureCacheKey("image-a", "extractor-b", DinoRect{0.0, 0.0, 1.0, 1.0});
    ASSERT_NE(key_a, key_b);
    ASSERT_NE(key_a, key_c);

    cache.insert(key_a, std::make_shared<DinoFeatureGrid>(grid));
    cache.insert(key_b, std::make_shared<DinoFeatureGrid>(grid));

    EXPECT_EQ(cache.bytes(), 2U * sizeof(float));
    EXPECT_EQ(cache.find(key_a), nullptr);
    EXPECT_NE(cache.find(key_b), nullptr);
    EXPECT_EQ(cache.hits(), 1U);
    EXPECT_EQ(cache.misses(), 1U);
}

TEST(DinoResourceTest, SearchCacheBundleReusesSameExtractorAndInvalidatesOnChange)
{
    const auto first = irt::features::priv::dinoAcquireSearchCaches("extractor-a", 64U, 64U);
    const auto second = irt::features::priv::dinoAcquireSearchCaches("extractor-a", 64U, 64U);
    ASSERT_EQ(first, second);

    DinoFeatureGrid grid;
    grid.tokens     = {1.0F};
    grid.valid_area = {1.0F};
    first->features.insert("candidate", std::make_shared<DinoFeatureGrid>(grid));
    EXPECT_NE(second->features.find("candidate"), nullptr);

    const auto changed_budget = irt::features::priv::dinoAcquireSearchCaches("extractor-a", 128U, 64U);
    EXPECT_NE(changed_budget, first);
    EXPECT_EQ(changed_budget->features.find("candidate"), nullptr);

    const auto changed = irt::features::priv::dinoAcquireSearchCaches("extractor-b", 64U, 64U);
    EXPECT_NE(changed, changed_budget);
    EXPECT_EQ(changed->features.find("candidate"), nullptr);
}


TEST(DinoResourceTest, ImageCacheAccountsForDecodedRowStride)
{
    auto make_image = []
    {
        auto image = std::make_shared<DinoCanonicalImage>();
        image->image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(7, 8, 9)).clone();
        return image;
    };

    DinoImageCache cache(12U);
    cache.insert("image-a", make_image());
    cache.insert("image-b", make_image());

    EXPECT_EQ(cache.bytes(), 12U);
    EXPECT_EQ(cache.find("image-a"), nullptr);
    EXPECT_NE(cache.find("image-b"), nullptr);
}

TEST(DinoRegionSearchScan, SpatialTopTwoMatchesAcrossAvailableBackends)
{
    using namespace irt::features::priv;
    // Three views exercise ordinary, singleton, and block-tail ranges.
    constexpr int dim = 96, query_count = 3;
    std::vector<int8_t> codes(10U * dim, 0);
    std::vector<float> factors(10U, 1.f);
    for (size_t p = 0; p < 10; ++p) codes[p * dim + p % query_count] = 1;
    std::vector<float> query(query_count * dim, 0.f);
    for (int q = 0; q < query_count; ++q) query[q * dim + q] = 1;
    const size_t offsets[]{0, 5, 6}, counts[]{5, 1, 4};
    DinoCompactBlock block{codes.data(), factors.data(), 10, dim};
    std::vector<retrieval::Pair> reference(3 * query_count), actual(reference.size());
    DinoSimilarityReducer cpu(DinoSimilarityBackend::Cpu, dim);
    cpu.matchViewGroup(block, offsets, counts, 3, query.data(), query_count, reference.data());
    EXPECT_EQ(reference[0].first.index, 0);
    EXPECT_EQ(reference[0].second.index, 3);
    EXPECT_EQ(reference[3].second.index, -1);
    if (!dinoCudaSimilarityAvailable()) GTEST_SKIP() << "CUDA unavailable; CPU pair checks passed";
    DinoSimilarityReducer gpu(DinoSimilarityBackend::Cuda, dim);
    gpu.matchViewGroup(block, offsets, counts, 3, query.data(), query_count, actual.data());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(reference[i].first.index, actual[i].first.index);
        EXPECT_EQ(reference[i].second.index, actual[i].second.index);
        EXPECT_NEAR(reference[i].first.score, actual[i].first.score, 1e-5f);
        if (reference[i].second.index >= 0) EXPECT_NEAR(reference[i].second.score, actual[i].second.score, 1e-5f);
    }
}

TEST(DinoRegionSearchContract, V4BudgetAndBatchFieldsParse)
{
    const auto config = irt::features::dinoConfigFromYaml(
        "model: {name: dinov3_vits16, weights_path: unused.wts}\n"
        "regions: {coarse_dimension: 192, local_representatives: 128}\n"
        "search: {verify_k: 32}\n");
    EXPECT_EQ(config.coarse_dimension, 192);
    EXPECT_EQ(config.local_representatives, 128);
    EXPECT_EQ(config.fine_verify_k, 32U);
    const auto requests = irt::features::dinoSearchRequestsFromYaml(
        "- request_id: square\n  query_path: a.png\n  bbox: [0, 0, 40, 40]\n"
        "- request_id: small\n  query_path: b.png\n  bbox: [5, 5, 15, 15]\n");
    ASSERT_EQ(requests.size(), 2U);
    EXPECT_EQ(requests[1].request_id, "small");
    irt::features::DinoSearchResponse response;
    response.score_kind = "tight_global_and_grid";
    response.verified_candidates = 3;
    auto yaml = irt::features::dinoSearchResponseToYaml(response);
    EXPECT_NE(yaml.find("localized_candidates"), std::string::npos);
    EXPECT_NE(yaml.find("verification_candidates"), std::string::npos);
    EXPECT_NE(yaml.find("tight_global_and_grid"), std::string::npos);
}
