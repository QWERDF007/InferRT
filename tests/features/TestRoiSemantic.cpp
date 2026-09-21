#include <gtest/gtest.h>

#include "RoiEmbeddingCore.hpp"
#include <inferrt/features/RoiFeatureEncoder.hpp>
#include <inferrt/features/RoiSearch.hpp>
#include <inferrt/features/RoiCluster.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

using namespace irt::features;
using namespace irt::features::embedding;

double dotProduct(const std::vector<float> &a, const std::vector<float> &b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

} // namespace

// ============================================================================
// Core Embedding & Geometry Unit Tests (Dependency-free)
// ============================================================================

TEST(RoiSemanticCoreTest, RoundingAndLetterboxMapping)
{
    auto v = makeView({10, 20, 111, 70}, 200, 100, 512, 512, 0);
    EXPECT_NEAR(v.map({10, 20}).x, v.left, 1e-5);
    EXPECT_NEAR(v.map({111, 70}).y, v.top + v.resized_height, 1e-5);
    EXPECT_NEAR(v.sx, 512.0 / 101.0, 1e-5);
    EXPECT_GT(v.top, 0);
}

TEST(RoiSemanticCoreTest, RectangleFractionalArea)
{
    Shape s{{0.25, 0.5, 8.75, 9.25}, {}};
    auto  v = makeView(s.box, 10, 10, 16, 16, 0);
    auto  m = rasterMask(s, v);
    EXPECT_NEAR(std::accumulate(m.begin(), m.end(), 0.0), 8.5 * 8.75 * v.sx * v.sy, 1e-4);
}

TEST(RoiSemanticCoreTest, PaddingExcluded)
{
    Shape s{{0, 0, 100, 20}, {}};
    auto  v = makeView(s.box, 100, 20, 64, 64, 0);
    auto  m = rasterMask(s, v);
    for (int y = 0; y < v.top; ++y)
    {
        for (int x = 0; x < 64; ++x)
        {
            EXPECT_NEAR(m[static_cast<size_t>(y) * 64 + static_cast<size_t>(x)], 0.0f, 1e-5);
        }
    }
}

TEST(RoiSemanticCoreTest, ConcavePolygonNotBbox)
{
    Shape s{{}, {{0, 0}, {8, 0}, {8, 2}, {2, 2}, {2, 8}, {0, 8}}};
    auto  v = makeView(bounds(s), 8, 8, 64, 64, 0);
    auto  m = rasterMask(s, v);
    EXPECT_NEAR(std::accumulate(m.begin(), m.end(), 0.0), 28.0 * 64.0, 1e-5);
    EXPECT_NEAR(m[48 * 64 + 48], 0.0f, 1e-5);
}

TEST(RoiSemanticCoreTest, PolygonRectangleEquivalence)
{
    Shape a{{1, 2, 9, 10}, {}};
    Shape b{{}, {{1, 2}, {9, 2}, {9, 10}, {1, 10}}};
    auto  v = makeView(bounds(a), 12, 12, 32, 32, 0);
    auto  x = rasterMask(a, v);
    auto  y = rasterMask(b, v);
    ASSERT_EQ(x.size(), y.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x[i], y[i], 1e-5);
    }
}

TEST(RoiSemanticCoreTest, PolygonVertexOrder)
{
    Shape a{{}, {{0, 0}, {8, 0}, {8, 8}, {0, 8}}};
    auto  b = a;
    std::reverse(b.polygon.begin(), b.polygon.end());
    auto v = makeView(bounds(a), 8, 8, 32, 32, 0);
    EXPECT_EQ(rasterMask(a, v), rasterMask(b, v));
}

TEST(RoiSemanticCoreTest, ClippedRoiAndOutside)
{
    auto v = makeView({-20, -10, 10, 20}, 100, 100, 32, 32, 0);
    EXPECT_EQ(v.x, 0);
    EXPECT_EQ(v.y, 0);
    EXPECT_EQ(v.width, 10);
    EXPECT_EQ(v.height, 20);
    EXPECT_THROW(makeView({101, 101, 120, 120}, 100, 100, 32, 32, 0.5), std::invalid_argument);
}

TEST(RoiSemanticCoreTest, InvalidPolygonAndEmptyBox)
{
    EXPECT_THROW(bounds({{0, 0, 0, 2}, {}}), std::invalid_argument);
    EXPECT_THROW(bounds({{}, {{0, 0}, {1, 1}, {2, 2}}}), std::invalid_argument);
    EXPECT_THROW(bounds({{}, {{0, 0}, {1, 1}, {NAN, 2}}}), std::invalid_argument);
}

TEST(RoiSemanticCoreTest, ForegroundTokensOnly)
{
    float t[] = {1, 0, 0, 1, 0, 1, 0, 1};
    auto  d   = pool(t, 2, 2, 2, {1, 0, 0, 0});
    EXPECT_NEAR(d[0], 1.0f, 1e-5);
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    auto m = pool(t, 2, 2, 2, {1, 1, 1, 1});
    EXPECT_LT(dotProduct(d, m), 0.5);
}

TEST(RoiSemanticCoreTest, ZeroMaskAndZeroFeatureRejected)
{
    float t[] = {1, 0};
    EXPECT_THROW(pool(t, 1, 1, 2, {0}), std::invalid_argument);
    float z[] = {0, 0};
    EXPECT_THROW(pool(z, 1, 1, 2, {1}), std::invalid_argument);
}

TEST(RoiSemanticCoreTest, MaskedNanDoesNotLeak)
{
    float t[] = {1, 0, NAN, NAN};
    auto  d   = pool(t, 1, 2, 2, {1, 0});
    EXPECT_NEAR(d[0], 1.0f, 1e-5);
    EXPECT_THROW(pool(t, 1, 2, 2, {1, 1}), std::invalid_argument);
}

TEST(RoiSemanticCoreTest, DifferentTokenCountsSameDimensionAndMean)
{
    float a[] = {1, 2, 1, 2};
    float b[] = {1, 2, 1, 2, 1, 2, 1, 2};
    auto  x   = pool(a, 1, 2, 2, {1, 1});
    auto  y   = pool(b, 2, 2, 2, {1, 1, 1, 1});
    EXPECT_EQ(x.size(), 2U);
    EXPECT_EQ(y.size(), 2U);
    EXPECT_NEAR(dotProduct(x, y), 1.0, 1e-5);
}

TEST(RoiSemanticCoreTest, TokenNormDoesNotDominate)
{
    float t[] = {100, 0, 0, 1};
    auto  d   = pool(t, 1, 2, 2, {1, 1});
    EXPECT_NEAR(d[0], d[1], 1e-5);
}

TEST(RoiSemanticCoreTest, SpatialDescriptorFixedAndUnit)
{
    float t[] = {1, 0, 0, 1, 1, 0, 0, 1};
    auto  d   = pool(t, 2, 2, 2, {1, 0, 0, 0}, 0.2);
    EXPECT_EQ(d.size(), 10U);
    EXPECT_NEAR(dotProduct(d, d), 1.0, 1e-5);
}

TEST(RoiSemanticCoreTest, ViewBudgetAndCompletePartitions)
{
    Shape s{{0, 0, 1000, 10}, {}};
    auto  v = planViews(s, 1000, 10, 512, 512, 16, 0, 3);
    EXPECT_EQ(v.size(), 4U);
    EXPECT_EQ(v[1].x, 0);
    EXPECT_EQ(v.back().x + v.back().width, 1000);
    for (size_t i = 2; i < v.size(); ++i)
    {
        EXPECT_LE(v[i].x, v[i - 1].x + v[i - 1].width);
    }
    EXPECT_EQ(planViews(s, 1000, 10, 512, 512, 16, 0, 0).size(), 1U);
}

TEST(RoiSemanticCoreTest, NormalRoiDoesNotAddViews)
{
    Shape s{{0, 0, 200, 200}, {}};
    EXPECT_EQ(planViews(s, 200, 200, 512, 512, 16, 0, 3).size(), 1U);
}

TEST(RoiSemanticCoreTest, ViewFusionScaleAndOrder)
{
    std::vector<float> a{1, 0}, b{0, 1};
    auto               x = fuse({a, b, a}, {1, 2, 3});
    auto               y = fuse({a, a, b}, {1, 3, 2});
    EXPECT_NEAR(dotProduct(x, y), 1.0, 1e-5);
    EXPECT_NEAR(dotProduct(x, x), 1.0, 1e-5);
}

TEST(RoiSemanticCoreTest, EuclideanAndCosineNeighborOrder)
{
    std::vector<float> a{1, 0}, b{0.8f, 0.6f}, c{0, 1};
    auto dist = [](const auto &x, const auto &y) {
        double s = 0;
        for (size_t i = 0; i < x.size(); ++i) s += (x[i] - y[i]) * (x[i] - y[i]);
        return s;
    };
    EXPECT_NEAR(dist(a, b), 2.0 - 2.0 * dotProduct(a, b), 1e-5);
    EXPECT_GT(dotProduct(a, b), dotProduct(a, c));
    EXPECT_LT(dist(a, b), dist(a, c));
}

TEST(RoiSemanticCoreTest, DescriptorIndependentOfUnrelatedSamples)
{
    float a[] = {1, 2, 3, 4}, b[] = {-1, 4, 9, 2};
    auto  x   = pool(a, 1, 2, 2, {1, 1});
    pool(b, 1, 2, 2, {1, 1});
    auto y = pool(a, 1, 2, 2, {1, 1});
    EXPECT_EQ(x, y);
}

TEST(RoiSemanticCoreTest, StreamedRowsKeepOriginalOrder)
{
    std::vector<float> dst(6, 0);
    scatterRows({2, 0}, {5, 6, 1, 2}, 2, dst.data(), 3);
    scatterRows({1}, {3, 4}, 2, dst.data(), 3);
    EXPECT_EQ(dst, (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(RoiSemanticCoreTest, StreamedRowsRejectInvalidDestination)
{
    float dst[4]{};
    EXPECT_THROW(scatterRows({2}, {1, 2}, 2, dst, 2), std::invalid_argument);
    EXPECT_THROW(scatterRows({0}, {1}, 2, dst, 2), std::invalid_argument);
}

// ============================================================================
// End-to-End Integration Tests (Skipped if no model weights)
// ============================================================================

TEST(RoiSemanticTest, RealBackbonePolygonBatchOrderAndExactSearch)
{
    const char *weights = std::getenv("ROI_DINO_WEIGHTS");
    if (!weights)
    {
        GTEST_SKIP() << "Set ROI_DINO_WEIGHTS for a real DINOv3 S16 512px engine/weights";
    }
    const auto root = std::filesystem::temp_directory_path() / "inferrt_roi_semantic_test";
    std::filesystem::create_directories(root);
    cv::Mat image(160, 240, CV_8UC3, cv::Scalar(45, 80, 120));
    cv::circle(image, {80, 70}, 25, {180, 200, 220}, -1);
    cv::imwrite((root / "test.png").string(), image);

    const RoiFeatureItem rectangle{1, root / "test.png", {40, 30, 120, 110}};
    RoiFeatureItem       polygon = rectangle;
    polygon.roi_id               = 2;
    polygon.roi                  = {};
    polygon.polygon              = {{40, 30}, {120, 30}, {120, 110}, {40, 110}};
    RoiFeatureItem other{3, root / "test.png", {140, 10, 225, 145}};

    {
        RoiFeatureConfig fc;
        fc.model_batch_size = 4;
        RoiFeatureEncoder encoder(fc, weights);
        const auto one   = encoder.extract({polygon});
        const auto batch = encoder.extract({other, rectangle, polygon});

        ASSERT_EQ(one.size(), static_cast<size_t>(encoder.featureDim()));
        ASSERT_EQ(one.size(), 384U);
        ASSERT_EQ(batch.size(), one.size() * 3);

        for (size_t c = 0; c < one.size(); ++c)
        {
            EXPECT_NEAR(one[c], batch[one.size() + c], 3e-4);     // same rectangle represented as polygon
            EXPECT_NEAR(one[c], batch[2 * one.size() + c], 3e-4); // unrelated ROI/order/batch must not change basis
        }
    }

    RoiSearchConfig sc;
    sc.model_batch_size = 4;
    RoiSearch search(sc);
    search.build(weights, {polygon, other}, root / "semantic.faiss");
    const auto hits = search.search(polygon, 2);
    ASSERT_EQ(hits.size(), 2U);
    EXPECT_EQ(hits[0].roi_id, 2);
    EXPECT_NEAR(hits[0].score, 1.0f, 1e-3);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST(RoiSemanticTest, SharedLibraryCrossImageBatchAndNoRepeatedInference)
{
    const char *weights = std::getenv("ROI_DINO_WEIGHTS");
    if (!weights)
    {
        GTEST_SKIP() << "Requires real DINOv3 S16/512, dynamic batch 1..4";
    }
    const auto root = std::filesystem::temp_directory_path() / "inferrt_roi_shared_test";
    std::filesystem::create_directories(root);
    std::vector<RoiSearchItem> items;
    for (int i = 0; i < 4; ++i)
    {
        cv::Mat image(160, 200, CV_8UC3, cv::Scalar(30 + 20 * i, 70, 120));
        cv::circle(image, {70, 70}, 15 + 5 * i, {200, 200, 200}, -1);
        auto path = root / (std::to_string(i) + ".png");
        cv::imwrite(path.string(), image);
        items.push_back({100 + i, path, {20, 20, 140, 140}});
    }

    RoiSearchConfig config;
    config.model_batch_size = 4;
    RoiSearch library(config);
    library.build(weights, items, root / "shared.faiss");
    auto built = library.featureWorkStats();
    ASSERT_TRUE(built.available);
    EXPECT_EQ(built.decoded_images, 4U);
    EXPECT_EQ(built.encoded_views, 4U);
    EXPECT_EQ(built.forward_batches, 1U);

    library.searchByRoiId(100, 2);
    library.repeatSearch(3);

    RoiClusterConfig cc;
    cc.hdbscan.min_cluster_size     = 2;
    cc.hdbscan.min_samples          = 1;
    cc.hdbscan.allow_single_cluster = true;
    RoiCluster cluster(cc);
    auto       result = cluster.cluster(weights, items, [](const RoiClusterProgress &p) {
        EXPECT_NE(p.stage, RoiClusterStage::Unknown);
    });
    EXPECT_EQ(result.assignments.size(), items.size());

    library.search(items[0], 2); // explicit new external-query path encodes only this ROI
    EXPECT_EQ(library.featureWorkStats().encoded_views, built.encoded_views + 1);
    EXPECT_EQ(library.featureWorkStats().decoded_images, built.decoded_images + 1);
    EXPECT_EQ(library.featureWorkStats().forward_batches, built.forward_batches + 1);

    // Re-open in the same object with an unusable weights path. Cached-vector operations
    // must still succeed; a hidden attempt to construct a model would fail this test.
    library.load(root / "deliberately_missing.wts", root / "shared.faiss");
    EXPECT_FALSE(std::filesystem::exists(root / "deliberately_missing.wts"));
    EXPECT_EQ(library.searchByRoiId(100, 2).size(), 2U);
    EXPECT_EQ(library.repeatSearch(3).size(), 3U);
    EXPECT_EQ(library.featureWorkStats().forward_batches, 0U);
    EXPECT_EQ(library.featureWorkStats().decoded_images, 0U);

    std::filesystem::remove_all(root);
}
