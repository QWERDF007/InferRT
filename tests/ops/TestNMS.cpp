#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/NMS.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

void expectEqualVector(const std::vector<int64_t> &actual, const std::vector<int64_t> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(actual[i], expected[i]) << "index=" << i;
    }
}

} // namespace

TEST(NMSTest, SuppressesLowerScoringOverlaps)
{
    const std::vector<float> boxes{
        0.0f, 0.0f, 10.0f, 10.0f,
        1.0f, 1.0f, 11.0f, 11.0f,
        20.0f, 20.0f, 30.0f, 30.0f,
    };
    const std::vector<float> scores{0.9f, 0.8f, 0.7f};

    const auto keep = irt::ops::nms(boxes.data(), scores.data(), 3, 0.5f);

    expectEqualVector(keep, {0, 2});
}

TEST(NMSTest, ReturnsIndicesSortedByScore)
{
    const std::vector<float> boxes{
        0.0f, 0.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 5.0f, 5.0f,
    };
    const std::vector<float> scores{0.1f, 0.9f, 0.2f};

    const auto keep = irt::ops::nms(boxes.data(), scores.data(), 3, 0.0f);

    expectEqualVector(keep, {1, 2, 0});
}

TEST(NMSTest, KeepsBoxesWithZeroIoUAtZeroThreshold)
{
    const std::vector<float> boxes{
        0.0f, 0.0f, 10.0f, 10.0f,
        10.0f, 0.0f, 20.0f, 10.0f,
    };
    const std::vector<float> scores{0.9f, 0.8f};

    const auto keep = irt::ops::nms(boxes.data(), scores.data(), 2, 0.0f);

    expectEqualVector(keep, {0, 1});
}

TEST(NMSTest, SupportsEmptyInput)
{
    const auto keep = irt::ops::nms(nullptr, nullptr, 0, 0.5f);

    EXPECT_TRUE(keep.empty());
}

TEST(NMSTest, RejectsInvalidArguments)
{
    const std::vector<float> boxes{0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> scores{0.5f};

    EXPECT_THROW((void)irt::ops::nms(nullptr, scores.data(), 1, 0.5f), irt::Exception);
    EXPECT_THROW((void)irt::ops::nms(boxes.data(), nullptr, 1, 0.5f), irt::Exception);
    EXPECT_THROW((void)irt::ops::nms(boxes.data(), scores.data(), -1, 0.5f), irt::Exception);
    EXPECT_THROW((void)irt::ops::nms(boxes.data(), scores.data(), 1, std::numeric_limits<float>::quiet_NaN()),
                 irt::Exception);
}
