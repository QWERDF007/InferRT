#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/RoIAlign.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace {

void expectNearVector(const std::vector<float> &actual, const std::vector<float> &expected, float atol = 1e-5f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index=" << i;
    }
}

} // namespace

TEST(RoIAlignTest, MatchesTorchvisionLegacySamplingRatioOne)
{
    const std::array<int64_t, 4> shape{1, 1, 4, 4};
    const std::vector<float>     input{
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<float> rois{0.0f, 0.0f, 0.0f, 3.0f, 3.0f};

    const irt::ops::RoIAlign roi_align(2, 2, 1.0f, 1, false);
    const auto               output = roi_align.forward(input.data(), shape.data(), rois.data(), 1);

    expectNearVector(output, {3.75f, 5.25f, 9.75f, 11.25f});
}

TEST(RoIAlignTest, MatchesTorchvisionAlignedSamplingRatioTwo)
{
    const std::array<int64_t, 4> shape{1, 1, 4, 4};
    const std::vector<float>     input{
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<float> rois{0.0f, 0.5f, 0.5f, 2.5f, 2.5f};

    const irt::ops::RoIAlign roi_align({2, 2}, 1.0f, 2, true);
    const auto               output = roi_align.forward(input.data(), shape.data(), rois.data(), 1);

    expectNearVector(output, {2.5f, 3.5f, 6.5f, 7.5f});
}

TEST(RoIAlignTest, SupportsAdaptiveSampling)
{
    const std::array<int64_t, 4> shape{1, 1, 4, 4};
    const std::vector<float>     input{
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<float> rois{0.0f, 0.0f, 0.0f, 3.0f, 3.0f};

    const irt::ops::RoIAlign roi_align(2, 2, 1.0f, -1, false);
    const auto               output = roi_align.forward(input.data(), shape.data(), rois.data(), 1);

    expectNearVector(output, {3.75f, 5.25f, 9.75f, 11.25f});
}

TEST(RoIAlignTest, SupportsMultipleBatchesAndChannels)
{
    const std::array<int64_t, 4> shape{2, 2, 2, 2};
    const std::vector<float>     input{
        1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f, 5.0f, 6.0f, 7.0f, 8.0f, 50.0f, 60.0f, 70.0f, 80.0f,
    };
    const std::vector<float> rois{
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    };

    const irt::ops::RoIAlign roi_align(1, 1, 1.0f, 1, false);
    const auto               output = roi_align.forward(input.data(), shape.data(), rois.data(), 2);

    expectNearVector(output, {2.5f, 25.0f, 6.5f, 65.0f});
}

TEST(RoIAlignTest, RejectsInvalidArguments)
{
    EXPECT_THROW((irt::ops::RoIAlign(0, 2, 1.0f, 1, false)), irt::Exception);

    const irt::ops::RoIAlign     roi_align(2, 2, 1.0f, 1, false);
    const std::array<int64_t, 4> shape{1, 1, 2, 2};
    const std::vector<float>     input{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float>     rois{1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float>           output(4);

    EXPECT_THROW(roi_align.forward(input.data(), shape.data(), rois.data(), 1, output.data()), irt::Exception);
    EXPECT_THROW((void)roi_align.forward(input.data(), shape.data(), rois.data(), -1), irt::Exception);
}
