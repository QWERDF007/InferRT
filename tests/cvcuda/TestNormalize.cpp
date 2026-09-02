/**
 * @file TestNormalize.cpp
 * @brief Normalize 算子的单元测试。
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpNormalize.h>
#include <inferrt/cvcuda/OpNormalize.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

template<size_t ParameterCount>
std::vector<float> makeReference(const std::vector<uint8_t> &source, int width, int height, int channels,
                                 const std::array<float, ParameterCount> &mean,
                                 const std::array<float, ParameterCount> &stddev)
{
    const int pixels = width * height;
    std::vector<float> output(static_cast<size_t>(pixels) * channels);
    for (int pixel = 0; pixel < pixels; ++pixel)
    {
        for (int channel = 0; channel < channels; ++channel)
        {
            output[static_cast<size_t>(channel) * pixels + pixel]
                = (static_cast<float>(source[static_cast<size_t>(pixel) * channels + channel]) / 255.0F
                   - mean[static_cast<size_t>(channel)])
                / stddev[static_cast<size_t>(channel)];
        }
    }
    return output;
}

template<typename Caller, size_t ParameterCount>
void runNormalizeTest(const int width, const int height, const int channels,
                      const std::array<float, ParameterCount> &mean,
                      const std::array<float, ParameterCount> &stddev, Caller caller)
{
    const size_t source_elements = static_cast<size_t>(width) * height * channels;
    std::vector<uint8_t> source(source_elements);
    for (size_t index = 0; index < source.size(); ++index)
    {
        source[index] = static_cast<uint8_t>((index * 47 + 19) % 256);
    }
    const auto expected = makeReference(source, width, height, channels, mean, stddev);

    uint8_t *d_source = nullptr;
    float   *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_source, source.size() * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, expected.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_source, source.data(), source.size() * sizeof(uint8_t), cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_TRUE(AssertInferRTSuccess(
        caller(d_source, d_output, cv::Size(width, height), channels, mean.data(), stddev.data(), nullptr)));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> output(expected.size());
    ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_NEAR(output[index], expected[index], 1e-6F) << "at index " << index;
    }

    cudaFree(d_source);
    cudaFree(d_output);
}

} // namespace

TEST(NormalizeFunctionTest, ConvertsRgbHwcToNchwAndNormalizes)
{
    const std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    const std::array<float, 3> stddev{0.229F, 0.224F, 0.225F};
    runNormalizeTest(13, 7, 3, mean, stddev,
                     [](const uint8_t *source, float *output, cv::Size size, int channels, const float *mean_values,
                        const float *stddev_values, cudaStream_t stream)
                     { return irt::cvcuda::normalize(source, output, size, channels, mean_values, stddev_values, stream); });
}

TEST(NormalizeClassTest, SupportsSingleChannelImages)
{
    irt::cvcuda::Normalize op;
    const std::array<float, 3> mean{0.5F, 0.0F, 0.0F};
    const std::array<float, 3> stddev{0.25F, 1.0F, 1.0F};
    runNormalizeTest(11, 5, 1, mean, stddev,
                     [&op](const uint8_t *source, float *output, cv::Size size, int channels, const float *mean_values,
                           const float *stddev_values, cudaStream_t stream)
                     { return op(source, output, size, channels, mean_values, stddev_values, stream); });
}

TEST(NormalizeFunctionTest, ConvertsBgraHwcToNchwAndNormalizes)
{
    const std::array<float, 4> mean{0.1F, 0.2F, 0.3F, 0.4F};
    const std::array<float, 4> stddev{0.5F, 0.6F, 0.7F, 0.8F};
    runNormalizeTest(13, 7, 4, mean, stddev,
                     [](const uint8_t *source, float *output, cv::Size size, int channels, const float *mean_values,
                        const float *stddev_values, cudaStream_t stream)
                     {
                         return irt::cvcuda::normalize(source, output, size, channels, mean_values, stddev_values,
                                                        stream);
                     });
}

TEST(NormalizeFunctionTest, HonorsExplicitPreprocessScale)
{
    const std::vector<uint8_t> source{10U, 20U, 30U, 40U, 50U, 60U};
    const std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    const std::array<float, 3> stddev{1.0F, 1.0F, 1.0F};
    const std::vector<float> expected{5.0F, 20.0F, 10.0F, 25.0F, 15.0F, 30.0F};

    uint8_t *d_source = nullptr;
    float   *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_source, source.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, expected.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_TRUE(AssertInferRTSuccess(irt::cvcuda::normalize(d_source, d_output, cv::Size(2, 1), 3, mean.data(),
                                                             stddev.data(), 0.5F, nullptr)));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> output(expected.size());
    ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_FLOAT_EQ(output[index], expected[index]) << "at index " << index;
    }

    cudaFree(d_source);
    cudaFree(d_output);
}

TEST(NormalizeFunctionEdgeCaseTest, RejectsZeroStddev)
{
    uint8_t *d_source = nullptr;
    float   *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_source, 3), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, 3 * sizeof(float)), cudaSuccess);

    const std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    const std::array<float, 3> stddev{1.0F, 0.0F, 1.0F};
    EXPECT_EQ(irt::cvcuda::normalize(d_source, d_output, cv::Size(1, 1), 3, mean.data(), stddev.data()),
              IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_source);
    cudaFree(d_output);
}
