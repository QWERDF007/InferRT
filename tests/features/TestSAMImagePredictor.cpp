/**
 * @file TestSAMImagePredictor.cpp
 * @brief SAMImagePredictor API and mask postprocessing tests.
 */

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/SAMImagePredictor.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace {

/**
 * @brief 断言调用抛出指定错误码的 InferRT 异常。
 * @tparam Fn 可调用对象类型。
 * @param fn 待执行调用。
 * @param expected_code 期望错误码。
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
 * @brief 构造不裁剪、不 padding 的直接 resize 后处理几何。
 * @param width 原图和模型输入宽度。
 * @param height 原图和模型输入高度。
 * @return SAM mask 后处理几何信息。
 */
irt::features::SAMMaskPostprocessGeometry makeDirectGeometry(int width, int height)
{
    irt::features::SAMMaskPostprocessGeometry geometry;
    geometry.original_width  = width;
    geometry.original_height = height;
    geometry.model_width     = width;
    geometry.model_height    = height;
    geometry.resized_width   = width;
    geometry.resized_height  = height;
    geometry.resize_mode     = irt::features::SAMImageResizeMode::StretchSquare;
    return geometry;
}

} // namespace

TEST(SAMImagePredictorTest, DefaultConstructsNotReadyPredictor)
{
    const irt::features::SAMImagePredictor predictor;

    EXPECT_EQ(predictor.config().model_name, irt::features::kDefaultSAMImagePredictorModelName);
    EXPECT_EQ(predictor.config().model_backend, irt::model::ModelBackend::TensorRT);
    EXPECT_EQ(predictor.config().model_device, irt::model::ModelDevice::GPU);
    EXPECT_EQ(predictor.config().resize_mode, irt::features::SAMImageResizeMode::Auto);
    EXPECT_TRUE(predictor.config().use_sam2_mask_postprocess);
    EXPECT_FALSE(predictor.isReady());
}

TEST(SAMImagePredictorTest, ConstructorStoresMaskPostprocessOverride)
{
    irt::features::SAMImagePredictorConfig config;
    config.use_sam2_mask_postprocess = false;

    const irt::features::SAMImagePredictor predictor(config);

    EXPECT_FALSE(predictor.config().use_sam2_mask_postprocess);
}

TEST(SAMImagePredictorTest, ConstructorRejectsTensorRtCpuDevice)
{
    irt::features::SAMImagePredictorConfig config;
    config.model_device = irt::model::ModelDevice::CPU;

    expectIrtExceptionCode([&] { irt::features::SAMImagePredictor predictor(config); },
                           irt::Status::ERROR_NOT_IMPLEMENTED);
}

TEST(SAMImagePredictorTest, PostprocessThresholdsMasksAndClampsReturnedLowResLogits)
{
    const std::vector<float> low_res_masks{-40.0F, -0.1F, 0.2F, 40.0F};
    const std::vector<float> iou_predictions{0.25F};
    const auto               geometry = makeDirectGeometry(2, 2);

    const auto prediction = irt::features::SAMImagePredictor::postprocessMasks(
        low_res_masks, 1, 2, 2, iou_predictions, geometry);

    EXPECT_EQ(prediction.mask_count, 1);
    EXPECT_EQ(prediction.width, 2);
    EXPECT_EQ(prediction.height, 2);
    EXPECT_FALSE(prediction.masks_are_logits);
    EXPECT_EQ(prediction.masks, (std::vector<float>{0.0F, 0.0F, 1.0F, 1.0F}));
    EXPECT_EQ(prediction.binary_masks, (std::vector<std::uint8_t>{0U, 0U, 1U, 1U}));
    EXPECT_EQ(prediction.low_res_masks, (std::vector<float>{-32.0F, -0.1F, 0.2F, 32.0F}));
    EXPECT_EQ(prediction.iou_predictions, iou_predictions);
}

TEST(SAMImagePredictorTest, PostprocessReturnLogitsKeepsHighResValuesUnclamped)
{
    const std::vector<float> low_res_masks{40.0F};
    const auto               geometry = makeDirectGeometry(3, 2);

    irt::features::SAMImagePredictOptions options;
    options.return_logits = true;

    const auto prediction
        = irt::features::SAMImagePredictor::postprocessMasks(low_res_masks, 1, 1, 1, {}, geometry, options);

    EXPECT_TRUE(prediction.masks_are_logits);
    EXPECT_TRUE(prediction.binary_masks.empty());
    ASSERT_EQ(prediction.masks.size(), 6U);
    EXPECT_TRUE(std::all_of(prediction.masks.begin(), prediction.masks.end(),
                            [](float value) { return value == 40.0F; }));
    EXPECT_EQ(prediction.low_res_masks, (std::vector<float>{32.0F}));
}

TEST(SAMImagePredictorTest, PostprocessFillsSmallHolesInLowResLogitSpace)
{
    const std::vector<float> low_res_masks{
        1.0F, 1.0F, 1.0F,
        1.0F, -0.5F, 1.0F,
        1.0F, 1.0F, 1.0F,
    };
    const auto geometry = makeDirectGeometry(3, 3);

    irt::features::SAMImagePredictOptions options;
    options.return_logits  = true;
    options.max_hole_area  = 1;
    options.mask_threshold = 0.0F;

    const auto prediction
        = irt::features::SAMImagePredictor::postprocessMasks(low_res_masks, 1, 3, 3, {}, geometry, options);

    ASSERT_EQ(prediction.masks.size(), 9U);
    EXPECT_FLOAT_EQ(prediction.masks[4], 10.0F);
    EXPECT_FLOAT_EQ(prediction.low_res_masks[4], -0.5F);
}

TEST(SAMImagePredictorTest, PostprocessRemovesSmallSprinklesInLowResLogitSpace)
{
    const std::vector<float> low_res_masks{
        -1.0F, -1.0F, -1.0F,
        -1.0F, 0.5F, -1.0F,
        -1.0F, -1.0F, -1.0F,
    };
    const auto geometry = makeDirectGeometry(3, 3);

    irt::features::SAMImagePredictOptions options;
    options.return_logits      = true;
    options.max_sprinkle_area  = 1;
    options.mask_threshold     = 0.0F;

    const auto prediction
        = irt::features::SAMImagePredictor::postprocessMasks(low_res_masks, 1, 3, 3, {}, geometry, options);

    ASSERT_EQ(prediction.masks.size(), 9U);
    EXPECT_FLOAT_EQ(prediction.masks[4], -10.0F);
    EXPECT_FLOAT_EQ(prediction.low_res_masks[4], 0.5F);
}

TEST(SAMImagePredictorTest, PostprocessRejectsInvalidMaskSize)
{
    const std::vector<float> low_res_masks{0.0F, 1.0F, 2.0F};
    const auto               geometry = makeDirectGeometry(2, 2);

    expectIrtExceptionCode(
        [&] { (void)irt::features::SAMImagePredictor::postprocessMasks(low_res_masks, 1, 2, 2, {}, geometry); },
        irt::Status::ERROR_INVALID_ARGUMENT);
}
