#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/ModelContract.hpp>
#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/core/Tensor.hpp>

#include <vector>
#include <span>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace irt;

TEST(TensorTest, DataTypePropertiesAndSizes)
{
    EXPECT_EQ(dataTypeSize(TensorDataType::U8), 1U);
    EXPECT_EQ(dataTypeSize(TensorDataType::I8), 1U);
    EXPECT_EQ(dataTypeSize(TensorDataType::Bool), 1U);
    EXPECT_EQ(dataTypeSize(TensorDataType::F16), 2U);
    EXPECT_EQ(dataTypeSize(TensorDataType::F32), 4U);
    EXPECT_EQ(dataTypeSize(TensorDataType::I32), 4U);
    EXPECT_EQ(dataTypeSize(TensorDataType::I64), 8U);

    EXPECT_EQ(dataTypeToString(TensorDataType::U8), "uint8");
    EXPECT_EQ(dataTypeToString(TensorDataType::I8), "int8");
    EXPECT_EQ(dataTypeToString(TensorDataType::F32), "float32");
    EXPECT_EQ(dataTypeToString(TensorDataType::I32), "int32");
    EXPECT_EQ(dataTypeToString(TensorDataType::I64), "int64");
    EXPECT_EQ(dataTypeToString(TensorDataType::Bool), "bool");
}

TEST(TensorTest, ShapeElementCountValid)
{
    Shape shape{1, 3, 224, 224};
    EXPECT_EQ(shape.rank(), 4U);
    EXPECT_FALSE(shape.empty());
    EXPECT_FALSE(shape.isDynamic());
    EXPECT_EQ(shape.elementCount(), 1U * 3U * 224U * 224U);

    Shape empty_shape{};
    EXPECT_THROW((void)empty_shape.elementCount(), irt::Exception);
}

TEST(TensorTest, ShapeRejectsNegativeOrZeroDimensions)
{
    Shape dynamic_shape{-1, 3, 224, 224};
    EXPECT_TRUE(dynamic_shape.isDynamic());
    EXPECT_THROW((void)dynamic_shape.elementCount(), irt::Exception);

    Shape zero_shape{1, 0, 224, 224};
    EXPECT_THROW((void)zero_shape.elementCount(), irt::Exception);
}

TEST(TensorTest, ShapeDetectsMultiplicationOverflow)
{
    Shape huge_shape{static_cast<int64_t>(1ULL << 40), static_cast<int64_t>(1ULL << 30)};
    EXPECT_THROW((void)huge_shape.elementCount(), irt::Exception);
}

TEST(TensorTest, RuntimeShapeValidationAllowsDeclaredDynamicDimensions)
{
    const Shape declared{-1, 3, 8, 10};

    EXPECT_NO_THROW(validateRuntimeShape(declared, Shape{1, 3, 8, 10}, "input"));
    EXPECT_NO_THROW(validateRuntimeShape(declared, Shape{4, 3, 8, 10}, "input"));
    EXPECT_THROW(validateRuntimeShape(declared, Shape{1, 3, 9, 10}, "input"), irt::Exception);
    EXPECT_THROW(validateRuntimeShape(declared, Shape{1, 3, 8}, "input"), irt::Exception);
    EXPECT_THROW(validateRuntimeShape(declared, Shape{0, 3, 8, 10}, "input"), irt::Exception);
}

TEST(TensorTest, CheckedSizeHelpersRejectOverflowAndNarrowing)
{
    EXPECT_EQ(checkedSizeAdd(4, 5, "test"), 9U);
    EXPECT_EQ(checkedSizeMul(4, 5, "test"), 20U);
    EXPECT_EQ(checkedSizeToInt(42, "test"), 42);
    EXPECT_EQ(checkedSizeToInt64(42, "test"), 42);
    EXPECT_EQ(checkedSizeToStreamoff(42, "test"), 42);
    EXPECT_EQ(checkedSizeToStreamsize(42, "test"), 42);
    EXPECT_EQ(checkedInt64ToSize(42, "test"), 42U);
    EXPECT_THROW((void)checkedInt64ToSize(-1, "test"), irt::Exception);
    if constexpr (std::numeric_limits<size_t>::digits < std::numeric_limits<int64_t>::digits)
    {
        EXPECT_THROW((void)checkedInt64ToSize(std::numeric_limits<int64_t>::max(), "test"), irt::Exception);
    }

    EXPECT_THROW(checkedSizeAdd(std::numeric_limits<size_t>::max(), 1, "test"), irt::Exception);
    EXPECT_THROW(checkedSizeMul(std::numeric_limits<size_t>::max(), 2, "test"), irt::Exception);
    EXPECT_THROW(checkedSizeToInt(static_cast<size_t>(std::numeric_limits<int>::max()) + 1, "test"), irt::Exception);
    EXPECT_THROW(checkedSizeToInt64(static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1, "test"), irt::Exception);
}

TEST(TensorTest, TensorDescBytesPerRequest)
{
    TensorDesc desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3};
    EXPECT_EQ(desc.bytesPerRequest(), 224U * 224U * 3U * sizeof(float));

    TensorDesc invalid_desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 0, 224, 3};
    EXPECT_THROW((void)invalid_desc.bytesPerRequest(), irt::Exception);
    EXPECT_THROW(invalid_desc.validate("invalid_tensor"), irt::Exception);
}

TEST(TensorTest, TensorViewBoundsCheck)
{
    std::vector<uint8_t> buffer(224 * 224 * 3 * sizeof(float) * 4);
    TensorDesc desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3};
    TensorView view{buffer.data(), desc, desc.bytesPerRequest(), 4};

    EXPECT_NE(view.dataForRequest(0), nullptr);
    EXPECT_NE(view.dataForRequest(3), nullptr);
    EXPECT_EQ(view.dataForRequest(4), nullptr);
    EXPECT_EQ(view.dataForRequest(-1), nullptr);

    const TensorView const_view = view;
    EXPECT_NE(const_view.dataForRequest(0), nullptr);
    EXPECT_EQ(const_view.dataForRequest(4), nullptr);
    EXPECT_EQ(const_view.dataForRequest(-1), nullptr);

    TensorView null_view{nullptr, desc, desc.bytesPerRequest(), 4};
    EXPECT_EQ(null_view.dataForRequest(0), nullptr);
}

TEST(TensorTest, BufferViewHonorsBackingCapacity)
{
    std::vector<uint8_t> buffer(8);
    TensorDesc desc{TensorDataType::U8, TensorLayout::Opaque, MemoryKind::HOST, Shape{4}};
    BufferView view{buffer.data(), desc, 4, 2, buffer.size()};

    EXPECT_NE(view.dataForRequest(0), nullptr);
    EXPECT_NE(view.dataForRequest(1), nullptr);
    EXPECT_EQ(view.dataForRequest(2), nullptr);
    EXPECT_EQ(view.byteSize(), buffer.size());

    view.capacity_bytes = 7;
    EXPECT_EQ(view.dataForRequest(1), nullptr);
    EXPECT_THROW((void)view.byteSize(), irt::Exception);
}

TEST(TensorTest, BufferViewRejectsInvalidBatchCapacity)
{
    std::vector<uint8_t> buffer(4);
    TensorDesc           desc{TensorDataType::U8, TensorLayout::Opaque, MemoryKind::HOST, Shape{4}};

    BufferView no_batch{buffer.data(), desc, 4, 0, buffer.size()};
    EXPECT_THROW((void)no_batch.byteSize(), irt::Exception);

    BufferView negative_batch{buffer.data(), desc, 4, -1, buffer.size()};
    EXPECT_THROW((void)negative_batch.byteSize(), irt::Exception);
}

TEST(TensorTest, EmptyBufferViewRemainsAnEmptyValue)
{
    BufferView empty;
    EXPECT_EQ(empty.byteSize(), 0U);
}

TEST(TensorTest, GenericShapeTensorDescAndBufferView)
{
    Shape shape{2, 16, 512};
    TensorDesc desc{TensorDataType::F32, TensorLayout::Opaque, MemoryKind::HOST, shape};
    EXPECT_EQ(desc.elementCount(), 2U * 16U * 512U);
    EXPECT_EQ(desc.byteSize(), 2U * 16U * 512U * sizeof(float));

    std::vector<float> data(desc.elementCount());
    BufferView buf{data.data(), desc, desc.byteSize(), 1};
    EXPECT_EQ(buf.byteSize(), desc.byteSize());
    EXPECT_NE(buf.dataForRequest(0), nullptr);
    EXPECT_EQ(buf.dataForRequest(1), nullptr);
}

TEST(TensorTest, TensorDescUsesShapeAsTheSingleGeometryRepresentation)
{
    TensorDesc desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, Shape{3, 8, 8}};
    EXPECT_EQ(desc.channels(), 3);
    EXPECT_EQ(desc.height(), 8);
    EXPECT_EQ(desc.width(), 8);
    EXPECT_NO_THROW(desc.validate("shape-only"));
}

TEST(TensorTest, PreprocessSpecValidation)
{
    PreprocessSpec spec;
    spec.input_width = 224;
    spec.input_height = 224;
    spec.input_channels = 3;
    spec.mean = {0.485f, 0.456f, 0.406f};
    spec.stddev = {0.229f, 0.224f, 0.225f};
    EXPECT_NO_THROW(spec.validate());

    // 零 stddev 拒绝
    spec.stddev = {0.229f, 0.0f, 0.225f};
    EXPECT_THROW(spec.validate(), irt::Exception);

    // 通道数不匹配拒绝
    spec.stddev = {0.229f, 0.224f};
    EXPECT_THROW(spec.validate(), irt::Exception);

    // 非法尺寸拒绝
    spec.stddev = {0.229f, 0.224f, 0.225f};
    spec.input_width = 0;
    EXPECT_THROW(spec.validate(), irt::Exception);
}

TEST(PreprocessGeometryTest, UsesOneRuleForLetterboxAndCenterCrop)
{
    irt::PreprocessSpec spec;
    spec.input_width      = 10;
    spec.input_height     = 10;
    spec.input_channels   = 3;
    spec.source_channels  = 3;
    spec.mean             = {0.0F, 0.0F, 0.0F};
    spec.stddev           = {1.0F, 1.0F, 1.0F};
    spec.padding_mode     = irt::PaddingMode::Letterbox;
    spec.source_width     = 7;
    spec.source_height    = 3;

    const auto letterbox = irt::resolvePreprocessGeometry(spec, 7, 3);
    EXPECT_EQ(letterbox.original_width, 7);
    EXPECT_EQ(letterbox.original_height, 3);
    EXPECT_EQ(letterbox.resized_width, 10);
    EXPECT_EQ(letterbox.resized_height, 4);
    EXPECT_EQ(letterbox.pad_left, 0);
    EXPECT_EQ(letterbox.pad_top, 3);
    EXPECT_EQ(letterbox.crop_left, 0);
    EXPECT_EQ(letterbox.crop_top, 0);
    EXPECT_FLOAT_EQ(letterbox.scale, 10.0F / 7.0F);

    spec.padding_mode = irt::PaddingMode::CenterCrop;
    spec.source_width  = 3;
    spec.source_height = 7;
    const auto crop    = irt::resolvePreprocessGeometry(spec, 3, 7);
    EXPECT_EQ(crop.resized_width, 10);
    EXPECT_EQ(crop.resized_height, 23);
    EXPECT_EQ(crop.pad_left, 0);
    EXPECT_EQ(crop.pad_top, 0);
    EXPECT_EQ(crop.crop_left, 0);
    EXPECT_EQ(crop.crop_top, 6);
}

TEST(PreprocessGeometryTest, RejectsDeclaredSourceSizeMismatch)
{
    irt::PreprocessSpec spec;
    spec.input_width     = 10;
    spec.input_height    = 10;
    spec.input_channels  = 3;
    spec.source_channels = 3;
    spec.mean            = {0.0F, 0.0F, 0.0F};
    spec.stddev          = {1.0F, 1.0F, 1.0F};
    spec.source_width    = 640;
    spec.source_height   = 480;

    EXPECT_THROW((void)irt::resolvePreprocessGeometry(spec, 320, 240), irt::Exception);
    EXPECT_THROW((void)irt::resolvePreprocessGeometry(spec, 640, 0), irt::Exception);
}

TEST(ModelContractTest, CoreExecutionContractUsesOpaqueStreamAndBuffers)
{
    class StubModel final : public IExecutableModel
    {
    public:
        std::vector<TensorInfo> inputs() const override { return {{"input", {}, TensorIOMode::Input}}; }
        std::vector<TensorInfo> outputs() const override { return {{"output", {}, TensorIOMode::Output}}; }
        ExecutionCapabilities capabilities() const noexcept override { return {}; }
        void setInputShape(const std::string &, Shape shape) override { shape_ = std::move(shape); }
        void execute(std::span<const BufferView> buffers, ExecuteOptions options = {}) override
        {
            ASSERT_EQ(buffers.size(), 1U);
            EXPECT_EQ(options.stream, 0U);
            executed_ = true;
        }
        std::unique_ptr<ITensorRuntimeSession> createSession() const override { return nullptr; }
        void executeSession(ITensorRuntimeSession &, std::span<const BufferView>, ExecuteOptions) const override {}
        Shape shape_;
        bool  executed_{false};
    } model;

    model.setInputShape("input", Shape{1, 3, 2, 2});
    TensorDesc desc{TensorDataType::F32, TensorLayout::Opaque, MemoryKind::HOST, Shape{1, 3, 2, 2}};
    std::vector<float> storage(desc.elementCount());
    BufferView buffer{storage.data(), desc, desc.byteSize(), 1, desc.byteSize()};
    model.execute(std::span<const BufferView>(&buffer, 1));
    EXPECT_TRUE(model.executed_);
}

TEST(ModelContractTest, ExecutableModelsAndPlansShareOneCapabilityContract)
{
    static_assert(std::is_base_of_v<IExecutionPlan, IExecutableModel>);

    class StubSession final : public ITensorRuntimeSession
    {
    public:
        Shape tensorShape(const std::string &) const override { return Shape{1, 1}; }
        TensorDataType tensorDataType(const std::string &) const override { return TensorDataType::F32; }
        void setTensorShape(const std::string &, const Shape &) override {}
        void execute(std::span<const BufferView>, ExecuteOptions) override {}
    };

    class StubModel final : public IExecutableModel
    {
    public:
        std::vector<TensorInfo> inputs() const override
        {
            return {{"input", {TensorDataType::F32, TensorLayout::Opaque, MemoryKind::HOST, Shape{1, 1}},
                     TensorIOMode::Input}};
        }
        std::vector<TensorInfo> outputs() const override
        {
            return {{"output", {TensorDataType::F32, TensorLayout::Opaque, MemoryKind::HOST, Shape{1, 1}},
                     TensorIOMode::Output}};
        }
        ExecutionCapabilities capabilities() const noexcept override
        {
            return {.supports_dynamic_batch = true, .supports_feature_outputs = true, .fixed_batch_size = 0};
        }
        void setInputShape(const std::string &, Shape) override {}
        void execute(std::span<const BufferView>, ExecuteOptions) override {}
        std::unique_ptr<ITensorRuntimeSession> createSession() const override
        {
            return std::make_unique<StubSession>();
        }
        void executeSession(ITensorRuntimeSession &session, std::span<const BufferView> buffers,
                            ExecuteOptions options) const override
        {
            session.execute(buffers, options);
        }
    } model;

    static_assert(std::is_move_constructible_v<StubModel>);
    static_assert(std::is_move_assignable_v<StubModel>);

    const IExecutionPlan &plan = model;
    const auto capabilities = plan.capabilities();
    EXPECT_TRUE(capabilities.supports_dynamic_batch);
    EXPECT_TRUE(capabilities.supports_feature_outputs);
    EXPECT_EQ(capabilities.fixed_batch_size, 0);
    EXPECT_NE(plan.createSession(), nullptr);
}

TEST(ModelContractTest, AcceptsConcreteBuffersForDeclaredDynamicDimensions)
{
    const std::vector<irt::TensorInfo> inputs{
        {"input",
         {irt::TensorDataType::F32, irt::TensorLayout::NCHW, irt::MemoryKind::HOST,
          irt::Shape{-1, 3, -1, 2}},
         irt::TensorIOMode::Input}};
    const std::vector<irt::TensorInfo> outputs{
        {"output",
         {irt::TensorDataType::F32, irt::TensorLayout::NCHW, irt::MemoryKind::HOST,
          irt::Shape{-1, 1}},
         irt::TensorIOMode::Output}};

    std::vector<float> input_storage(4U * 3U * 5U * 2U, 1.0F);
    std::vector<float> output_storage(4U, 0.0F);
    const irt::TensorDesc input_desc{irt::TensorDataType::F32, irt::TensorLayout::NCHW,
                                     irt::MemoryKind::HOST, irt::Shape{4, 3, 5, 2}};
    const irt::TensorDesc output_desc{irt::TensorDataType::F32, irt::TensorLayout::NCHW,
                                      irt::MemoryKind::HOST, irt::Shape{4, 1}};
    const std::vector<irt::BufferView> buffers{
        {input_storage.data(), input_desc, input_desc.byteSize(), 1, input_desc.byteSize(), "input"},
        {output_storage.data(), output_desc, output_desc.byteSize(), 1, output_desc.byteSize(), "output"},
    };

    const auto normalized = irt::normalizeExecutionBuffers(buffers, inputs, outputs);
    ASSERT_EQ(normalized.size(), 2U);
    EXPECT_EQ(normalized[0].tensor_name, "input");
    EXPECT_EQ(normalized[1].tensor_name, "output");
}

} // namespace
