/**
 * @file DinoBackbone.cpp
 * @brief 冻结 DINO 骨干适配器实现。
 */

#include "DinoBackbone.hpp"
#include "DinoPaths.hpp"
#include "DinoProfile.hpp"
#include "../ModelShapeAdapter.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Version.h>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <new>
#include <utility>

namespace irt::features::priv {

namespace {

constexpr float kMinimumTokenNorm = 1e-8F;

std::string dimsToCsv(const nvinfer1::Dims &dims)
{
    std::string result;
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (index > 0)
        {
            result += ',';
        }
        result += std::to_string(dims.d[index]);
    }
    return result;
}

bool usesTensorRtRuntime(const DinoRegionSearchConfig &config) noexcept
{
    return config.runtime.model_runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT
        && !config.runtime.model_runtime.isCpu();
}

std::string preprocessDescription(const irt::PreprocessSpec &spec)
{
    std::string result;
    result += "bgr8->rgb32f";
    result += ";scale=" + std::to_string(spec.scale);
    for (size_t index = 0; index < spec.mean.size(); ++index)
    {
        result += ";mean" + std::to_string(index) + "=" + std::to_string(spec.mean[index]);
    }
    for (size_t index = 0; index < spec.stddev.size(); ++index)
    {
        result += ";std" + std::to_string(index) + "=" + std::to_string(spec.stddev[index]);
    }
    result += ";resize=linear;letterbox=topleft;pad=mean";
    return result;
}

bool isPatchTokenOutput(const nvinfer1::Dims &dims)
{
    return dims.nbDims == 3 && dims.d[0] > 0 && dims.d[1] > 0 && dims.d[2] > 0;
}

struct DinoBackboneCacheKey final
{
    std::string                       model_name{};
    std::filesystem::path             weights_path{};
    irt::model::ModelRuntime          model_runtime{};
    irt::model::ModelPrecision        model_precision{irt::model::ModelPrecision::FP32};
    int                               encoder_edge{0};
    size_t                            model_batch_size{0};
    uintmax_t                         weights_size{0};
    std::filesystem::file_time_type   weights_mtime{};

    bool same(const DinoBackboneCacheKey &other) const
    {
        return model_name == other.model_name && weights_path == other.weights_path
            && model_runtime == other.model_runtime && model_precision == other.model_precision
            && encoder_edge == other.encoder_edge && model_batch_size == other.model_batch_size
            && weights_size == other.weights_size && weights_mtime == other.weights_mtime;
    }
};

DinoBackboneCacheKey makeBackboneCacheKey(const DinoRegionSearchConfig &config)
{
    DinoBackboneCacheKey key;
    key.model_name        = config.model.model_name;
    key.weights_path      = std::filesystem::absolute(config.model.weights_file).lexically_normal();
    key.model_runtime     = config.runtime.model_runtime;
    key.model_precision   = config.runtime.model_precision;
    key.encoder_edge      = config.model.encoder_edge;
    key.model_batch_size  = config.runtime.model_batch_size;
    std::error_code error;
    key.weights_size = std::filesystem::file_size(key.weights_path, error);
    error.clear();
    key.weights_mtime = std::filesystem::last_write_time(key.weights_path, error);
    return key;
}

class DinoBackboneRegistry final
{
public:
    std::shared_ptr<DinoBackbone> acquire(const DinoRegionSearchConfig &config)
    {
        const auto requested = makeBackboneCacheKey(config);
        std::lock_guard lock(mutex_);
        if (active_ != nullptr && has_key_ && key_.same(requested))
        {
            return active_;
        }

        auto next = std::make_shared<DinoBackbone>(config);
        key_       = requested;
        has_key_   = true;
        active_    = std::move(next);
        return active_;
    }

    void reset()
    {
        std::lock_guard lock(mutex_);
        active_.reset();
        has_key_ = false;
        key_     = {};
    }

private:
    std::mutex                    mutex_{};
    DinoBackboneCacheKey          key_{};
    bool                          has_key_{false};
    std::shared_ptr<DinoBackbone> active_{};
};

DinoBackboneRegistry &backboneRegistry()
{
    // TensorRT/CUDA DLLs may tear down their runtime before DLL-local statics;
    // keep the process cache alive so its model is reclaimed by process teardown.
    static DinoBackboneRegistry *instance = new DinoBackboneRegistry();
    return *instance;
}

} // namespace

std::string DinoExtractorSignature::cacheKey() const
{
    return model_name + "|" + weights_path + "|" + std::to_string(weights_size) + "|"
         + std::to_string(weights_mtime) + "|" + runtime + "|" + precision + "|" + input_tensor + "|"
         + input_shape + "|" + output_tensor + "|" + output_shape + "|" + extractor_layer + "|" + preprocess + "|"
         + revision + "|" + std::to_string(patch_size) + "|" + std::to_string(channels) + "|"
         + std::to_string(encoder_edge) + "|" + std::to_string(grid_height) + "|" + std::to_string(grid_width);
}

DinoBackbone::DinoBackbone(const DinoRegionSearchConfig &config)
    : config_(config)
{
    dinoValidateConfig(config_);

    // 骨干 patch 边长由模型别名决定：DINOv3 为 16，DINOv2 为 14。
    const bool is_dinov3 = config_.model.model_name.find("dinov3") != std::string::npos;
    const int  patch_size = is_dinov3 ? 16 : 14;
    const int  encoder_edge = dinoResolveEncoderEdge(config_, patch_size);
    preprocess_spec_ = dinoViewPreprocessSpec(encoder_edge, patch_size);

    constexpr const char *kPatchTokenFeature = "x_norm_patchtokens";

    auto model_config = std::make_unique<irt::model::IModelConfig>();
    model_config->setFeatureTensorNames({kPatchTokenFeature});
    model_config->setOutputTensorNames({kPatchTokenFeature});
    model_config->setFeatureOnly(true);
    model_config->setRuntime(config_.runtime.model_runtime);
    model_config->setPrecision(config_.runtime.model_precision);
    model_config->setDynamicBatchRange(1, static_cast<int>(config_.runtime.model_batch_size),
                                       static_cast<int>(config_.runtime.model_batch_size));
    model_config->setInputShape(irt::Shape{1, 3, encoder_edge, encoder_edge});

    model_ = irt::model::CreateModel(config_.model.model_name, std::move(model_config));
    if (!model_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create backbone model: %s",
                             config_.model.model_name.c_str());
    }
    model_->setLogLevel(irt::model::LogLevel::Warning);
    if (!std::filesystem::exists(config_.model.weights_file))
    {
        throw irt::Exception(irt::Status::NOT_READY, "Backbone weights file does not exist: %s",
                             config_.model.weights_file.string().c_str());
    }

    const auto weight_identity = makeBackboneCacheKey(config_);
    model_->buildOrLoad(config_.model.weights_file.string());

    use_device_buffers_ = usesTensorRtRuntime(config_);
    if (use_device_buffers_)
    {
        irt::model::setCudaDevice(config_.runtime.model_runtime.deviceId());
    }

    const auto input_names = model_->ioTensorNames(irt::TensorIOMode::Input);
    const auto output_names = model_->ioTensorNames(irt::TensorIOMode::Output);
    if (input_names.size() != 1U || output_names.size() != 1U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Backbone requires exactly one input and one patch-token output");
    }
    input_name_  = input_names.front();
    output_name_ = output_names.front();

    if (model_->tensorDataType(input_name_) != irt::TensorDataType::F32
        || model_->tensorDataType(output_name_) != irt::TensorDataType::F32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Backbone expects float32 input and output tensors");
    }

    input_shape_ = toTensorRtDims(model_->tensorShape(input_name_));
    if (input_shape_.nbDims != 4 || input_shape_.d[1] != 3 || input_shape_.d[2] <= 0 || input_shape_.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Backbone expects NCHW input, got %s",
                             dimsToCsv(input_shape_).c_str());
    }
    if (input_shape_.d[2] % patch_size != 0 || input_shape_.d[3] % patch_size != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Backbone input %dx%d must be divisible by patch size %d", input_shape_.d[2],
                             input_shape_.d[3], patch_size);
    }

    output_shape_ = toTensorRtDims(model_->tensorShape(output_name_));
    if (!isPatchTokenOutput(output_shape_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Backbone patch token output must be BxNxD, got %s", dimsToCsv(output_shape_).c_str());
    }

    max_batch_size_ = static_cast<size_t>(input_shape_.d[0]);
    const auto grid_height = static_cast<int>(input_shape_.d[2]) / patch_size;
    const auto grid_width  = static_cast<int>(input_shape_.d[3]) / patch_size;
    token_count_           = static_cast<size_t>(grid_height) * static_cast<size_t>(grid_width);
    input_elements_per_sample_ = static_cast<size_t>(input_shape_.d[1]) * static_cast<size_t>(input_shape_.d[2])
                               * static_cast<size_t>(input_shape_.d[3]);

    if (static_cast<size_t>(output_shape_.d[1]) != token_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Backbone patch token grid mismatch: output has %d tokens, expected %zu",
                             output_shape_.d[1], token_count_);
    }

    signature_.model_name    = config_.model.model_name;
    signature_.weights_path  = dinoPathToUtf8(std::filesystem::absolute(config_.model.weights_file).lexically_normal());
    signature_.weights_size  = weight_identity.weights_size;
    signature_.weights_mtime = static_cast<int64_t>(weight_identity.weights_mtime.time_since_epoch().count());
    signature_.runtime       = config_.runtime.model_runtime.toString();
    signature_.precision     = irt::model::modelPrecisionName(config_.runtime.model_precision);
    signature_.input_tensor  = input_name_;
    signature_.output_tensor = output_name_;
    signature_.input_shape   = dimsToCsv(input_shape_);
    signature_.output_shape  = dimsToCsv(output_shape_);
    signature_.extractor_layer = std::string(kPatchTokenFeature) + " (last layer, normalized spatial tokens)";
    signature_.preprocess    = preprocessDescription(preprocess_spec_);
    signature_.revision      = irt::GetVersionString() + "|trt-" + std::to_string(NV_TENSORRT_MAJOR);
    signature_.patch_size    = patch_size;
    signature_.channels      = static_cast<int>(output_shape_.d[2]);
    signature_.encoder_edge  = encoder_edge;
    signature_.grid_height   = grid_height;
    signature_.grid_width    = grid_width;

    if (use_device_buffers_)
    {
        device_input_.resize(max_batch_size_ * input_elements_per_sample_, irt::TensorDataType::F32);
        device_output_.resize(
            max_batch_size_ * token_count_ * static_cast<size_t>(signature_.channels), irt::TensorDataType::F32);
    }
    host_input_.resize(max_batch_size_ * input_elements_per_sample_);
    host_output_.resize(max_batch_size_ * token_count_ * static_cast<size_t>(signature_.channels));
}

DinoBackbone::~DinoBackbone()
{
    // 先释放模型上下文，再释放 CUDA 缓冲区，避免后端仍持有执行资源。
    model_.reset();
}

std::shared_ptr<DinoBackbone> dinoAcquireBackbone(const DinoRegionSearchConfig &config)
{
    return backboneRegistry().acquire(config);
}

void dinoResetBackbones()
{
    backboneRegistry().reset();
}

std::vector<DinoFeatureGrid> DinoBackbone::extract(const std::vector<DinoViewRaster> &rasters)
{
    std::vector<DinoFeatureGrid> grids;
    grids.reserve(rasters.size());
    const auto append = [&](const size_t begin, const size_t count)
    {
        auto chunk = extractChunk(rasters, begin, count);
        grids.insert(grids.end(), std::make_move_iterator(chunk.begin()), std::make_move_iterator(chunk.end()));
    };

    for (size_t begin = 0; begin < rasters.size(); begin += max_batch_size_)
    {
        const size_t count = std::min(max_batch_size_, rasters.size() - begin);
        try
        {
            append(begin, count);
        }
        catch (const irt::Exception &error)
        {
            if (error.code() != irt::Status::ERROR_OUT_OF_MEMORY || count <= 1U)
            {
                throw;
            }
            const size_t retry_batch = std::max<size_t>(1U, count / 2U);
            for (size_t offset = 0; offset < count; offset += retry_batch)
            {
                append(begin + offset, std::min(retry_batch, count - offset));
            }
        }
        catch (const std::bad_alloc &)
        {
            if (count <= 1U)
            {
                throw;
            }
            const size_t retry_batch = std::max<size_t>(1U, count / 2U);
            for (size_t offset = 0; offset < count; offset += retry_batch)
            {
                append(begin + offset, std::min(retry_batch, count - offset));
            }
        }
    }
    return grids;
}

std::vector<DinoFeatureGrid> DinoBackbone::extractChunk(const std::vector<DinoViewRaster> &rasters,
                                                        const size_t begin, const size_t count)
{
    for (size_t index = 0; index < count; ++index)
    {
        const auto &raster = rasters[begin + index];
        if (raster.chw.size() != input_elements_per_sample_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "View raster element count does not match backbone input");
        }
        if (raster.plan.patch_size != signature_.patch_size
            || raster.plan.grid_height != signature_.grid_height
            || raster.plan.grid_width != signature_.grid_width)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "View geometry does not match backbone patch grid");
        }
    }

    auto input_dims  = input_shape_;
    input_dims.d[0]  = static_cast<int32_t>(count);
    model_->setTensorShape(input_name_, toCoreShape(input_dims));
    const auto runtime_output_dims = toTensorRtDims(model_->tensorShape(output_name_));
    if (runtime_output_dims.nbDims != 3 || runtime_output_dims.d[0] != static_cast<int32_t>(count))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Backbone runtime patch token shape does not preserve batch");
    }

    const auto channel_count = static_cast<size_t>(signature_.channels);
    const auto expected_elements = count * token_count_ * channel_count;

    for (size_t index = 0; index < count; ++index)
    {
        const auto &raster = rasters[begin + index];
        std::copy(raster.chw.begin(), raster.chw.end(),
                  host_input_.begin() + static_cast<std::ptrdiff_t>(index * input_elements_per_sample_));
    }

    const auto input_names  = model_->ioTensorNames(irt::TensorIOMode::Input);
    const auto output_names = model_->ioTensorNames(irt::TensorIOMode::Output);

    if (use_device_buffers_)
    {
        const auto stream_handle = model_->resolveExecutionStream();
        const auto stream        = reinterpret_cast<cudaStream_t>(stream_handle);
        if (stream == nullptr)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Backbone requires a valid CUDA stream");
        }
        irt::model::checkCuda(cudaMemcpyAsync(device_input_.data(), host_input_.data(),
                                              count * input_elements_per_sample_ * sizeof(float),
                                              cudaMemcpyHostToDevice, stream),
                              "cudaMemcpyAsync(backbone input)");
        std::vector<irt::BufferView> buffers{
            irt::BufferView::fromBytes(device_input_.data(), device_input_.sizeBytes(), irt::MemoryKind::DEVICE,
                                       input_names.front()),
            irt::BufferView::fromBytes(device_output_.data(), device_output_.sizeBytes(), irt::MemoryKind::DEVICE,
                                       output_names.front())};
        model_->forwardFeatures(buffers, stream_handle, true);
        irt::model::checkCuda(cudaMemcpyAsync(host_output_.data(), device_output_.data(),
                                              expected_elements * sizeof(float), cudaMemcpyDeviceToHost, stream),
                              "cudaMemcpyAsync(backbone output)");
        irt::model::checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(backbone)");
    }
    else
    {
        std::vector<irt::BufferView> buffers{
            irt::BufferView::fromBytes(host_input_.data(), count * input_elements_per_sample_ * sizeof(float),
                                       irt::MemoryKind::HOST, input_names.front()),
            irt::BufferView::fromBytes(host_output_.data(), expected_elements * sizeof(float), irt::MemoryKind::HOST,
                                       output_names.front())};
        model_->forwardFeatures(buffers, 0, false);
    }
    ++forward_count_;

    std::vector<DinoFeatureGrid> grids;
    grids.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const auto &raster = rasters[begin + index];
        DinoFeatureGrid grid;
        grid.plan      = raster.plan;
        grid.channels  = signature_.channels;
        grid.valid_area = dinoPatchValidArea(raster.plan);
        grid.tokens.resize(token_count_ * channel_count);

        const float *sample = host_output_.data() + index * token_count_ * channel_count;
        for (size_t token = 0; token < token_count_; ++token)
        {
            const float *source = sample + token * channel_count;
            float       *target = grid.tokens.data() + token * channel_count;
            double       norm   = 0.0;
            for (size_t channel = 0; channel < channel_count; ++channel)
            {
                norm += static_cast<double>(source[channel]) * static_cast<double>(source[channel]);
            }
            norm = std::sqrt(norm);
            if (norm < kMinimumTokenNorm)
            {
                // 退化描述不参与池化与局部匹配，直接标为无效。
                std::fill(target, target + channel_count, 0.0F);
                grid.valid_area[token] = 0.0F;
                continue;
            }
            const float inverse = static_cast<float>(1.0 / norm);
            for (size_t channel = 0; channel < channel_count; ++channel)
            {
                target[channel] = source[channel] * inverse;
            }
        }
        grids.push_back(std::move(grid));
    }
    return grids;
}

} // namespace irt::features::priv
