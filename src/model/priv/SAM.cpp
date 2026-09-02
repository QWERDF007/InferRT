#include "SAM.hpp"

#include "SAMCommon.hpp"
#include "SAMDecoders.hpp"
#include "SAMEncoders.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/ModelFactory.h>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace irt::model {

namespace {

SAMGeometry resolveSAMGeometry(const SAMSpec &spec, const IModelConfig &config)
{
    const auto &input_shapes = config.inputShapes();
    if (input_shapes.size() != kDefaultInputNames.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM expects %zu input tensors, got %zu",
                             kDefaultInputNames.size(), input_shapes.size());
    }
    if (config.inputTensorNames().size() != input_shapes.size())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "SAM input tensor name count must match input shape count");
    }
    if (config.outputTensorNames().size() != kDefaultOutputNames.size() && !config.featureOnly())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM expects %zu output tensors, got %zu",
                             kDefaultOutputNames.size(), config.outputTensorNames().size());
    }

    const auto &image_shape = input_shapes[0];
    SAMGeometry geometry{};
    geometry.batch    = static_cast<int>(image_shape[0]);
    geometry.channels = static_cast<int>(image_shape[1]);
    geometry.image_h  = static_cast<int>(image_shape[2]);
    geometry.image_w  = static_cast<int>(image_shape[3]);
    if (geometry.batch <= 0 || geometry.channels != 3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM image input must be Nx3xHxW, got %lldx%lldx%lldx%lld",
                             static_cast<long long>(image_shape[0]), static_cast<long long>(image_shape[1]),
                             static_cast<long long>(image_shape[2]), static_cast<long long>(image_shape[3]));
    }
    if (geometry.image_h != spec.image_size || geometry.image_w != spec.image_size)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "%s expects fixed image size %dx%d, got %dx%d",
                             spec.display_name, spec.image_size, spec.image_size, geometry.image_h, geometry.image_w);
    }
    if (geometry.image_h % kSamPatchSize != 0 || geometry.image_w % kSamPatchSize != 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM image size must be divisible by patch size %d",
                             kSamPatchSize);
    }
    geometry.grid_h      = geometry.image_h / kSamPatchSize;
    geometry.grid_w      = geometry.image_w / kSamPatchSize;
    geometry.grid_tokens = geometry.grid_h * geometry.grid_w;

    const auto &coords_shape = input_shapes[1];
    if (coords_shape[0] != geometry.batch || coords_shape[1] != spec.max_points || coords_shape[2] != 2
        || coords_shape[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM point_coords must be Nx%dx2x1 and share image batch",
                             spec.max_points);
    }

    const auto &labels_shape = input_shapes[2];
    if (labels_shape[0] != geometry.batch || labels_shape[1] != spec.max_points || labels_shape[2] != 1
        || labels_shape[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM point_labels must be Nx%dx1x1 and share image batch",
                             spec.max_points);
    }

    const auto &mask_shape = input_shapes[3];
    if (mask_shape[0] != geometry.batch || mask_shape[1] != 1 || mask_shape[2] != spec.mask_size
        || mask_shape[3] != spec.mask_size)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM mask_input must be Nx1x%dx%d and share image batch",
                             spec.mask_size, spec.mask_size);
    }

    const auto &has_mask_shape = input_shapes[4];
    if (has_mask_shape[0] != geometry.batch || has_mask_shape[1] != 1 || has_mask_shape[2] != 1
        || has_mask_shape[3] != 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "SAM has_mask_input must be Nx1x1x1 and share image batch");
    }

    return geometry;
}

std::vector<irt::Shape> defaultInputShapes(const SAMSpec &spec)
{
    return {
        irt::Shape{1, 3, spec.image_size, spec.image_size},
        irt::Shape{1, spec.max_points, 2, 1},
        irt::Shape{1, spec.max_points, 1, 1},
        irt::Shape{1, 1, spec.mask_size, spec.mask_size},
        irt::Shape{1, 1, 1, 1},
    };
}

template<size_t N>
std::vector<std::string> toStringVector(const std::array<const char *, N> &values)
{
    std::vector<std::string> result;
    result.reserve(values.size());
    for (const char *value : values)
    {
        result.emplace_back(value);
    }
    return result;
}

bool usesDefaultSingleImageShape(const IModelConfig &config)
{
    if (config.inputShapes().size() != 1)
    {
        return false;
    }
    const auto &shape = config.inputShape();
    return shape.rank() == 4 && shape[0] == 1 && shape[1] == 3 && shape[2] == kDefaultImageNetSize
        && shape[3] == kDefaultImageNetSize;
}

SAMSpec makeSAMSpec(const char *display_name, int embed_dim, int depth, int heads,
                    std::initializer_list<int> global_attn)
{
    return {display_name,
            kSamImageSize,
            kSamMaskSize,
            kSamMaxPoints,
            kSamOutputMasks,
            embed_dim,
            depth,
            heads,
            std::set<int>(global_attn.begin(), global_attn.end()),
            {},
            {},
            {},
            0,
            0,
            SAMFamily::SAM};
}

SAMSpec makeSAM2Spec(const char *display_name, int embed_dim, int heads, std::vector<int> stages,
                     std::initializer_list<int> global_attn, std::vector<int> window_spec, int pos_embed_size)
{
    const int depth = std::accumulate(stages.begin(), stages.end(), 0);
    return {
        display_name,
        kSamImageSize,
        kSamMaskSize,
        kSamMaxPoints,
        kSamOutputMasks,
        embed_dim,
        depth,
        heads,
        std::set<int>(global_attn.begin(), global_attn.end()),
        std::move(stages),
        std::move(window_spec),
        {embed_dim * 8, embed_dim * 4, embed_dim * 2, embed_dim},
        pos_embed_size,
        3,
        SAMFamily::SAM2
    };
}

SAMSpec makeEdgeSAMSpec()
{
    return {"EdgeSAM",
            kSamImageSize,
            kSamMaskSize,
            kSamMaxPoints,
            kSamOutputMasks,
            kSamPromptDim,
            static_cast<int>(edgeSAMRepViTM1Blocks().size()),
            0,
            {},
            {},
            {},
            {},
            0,
            0,
            SAMFamily::EdgeSAM};
}

SAMSpec makeSAM3Spec(const char *display_name)
{
    return {display_name,   1008, kSamMaskSize, kSamMaxPoints, kSamOutputMasks, 1024, 0, 0, {}, {}, {}, {}, 0, 0,
            SAMFamily::SAM3};
}

} // namespace

SAMSegmentationModel::SAMSegmentationModel(SAMSpec spec)
    : spec_(std::move(spec))
{
    auto config = std::make_unique<IModelConfig>();
    normalizeModelConfig(*config);
    setModelConfig(std::move(config));
}

std::string SAMSegmentationModel::name() const noexcept
{
    return spec_.display_name;
}

void SAMSegmentationModel::normalizeModelConfig(IModelConfig &config) const
{
    if (config.inputTensorNames() == std::vector<std::string>{"input"})
    {
        config.setInputTensorNames(toStringVector(kDefaultInputNames));
    }
    if (usesDefaultSingleImageShape(config))
    {
        config.setInputShapes(defaultInputShapes(spec_));
    }
    if (config.outputTensorNames() == std::vector<std::string>{"output"})
    {
        config.setOutputTensorNames(toStringVector(kDefaultOutputNames));
    }
    if (config.numClasses() == kDefaultNumClasses)
    {
        config.setNumClasses(spec_.multimask_outputs);
    }
}

void SAMSegmentationModel::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    const auto geometry = resolveSAMGeometry(spec_, modelConfig());
    if (spec_.family == SAMFamily::SAM3)
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED,
                             "%s official TensorRT graph requires its native SAM3 image encoder and is not "
                             "silently mapped to SAM v1",
                             spec_.display_name);
    }

    NamedTensorMap        named_tensors;
    nvinfer1::ITensor    *image_embedding = nullptr;
    nvinfer1::ITensor    *high_res_s0     = nullptr;
    nvinfer1::ITensor    *high_res_s1     = nullptr;
    SAMMaskDecoderOptions decoder_options{};
    if (spec_.family == SAMFamily::SAM)
    {
        SAMViTSpec vit_spec{spec_.encoder_embed_dim, spec_.encoder_depth, spec_.encoder_num_heads,
                            spec_.global_attn_indexes};
        image_embedding = addSAMImageEncoder(*this, network, weights_map, geometry, vit_spec, named_tensors);
        decoder_options = {
            {"prompt_encoder", "mask_decoder"},
            kSamEmbedGrid,
            false,
            false,
            false
        };
    }
    else if (spec_.family == SAMFamily::EdgeSAM)
    {
        image_embedding = addEdgeSAMImageEncoder(*this, network, weights_map, geometry, named_tensors);
        decoder_options = {
            {"prompt_encoder", "mask_decoder"},
            kSamEmbedGrid,
            false,
            false,
            false
        };
    }
    else
    {
        SAM2HieraSpec hiera_spec{
            spec_.encoder_embed_dim,   spec_.encoder_num_heads, spec_.hiera_stages,   spec_.hiera_window_spec,
            spec_.global_attn_indexes, spec_.backbone_channels, spec_.pos_embed_size, spec_.q_pool};
        image_embedding = addSAM2ImageEncoder(*this, network, weights_map, geometry, hiera_spec, high_res_s0,
                                              high_res_s1, named_tensors);
        decoder_options = {
            {"sam_prompt_encoder", "sam_mask_decoder"},
            kSamEmbedGrid,
            true,
            true,
            true
        };
    }

    auto *sparse_prompt = addPointPromptEmbedding(*this, network, weights_map, geometry,
                                                  decoder_options.prefixes.prompt, named_tensors);
    auto *dense_prompt
        = addDensePromptEmbedding(*this, network, weights_map, decoder_options.prefixes.prompt, named_tensors);
    auto *image_pe            = addDensePromptPE(network, weights_map, decoder_options.prefixes.prompt);
    named_tensors["dense_pe"] = image_pe;

    nvinfer1::ITensor *masks           = nullptr;
    nvinfer1::ITensor *iou_predictions = nullptr;
    addSAMMaskDecoder(network, weights_map, *image_embedding, *image_pe, *sparse_prompt, *dense_prompt, decoder_options,
                      high_res_s0, high_res_s1, masks, iou_predictions, named_tensors);

    if (isBuildingFeatureEngine())
    {
        markFeatureOutputTensors(network, named_tensors);
        return;
    }

    auto *low_res_clone = requireLayer(network->addShuffle(*masks), "Failed to clone SAM low_res_masks output");
    low_res_clone->setReshapeDimensions(nvinfer1::Dims4{0, kSamOutputMasks, kSamMaskSize, kSamMaskSize});
    markOutputTensors(network, {masks, iou_predictions, low_res_clone->getOutput(0)});
}

SAM::SAM()
    : SAMSegmentationModel(makeSAMSpec("SAMViTH", 1280, 32, 16, {7, 15, 23, 31}))
{
}

SAMViTB::SAMViTB()
    : SAMSegmentationModel(makeSAMSpec("SAMViTB", 768, 12, 12, {2, 5, 8, 11}))
{
}

SAMViTL::SAMViTL()
    : SAMSegmentationModel(makeSAMSpec("SAMViTL", 1024, 24, 16, {5, 11, 17, 23}))
{
}

SAMViTH::SAMViTH()
    : SAMSegmentationModel(makeSAMSpec("SAMViTH", 1280, 32, 16, {7, 15, 23, 31}))
{
}

EdgeSAM::EdgeSAM()
    : SAMSegmentationModel(makeEdgeSAMSpec())
{
}

SAM2::SAM2()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM2HieraTiny::SAM2HieraTiny()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraTiny", 96, 1, {1, 2, 7, 2}, {5, 7, 9}, {8, 4, 14, 7}, 7))
{
}

SAM2HieraSmall::SAM2HieraSmall()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraSmall", 96, 1, {1, 2, 11, 2}, {7, 10, 13}, {8, 4, 14, 7}, 7))
{
}

SAM2HieraBasePlus::SAM2HieraBasePlus()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraBasePlus", 112, 2, {2, 3, 16, 3}, {12, 16, 20}, {8, 4, 14, 7}, 14))
{
}

SAM2HieraLarge::SAM2HieraLarge()
    : SAMSegmentationModel(makeSAM2Spec("SAM2HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM21HieraTiny::SAM21HieraTiny()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraTiny", 96, 1, {1, 2, 7, 2}, {5, 7, 9}, {8, 4, 14, 7}, 7))
{
}

SAM21HieraSmall::SAM21HieraSmall()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraSmall", 96, 1, {1, 2, 11, 2}, {7, 10, 13}, {8, 4, 14, 7}, 7))
{
}

SAM21HieraBasePlus::SAM21HieraBasePlus()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraBasePlus", 112, 2, {2, 3, 16, 3}, {12, 16, 20}, {8, 4, 14, 7}, 14))
{
}

SAM21HieraLarge::SAM21HieraLarge()
    : SAMSegmentationModel(makeSAM2Spec("SAM2.1HieraLarge", 144, 2, {2, 6, 36, 4}, {23, 33, 43}, {8, 4, 16, 8}, 7))
{
}

SAM3::SAM3()
    : SAMSegmentationModel(makeSAM3Spec("SAM3Image"))
{
}

SAM3Image::SAM3Image()
    : SAMSegmentationModel(makeSAM3Spec("SAM3Image"))
{
}

} // namespace irt::model

INFERRT_REGISTER_MODEL(SAM)
INFERRT_REGISTER_MODEL(SAMViTB)
INFERRT_REGISTER_MODEL(SAMViTL)
INFERRT_REGISTER_MODEL(SAMViTH)
INFERRT_REGISTER_MODEL(EdgeSAM)
INFERRT_REGISTER_MODEL(SAM2)
INFERRT_REGISTER_MODEL(SAM2HieraTiny)
INFERRT_REGISTER_MODEL(SAM2HieraSmall)
INFERRT_REGISTER_MODEL(SAM2HieraBasePlus)
INFERRT_REGISTER_MODEL(SAM2HieraLarge)
INFERRT_REGISTER_MODEL(SAM21HieraTiny)
INFERRT_REGISTER_MODEL(SAM21HieraSmall)
INFERRT_REGISTER_MODEL(SAM21HieraBasePlus)
INFERRT_REGISTER_MODEL(SAM21HieraLarge)
INFERRT_REGISTER_MODEL(SAM3)
INFERRT_REGISTER_MODEL(SAM3Image)
