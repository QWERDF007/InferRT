#pragma once

#include "IModelImpl.hpp"
#include "SAMCommon.hpp"

namespace irt::model {

class SAMSegmentationModel;

nvinfer1::ITensor *addDensePromptPE(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    const std::string &prompt_prefix);

nvinfer1::ITensor *addPointPromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map, const SAMGeometry &geometry,
                                           const std::string                &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors);

nvinfer1::ITensor *addDensePromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map, const std::string &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors);

void addSAMMaskDecoder(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                       nvinfer1::ITensor &image_embedding, nvinfer1::ITensor &image_pe,
                       nvinfer1::ITensor &sparse_prompt, nvinfer1::ITensor &dense_prompt,
                       const SAMMaskDecoderOptions &options, nvinfer1::ITensor *high_res_s0,
                       nvinfer1::ITensor *high_res_s1, nvinfer1::ITensor *&masks, nvinfer1::ITensor *&iou_predictions,
                       priv::IModelImpl::NamedTensorMap &named_tensors);

} // namespace irt::model
