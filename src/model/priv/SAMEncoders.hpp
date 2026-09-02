#pragma once

#include "IModelImpl.hpp"
#include "SAMCommon.hpp"

namespace irt::model {

class SAMSegmentationModel;

const SAMViTSpec &samViTBaseSpec();
const SAMViTSpec &samViTLargeSpec();
const SAMViTSpec &samViTHugeSpec();

const SAM2HieraSpec &sam2HieraTinySpec();
const SAM2HieraSpec &sam2HieraSmallSpec();
const SAM2HieraSpec &sam2HieraBasePlusSpec();
const SAM2HieraSpec &sam2HieraLargeSpec();

const std::vector<EdgeSAMRepViTBlockSpec> &edgeSAMRepViTM1Blocks();

nvinfer1::ITensor *addSAMImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                      const WeightsMap &weights_map, const SAMGeometry &geometry,
                                      const SAMViTSpec &spec, priv::IModelImpl::NamedTensorMap &named_tensors);

nvinfer1::ITensor *addSAM2ImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                       const WeightsMap &weights_map, const SAMGeometry &geometry,
                                       const SAM2HieraSpec &spec, nvinfer1::ITensor *&high_res_s0,
                                       nvinfer1::ITensor *&high_res_s1, priv::IModelImpl::NamedTensorMap &named_tensors);

nvinfer1::ITensor *addEdgeSAMImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                          const WeightsMap &weights_map, const SAMGeometry &geometry,
                                          priv::IModelImpl::NamedTensorMap &named_tensors);

} // namespace irt::model
