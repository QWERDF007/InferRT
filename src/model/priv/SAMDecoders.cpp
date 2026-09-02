#include "SAMDecoders.hpp"

namespace irt::model {

namespace {

nvinfer1::ITensor *addLabelGate(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &labels, float target)
{
    auto *target_const = addScalar(network, labels, target);
    auto *diff = requireLayer(network->addElementWise(labels, *target_const, E::kSUB), "Failed to add SAM label diff");
    auto *abs  = requireLayer(network->addUnary(*diff->getOutput(0), U::kABS), "Failed to add SAM label abs");
    auto *one  = addScalar(network, labels, 1.0F);
    auto *raw
        = requireLayer(network->addElementWise(*one, *abs->getOutput(0), E::kSUB), "Failed to add SAM label gate");
    auto *zero = addScalar(network, labels, 0.0F);
    return requireLayer(network->addElementWise(*raw->getOutput(0), *zero, E::kMAX), "Failed to add SAM label clamp")
        ->getOutput(0);
}

nvinfer1::ITensor *addGatedEmbedding(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &gate, const std::string &key)
{
    auto *embedding = requireLayer(
        network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim}, requireWeight(weights_map, key, kSamPromptDim)),
        "Failed to add SAM prompt embedding");
    return requireLayer(network->addElementWise(*embedding->getOutput(0), gate, E::kPROD),
                        "Failed to add SAM gated prompt embedding")
        ->getOutput(0);
}

nvinfer1::ITensor *addDecoderAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                       nvinfer1::ITensor &q_input, nvinfer1::ITensor &k_input,
                                       nvinfer1::ITensor &v_input, const std::string &prefix, int q_tokens,
                                       int k_tokens, int downsample_rate)
{
    const int internal_dim = kSamPromptDim / downsample_rate;
    auto     *q    = addLinear3D(network, weights_map, q_input, prefix + ".q_proj", kSamPromptDim, internal_dim);
    auto     *k    = addLinear3D(network, weights_map, k_input, prefix + ".k_proj", kSamPromptDim, internal_dim);
    auto     *v    = addLinear3D(network, weights_map, v_input, prefix + ".v_proj", kSamPromptDim, internal_dim);
    auto     *attn = addTokenAttention(network, *q, *k, *v, 0, q_tokens, k_tokens, kSamTwoWayHeads, internal_dim);
    return addLinear3D(network, weights_map, *attn, prefix + ".out_proj", internal_dim, kSamPromptDim);
}

void addTwoWayBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map, nvinfer1::ITensor *&queries,
                    nvinfer1::ITensor *&keys, nvinfer1::ITensor &query_pe, nvinfer1::ITensor &key_pe, int index,
                    int query_tokens, int image_tokens, const std::string &mask_prefix)
{
    const std::string prefix = mask_prefix + ".transformer.layers." + std::to_string(index);
    if (index == 0)
    {
        auto *self_attn = addDecoderAttention(network, weights_map, *queries, *queries, *queries, prefix + ".self_attn",
                                              query_tokens, query_tokens, 1);
        queries         = self_attn;
    }
    else
    {
        auto *q
            = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM), "Failed to add SAM decoder query PE")
                  ->getOutput(0);
        auto *self_attn = addDecoderAttention(network, weights_map, *q, *q, *queries, prefix + ".self_attn",
                                              query_tokens, query_tokens, 1);
        queries         = requireLayer(network->addElementWise(*queries, *self_attn, E::kSUM),
                                       "Failed to add SAM decoder self-attn residual")
                      ->getOutput(0);
    }
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm1", kSamPromptDim);

    auto *q_cross
        = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM), "Failed to add SAM token->image q")
              ->getOutput(0);
    auto *k_cross = requireLayer(network->addElementWise(*keys, key_pe, E::kSUM), "Failed to add SAM token->image k")
                        ->getOutput(0);
    auto *cross = addDecoderAttention(network, weights_map, *q_cross, *k_cross, *keys,
                                      prefix + ".cross_attn_token_to_image", query_tokens, image_tokens, 2);
    queries
        = requireLayer(network->addElementWise(*queries, *cross, E::kSUM), "Failed to add SAM token->image residual")
              ->getOutput(0);
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm2", kSamPromptDim);

    const auto mlp0_prefix = resolveLinearPrefix(weights_map, {prefix + ".mlp.lin1", prefix + ".mlp.layers.0"});
    const auto mlp1_prefix = resolveLinearPrefix(weights_map, {prefix + ".mlp.lin2", prefix + ".mlp.layers.1"});
    auto      *mlp0        = addLinear3D(network, weights_map, *queries, mlp0_prefix, kSamPromptDim, kSamTwoWayMlpDim);
    auto      *relu        = requireLayer(network->addActivation(*mlp0, nvinfer1::ActivationType::kRELU),
                                          "Failed to add SAM decoder MLP ReLU");
    auto *mlp1 = addLinear3D(network, weights_map, *relu->getOutput(0), mlp1_prefix, kSamTwoWayMlpDim, kSamPromptDim);
    queries = requireLayer(network->addElementWise(*queries, *mlp1, E::kSUM), "Failed to add SAM decoder MLP residual")
                  ->getOutput(0);
    queries = addLayerNormLastDim(network, weights_map, *queries, prefix + ".norm3", kSamPromptDim);

    auto *q_image = requireLayer(network->addElementWise(*keys, key_pe, E::kSUM), "Failed to add SAM image->token q")
                        ->getOutput(0);
    auto *k_token
        = requireLayer(network->addElementWise(*queries, query_pe, E::kSUM), "Failed to add SAM image->token k")
              ->getOutput(0);
    auto *image_cross = addDecoderAttention(network, weights_map, *q_image, *k_token, *queries,
                                            prefix + ".cross_attn_image_to_token", image_tokens, query_tokens, 2);
    keys = requireLayer(network->addElementWise(*keys, *image_cross, E::kSUM), "Failed to add SAM image token residual")
               ->getOutput(0);
    keys = addLayerNormLastDim(network, weights_map, *keys, prefix + ".norm4", kSamPromptDim);
}

} // namespace

nvinfer1::ITensor *addDensePromptPE(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                    const std::string &prompt_prefix)
{
    const auto &gaussian = requireWeight(weights_map, prompt_prefix + ".pe_layer.positional_encoding_gaussian_matrix",
                                         2 * (kSamPromptDim / 2));
    const auto *g        = static_cast<const float *>(gaussian.values);

    const size_t pe_count = irt::checkedSizeProduct({static_cast<size_t>(kSamPromptDim),
                                                     static_cast<size_t>(kSamEmbedGrid),
                                                     static_cast<size_t>(kSamEmbedGrid)},
                                                    "SAM dense prompt position encoding");
    std::vector<float> values(pe_count);
    for (int y = 0; y < kSamEmbedGrid; ++y)
    {
        for (int x = 0; x < kSamEmbedGrid; ++x)
        {
            const float nx = (static_cast<float>(x) + 0.5F) / static_cast<float>(kSamEmbedGrid);
            const float ny = (static_cast<float>(y) + 0.5F) / static_cast<float>(kSamEmbedGrid);
            const float px = 2.0F * nx - 1.0F;
            const float py = 2.0F * ny - 1.0F;
            for (int c = 0; c < kSamPromptDim / 2; ++c)
            {
                const float projected = 2.0F * kPi * (px * g[c] + py * g[(kSamPromptDim / 2) + c]);
                const auto  base      = static_cast<size_t>(y) * kSamEmbedGrid + x;
                values[static_cast<size_t>(c) * kSamEmbedGrid * kSamEmbedGrid + base] = std::sin(projected);
                values[static_cast<size_t>(c + kSamPromptDim / 2) * kSamEmbedGrid * kSamEmbedGrid + base]
                    = std::cos(projected);
            }
        }
    }

    auto *pe = requireLayer(network->addConstant(nvinfer1::Dims4{1, kSamPromptDim, kSamEmbedGrid, kSamEmbedGrid},
                                                 ownedFloatVector(std::move(values))),
                            "Failed to add SAM dense prompt PE");
    return pe->getOutput(0);
}

nvinfer1::ITensor *addPointPromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map, const SAMGeometry &geometry,
                                           const std::string                &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *point_coords            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 1);
    auto *point_labels            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 2);
    named_tensors["point_coords"] = point_coords;
    named_tensors["point_labels"] = point_labels;

    auto *coords = requireLayer(network->addShuffle(*point_coords), "Failed to reshape SAM point coords");
    coords->setReshapeDimensions(nvinfer1::Dims3{0, kSamMaxPoints, 2});
    auto              *half    = addScalar(network, *coords->getOutput(0), 0.5F);
    auto              *shifted = requireLayer(network->addElementWise(*coords->getOutput(0), *half, E::kSUM),
                                              "Failed to shift SAM point coords");
    std::vector<float> inv_size{1.0F / static_cast<float>(geometry.image_w),
                                1.0F / static_cast<float>(geometry.image_h)};
    auto *norm_const = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 2}, ownedFloatVector(inv_size)),
                                    "Failed to add SAM point norm");
    auto *normalized
        = requireLayer(network->addElementWise(*shifted->getOutput(0), *norm_const->getOutput(0), E::kPROD),
                       "Failed to normalize SAM point coords");
    auto *pad_coord       = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 2}, ownedFloatVector({0.0F, 0.0F})),
                                         "Failed to add SAM padding point");
    auto *pad_coord_batch = broadcastFirstDimLike(network, *pad_coord->getOutput(0), *normalized->getOutput(0));
    std::array<nvinfer1::ITensor *, 2> coord_tensors{normalized->getOutput(0), pad_coord_batch};
    auto *coords_cat = requireLayer(network->addConcatenation(coord_tensors.data(), 2), "Failed to concat SAM coords");
    coords_cat->setAxis(1);

    auto *two           = addScalar(network, *coords_cat->getOutput(0), 2.0F);
    auto *one           = addScalar(network, *coords_cat->getOutput(0), 1.0F);
    auto *double_coords = requireLayer(network->addElementWise(*coords_cat->getOutput(0), *two, E::kPROD),
                                       "Failed to double SAM coords");
    auto *pe_coords     = requireLayer(network->addElementWise(*double_coords->getOutput(0), *one, E::kSUB),
                                       "Failed to center SAM coords");

    auto *gaussian = requireLayer(
        network->addConstant(nvinfer1::Dims3{1, 2, kSamPromptDim / 2},
                             requireWeight(weights_map, prompt_prefix + ".pe_layer.positional_encoding_gaussian_matrix",
                                           2 * (kSamPromptDim / 2))),
        "Failed to add SAM point PE gaussian");
    auto *projected = requireLayer(
        network->addMatrixMultiply(*pe_coords->getOutput(0), M::kNONE, *gaussian->getOutput(0), M::kNONE),
        "Failed to add SAM point PE projection");
    auto *two_pi = addScalar(network, *projected->getOutput(0), 2.0F * kPi);
    auto *phase  = requireLayer(network->addElementWise(*projected->getOutput(0), *two_pi, E::kPROD),
                                "Failed to scale SAM point PE");
    auto *sin    = requireLayer(network->addUnary(*phase->getOutput(0), U::kSIN), "Failed to add SAM point PE sin");
    auto *cos    = requireLayer(network->addUnary(*phase->getOutput(0), U::kCOS), "Failed to add SAM point PE cos");
    std::array<nvinfer1::ITensor *, 2> pe_parts{sin->getOutput(0), cos->getOutput(0)};
    auto *pe = requireLayer(network->addConcatenation(pe_parts.data(), 2), "Failed to concat SAM point PE");
    pe->setAxis(2);

    auto *labels = requireLayer(network->addShuffle(*point_labels), "Failed to reshape SAM point labels");
    labels->setReshapeDimensions(nvinfer1::Dims3{0, kSamMaxPoints, 1});
    auto *pad_label       = requireLayer(network->addConstant(nvinfer1::Dims3{1, 1, 1}, ownedScalarWeight(-1.0F)),
                                         "Failed to add SAM padding label");
    auto *pad_label_batch = broadcastFirstDimLike(network, *pad_label->getOutput(0), *labels->getOutput(0));
    std::array<nvinfer1::ITensor *, 2> label_tensors{labels->getOutput(0), pad_label_batch};
    auto *labels_cat = requireLayer(network->addConcatenation(label_tensors.data(), 2), "Failed to concat SAM labels");
    labels_cat->setAxis(1);

    auto *neg_gate    = addLabelGate(network, *labels_cat->getOutput(0), -1.0F);
    auto *zero_gate   = addLabelGate(network, *labels_cat->getOutput(0), 0.0F);
    auto *pos_gate    = addLabelGate(network, *labels_cat->getOutput(0), 1.0F);
    auto *box_tl_gate = addLabelGate(network, *labels_cat->getOutput(0), 2.0F);
    auto *box_br_gate = addLabelGate(network, *labels_cat->getOutput(0), 3.0F);
    auto *keep_pe     = requireLayer(network->addElementWise(*addScalar(network, *neg_gate, 1.0F), *neg_gate, E::kSUB),
                                     "Failed to add SAM point PE keep");
    auto *pe_kept     = requireLayer(network->addElementWise(*pe->getOutput(0), *keep_pe->getOutput(0), E::kPROD),
                                     "Failed to apply SAM point PE keep");

    auto *not_point = addGatedEmbedding(network, weights_map, *neg_gate, prompt_prefix + ".not_a_point_embed.weight");
    auto *neg_point = addGatedEmbedding(network, weights_map, *zero_gate, prompt_prefix + ".point_embeddings.0.weight");
    auto *pos_point = addGatedEmbedding(network, weights_map, *pos_gate, prompt_prefix + ".point_embeddings.1.weight");
    auto *box_tl = addGatedEmbedding(network, weights_map, *box_tl_gate, prompt_prefix + ".point_embeddings.2.weight");
    auto *box_br = addGatedEmbedding(network, weights_map, *box_br_gate, prompt_prefix + ".point_embeddings.3.weight");
    auto *tmp    = requireLayer(network->addElementWise(*pe_kept->getOutput(0), *not_point, E::kSUM),
                                "Failed to add SAM not-a-point embedding");
    tmp          = requireLayer(network->addElementWise(*tmp->getOutput(0), *neg_point, E::kSUM),
                                "Failed to add SAM negative point embedding");
    tmp          = requireLayer(network->addElementWise(*tmp->getOutput(0), *pos_point, E::kSUM),
                                "Failed to add SAM positive point embedding");
    tmp          = requireLayer(network->addElementWise(*tmp->getOutput(0), *box_tl, E::kSUM),
                                "Failed to add SAM box top-left embedding");
    auto *sparse = requireLayer(network->addElementWise(*tmp->getOutput(0), *box_br, E::kSUM),
                                "Failed to add SAM box bottom-right embedding")
                       ->getOutput(0);
    named_tensors["sparse_prompt_embedding"] = sparse;
    return sparse;
}

nvinfer1::ITensor *addDensePromptEmbedding(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                           const WeightsMap &weights_map, const std::string &prompt_prefix,
                                           priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *mask_input                = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 3);
    auto *has_mask_input            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 4);
    named_tensors["mask_input"]     = mask_input;
    named_tensors["has_mask_input"] = has_mask_input;

    auto *conv0
        = requireLayer(network->addConvolutionNd(
                           *mask_input, 4, nvinfer1::DimsHW{2, 2},
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.0.weight",
                                         checkedWeightProduct({4, 1, 2, 2}, "SAM mask prompt convolution weight")),
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.0.bias", 4)),
                       "Failed to add SAM mask prompt conv0");
    conv0->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto *norm0 = addLayerNorm2d(network, weights_map, *conv0->getOutput(0), prompt_prefix + ".mask_downscaling.1", 4);
    auto *gelu0 = addGeluExact(network, *norm0);
    auto *conv1
        = requireLayer(network->addConvolutionNd(
                           *gelu0, 16, nvinfer1::DimsHW{2, 2},
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.3.weight",
                                         checkedWeightProduct({16, 4, 2, 2}, "SAM mask prompt convolution weight")),
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.3.bias", 16)),
                       "Failed to add SAM mask prompt conv1");
    conv1->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto *norm1 = addLayerNorm2d(network, weights_map, *conv1->getOutput(0), prompt_prefix + ".mask_downscaling.4", 16);
    auto *gelu1 = addGeluExact(network, *norm1);
    auto *mask_embedding
        = requireLayer(network->addConvolutionNd(
                           *gelu1, kSamPromptDim, nvinfer1::DimsHW{1, 1},
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.6.weight",
                                         checkedWeightProduct({kSamPromptDim, 16}, "SAM mask prompt convolution weight")),
                           requireWeight(weights_map, prompt_prefix + ".mask_downscaling.6.bias", kSamPromptDim)),
                       "Failed to add SAM mask prompt conv2");

    auto *has_part = requireLayer(network->addElementWise(*mask_embedding->getOutput(0), *has_mask_input, E::kPROD),
                                  "Failed to apply SAM has-mask gate");
    auto *one      = addScalar(network, *has_mask_input, 1.0F);
    auto *no_mask_gate
        = requireLayer(network->addElementWise(*one, *has_mask_input, E::kSUB), "Failed to add SAM no-mask gate");
    auto *no_mask = requireLayer(
        network->addConstant(nvinfer1::Dims4{1, kSamPromptDim, 1, 1},
                             requireWeight(weights_map, prompt_prefix + ".no_mask_embed.weight", kSamPromptDim)),
        "Failed to add SAM no-mask embedding");
    auto *no_mask_part
        = requireLayer(network->addElementWise(*no_mask->getOutput(0), *no_mask_gate->getOutput(0), E::kPROD),
                       "Failed to apply SAM no-mask gate");
    auto *dense = requireLayer(network->addElementWise(*has_part->getOutput(0), *no_mask_part->getOutput(0), E::kSUM),
                               "Failed to add SAM dense prompt embedding")
                      ->getOutput(0);
    named_tensors["dense_prompt_embedding"] = dense;
    return dense;
}

void addSAMMaskDecoder(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                       nvinfer1::ITensor &image_embedding, nvinfer1::ITensor &image_pe,
                       nvinfer1::ITensor &sparse_prompt, nvinfer1::ITensor &dense_prompt,
                       const SAMMaskDecoderOptions &options, nvinfer1::ITensor *high_res_s0,
                       nvinfer1::ITensor *high_res_s1, nvinfer1::ITensor *&masks, nvinfer1::ITensor *&iou_predictions,
                       priv::IModelImpl::NamedTensorMap &named_tensors)
{
    const auto &mask_prefix = options.prefixes.mask;
    if (options.use_high_res_features && (high_res_s0 == nullptr || high_res_s1 == nullptr))
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "SAM2 mask decoder requires high-resolution FPN features");
    }

    std::vector<nvinfer1::ITensor *> output_token_parts;
    output_token_parts.reserve(options.pred_obj_scores ? 3 : 2);
    if (options.pred_obj_scores)
    {
        auto *obj_token = requireLayer(
            network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim},
                                 requireWeight(weights_map, mask_prefix + ".obj_score_token.weight", kSamPromptDim)),
            "Failed to add SAM2 object score token");
        output_token_parts.push_back(obj_token->getOutput(0));
    }
    auto *iou_token = requireLayer(
        network->addConstant(nvinfer1::Dims3{1, 1, kSamPromptDim},
                             requireWeight(weights_map, mask_prefix + ".iou_token.weight", kSamPromptDim)),
        "Failed to add SAM iou token");
    output_token_parts.push_back(iou_token->getOutput(0));
    auto *mask_tokens
        = requireLayer(network->addConstant(nvinfer1::Dims3{1, kSamMaskTokens, kSamPromptDim},
                                            requireWeight(weights_map, mask_prefix + ".mask_tokens.weight",
                                                          checkedWeightProduct({kSamMaskTokens, kSamPromptDim},
                                                                               "SAM mask token weight"))),
                       "Failed to add SAM mask tokens");
    output_token_parts.push_back(mask_tokens->getOutput(0));
    auto *out_cat = requireLayer(
        network->addConcatenation(output_token_parts.data(), static_cast<int32_t>(output_token_parts.size())),
        "Failed to concat SAM output tokens");
    out_cat->setAxis(1);
    auto *output_tokens = broadcastFirstDimLike(network, *out_cat->getOutput(0), sparse_prompt);
    std::array<nvinfer1::ITensor *, 2> all_tokens{output_tokens, &sparse_prompt};
    auto                              *token_cat
        = requireLayer(network->addConcatenation(all_tokens.data(), 2), "Failed to concat SAM decoder tokens");
    token_cat->setAxis(1);
    auto *queries = token_cat->getOutput(0);

    auto *src = requireLayer(network->addElementWise(image_embedding, dense_prompt, E::kSUM),
                             "Failed to add SAM dense prompt to image embedding")
                    ->getOutput(0);
    auto *keys = requireLayer(network->addShuffle(*src), "Failed to flatten SAM decoder image tokens");
    keys->setReshapeDimensions(nvinfer1::Dims3{0, kSamPromptDim, options.image_grid * options.image_grid});
    keys->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});

    auto *pe_tokens = requireLayer(network->addShuffle(image_pe), "Failed to flatten SAM decoder image PE");
    pe_tokens->setReshapeDimensions(nvinfer1::Dims3{0, kSamPromptDim, options.image_grid * options.image_grid});
    pe_tokens->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    nvinfer1::ITensor *key_tokens          = keys->getOutput(0);
    nvinfer1::ITensor *query_tokens_tensor = queries;

    const int output_token_count = (options.pred_obj_scores ? 1 : 0) + 1 + kSamMaskTokens;
    const int iou_token_index    = options.pred_obj_scores ? 1 : 0;
    const int mask_token_index   = iou_token_index + 1;
    const int query_tokens       = output_token_count + kSamPointTokens;
    const int image_tokens       = options.image_grid * options.image_grid;
    for (int i = 0; i < kSamTwoWayDepth; ++i)
    {
        addTwoWayBlock(network, weights_map, query_tokens_tensor, key_tokens, *queries, *pe_tokens->getOutput(0), i,
                       query_tokens, image_tokens, mask_prefix);
    }

    auto *q_final = requireLayer(network->addElementWise(*query_tokens_tensor, *queries, E::kSUM),
                                 "Failed to add SAM final query PE")
                        ->getOutput(0);
    auto *k_final = requireLayer(network->addElementWise(*key_tokens, *pe_tokens->getOutput(0), E::kSUM),
                                 "Failed to add SAM final key PE")
                        ->getOutput(0);
    auto *final_attn
        = addDecoderAttention(network, weights_map, *q_final, *k_final, *key_tokens,
                              mask_prefix + ".transformer.final_attn_token_to_image", query_tokens, image_tokens, 2);
    query_tokens_tensor = requireLayer(network->addElementWise(*query_tokens_tensor, *final_attn, E::kSUM),
                                       "Failed to add SAM final attention residual")
                              ->getOutput(0);
    query_tokens_tensor                  = addLayerNormLastDim(network, weights_map, *query_tokens_tensor,
                                                               mask_prefix + ".transformer.norm_final_attn", kSamPromptDim);
    named_tensors["mask_decoder_tokens"] = query_tokens_tensor;

    auto *iou_token_out   = slicePreserveFirstDim(network, *query_tokens_tensor, nvinfer1::Dims3{0, iou_token_index, 0},
                                                  {1, kSamPromptDim});
    auto *mask_tokens_out = slicePreserveFirstDim(
        network, *query_tokens_tensor, nvinfer1::Dims3{0, mask_token_index, 0}, {kSamMaskTokens, kSamPromptDim});

    auto *src_view = requireLayer(network->addShuffle(*key_tokens), "Failed to restore SAM decoder image tokens");
    src_view->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
    src_view->setReshapeDimensions(nvinfer1::Dims4{0, kSamPromptDim, options.image_grid, options.image_grid});

    auto *up0
        = requireLayer(network->addDeconvolutionNd(
                           *src_view->getOutput(0), kSamPromptDim / 4, nvinfer1::DimsHW{2, 2},
                           requireWeight(weights_map, mask_prefix + ".output_upscaling.0.weight",
                                         checkedWeightProduct({kSamPromptDim, kSamPromptDim / 4, 2, 2},
                                                              "SAM mask upscaling weight")),
                           requireWeight(weights_map, mask_prefix + ".output_upscaling.0.bias", kSamPromptDim / 4)),
                       "Failed to add SAM mask upscaling deconv0");
    up0->setStrideNd(nvinfer1::DimsHW{2, 2});
    nvinfer1::ITensor *up0_input = up0->getOutput(0);
    if (options.use_high_res_features)
    {
        up0_input = requireLayer(network->addElementWise(*up0_input, *high_res_s1, E::kSUM),
                                 "Failed to add SAM2 high-res s1 feature")
                        ->getOutput(0);
    }
    auto *up_norm
        = addLayerNorm2d(network, weights_map, *up0_input, mask_prefix + ".output_upscaling.1", kSamPromptDim / 4);
    auto *up_gelu = addGeluExact(network, *up_norm);
    auto *up1
        = requireLayer(network->addDeconvolutionNd(
                           *up_gelu, kSamPromptDim / 8, nvinfer1::DimsHW{2, 2},
                           requireWeight(weights_map, mask_prefix + ".output_upscaling.3.weight",
                                         checkedWeightProduct({kSamPromptDim / 4, kSamPromptDim / 8, 2, 2},
                                                              "SAM mask upscaling weight")),
                           requireWeight(weights_map, mask_prefix + ".output_upscaling.3.bias", kSamPromptDim / 8)),
                       "Failed to add SAM mask upscaling deconv1");
    up1->setStrideNd(nvinfer1::DimsHW{2, 2});
    nvinfer1::ITensor *up1_input = up1->getOutput(0);
    if (options.use_high_res_features)
    {
        up1_input = requireLayer(network->addElementWise(*up1_input, *high_res_s0, E::kSUM),
                                 "Failed to add SAM2 high-res s0 feature")
                        ->getOutput(0);
    }
    auto *upscaled                      = addGeluExact(network, *up1_input);
    named_tensors["upscaled_embedding"] = upscaled;

    std::vector<nvinfer1::ITensor *> hyper_outputs;
    hyper_outputs.reserve(kSamMaskTokens);
    for (int i = 0; i < kSamMaskTokens; ++i)
    {
        const auto prefix = mask_prefix + ".output_hypernetworks_mlps." + std::to_string(i) + ".layers.";
        auto *token = slicePreserveFirstDim(network, *mask_tokens_out, nvinfer1::Dims3{0, i, 0}, {1, kSamPromptDim});
        auto *h0    = addLinear3D(network, weights_map, *token, prefix + "0", kSamPromptDim, kSamPromptDim);
        auto *r0    = requireLayer(network->addActivation(*h0, nvinfer1::ActivationType::kRELU),
                                   "Failed to add SAM hyper ReLU0");
        auto *h1    = addLinear3D(network, weights_map, *r0->getOutput(0), prefix + "1", kSamPromptDim, kSamPromptDim);
        auto *r1    = requireLayer(network->addActivation(*h1, nvinfer1::ActivationType::kRELU),
                                   "Failed to add SAM hyper ReLU1");
        hyper_outputs.push_back(
            addLinear3D(network, weights_map, *r1->getOutput(0), prefix + "2", kSamPromptDim, kSamPromptDim / 8));
    }
    auto *hyper
        = requireLayer(network->addConcatenation(hyper_outputs.data(), static_cast<int32_t>(hyper_outputs.size())),
                       "Failed to concat SAM hyper outputs");
    hyper->setAxis(1);

    auto *up_flat = requireLayer(network->addShuffle(*upscaled), "Failed to flatten SAM upscaled embedding");
    up_flat->setReshapeDimensions(nvinfer1::Dims3{0, kSamPromptDim / 8, kSamMaskSize * kSamMaskSize});
    auto *mask_logits
        = requireLayer(network->addMatrixMultiply(*hyper->getOutput(0), M::kNONE, *up_flat->getOutput(0), M::kNONE),
                       "Failed to add SAM hyper mask matmul");
    auto *mask_view = requireLayer(network->addShuffle(*mask_logits->getOutput(0)), "Failed to reshape SAM masks");
    mask_view->setReshapeDimensions(nvinfer1::Dims4{0, kSamMaskTokens, kSamMaskSize, kSamMaskSize});
    auto *iou0 = addLinear3D(network, weights_map, *iou_token_out, mask_prefix + ".iou_prediction_head.layers.0",
                             kSamPromptDim, kSamPromptDim);
    auto *iou_r0
        = requireLayer(network->addActivation(*iou0, nvinfer1::ActivationType::kRELU), "Failed to add SAM iou ReLU0");
    auto *iou1 = addLinear3D(network, weights_map, *iou_r0->getOutput(0), mask_prefix + ".iou_prediction_head.layers.1",
                             kSamPromptDim, kSamPromptDim);
    auto *iou_r1
        = requireLayer(network->addActivation(*iou1, nvinfer1::ActivationType::kRELU), "Failed to add SAM iou ReLU1");
    auto *iou2 = addLinear3D(network, weights_map, *iou_r1->getOutput(0), mask_prefix + ".iou_prediction_head.layers.2",
                             kSamPromptDim, kSamMaskTokens);
    nvinfer1::ITensor *iou_logits = iou2;
    if (options.sigmoid_iou)
    {
        iou_logits = requireLayer(network->addActivation(*iou_logits, nvinfer1::ActivationType::kSIGMOID),
                                  "Failed to add SAM2 iou sigmoid")
                         ->getOutput(0);
    }
    auto *iou_slice = requireLayer(network->addSlice(*iou_logits, nvinfer1::Dims3{0, 0, 0},
                                                     nvinfer1::Dims3{1, 1, kSamOutputMasks}, nvinfer1::Dims3{1, 1, 1}),
                                   "Failed to slice SAM iou predictions");
    iou_slice->setInput(2, *shapeWithFirstDimOf(network, *iou_logits, {1, kSamOutputMasks}));
    auto *iou_view
        = requireLayer(network->addShuffle(*iou_slice->getOutput(0)), "Failed to reshape SAM iou predictions");
    iou_view->setReshapeDimensions(nvinfer1::Dims4{0, kSamOutputMasks, 1, 1});

    masks                            = mask_view->getOutput(0);
    iou_predictions                  = iou_view->getOutput(0);
    named_tensors["low_res_masks"]   = masks;
    named_tensors["masks"]           = masks;
    named_tensors["iou_predictions"] = iou_predictions;
}

} // namespace irt::model
