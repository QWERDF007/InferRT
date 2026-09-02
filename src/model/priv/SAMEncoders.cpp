#include "SAMEncoders.hpp"

#include <limits>

namespace irt::model {

namespace {

nvinfer1::Weights makeRelativePositionWeight(const WeightsMap &weights_map, const std::string &key, int q_size,
                                             int k_size, int head_dim)
{
    if (q_size <= 0 || k_size <= 0 || head_dim <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM relative position dimensions must be positive");
    }
    const auto &weight           = requireWeight(weights_map, key);
    const int max_size = std::max(q_size, k_size);
    if (static_cast<int64_t>(max_size) * 2 > static_cast<int64_t>(std::numeric_limits<int>::max()) + 1)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM relative position dimensions are too large");
    }
    const int   expected_pos_len = 2 * max_size - 1;
    const int64_t expected_count = checkedWeightProduct({expected_pos_len, head_dim}, "SAM relative position weight");
    if (weight.count != expected_count)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Unexpected SAM relative position shape for %s: got %lld values, expected %d x %d",
                             key.c_str(), static_cast<long long>(weight.count), expected_pos_len, head_dim);
    }

    const auto        *source = static_cast<const float *>(weight.values);
    std::vector<float> values(irt::checkedSizeProduct({static_cast<size_t>(q_size), static_cast<size_t>(head_dim),
                                                       static_cast<size_t>(k_size)},
                                                      "SAM relative position table"));
    const float        q_scale = std::max(static_cast<float>(k_size) / static_cast<float>(q_size), 1.0F);
    const float        k_scale = std::max(static_cast<float>(q_size) / static_cast<float>(k_size), 1.0F);
    const float        offset  = static_cast<float>(k_size - 1) * k_scale;
    for (int q = 0; q < q_size; ++q)
    {
        for (int k = 0; k < k_size; ++k)
        {
            const int rel_index
                = static_cast<int>((static_cast<float>(q) * q_scale - static_cast<float>(k) * k_scale) + offset);
            for (int d = 0; d < head_dim; ++d)
            {
                values[(static_cast<size_t>(q) * head_dim + d) * k_size + k]
                    = source[static_cast<size_t>(rel_index) * head_dim + d];
            }
        }
    }
    return ownedFloatVector(std::move(values));
}

nvinfer1::ITensor *addImageRelativePosition(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                            nvinfer1::ITensor &attn, nvinfer1::ITensor &q_heads,
                                            const std::string &prefix, int batch, int q_h, int q_w, int num_heads,
                                            int head_dim)
{
    (void)batch;
    auto *attn_view = requireLayer(network->addShuffle(attn), "Failed to reshape SAM relative position attention");
    attn_view->setReshapeDimensions(makeDims({0, num_heads, q_h, q_w, q_h, q_w}));

    auto *q_view = requireLayer(network->addShuffle(q_heads), "Failed to reshape SAM relative position query");
    q_view->setReshapeDimensions(makeDims({0, num_heads, q_h, q_w, head_dim}));

    auto *rel_h = requireLayer(
        network->addConstant(makeDims({1, 1, q_h, head_dim, q_h}),
                             makeRelativePositionWeight(weights_map, prefix + ".attn.rel_pos_h", q_h, q_h, head_dim)),
        "Failed to add SAM rel_pos_h constant");
    auto *rel_h_scores
        = requireLayer(network->addMatrixMultiply(*q_view->getOutput(0), M::kNONE, *rel_h->getOutput(0), M::kNONE),
                       "Failed to add SAM rel_pos_h scores");
    auto *rel_h_view
        = requireLayer(network->addShuffle(*rel_h_scores->getOutput(0)), "Failed to expand SAM rel_pos_h scores");
    rel_h_view->setReshapeDimensions(makeDims({0, num_heads, q_h, q_w, q_h, 1}));

    auto *q_w_view
        = requireLayer(network->addShuffle(*q_view->getOutput(0)), "Failed to transpose SAM relative position query");
    q_w_view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4});
    auto *rel_w = requireLayer(
        network->addConstant(makeDims({1, 1, q_w, head_dim, q_w}),
                             makeRelativePositionWeight(weights_map, prefix + ".attn.rel_pos_w", q_w, q_w, head_dim)),
        "Failed to add SAM rel_pos_w constant");
    auto *rel_w_scores
        = requireLayer(network->addMatrixMultiply(*q_w_view->getOutput(0), M::kNONE, *rel_w->getOutput(0), M::kNONE),
                       "Failed to add SAM rel_pos_w scores");
    auto *rel_w_transpose
        = requireLayer(network->addShuffle(*rel_w_scores->getOutput(0)), "Failed to transpose SAM rel_pos_w scores");
    rel_w_transpose->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4});
    auto *rel_w_view
        = requireLayer(network->addShuffle(*rel_w_transpose->getOutput(0)), "Failed to expand SAM rel_pos_w scores");
    rel_w_view->setReshapeDimensions(makeDims({0, num_heads, q_h, q_w, 1, q_w}));

    auto *with_h  = requireLayer(network->addElementWise(*attn_view->getOutput(0), *rel_h_view->getOutput(0), E::kSUM),
                                 "Failed to add SAM rel_pos_h to attention");
    auto *with_hw = requireLayer(network->addElementWise(*with_h->getOutput(0), *rel_w_view->getOutput(0), E::kSUM),
                                 "Failed to add SAM rel_pos_w to attention");
    auto *scores  = requireLayer(network->addShuffle(*with_hw->getOutput(0)),
                                 "Failed to flatten SAM relative position attention");
    scores->setReshapeDimensions(nvinfer1::Dims4{0, num_heads, q_h * q_w, q_h * q_w});
    return scores->getOutput(0);
}

nvinfer1::ITensor *addImageAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &tokens, const std::string &prefix, int batch, int height,
                                     int width, const SAMViTSpec &spec)
{
    const int token_count = height * width;
    const int head_dim    = spec.embed_dim / spec.num_heads;
    auto     *qkv = addLinear3D(network, weights_map, tokens, prefix + ".attn.qkv", spec.embed_dim, 3 * spec.embed_dim);
    nvinfer1::ITensor *q = nullptr;
    nvinfer1::ITensor *k = nullptr;
    nvinfer1::ITensor *v = nullptr;
    splitQkv(network, *qkv, batch, token_count, spec.embed_dim, q, k, v);
    auto *q_heads = reshapeToHeads(network, *q, batch, token_count, spec.num_heads, head_dim);
    auto *k_heads = reshapeToHeads(network, *k, batch, token_count, spec.num_heads, head_dim);
    auto *v_heads = reshapeToHeads(network, *v, batch, token_count, spec.num_heads, head_dim);

    auto *qk           = requireLayer(network->addMatrixMultiply(*q_heads, M::kNONE, *k_heads, M::kTRANSPOSE),
                                      "Failed to add SAM image attention qk");
    auto *scale        = addScalar(network, *qk->getOutput(0), 1.0F / std::sqrt(static_cast<float>(head_dim)));
    auto *scaled       = requireLayer(network->addElementWise(*qk->getOutput(0), *scale, E::kPROD),
                                      "Failed to add SAM image attention scale");
    auto *with_rel_pos = addImageRelativePosition(network, weights_map, *scaled->getOutput(0), *q_heads, prefix, batch,
                                                  height, width, spec.num_heads, head_dim);
    auto *softmax      = requireLayer(network->addSoftMax(*with_rel_pos), "Failed to add SAM image attention softmax");
    softmax->setAxes(1U << 3);
    auto *attended = requireLayer(network->addMatrixMultiply(*softmax->getOutput(0), M::kNONE, *v_heads, M::kNONE),
                                  "Failed to add SAM image attention value matmul");
    auto *attn     = mergeHeads(network, *attended->getOutput(0), batch, token_count, spec.embed_dim);
    return addLinear3D(network, weights_map, *attn, prefix + ".attn.proj", spec.embed_dim, spec.embed_dim);
}

nvinfer1::ITensor *padNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch, int height,
                           int width, int channels, int padded_h, int padded_w)
{
    (void)batch;
    if (height == padded_h && width == padded_w)
    {
        return &input;
    }
    auto *slice
        = requireLayer(network->addSlice(input, nvinfer1::Dims4{0, 0, 0, 0},
                                         nvinfer1::Dims4{1, padded_h, padded_w, channels}, nvinfer1::Dims4{1, 1, 1, 1}),
                       "Failed to add SAM window padding");
    slice->setInput(2, *shapeWithFirstDimOf(network, input, {padded_h, padded_w, channels}));
    slice->setMode(nvinfer1::SampleMode::kFILL);
    auto *zero = requireLayer(network->addConstant(nvinfer1::Dims4{1, 1, 1, 1}, ownedScalarWeight(0.0F)),
                              "Failed to add SAM padding zero");
    slice->setInput(4, *zero->getOutput(0));
    return slice->getOutput(0);
}

nvinfer1::ITensor *partitionWindows(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input, int batch,
                                    int height, int width, int channels, int window_size, int &padded_h, int &padded_w)
{
    padded_h     = ((height + window_size - 1) / window_size) * window_size;
    padded_w     = ((width + window_size - 1) / window_size) * window_size;
    auto *padded = padNHWC(network, input, batch, height, width, channels, padded_h, padded_w);

    const int windows_h = padded_h / window_size;
    const int windows_w = padded_w / window_size;
    auto     *view      = requireLayer(network->addShuffle(*padded), "Failed to add SAM window view");
    view->setReshapeDimensions(makeDims({0, windows_h, window_size, windows_w, window_size, channels}));
    view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *windows = requireLayer(network->addShuffle(*view->getOutput(0)), "Failed to add SAM window flatten");
    windows->setReshapeDimensions(nvinfer1::Dims4{-1, window_size, window_size, channels});
    return windows->getOutput(0);
}

nvinfer1::ITensor *unpartitionWindows(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &windows, int batch,
                                      int height, int width, int channels, int window_size, int padded_h, int padded_w)
{
    (void)batch;
    const int windows_h = padded_h / window_size;
    const int windows_w = padded_w / window_size;
    auto     *view      = requireLayer(network->addShuffle(windows), "Failed to add SAM window restore view");
    view->setReshapeDimensions(makeDims({-1, windows_h, windows_w, window_size, window_size, channels}));
    view->setSecondTranspose(nvinfer1::Permutation{0, 1, 3, 2, 4, 5});

    auto *merged = requireLayer(network->addShuffle(*view->getOutput(0)), "Failed to add SAM window restore merge");
    merged->setReshapeDimensions(nvinfer1::Dims4{0, padded_h, padded_w, channels});
    if (height == padded_h && width == padded_w)
    {
        return merged->getOutput(0);
    }
    auto *crop
        = requireLayer(network->addSlice(*merged->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                                         nvinfer1::Dims4{1, height, width, channels}, nvinfer1::Dims4{1, 1, 1, 1}),
                       "Failed to add SAM window crop");
    crop->setInput(2, *shapeWithFirstDimOf(network, *merged->getOutput(0), {height, width, channels}));
    return crop->getOutput(0);
}

nvinfer1::ITensor *addImageBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, int index, const SAMGeometry &geometry,
                                 const SAMViTSpec &spec)
{
    const std::string prefix = "image_encoder.blocks." + std::to_string(index);
    auto             *norm1  = addLayerNormLastDim(network, weights_map, input, prefix + ".norm1", spec.embed_dim);

    nvinfer1::ITensor *attn_out = nullptr;
    if (spec.global_attn_indexes.count(index) != 0)
    {
        auto *tokens = flattenNHWC(network, *norm1, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
        auto *attn_tokens = addImageAttention(network, weights_map, *tokens, prefix, geometry.batch, geometry.grid_h,
                                              geometry.grid_w, spec);
        attn_out
            = unflattenNHWC(network, *attn_tokens, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    }
    else
    {
        int       padded_h     = 0;
        int       padded_w     = 0;
        auto     *windows      = partitionWindows(network, *norm1, geometry.batch, geometry.grid_h, geometry.grid_w,
                                                  spec.embed_dim, 14, padded_h, padded_w);
        const int window_batch = (padded_h / 14) * (padded_w / 14) * geometry.batch;
        auto     *tokens       = flattenNHWC(network, *windows, window_batch, 14, 14, spec.embed_dim);
        auto     *attn_tokens  = addImageAttention(network, weights_map, *tokens, prefix, window_batch, 14, 14, spec);
        auto     *attn_windows = unflattenNHWC(network, *attn_tokens, window_batch, 14, 14, spec.embed_dim);
        attn_out = unpartitionWindows(network, *attn_windows, geometry.batch, geometry.grid_h, geometry.grid_w,
                                      spec.embed_dim, 14, padded_h, padded_w);
    }

    auto *attn_residual
        = requireLayer(network->addElementWise(input, *attn_out, E::kSUM), "Failed to add SAM image attention residual")
              ->getOutput(0);

    auto *norm2  = addLayerNormLastDim(network, weights_map, *attn_residual, prefix + ".norm2", spec.embed_dim);
    auto *tokens = flattenNHWC(network, *norm2, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    auto *fc1    = addLinear3D(network, weights_map, *tokens, prefix + ".mlp.lin1", spec.embed_dim, spec.embed_dim * 4);
    auto *gelu   = addGeluExact(network, *fc1);
    auto *fc2    = addLinear3D(network, weights_map, *gelu, prefix + ".mlp.lin2", spec.embed_dim * 4, spec.embed_dim);
    auto *mlp    = unflattenNHWC(network, *fc2, geometry.batch, geometry.grid_h, geometry.grid_w, spec.embed_dim);
    return requireLayer(network->addElementWise(*attn_residual, *mlp, E::kSUM), "Failed to add SAM image MLP residual")
        ->getOutput(0);
}

bool containsIndex(const std::vector<int> &values, int target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

nvinfer1::ITensor *addLinearNHWC(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                 nvinfer1::ITensor &input, const std::string &prefix, int batch, int height, int width,
                                 int in_channels, int out_channels)
{
    auto *tokens = flattenNHWC(network, input, batch, height, width, in_channels);
    auto *linear = addLinear3D(network, weights_map, *tokens, prefix, in_channels, out_channels);
    return unflattenNHWC(network, *linear, batch, height, width, out_channels);
}

nvinfer1::ITensor *addMaxPoolNHWC(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input)
{
    auto *nchw = nhwcToNchw(network, input);
    auto *pool = requireLayer(network->addPoolingNd(*nchw, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2}),
                              "Failed to add SAM2 Hiera max pool");
    pool->setStrideNd(nvinfer1::DimsHW{2, 2});
    return nchwToNhwc(network, *pool->getOutput(0));
}

void setResizeOutputLike(nvinfer1::INetworkDefinition *network, nvinfer1::IResizeLayer &resize,
                         nvinfer1::ITensor &input, const nvinfer1::ITensor &reference)
{
    auto       dims     = input.getDimensions();
    const auto ref_dims = reference.getDimensions();
    if (dims.nbDims != 4 || ref_dims.nbDims != 4)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "SAM resize expects NCHW tensors");
    }

    dims.d[2] = ref_dims.d[2];
    dims.d[3] = ref_dims.d[3];
    if (dims.d[0] < 0)
    {
        resize.setInput(1, *shapeWithFirstDimOf(network, input, {dims.d[1], dims.d[2], dims.d[3]}));
        return;
    }
    resize.setOutputDimensions(dims);
}

nvinfer1::ITensor *addNearestResizeLike(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                        const nvinfer1::ITensor &reference)
{
    auto *resize = requireLayer(network->addResize(input), "Failed to add SAM2 nearest resize");
    resize->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    setResizeOutputLike(network, *resize, input, reference);
    return resize->getOutput(0);
}

nvinfer1::ITensor *addCubicResizeLike(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
                                      const nvinfer1::ITensor &reference)
{
    auto *resize = requireLayer(network->addResize(input), "Failed to add EdgeSAM bicubic resize");
    resize->setResizeMode(nvinfer1::InterpolationMode::kCUBIC);
    resize->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    resize->setCubicCoeff(-0.75F);
    setResizeOutputLike(network, *resize, input, reference);
    return resize->getOutput(0);
}

nvinfer1::ITensor *addSAM2HieraPositionEmbedding(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                                 const SAM2HieraSpec &spec, int grid_h, int grid_w)
{
    const std::string prefix = "image_encoder.trunk";
    auto             *base   = requireLayer(
        network->addConstant(
            nvinfer1::Dims4{1, spec.embed_dim, spec.pos_embed_size, spec.pos_embed_size},
            requireWeight(weights_map, prefix + ".pos_embed",
            checkedWeightProduct({spec.embed_dim, spec.pos_embed_size, spec.pos_embed_size},
                                 "SAM2 position embedding weight"))),
        "Failed to add SAM2 Hiera base position embedding");
    auto *resize = requireLayer(network->addResize(*base->getOutput(0)), "Failed to resize SAM2 Hiera pos_embed");
    resize->setResizeMode(nvinfer1::InterpolationMode::kCUBIC);
    resize->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    resize->setCubicCoeff(-0.75F);
    resize->setOutputDimensions(nvinfer1::Dims4{1, spec.embed_dim, grid_h, grid_w});

    const int          window_size   = spec.window_spec.front();
    const auto        &window_weight = requireWeight(
        weights_map, prefix + ".pos_embed_window",
        checkedWeightProduct({spec.embed_dim, window_size, window_size}, "SAM2 window position weight"));
    const auto        *window_values = static_cast<const float *>(window_weight.values);
    std::vector<float> tiled(irt::checkedSizeProduct({static_cast<size_t>(spec.embed_dim), static_cast<size_t>(grid_h),
                                                      static_cast<size_t>(grid_w)},
                                                     "SAM2 tiled position embedding"));
    for (int c = 0; c < spec.embed_dim; ++c)
    {
        for (int y = 0; y < grid_h; ++y)
        {
            for (int x = 0; x < grid_w; ++x)
            {
                const auto dst = (static_cast<size_t>(c) * grid_h + y) * grid_w + x;
                const auto src
                    = (static_cast<size_t>(c) * window_size + (y % window_size)) * window_size + (x % window_size);
                tiled[dst] = window_values[src];
            }
        }
    }
    auto *window = requireLayer(
        network->addConstant(nvinfer1::Dims4{1, spec.embed_dim, grid_h, grid_w}, ownedFloatVector(std::move(tiled))),
        "Failed to add SAM2 Hiera window position embedding");
    auto *sum = requireLayer(network->addElementWise(*resize->getOutput(0), *window->getOutput(0), E::kSUM),
                             "Failed to add SAM2 Hiera position embedding");
    return nchwToNhwc(network, *sum->getOutput(0));
}

nvinfer1::ITensor *addSAM2HieraAttention(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                         nvinfer1::ITensor &input, const std::string &prefix, int batch, int height,
                                         int width, int dim_in, int dim_out, int num_heads, bool q_pool, int &out_h,
                                         int &out_w)
{
    const int          token_count = height * width;
    auto              *tokens      = flattenNHWC(network, input, batch, height, width, dim_in);
    auto              *qkv = addLinear3D(network, weights_map, *tokens, prefix + ".attn.qkv", dim_in, 3 * dim_out);
    nvinfer1::ITensor *q   = nullptr;
    nvinfer1::ITensor *k   = nullptr;
    nvinfer1::ITensor *v   = nullptr;
    splitQkv(network, *qkv, batch, token_count, dim_out, q, k, v);

    out_h = height;
    out_w = width;
    if (q_pool)
    {
        auto *q_map    = unflattenNHWC(network, *q, batch, height, width, dim_out);
        auto *pooled_q = addMaxPoolNHWC(network, *q_map);
        out_h /= 2;
        out_w /= 2;
        q = flattenNHWC(network, *pooled_q, batch, out_h, out_w, dim_out);
    }

    auto *attn = addTokenAttention(network, *q, *k, *v, batch, out_h * out_w, token_count, num_heads, dim_out);
    auto *proj = addLinear3D(network, weights_map, *attn, prefix + ".attn.proj", dim_out, dim_out);
    return unflattenNHWC(network, *proj, batch, out_h, out_w, dim_out);
}

nvinfer1::ITensor *addSAM2HieraBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                     nvinfer1::ITensor &input, int index, int batch, int &height, int &width,
                                     int dim_in, int dim_out, int num_heads, int window_size, bool q_pool)
{
    const std::string prefix = "image_encoder.trunk.blocks." + std::to_string(index);
    auto             *norm1  = addLayerNormLastDim(network, weights_map, input, prefix + ".norm1", dim_in);

    nvinfer1::ITensor *shortcut = &input;
    if (dim_in != dim_out)
    {
        shortcut = addLinearNHWC(network, weights_map, *norm1, prefix + ".proj", batch, height, width, dim_in, dim_out);
        if (q_pool)
        {
            shortcut = addMaxPoolNHWC(network, *shortcut);
        }
    }

    nvinfer1::ITensor *attn_input = norm1;
    int                attn_batch = batch;
    int                attn_h     = height;
    int                attn_w     = width;
    int                padded_h   = height;
    int                padded_w   = width;
    if (window_size > 0)
    {
        attn_input = partitionWindows(network, *norm1, batch, height, width, dim_in, window_size, padded_h, padded_w);
        attn_batch = batch * (padded_h / window_size) * (padded_w / window_size);
        attn_h     = window_size;
        attn_w     = window_size;
    }

    int   out_h = attn_h;
    int   out_w = attn_w;
    auto *attn  = addSAM2HieraAttention(network, weights_map, *attn_input, prefix, attn_batch, attn_h, attn_w, dim_in,
                                        dim_out, num_heads, q_pool, out_h, out_w);
    if (window_size > 0)
    {
        const int restore_window = q_pool ? window_size / 2 : window_size;
        const int target_h       = q_pool ? height / 2 : height;
        const int target_w       = q_pool ? width / 2 : width;
        const int restore_h      = q_pool ? padded_h / 2 : padded_h;
        const int restore_w      = q_pool ? padded_w / 2 : padded_w;
        attn  = unpartitionWindows(network, *attn, batch, target_h, target_w, dim_out, restore_window, restore_h,
                                   restore_w);
        out_h = target_h;
        out_w = target_w;
    }

    auto *x = requireLayer(network->addElementWise(*shortcut, *attn, E::kSUM),
                           "Failed to add SAM2 Hiera attention residual")
                  ->getOutput(0);
    auto *norm2 = addLayerNormLastDim(network, weights_map, *x, prefix + ".norm2", dim_out);
    auto *mlp0  = addLinearNHWC(network, weights_map, *norm2, prefix + ".mlp.layers.0", batch, out_h, out_w, dim_out,
                                dim_out * 4);
    auto *gelu  = addGeluExact(network, *mlp0);
    auto *mlp1  = addLinearNHWC(network, weights_map, *gelu, prefix + ".mlp.layers.1", batch, out_h, out_w, dim_out * 4,
                                dim_out);
    height      = out_h;
    width       = out_w;
    return requireLayer(network->addElementWise(*x, *mlp1, E::kSUM), "Failed to add SAM2 Hiera MLP residual")
        ->getOutput(0);
}

std::vector<nvinfer1::ITensor *> addSAM2HieraTrunk(const SAMSegmentationModel   &impl,
                                                   nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                                   const SAMGeometry &geometry, const SAM2HieraSpec &spec,
                                                   priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *image            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 0);
    named_tensors["image"] = image;

    auto *patch
        = requireLayer(network->addConvolutionNd(
                           *image, spec.embed_dim, nvinfer1::DimsHW{7, 7},
                           requireWeight(weights_map, "image_encoder.trunk.patch_embed.proj.weight",
                                         checkedWeightProduct({spec.embed_dim, geometry.channels, 7, 7},
                                                              "SAM2 patch embedding weight")),
                           requireWeight(weights_map, "image_encoder.trunk.patch_embed.proj.bias", spec.embed_dim)),
                       "Failed to add SAM2 Hiera patch embedding");
    patch->setStrideNd(nvinfer1::DimsHW{kSam2PatchSize, kSam2PatchSize});
    patch->setPaddingNd(nvinfer1::DimsHW{3, 3});

    int   height = geometry.image_h / kSam2PatchSize;
    int   width  = geometry.image_w / kSam2PatchSize;
    auto *x      = nchwToNhwc(network, *patch->getOutput(0));
    auto *pos    = addSAM2HieraPositionEmbedding(network, weights_map, spec, height, width);
    x = requireLayer(network->addElementWise(*x, *pos, E::kSUM), "Failed to add SAM2 Hiera position embedding")
            ->getOutput(0);

    std::vector<int> stage_ends;
    stage_ends.reserve(spec.stages.size());
    int depth = 0;
    for (const int blocks : spec.stages)
    {
        depth += blocks;
        stage_ends.push_back(depth - 1);
    }
    std::vector<int> q_pool_blocks;
    for (size_t i = 0; i + 1 < stage_ends.size() && static_cast<int>(q_pool_blocks.size()) < spec.q_pool; ++i)
    {
        q_pool_blocks.push_back(stage_ends[i] + 1);
    }

    std::vector<nvinfer1::ITensor *> outputs;
    int                              cur_stage = 1;
    int                              dim       = spec.embed_dim;
    int                              heads     = spec.num_heads;
    for (int i = 0; i < depth; ++i)
    {
        int window_size = spec.window_spec.at(static_cast<size_t>(cur_stage - 1));
        if (spec.global_attn_indexes.count(i) != 0)
        {
            window_size = 0;
        }

        int dim_out = dim;
        if (containsIndex(stage_ends, i - 1))
        {
            dim_out = dim * 2;
            heads *= 2;
            ++cur_stage;
        }
        const bool q_pool = containsIndex(q_pool_blocks, i);
        x   = addSAM2HieraBlock(network, weights_map, *x, i, geometry.batch, height, width, dim, dim_out, heads,
                                window_size, q_pool);
        dim = dim_out;
        named_tensors["image_encoder.trunk.blocks." + std::to_string(i)] = x;
        if (containsIndex(stage_ends, i))
        {
            auto *stage = nhwcToNchw(network, *x);
            outputs.push_back(stage);
            named_tensors["image_encoder.trunk.stage" + std::to_string(outputs.size())] = stage;
        }
    }
    return outputs;
}

int makeDivisibleBy8(int value)
{
    constexpr int divisor = 8;
    int           result  = std::max(divisor, (value + divisor / 2) / divisor * divisor);
    if (result < static_cast<int>(0.9F * static_cast<float>(value)))
    {
        result += divisor;
    }
    return result;
}

nvinfer1::ITensor *addEdgeSAMSE(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    const int reduced_channels = makeDivisibleBy8(static_cast<int>(static_cast<float>(channels) * 0.25F));
    auto *pool = requireLayer(network->addReduce(input, nvinfer1::ReduceOperation::kAVG, (1U << 2) | (1U << 3), true),
                              "Failed to add EdgeSAM SE global average pooling");
    auto *fc1  = requireLayer(
        network->addConvolutionNd(
            *pool->getOutput(0), reduced_channels, nvinfer1::DimsHW{1, 1},
            requireWeight(weights_map, prefix + ".fc1.weight",
                          checkedWeightProduct({reduced_channels, channels}, "EdgeSAM SE fc1 weight")),
            requireWeight(weights_map, prefix + ".fc1.bias", reduced_channels)),
        "Failed to add EdgeSAM SE fc1");
    auto *relu = requireLayer(network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU),
                              "Failed to add EdgeSAM SE ReLU");
    auto *fc2  = requireLayer(network->addConvolutionNd(*relu->getOutput(0), channels, nvinfer1::DimsHW{1, 1},
                                                        requireWeight(weights_map, prefix + ".fc2.weight",
                                                                      checkedWeightProduct({channels, reduced_channels},
                                                                                           "EdgeSAM SE fc2 weight")),
                                                        requireWeight(weights_map, prefix + ".fc2.bias", channels)),
                              "Failed to add EdgeSAM SE fc2");
    auto *gate = requireLayer(network->addActivation(*fc2->getOutput(0), nvinfer1::ActivationType::kSIGMOID),
                              "Failed to add EdgeSAM SE sigmoid");
    return requireLayer(network->addElementWise(input, *gate->getOutput(0), E::kPROD),
                        "Failed to apply EdgeSAM SE gate")
        ->getOutput(0);
}

nvinfer1::ITensor *addEdgeSAMRepVGGDW(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                      nvinfer1::ITensor &input, const std::string &prefix, int channels)
{
    auto *conv3
        = addEdgeSAMConvBN(network, weights_map, input, prefix + ".conv", channels, channels, 3, 1, 1, channels);
    auto *conv1
        = addEdgeSAMConvBN(network, weights_map, input, prefix + ".conv1", channels, channels, 1, 1, 0, channels);
    auto *sum = requireLayer(network->addElementWise(*conv3, *conv1, E::kSUM),
                             "Failed to add EdgeSAM RepVGG depthwise branches");
    return requireLayer(network->addElementWise(*sum->getOutput(0), input, E::kSUM),
                        "Failed to add EdgeSAM RepVGG identity")
        ->getOutput(0);
}

nvinfer1::ITensor *addEdgeSAMChannelMixer(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                          nvinfer1::ITensor &input, const std::string &prefix, int channels,
                                          int hidden_channels)
{
    auto *expand  = addEdgeSAMConvBN(network, weights_map, input, prefix + ".channel_mixer.m.0", channels,
                                     hidden_channels, 1, 1, 0);
    auto *gelu    = addGeluExact(network, *expand);
    auto *project = addEdgeSAMConvBN(network, weights_map, *gelu, prefix + ".channel_mixer.m.2", hidden_channels,
                                     channels, 1, 1, 0);
    return requireLayer(network->addElementWise(input, *project, E::kSUM),
                        "Failed to add EdgeSAM channel mixer residual")
        ->getOutput(0);
}

nvinfer1::ITensor *addEdgeSAMRepViTBlock(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map,
                                         nvinfer1::ITensor &input, int feature_index, int input_channels,
                                         const EdgeSAMRepViTBlockSpec &spec)
{
    const std::string prefix          = "image_encoder.features." + std::to_string(feature_index);
    const int         output_channels = makeDivisibleBy8(spec.out_channels);
    const int         hidden_channels = makeDivisibleBy8(output_channels * spec.expansion);

    nvinfer1::ITensor *mixed = nullptr;
    if (spec.stride == 2)
    {
        mixed = addEdgeSAMConvBN(network, weights_map, input, prefix + ".token_mixer.0", input_channels, input_channels,
                                 spec.kernel_size, spec.stride, (spec.kernel_size - 1) / 2, input_channels);
        if (spec.use_se)
        {
            mixed = addEdgeSAMSE(network, weights_map, *mixed, prefix + ".token_mixer.1", input_channels);
        }
        mixed = addEdgeSAMConvBN(network, weights_map, *mixed, prefix + ".token_mixer.2", input_channels,
                                 output_channels, 1, 1, 0);
    }
    else
    {
        if (input_channels != output_channels)
        {
            throw irt::Exception(Status::ERROR_INTERNAL, "EdgeSAM stride=1 RepViT block requires identity channels");
        }
        mixed = addEdgeSAMRepVGGDW(network, weights_map, input, prefix + ".token_mixer.0", input_channels);
        if (spec.use_se)
        {
            mixed = addEdgeSAMSE(network, weights_map, *mixed, prefix + ".token_mixer.1", input_channels);
        }
    }
    return addEdgeSAMChannelMixer(network, weights_map, *mixed, prefix, output_channels, hidden_channels);
}

} // namespace

const std::vector<EdgeSAMRepViTBlockSpec> &edgeSAMRepViTM1Blocks()
{
    static const std::vector<EdgeSAMRepViTBlockSpec> blocks{
        {3, 2,  48,  true, 1},
        {3, 2,  48, false, 1},
        {3, 2,  48, false, 1},
        {3, 2,  96, false, 2},
        {3, 2,  96,  true, 1},
        {3, 2,  96, false, 1},
        {3, 2,  96, false, 1},
        {3, 2, 192, false, 2},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192,  true, 1},
        {3, 2, 192, false, 1},
        {3, 2, 192, false, 1},
        {3, 2, 384, false, 2},
        {3, 2, 384,  true, 1},
        {3, 2, 384, false, 1},
    };
    return blocks;
}

nvinfer1::ITensor *addSAMImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                      const WeightsMap &weights_map, const SAMGeometry &geometry,
                                      const SAMViTSpec &spec, priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *image            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 0);
    named_tensors["image"] = image;

    auto *patch = requireLayer(
        network->addConvolutionNd(
            *image, spec.embed_dim, nvinfer1::DimsHW{kSamPatchSize, kSamPatchSize},
            requireWeight(weights_map, "image_encoder.patch_embed.proj.weight",
                          checkedWeightProduct({spec.embed_dim, geometry.channels, kSamPatchSize, kSamPatchSize},
                                               "SAM patch embedding weight")),
            requireWeight(weights_map, "image_encoder.patch_embed.proj.bias", spec.embed_dim)),
        "Failed to add SAM patch embedding");
    patch->setStrideNd(nvinfer1::DimsHW{kSamPatchSize, kSamPatchSize});
    auto *x = nchwToNhwc(network, *patch->getOutput(0));

    auto *pos
        = requireLayer(network->addConstant(nvinfer1::Dims4{1, geometry.grid_h, geometry.grid_w, spec.embed_dim},
                                            requireWeight(weights_map, "image_encoder.pos_embed",
                                                          checkedWeightProduct({geometry.grid_tokens, spec.embed_dim},
                                                                               "SAM position embedding weight"))),
                       "Failed to add SAM image pos_embed");
    x = requireLayer(network->addElementWise(*x, *pos->getOutput(0), E::kSUM), "Failed to add SAM image pos embedding")
            ->getOutput(0);
    named_tensors["image_tokens"] = x;

    for (int i = 0; i < spec.depth; ++i)
    {
        x = addImageBlock(network, weights_map, *x, i, geometry, spec);
        named_tensors["image_encoder.block" + std::to_string(i)] = x;
    }

    auto *nchw = nhwcToNchw(network, *x);
    auto *neck0
        = requireLayer(network->addConvolutionNd(*nchw, kSamPromptDim, nvinfer1::DimsHW{1, 1},
                                                 requireWeight(weights_map, "image_encoder.neck.0.weight",
                                                               checkedWeightProduct({kSamPromptDim, spec.embed_dim},
                                                                                    "SAM neck weight")),
                                                 emptyWeights()),
                       "Failed to add SAM neck conv0");
    auto *neck1 = addLayerNorm2d(network, weights_map, *neck0->getOutput(0), "image_encoder.neck.1", kSamPromptDim);
    auto *neck2 = requireLayer(
        network->addConvolutionNd(*neck1, kSamPromptDim, nvinfer1::DimsHW{3, 3},
                                  requireWeight(weights_map, "image_encoder.neck.2.weight",
                                                checkedWeightProduct({kSamPromptDim, kSamPromptDim, 3, 3},
                                                                     "SAM neck convolution weight")),
                                  emptyWeights()),
        "Failed to add SAM neck conv1");
    neck2->setPaddingNd(nvinfer1::DimsHW{1, 1});
    auto *embedding = addLayerNorm2d(network, weights_map, *neck2->getOutput(0), "image_encoder.neck.3", kSamPromptDim);
    named_tensors["image_embedding"] = embedding;
    return embedding;
}

nvinfer1::ITensor *addSAM2ImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                       const WeightsMap &weights_map, const SAMGeometry &geometry,
                                       const SAM2HieraSpec &spec, nvinfer1::ITensor *&high_res_s0,
                                       nvinfer1::ITensor *&high_res_s1, priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto trunk_outputs = addSAM2HieraTrunk(impl, network, weights_map, geometry, spec, named_tensors);
    if (trunk_outputs.size() != 4 || spec.backbone_channels.size() != 4)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "SAM2 Hiera expects four FPN levels");
    }

    std::vector<nvinfer1::ITensor *> fpn(4, nullptr);
    nvinfer1::ITensor               *prev = nullptr;
    const int                        n    = static_cast<int>(trunk_outputs.size()) - 1;
    for (int i = n; i >= 0; --i)
    {
        const int conv_index = n - i;
        const int channels   = spec.backbone_channels.at(static_cast<size_t>(conv_index));
        auto     *lateral    = requireLayer(
            network->addConvolutionNd(
                *trunk_outputs.at(static_cast<size_t>(i)), kSamPromptDim, nvinfer1::DimsHW{1, 1},
                requireWeight(weights_map, "image_encoder.neck.convs." + std::to_string(conv_index) + ".conv.weight",
                              checkedWeightProduct({kSamPromptDim, channels}, "SAM2 FPN projection weight")),
                requireWeight(weights_map, "image_encoder.neck.convs." + std::to_string(conv_index) + ".conv.bias",
                              kSamPromptDim)),
            "Failed to add SAM2 FPN lateral conv");
        nvinfer1::ITensor *out = lateral->getOutput(0);
        if ((i == 2 || i == 3) && prev != nullptr)
        {
            auto *top_down = addNearestResizeLike(network, *prev, *out);
            out            = requireLayer(network->addElementWise(*out, *top_down, E::kSUM),
                                          "Failed to add SAM2 FPN top-down feature")
                      ->getOutput(0);
        }
        fpn.at(static_cast<size_t>(i))                          = out;
        prev                                                    = out;
        named_tensors["image_encoder.fpn." + std::to_string(i)] = out;
    }

    auto *conv_s0 = requireLayer(
        network->addConvolutionNd(*fpn[0], kSamPromptDim / 8, nvinfer1::DimsHW{1, 1},
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s0.weight",
                                                checkedWeightProduct({kSamPromptDim / 8, kSamPromptDim},
                                                                     "SAM2 high-resolution s0 weight")),
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s0.bias", kSamPromptDim / 8)),
        "Failed to add SAM2 high-res s0 projection");
    auto *conv_s1 = requireLayer(
        network->addConvolutionNd(*fpn[1], kSamPromptDim / 4, nvinfer1::DimsHW{1, 1},
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s1.weight",
                                                checkedWeightProduct({kSamPromptDim / 4, kSamPromptDim},
                                                                     "SAM2 high-resolution s1 weight")),
                                  requireWeight(weights_map, "sam_mask_decoder.conv_s1.bias", kSamPromptDim / 4)),
        "Failed to add SAM2 high-res s1 projection");
    high_res_s0 = conv_s0->getOutput(0);
    high_res_s1 = conv_s1->getOutput(0);

    auto *no_mem_embed    = requireLayer(network->addConstant(nvinfer1::Dims4{1, kSamPromptDim, 1, 1},
                                                              requireWeight(weights_map, "no_mem_embed", kSamPromptDim)),
                                         "Failed to add SAM2 no-memory embedding");
    auto *image_embedding = requireLayer(network->addElementWise(*fpn[2], *no_mem_embed->getOutput(0), E::kSUM),
                                         "Failed to add SAM2 no-memory embedding to image embedding")
                                ->getOutput(0);

    named_tensors["high_res_s0"]     = high_res_s0;
    named_tensors["high_res_s1"]     = high_res_s1;
    named_tensors["image_embedding"] = image_embedding;
    return image_embedding;
}

nvinfer1::ITensor *addEdgeSAMImageEncoder(const SAMSegmentationModel &impl, nvinfer1::INetworkDefinition *network,
                                          const WeightsMap &weights_map, const SAMGeometry &geometry,
                                          priv::IModelImpl::NamedTensorMap &named_tensors)
{
    auto *image            = impl.addInputTensor(network, nvinfer1::DataType::kFLOAT, 0);
    named_tensors["image"] = image;

    const auto &blocks   = edgeSAMRepViTM1Blocks();
    int         channels = makeDivisibleBy8(blocks.front().out_channels);
    auto       *x = addEdgeSAMConvBN(network, weights_map, *image, "image_encoder.features.0.0", geometry.channels,
                                     channels / 2, 3, 2, 1);
    x             = addGeluExact(network, *x);
    x = addEdgeSAMConvBN(network, weights_map, *x, "image_encoder.features.0.2", channels / 2, channels, 3, 2, 1);
    named_tensors["image_encoder.stem"] = x;

    nvinfer1::ITensor *stage2 = nullptr;
    nvinfer1::ITensor *stage3 = nullptr;
    int                stage  = 0;
    for (size_t i = 0; i < blocks.size(); ++i)
    {
        x        = addEdgeSAMRepViTBlock(network, weights_map, *x, static_cast<int>(i) + 1, channels, blocks[i]);
        channels = makeDivisibleBy8(blocks[i].out_channels);
        named_tensors["image_encoder.features." + std::to_string(i + 1)] = x;

        const bool is_stage_end = i + 1 == blocks.size() || blocks[i + 1].out_channels != blocks[i].out_channels;
        if (is_stage_end)
        {
            const auto stage_name     = "image_encoder.stage" + std::to_string(stage);
            named_tensors[stage_name] = x;
            if (stage == 2)
            {
                stage2 = x;
            }
            else if (stage == 3)
            {
                stage3 = x;
            }
            ++stage;
        }
    }
    if (stage2 == nullptr || stage3 == nullptr)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "EdgeSAM RepViT-M1 expected stage2 and stage3 features");
    }

    auto *fuse_stage2 = addEdgeSAMConvNoBias(network, weights_map, *stage2, "image_encoder.fuse_stage2.weight", 192,
                                             kSamPromptDim, 1);
    auto *fuse_stage3_conv = addEdgeSAMConvNoBias(network, weights_map, *stage3,
                                                  "image_encoder.fuse_stage3.op_list.0.weight", 384, kSamPromptDim, 1);
    auto *fuse_stage3      = addCubicResizeLike(network, *fuse_stage3_conv, *fuse_stage2);
    auto *fused            = requireLayer(network->addElementWise(*fuse_stage2, *fuse_stage3, E::kSUM),
                                          "Failed to add EdgeSAM fused RepViT features")
                      ->getOutput(0);
    named_tensors["image_encoder.fused_features"] = fused;

    auto *neck0     = addEdgeSAMConvNoBias(network, weights_map, *fused, "image_encoder.neck.0.weight", kSamPromptDim,
                                           kSamPromptDim, 1);
    auto *neck1     = addLayerNorm2d(network, weights_map, *neck0, "image_encoder.neck.1", kSamPromptDim);
    auto *neck2     = addEdgeSAMConvNoBias(network, weights_map, *neck1, "image_encoder.neck.2.weight", kSamPromptDim,
                                           kSamPromptDim, 3, 1, 1);
    auto *embedding = addLayerNorm2d(network, weights_map, *neck2, "image_encoder.neck.3", kSamPromptDim);
    named_tensors["image_embedding"] = embedding;
    return embedding;
}

} // namespace irt::model
