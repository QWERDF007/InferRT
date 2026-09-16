/**
 * @file DinoIndexStore.cpp
 * @brief 根目录索引数组写入与读取实现。
 */

#include "DinoIndexStore.hpp"
#include "DinoBackboneMetadata.hpp"
#include <yaml-cpp/yaml.h>

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <cmath>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features::priv {

namespace {

constexpr const char *kIndexFile = "index.yaml";
constexpr const char *kViewsFile      = "views.npy";
constexpr const char *kOffsetsFile    = "offsets.npy";
constexpr const char *kRegionVectors  = "region_vectors.i8";
constexpr const char *kRegionScales   = "region_scales.f32";
constexpr const char *kRegionMeta     = "region_meta.npy";
constexpr const char *kLocalVectors   = "local_vectors.i8";
constexpr const char *kLocalScales    = "local_scales.f32";
constexpr const char *kLocalMeta = "local_meta.npy";


constexpr size_t kViewTableColumns = 18U;

DinoPackedMeta encodeRegionMeta(const DinoRegionDescriptor &descriptor)
{
    DinoPackedMeta meta;
    meta.view_id          = static_cast<uint32_t>(descriptor.view_id);
    meta.x0               = static_cast<uint16_t>(descriptor.grid_col);
    meta.y0               = static_cast<uint16_t>(descriptor.grid_row);
    meta.x1               = static_cast<uint16_t>(descriptor.grid_col + descriptor.grid_width);
    meta.y1               = static_cast<uint16_t>(descriptor.grid_row + descriptor.grid_height);
    meta.representative_x = 0;
    meta.representative_y = 0;
    meta.member_count     = 0;
    meta.flags            = 0;
    meta.radius           = 0.0F;
    float valid_fraction  = descriptor.valid_fraction;
    std::memcpy(&meta.reserved0, &valid_fraction, sizeof(valid_fraction));
    meta.reserved1 = 0; // 0 = 区域窗口
    return meta;
}

DinoPackedMeta encodeLocalMeta(const DinoLocalLeaf &leaf)
{
    DinoPackedMeta meta;
    meta.view_id          = static_cast<uint32_t>(leaf.view_id);
    meta.x0               = static_cast<uint16_t>(leaf.grid_col);
    meta.y0               = static_cast<uint16_t>(leaf.grid_row);
    meta.x1               = static_cast<uint16_t>(leaf.grid_col + leaf.grid_width);
    meta.y1               = static_cast<uint16_t>(leaf.grid_row + leaf.grid_height);
    meta.representative_x = static_cast<uint16_t>(leaf.rep_col);
    meta.representative_y = static_cast<uint16_t>(leaf.rep_row);
    meta.member_count     = static_cast<uint16_t>(leaf.member_count);
    meta.flags            = 0;
    meta.radius           = leaf.radius;
    meta.reserved0        = 0;
    meta.reserved1        = 1; // 1 = 局部叶节点
    return meta;
}

DinoRegionDescriptor decodeRegionMeta(const DinoPackedMeta &meta)
{
    DinoRegionDescriptor descriptor;
    descriptor.view_id      = static_cast<int>(meta.view_id);
    descriptor.grid_col     = meta.x0;
    descriptor.grid_row     = meta.y0;
    descriptor.grid_width   = meta.x1 >= meta.x0 ? meta.x1 - meta.x0 : 0;
    descriptor.grid_height  = meta.y1 >= meta.y0 ? meta.y1 - meta.y0 : 0;
    float valid_fraction    = 0.0F;
    std::memcpy(&valid_fraction, &meta.reserved0, sizeof(valid_fraction));
    descriptor.valid_fraction = valid_fraction;
    return descriptor;
}

DinoLocalLeaf decodeLocalMeta(const DinoPackedMeta &meta)
{
    DinoLocalLeaf leaf;
    leaf.view_id      = static_cast<int>(meta.view_id);
    leaf.grid_row     = meta.y0;
    leaf.grid_col     = meta.x0;
    leaf.grid_height  = meta.y1 >= meta.y0 ? meta.y1 - meta.y0 : 0;
    leaf.grid_width   = meta.x1 >= meta.x0 ? meta.x1 - meta.x0 : 0;
    leaf.rep_row      = meta.representative_y;
    leaf.rep_col      = meta.representative_x;
    leaf.member_count = meta.member_count;
    leaf.radius       = meta.radius;
    return leaf;
}

void validateDescriptorRect(const DinoIndexView &view, const int row, const int col, const int height,
                            const int width, const char *label)
{
    if (row < 0 || col < 0 || height <= 0 || width <= 0
        || static_cast<int64_t>(row) + static_cast<int64_t>(height) > view.grid_height
        || static_cast<int64_t>(col) + static_cast<int64_t>(width) > view.grid_width)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s metadata lies outside its view grid", label);
    }
}


void writeQuantizedVector(std::ofstream &vector_stream, std::ofstream &scale_stream, const std::vector<float> &vector,
                          uint64_t &byte_accounting, float &max_delta, double &delta_sum, uint64_t &delta_samples)
{
    const auto quantized = dinoQuantizeInt8(vector.data(), vector.size());
    const auto delta     = dinoQuantizationDelta(vector.data(), quantized);
    max_delta            = std::max(max_delta, delta);
    delta_sum += delta;
    ++delta_samples;

    vector_stream.write(reinterpret_cast<const char *>(quantized.codes.data()),
                        static_cast<std::streamsize>(quantized.codes.size()));
    scale_stream.write(reinterpret_cast<const char *>(&quantized.scale), sizeof(float));
    scale_stream.write(reinterpret_cast<const char *>(&quantized.inv_norm), sizeof(float));
    byte_accounting += quantized.codes.size() + 2U * sizeof(float) + kDinoMetaRecordBytes;
}

void writeRawVector(std::ofstream &vector_stream, std::ofstream &scale_stream, const std::vector<float> &vector,
                    uint64_t &byte_accounting)
{
    vector_stream.write(reinterpret_cast<const char *>(vector.data()),
                        static_cast<std::streamsize>(vector.size() * sizeof(float)));
    // FP32 参考路径不量化，保留中性因子以共用数组布局。
    const float identity = 1.0F;
    scale_stream.write(reinterpret_cast<const char *>(&identity), sizeof(float));
    scale_stream.write(reinterpret_cast<const char *>(&identity), sizeof(float));
    byte_accounting += vector.size() * sizeof(float) + 2U * sizeof(float) + kDinoMetaRecordBytes;
}


fs::path resolveIndexRoot(const fs::path &index_root)
{
    if (index_root.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index root must not be empty");
    }
    return fs::absolute(index_root).lexically_normal();
}

} // namespace

DinoIndexContract DinoIndexContract::fromConfig(const DinoRegionSearchConfig &config)
{
    DinoIndexContract contract;
    contract.schema_version = "1.0";
    contract.feature_version = "v4_hadamard_sparse_fine";
    contract.model_name = config.model.model_name;
    contract.weights_id = config.model.weights_id;
    const auto backbone = dinoDescribeBackbone(config);
    contract.patch_size = backbone.patch_size;
    contract.token_dimension = backbone.channels;
    contract.encoder_edge = backbone.encoder_edge;
    contract.gallery_tile_edges = config.gallery_views.gallery_tile_edges;
    contract.view_overlap = config.gallery_views.view_overlap;
    contract.region_window_ratios = config.descriptors.region_window_ratios;
    contract.region_window_stride_ratio = config.descriptors.region_window_stride_ratio;
    contract.region_min_valid_fraction = config.descriptors.region_min_valid_fraction;
    contract.coarse_dimension = config.descriptors.coarse_dimension;
    contract.local_representatives = config.descriptors.local_representatives;
    contract.merge_enabled = config.descriptors.merge_enabled;
    contract.merge_epsilon = config.descriptors.merge_epsilon;
    contract.max_leaf_side_patches = config.descriptors.max_leaf_side_patches;
    contract.quantize_int8 = config.descriptors.quantize_int8;
    contract.preprocess_description = backbone.preprocess;
    return contract;
}

YAML::Node DinoIndexContract::toYamlNode() const
{
    YAML::Node node;
    node["schema_version"] = schema_version;
    node["feature_version"] = feature_version;
    node["model_name"] = model_name;
    node["weights_id"] = weights_id;
    node["encoder_edge"] = encoder_edge;
    node["patch_size"] = patch_size;
    node["token_dimension"] = token_dimension;
    node["gallery_tile_edges"] = gallery_tile_edges;
    node["view_overlap"] = view_overlap;
    node["region_window_ratios"] = region_window_ratios;
    node["region_window_stride_ratio"] = region_window_stride_ratio;
    node["region_min_valid_fraction"] = region_min_valid_fraction;
    node["coarse_dimension"] = coarse_dimension;
    node["local_representatives"] = local_representatives;
    node["merge_enabled"] = merge_enabled;
    node["merge_epsilon"] = merge_epsilon;
    node["max_leaf_side_patches"] = max_leaf_side_patches;
    node["quantize_int8"] = quantize_int8;
    node["preprocess_description"] = preprocess_description;
    return node;
}

DinoIndexContract DinoIndexContract::fromYamlNode(const YAML::Node &node)
{
    if (!node || !node.IsMap())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index manifest must be a mapping");
    }
    DinoIndexContract c;
    c.schema_version = node["schema_version"].as<std::string>();
    c.feature_version = node["feature_version"].as<std::string>();
    c.model_name = node["model_name"].as<std::string>();
    c.weights_id = node["weights_id"].as<std::string>();
    c.encoder_edge = node["encoder_edge"].as<int>();
    c.patch_size = node["patch_size"].as<int>();
    c.token_dimension = node["token_dimension"].as<int>();
    c.gallery_tile_edges = node["gallery_tile_edges"].as<std::vector<int>>();
    c.view_overlap = node["view_overlap"].as<double>();
    c.region_window_ratios = node["region_window_ratios"].as<std::vector<double>>();
    c.region_window_stride_ratio = node["region_window_stride_ratio"].as<double>();
    c.region_min_valid_fraction = node["region_min_valid_fraction"].as<double>();
    c.coarse_dimension = node["coarse_dimension"].as<int>();
    c.local_representatives = node["local_representatives"].as<int>();
    c.merge_enabled = node["merge_enabled"].as<bool>();
    c.merge_epsilon = node["merge_epsilon"].as<double>();
    c.max_leaf_side_patches = node["max_leaf_side_patches"].as<int>();
    c.quantize_int8 = node["quantize_int8"].as<bool>();
    c.preprocess_description = node["preprocess_description"].as<std::string>();
    return c;
}

std::string DinoIndexContract::checkCompatibility(const DinoRegionSearchConfig &config) const
{
    const auto expected = fromConfig(config);
    if (*this == expected)
    {
        return {};
    }
    const auto actual_fields = toYamlNode();
    const auto expected_fields = expected.toYamlNode();
    for (const auto &field : expected_fields)
    {
        const auto name = field.first.as<std::string>();
        const auto actual_value = YAML::Dump(actual_fields[name]);
        const auto expected_value = YAML::Dump(field.second);
        if (actual_value != expected_value)
        {
            return "Index contract mismatch for field '" + name + "': index has " + actual_value
                 + ", request has " + expected_value + "; rebuild the index";
        }
    }
    return "Index contract differs from requested configuration; rebuild the index";
}

bool dinoIndexNeedsRebuild(const fs::path &index_root, const DinoRegionSearchConfig &config)
{
    config.validate();
    const auto root = resolveIndexRoot(index_root);
    const auto index_file = root / kIndexFile;
    if (!fs::exists(index_file))
    {
        return true;
    }
    try
    {
        const auto text = dinoReadTextFile(index_file);
        const auto node = YAML::Load(text);
        if (!node || !node.IsMap())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index metadata must be a mapping");
        }
        if (!node["manifest"])
        {
            return true;
        }
        const auto contract = DinoIndexContract::fromYamlNode(node["manifest"]);
        const auto mismatch = contract.checkCompatibility(config);
        if (!mismatch.empty())
        {
            return true;
        }
        const auto require_file = [&](const char *name)
        {
            if (!fs::is_regular_file(root / name))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Index payload is missing or not a file: %s", (root / name).string().c_str());
            }
        };
        for (const auto *name : {kViewsFile, kOffsetsFile, kRegionMeta, kLocalMeta, kRegionScales, kLocalScales})
        {
            require_file(name);
        }
        const bool quantized = contract.quantize_int8;
        require_file(quantized ? kRegionVectors : "region_vectors.f32");
        require_file(quantized ? kLocalVectors : "local_vectors.f32");
        return false;
    }
    catch (const irt::Exception &)
    {
        throw;
    }
    catch (const std::exception &e)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read index at %s: %s",
                             root.string().c_str(), e.what());
    }
}

DinoIndexWriter::DinoIndexWriter(const fs::path &index_root, const DinoIndexContract &contract)
    : index_root_(resolveIndexRoot(index_root))
    , contract_(contract)
    , descriptor_dim_(static_cast<size_t>(contract.coarse_dimension))
    , quantize_(contract.quantize_int8)
{
    if (descriptor_dim_ == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index descriptor dimension must be positive");
    }
    fs::create_directories(index_root_);
    fs::remove(index_root_ / kIndexFile);
    region_meta_writer_ = std::make_unique<DinoNpyWriter>(index_root_ / kRegionMeta, "|u1", 1U);
    local_meta_writer_  = std::make_unique<DinoNpyWriter>(index_root_ / kLocalMeta, "|u1", 1U);
    const auto open_raw = [&](const char *name)
    {
        auto stream = std::make_unique<std::ofstream>(index_root_ / name, std::ios::binary | std::ios::trunc);
        if (!*stream)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create index array: %s", name);
        }
        return stream;
    };
    region_vector_writer_ = open_raw(quantize_ ? kRegionVectors : "region_vectors.f32");
    local_vector_writer_  = open_raw(quantize_ ? kLocalVectors : "local_vectors.f32");
    region_scale_writer_  = open_raw(kRegionScales);
    local_scale_writer_   = open_raw(kLocalScales);
    fs::remove(index_root_ / (quantize_ ? "region_vectors.f32" : kRegionVectors));
    fs::remove(index_root_ / (quantize_ ? "local_vectors.f32" : kLocalVectors));
}

DinoIndexWriter::~DinoIndexWriter()
{
    if (!finished_)
    {
        closeWritersIfOpen();
    }
}

size_t DinoIndexWriter::imageViewCount(const size_t index) const
{
    if (index >= image_view_counts_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image stats index is out of bounds");
    }
    return image_view_counts_[index];
}

size_t DinoIndexWriter::imageRegionCount(const size_t index) const
{
    if (index >= image_region_counts_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image stats index is out of bounds");
    }
    return image_region_counts_[index];
}

size_t DinoIndexWriter::imageLocalCount(const size_t index) const
{
    if (index >= image_local_counts_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image stats index is out of bounds");
    }
    return image_local_counts_[index];
}

uint64_t DinoIndexWriter::imagePatchCount(const size_t index) const
{
    if (index >= image_patch_counts_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image stats index is out of bounds");
    }
    return image_patch_counts_[index];
}

void DinoIndexWriter::closeWritersIfOpen(){
    region_meta_writer_.reset();
    local_meta_writer_.reset();
    region_vector_writer_.reset();
    local_vector_writer_.reset();
    region_scale_writer_.reset();
    local_scale_writer_.reset();
}

void DinoIndexWriter::addImage(const DinoImageIdentity &record)
{
    if (record.width <= 0 || record.height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index image identity must contain positive dimensions");
    }
    for (const auto &image : images_)
    {
        if (image.image_id == record.image_id)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Index image identity duplicates an existing image id: %lld",
                                 static_cast<long long>(record.image_id));
        }
    }
    images_.push_back(record);
    image_view_counts_.push_back(0U);
    image_region_counts_.push_back(0U);
    image_local_counts_.push_back(0U);
    image_patch_counts_.push_back(0U);
}

int DinoIndexWriter::addView(const int image_index, const DinoViewPlan &plan)
{
    if (image_index < 0 || static_cast<size_t>(image_index) >= images_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "View references an unknown image index");
    }
    if (plan.grid_height <= 0 || plan.grid_width <= 0 || plan.grid_width > 65535 || plan.grid_height > 65535)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "View grid must fit the packed u16 meta layout, got %dx%d", plan.grid_width,
                             plan.grid_height);
    }

    DinoIndexView view;
    view.image_index        = image_index;
    view.source_rect        = plan.source_rect;
    view.canonical_to_input = plan.canonical_to_input;
    view.input_to_canonical = plan.input_to_canonical;
    view.input_width        = plan.input_width;
    view.input_height       = plan.input_height;
    view.grid_height        = plan.grid_height;
    view.grid_width         = plan.grid_width;
    view.patch_size         = plan.patch_size;
    view.is_full_view       = plan.is_full_view;
    view.scale_index        = plan.scale_index;

    if (views_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index contains too many views");
    }
    const auto view_id = static_cast<int>(views_.size());
    views_.push_back(view);
    ++image_view_counts_[static_cast<size_t>(image_index)];
    const auto patches = static_cast<uint64_t>(plan.grid_height) * static_cast<uint64_t>(plan.grid_width);
    image_patch_counts_[static_cast<size_t>(image_index)] += patches;
    original_patch_count_ += patches;
    region_offsets_.back() = static_cast<int64_t>(region_count_);
    local_offsets_.back() = static_cast<int64_t>(local_count_);
    region_offsets_.push_back(static_cast<int64_t>(region_count_));
    local_offsets_.push_back(static_cast<int64_t>(local_count_));
    return view_id;
}

void DinoIndexWriter::addRegion(const int view_id, const DinoRegionDescriptor &meta, const std::vector<float> &vector)
{
    if (view_id < 0 || static_cast<size_t>(view_id) >= views_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region descriptor references an unknown view");
    }
    if (vector.size() != descriptor_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region descriptor dimension mismatch");
    }

    const auto &view = views_[static_cast<size_t>(view_id)];
    validateDescriptorRect(view, meta.grid_row, meta.grid_col, meta.grid_height, meta.grid_width, "Region");
    if (!std::isfinite(meta.valid_fraction) || meta.valid_fraction < 0.0F || meta.valid_fraction > 1.0F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region valid fraction must be within [0, 1]");
    }
    auto encoded = meta;
    encoded.view_id = view_id;
    const auto packed = encodeRegionMeta(encoded);
    uint8_t    bytes[kDinoMetaRecordBytes]{};
    dinoEncodeMeta(packed, bytes);
    region_meta_writer_->append(bytes, kDinoMetaRecordBytes);

    if (quantize_)
    {
        writeQuantizedVector(*region_vector_writer_, *region_scale_writer_, vector, region_bytes_,
                             max_quantization_delta_, quantization_delta_sum_, quantization_samples_);
    }
    else
    {
        writeRawVector(*region_vector_writer_, *region_scale_writer_, vector, region_bytes_);
    }

    ++region_count_;
    if (views_[static_cast<size_t>(view_id)].image_index >= 0)
    {
        ++image_region_counts_[static_cast<size_t>(views_[static_cast<size_t>(view_id)].image_index)];
    }
}

void DinoIndexWriter::addLocal(const int view_id, const DinoLocalLeaf &meta, const std::vector<float> &vector)
{
    if (view_id < 0 || static_cast<size_t>(view_id) >= views_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local descriptor references an unknown view");
    }
    if (vector.size() != descriptor_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local descriptor dimension mismatch");
    }

    const auto &view = views_[static_cast<size_t>(view_id)];
    validateDescriptorRect(view, meta.grid_row, meta.grid_col, meta.grid_height, meta.grid_width, "Local");
    if (meta.member_count <= 0 || meta.member_count > 65535 || meta.rep_row < meta.grid_row
        || meta.rep_col < meta.grid_col || meta.rep_row >= meta.grid_row + meta.grid_height
        || meta.rep_col >= meta.grid_col + meta.grid_width || !std::isfinite(meta.radius) || meta.radius < 0.0F)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local metadata is outside the packed contract");
    }
    auto       encoded = meta;
    encoded.view_id    = view_id;
    const auto packed = encodeLocalMeta(encoded);
    uint8_t    bytes[kDinoMetaRecordBytes]{};
    dinoEncodeMeta(packed, bytes);
    local_meta_writer_->append(bytes, kDinoMetaRecordBytes);

    if (quantize_)
    {
        writeQuantizedVector(*local_vector_writer_, *local_scale_writer_, vector, local_bytes_,
                             max_quantization_delta_, quantization_delta_sum_, quantization_samples_);
    }
    else
    {
        writeRawVector(*local_vector_writer_, *local_scale_writer_, vector, local_bytes_);
    }

    max_merge_radius_ = std::max(max_merge_radius_, meta.radius);
    mean_merge_radius_ += meta.radius;
    ++radius_samples_;
    ++local_count_;
    if (views_[static_cast<size_t>(view_id)].image_index >= 0)
    {
        ++image_local_counts_[static_cast<size_t>(views_[static_cast<size_t>(view_id)].image_index)];
    }
}

DinoBuildReport DinoIndexWriter::finish()
{
    if (finished_)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Index writer has already finished");
    }
    if (views_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index build produced no views");
    }

    // 视图表与偏移表
    std::vector<double> view_rows(views_.size() * kViewTableColumns, 0.0);
    for (size_t index = 0; index < views_.size(); ++index)
    {
        const auto &view  = views_[index];
        auto       *row   = view_rows.data() + index * kViewTableColumns;
        row[0]            = static_cast<double>(view.image_index);
        row[1]            = view.source_rect.x0;
        row[2]            = view.source_rect.y0;
        row[3]            = view.source_rect.x1;
        row[4]            = view.source_rect.y1;
        row[5]            = view.input_width;
        row[6]            = view.input_height;
        row[7]            = view.grid_height;
        row[8]            = view.grid_width;
        row[9]            = view.patch_size;
        row[10]           = view.is_full_view ? 1.0 : 0.0;
        row[11]           = view.scale_index;
        for (size_t coefficient = 0; coefficient < 6U; ++coefficient)
        {
            row[12U + coefficient] = view.canonical_to_input.m[coefficient];
        }
    }
    DinoNpyWriter view_writer(index_root_ / kViewsFile, "<f8", sizeof(double));
    view_writer.append(view_rows.data(), view_rows.size() * sizeof(double));
    view_writer.finish({static_cast<int64_t>(views_.size()), static_cast<int64_t>(kViewTableColumns)});

    // 每个视图登记时记录的是“该视图之前的数量”，因此末项必须回填为最终总数，
    // 否则最后一个视图的描述区间会被截断、无法被读取。
    region_offsets_.back() = static_cast<int64_t>(region_count_);
    local_offsets_.back()  = static_cast<int64_t>(local_count_);


    std::vector<int64_t> offsets;
    offsets.reserve(region_offsets_.size() + local_offsets_.size());
    offsets.insert(offsets.end(), region_offsets_.begin(), region_offsets_.end());
    offsets.insert(offsets.end(), local_offsets_.begin(), local_offsets_.end());
    DinoNpyWriter offset_writer(index_root_ / kOffsetsFile, "<i8", sizeof(int64_t));
    offset_writer.append(offsets.data(), offsets.size() * sizeof(int64_t));
    offset_writer.finish({static_cast<int64_t>(offsets.size())});

    region_meta_writer_->finish({static_cast<int64_t>(region_count_), static_cast<int64_t>(kDinoMetaRecordBytes)});
    local_meta_writer_->finish({static_cast<int64_t>(local_count_), static_cast<int64_t>(kDinoMetaRecordBytes)});
    region_meta_writer_.reset();
    local_meta_writer_.reset();

    for (auto *stream : {region_vector_writer_.get(), local_vector_writer_.get(),
                         region_scale_writer_.get(), local_scale_writer_.get()})
    {
        stream->flush();
        stream->close();
        if (!*stream)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to finish index array");
        }
    }

    const char *region_vector_name = quantize_ ? kRegionVectors : "region_vectors.f32";
    const char *local_vector_name  = quantize_ ? kLocalVectors : "local_vectors.f32";

    YAML::Node metadata;
    metadata["manifest"] = contract_.toYamlNode();
    metadata["dimension"] = descriptor_dim_;
    metadata["quantized"] = quantize_;
    YAML::Node images(YAML::NodeType::Sequence);
    for (const auto &record : images_)
    {
        YAML::Node image;
        image["image_id"] = record.image_id;
        image["width"] = record.width;
        image["height"] = record.height;
        image["file_size"] = record.file_size;
        image["mtime_ns"] = record.mtime_ns;
        images.push_back(image);
    }
    metadata["images"] = images;
    std::ofstream metadata_stream(index_root_ / kIndexFile, std::ios::binary | std::ios::trunc);
    metadata_stream << metadata;
    metadata_stream.close();
    if (!metadata_stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write index.yaml");
    }
    uint64_t index_bytes = 0U;
    for (const auto *file : {kIndexFile, kViewsFile, kOffsetsFile, kRegionMeta, kLocalMeta,
                             region_vector_name, local_vector_name, kRegionScales, kLocalScales})
    {
        index_bytes += dinoFileSize(index_root_ / file);
    }

    DinoBuildReport report;
    report.image_count                = images_.size();
    report.view_count                 = views_.size();
    report.region_descriptor_count    = region_count_;
    report.local_descriptor_count     = local_count_;
    report.original_patch_count       = original_patch_count_;
    report.index_bytes                = index_bytes;


    finished_ = true;
    closeWritersIfOpen();
    return report;
}

void DinoIndexWriter::abort()
{
    closeWritersIfOpen();
    std::error_code error;
    fs::remove(index_root_ / kIndexFile, error);
    finished_ = true;
}

DinoIndexReader::DinoIndexReader(const fs::path &index_root)
    : index_root_(resolveIndexRoot(index_root))
{
    if (!fs::exists(index_root_ / kIndexFile))
    {
        throw irt::Exception(irt::Status::NOT_READY, "DINO index metadata is missing: %s",
                             (index_root_ / kIndexFile).string().c_str());
    }
    const auto metadata = YAML::Load(dinoReadTextFile(index_root_ / kIndexFile));
    if (!metadata["manifest"])
    {
        throw irt::Exception(irt::Status::NOT_READY,
                             "DINO index at '%s' is missing manifest contract; rebuild the index / 请重建索引",
                             index_root_.string().c_str());
    }
    contract_ = DinoIndexContract::fromYamlNode(metadata["manifest"]);
    descriptor_dim_ = metadata["dimension"] ? metadata["dimension"].as<size_t>() : static_cast<size_t>(contract_.coarse_dimension);
    quantized_ = metadata["quantized"] ? metadata["quantized"].as<bool>() : contract_.quantize_int8;
    const auto require_file = [&](const char *name)
    {
        if (!fs::exists(index_root_ / name))
        {
            throw irt::Exception(irt::Status::NOT_READY, "DINO index array is missing: %s",
                                 (index_root_ / name).string().c_str());
        }
    };
    require_file(kViewsFile);
    require_file(kOffsetsFile);
    require_file(kRegionMeta);
    require_file(kLocalMeta);
    require_file(kRegionScales);
    require_file(kLocalScales);
    require_file(quantized_ ? kRegionVectors : "region_vectors.f32");
    require_file(quantized_ ? kLocalVectors : "local_vectors.f32");
    if (descriptor_dim_ == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index descriptor dimension must be positive");
    }
    for (const auto &image : metadata["images"])
    {
        DinoImageIdentity record;
        record.image_id = image["image_id"].as<int64_t>();
        record.width = image["width"].as<int>();
        record.height = image["height"].as<int>();
        record.file_size = image["file_size"].as<int64_t>();
        record.mtime_ns = image["mtime_ns"].as<int64_t>();
        images_.push_back(std::move(record));
    }
    loadViews();

    const auto offsets_path = index_root_ / kOffsetsFile;
    DinoNpyReader offset_reader(offsets_path, "<i8");
    const auto    offset_bytes = offset_reader.readAll();
    const auto    offset_count = offset_bytes.size() / sizeof(int64_t);
    offsets_.resize(offset_count);
    if (offset_count > 0U)
    {
        std::memcpy(offsets_.data(), offset_bytes.data(), offset_bytes.size());
    }
    if (offsets_.size() != 2U * (views_.size() + 1U))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index offsets table does not match the view count");
    }

    region_meta_ = std::make_unique<DinoNpyReader>(index_root_ / kRegionMeta, "|u1");
    local_meta_ = std::make_unique<DinoNpyReader>(index_root_ / kLocalMeta, "|u1");
    for (const auto *meta : {region_meta_.get(), local_meta_.get()})
    {
        if (meta->shape().size() != 2U || meta->shape()[1] != static_cast<int64_t>(kDinoMetaRecordBytes))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index metadata has an unexpected record size");
        }
    }
    region_count_ = static_cast<size_t>(region_meta_->rowCount());
    local_count_ = static_cast<size_t>(local_meta_->rowCount());
    region_vectors_ = std::make_unique<std::ifstream>(index_root_ / (quantized_ ? kRegionVectors : "region_vectors.f32"),
                                                     std::ios::binary);
    local_vectors_ = std::make_unique<std::ifstream>(index_root_ / (quantized_ ? kLocalVectors : "local_vectors.f32"),
                                                    std::ios::binary);
    region_scales_ = std::make_unique<std::ifstream>(index_root_ / kRegionScales, std::ios::binary);
    local_scales_ = std::make_unique<std::ifstream>(index_root_ / kLocalScales, std::ios::binary);
    if (!*region_vectors_ || !*local_vectors_ || !*region_scales_ || !*local_scales_)
    {
        throw irt::Exception(irt::Status::NOT_READY, "Index descriptor arrays are missing");
    }
    for (const auto *file : {kIndexFile, kViewsFile, kOffsetsFile, kRegionMeta, kLocalMeta, kRegionScales, kLocalScales,
                             quantized_ ? kRegionVectors : "region_vectors.f32",
                             quantized_ ? kLocalVectors : "local_vectors.f32"})
    {
        index_bytes_ += dinoFileSize(index_root_ / file);
    }
}

void DinoIndexReader::validateContract(const DinoRegionSearchConfig &config) const
{
    config.validate();
    const auto mismatch = contract_.checkCompatibility(config);
    if (!mismatch.empty())
    {
        throw irt::Exception(irt::Status::NOT_READY, "%s", mismatch.c_str());
    }
}


void DinoIndexReader::loadViews()
{
    DinoNpyReader reader(index_root_ / kViewsFile, "<f8");
    if (reader.shape().size() != 2U || reader.shape()[1] != static_cast<int64_t>(kViewTableColumns))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index view table has an unexpected shape");
    }
    const auto bytes = reader.readAll();
    const auto rows  = static_cast<size_t>(reader.shape()[0]);
    views_.resize(rows);
    for (size_t index = 0; index < rows; ++index)
    {
        double row[kViewTableColumns];
        std::memcpy(row, bytes.data() + index * sizeof(row), sizeof(row));
        auto         &view = views_[index];
        view.image_index    = static_cast<int>(row[0]);
        view.source_rect    = DinoRect{row[1], row[2], row[3], row[4]};
        view.input_width    = static_cast<int>(row[5]);
        view.input_height   = static_cast<int>(row[6]);
        view.grid_height    = static_cast<int>(row[7]);
        view.grid_width     = static_cast<int>(row[8]);
        view.patch_size     = static_cast<int>(row[9]);
        view.is_full_view   = row[10] > 0.5;
        view.scale_index    = static_cast<int>(row[11]);
        for (size_t coefficient = 0; coefficient < 6U; ++coefficient)
        {
            view.canonical_to_input.m[coefficient] = row[12U + coefficient];
        }
        view.input_to_canonical = view.canonical_to_input.inverse();
    }
}

DinoDescriptorRange DinoIndexReader::regionRange(const size_t view_id) const
{
    if (view_id >= views_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region range query uses an unknown view id");
    }
    DinoDescriptorRange range;
    range.begin = static_cast<size_t>(offsets_[view_id]);
    const auto end = static_cast<size_t>(offsets_[view_id + 1U]);
    if (range.begin > end || end > region_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region offsets are out of bounds");
    }
    range.count = end - range.begin;
    return range;
}

DinoDescriptorRange DinoIndexReader::localRange(const size_t view_id) const
{
    if (view_id >= views_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local range query uses an unknown view id");
    }
    const size_t base = views_.size() + 1U;
    DinoDescriptorRange range;
    range.begin = static_cast<size_t>(offsets_[base + view_id]);
    const auto end = static_cast<size_t>(offsets_[base + view_id + 1U]);
    if (range.begin > end || end > local_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local offsets are out of bounds");
    }
    range.count = end - range.begin;
    return range;
}

DinoCompactBlock DinoIndexReader::readCompact(std::ifstream &vectors, std::ifstream &scales,
                                             const size_t total_count, const size_t begin, const size_t count,
                                             const char *label, DinoCompactScratch &scratch) const
{
    if (begin > total_count || count > total_count - begin
        || count > std::numeric_limits<size_t>::max() / descriptor_dim_
        || count > std::numeric_limits<size_t>::max() / (2U * sizeof(float))
        || begin > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / descriptor_dim_
        || begin > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / (2U * sizeof(float)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s vector range is out of bounds", label);
    }
    if (count == 0U)
    {
        return {};
    }

    scratch.codes.resize(count * descriptor_dim_);
    vectors.clear();
    vectors.seekg(static_cast<std::streamoff>(begin * descriptor_dim_), std::ios::beg);
    vectors.read(reinterpret_cast<char *>(scratch.codes.data()),
                 static_cast<std::streamsize>(scratch.codes.size()));
    if (vectors.gcount() != static_cast<std::streamsize>(scratch.codes.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s vector short read", label);
    }

    scratch.scales.resize(count * 2U);
    auto &stored = scratch.scales;
    scales.clear();
    scales.seekg(static_cast<std::streamoff>(begin * 2U * sizeof(float)), std::ios::beg);
    scales.read(reinterpret_cast<char *>(stored.data()), static_cast<std::streamsize>(stored.size() * sizeof(float)));
    if (scales.gcount() != static_cast<std::streamsize>(stored.size() * sizeof(float)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s scale short read", label);
    }

    scratch.factors.resize(count);
    for (size_t index = 0; index < count; ++index)
    {
        scratch.factors[index] = stored[index * 2U] * stored[index * 2U + 1U];
    }

    DinoCompactBlock block;
    block.codes     = scratch.codes.data();
    block.factors   = scratch.factors.data();
    block.count     = count;
    block.dimension = descriptor_dim_;
    return block;
}

DinoCompactBlock DinoIndexReader::readRegionCompact(const size_t begin, const size_t count,
                                                   DinoCompactScratch &scratch) const
{
    if (!quantized_)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION,
                             "Compact reads require INT8 storage");
    }
    return readCompact(*region_vectors_, *region_scales_, region_count_, begin, count, "Region", scratch);
}

DinoCompactBlock DinoIndexReader::readLocalCompact(const size_t begin, const size_t count,
                                                  DinoCompactScratch &scratch) const
{
    if (!quantized_)
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION,
                             "Compact reads require INT8 storage");
    }
    return readCompact(*local_vectors_, *local_scales_, local_count_, begin, count, "Local", scratch);
}

void DinoIndexReader::readRegionVectors(const size_t begin, const size_t count, std::vector<float> &out) const
{
    if (begin > region_count_ || count > region_count_ - begin
        || count > std::numeric_limits<size_t>::max() / sizeof(float) / descriptor_dim_
        || begin > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / sizeof(float) / descriptor_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region vector range is out of bounds");
    }
    out.resize(count * descriptor_dim_);
    if (count == 0U)
    {
        return;
    }

    if (!quantized_)
    {
        auto &stream = *region_vectors_;
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(begin * descriptor_dim_ * sizeof(float)), std::ios::beg);
        stream.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
        if (stream.gcount() != static_cast<std::streamsize>(out.size() * sizeof(float)))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region vector short read");
        }
        return;
    }

    DinoCompactScratch scratch;
    const auto        block = readRegionCompact(begin, count, scratch);
    for (size_t index = 0; index < count; ++index)
    {
        const float factor = block.factors[index];
        const auto *code   = block.codes + index * descriptor_dim_;
        auto       *target = out.data() + index * descriptor_dim_;
        for (size_t dimension = 0; dimension < descriptor_dim_; ++dimension)
        {
            target[dimension] = static_cast<float>(code[dimension]) * factor;
        }
    }
}

void DinoIndexReader::readLocalVectors(const size_t begin, const size_t count, std::vector<float> &out) const
{
    if (begin > local_count_ || count > local_count_ - begin
        || count > std::numeric_limits<size_t>::max() / sizeof(float) / descriptor_dim_
        || begin > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()) / sizeof(float) / descriptor_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local vector range is out of bounds");
    }
    out.resize(count * descriptor_dim_);
    if (count == 0U)
    {
        return;
    }

    if (!quantized_)
    {
        auto &stream = *local_vectors_;
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(begin * descriptor_dim_ * sizeof(float)), std::ios::beg);
        stream.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
        if (stream.gcount() != static_cast<std::streamsize>(out.size() * sizeof(float)))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local vector short read");
        }
        return;
    }

    DinoCompactScratch scratch;
    const auto        block = readLocalCompact(begin, count, scratch);
    for (size_t index = 0; index < count; ++index)
    {
        const float factor = block.factors[index];
        const auto *code   = block.codes + index * descriptor_dim_;
        auto       *target = out.data() + index * descriptor_dim_;
        for (size_t dimension = 0; dimension < descriptor_dim_; ++dimension)
        {
            target[dimension] = static_cast<float>(code[dimension]) * factor;
        }
    }
}

std::vector<DinoRegionDescriptor> DinoIndexReader::readRegionMeta(const size_t begin, const size_t count) const
{
    const auto bytes = region_meta_->readBytes(static_cast<int64_t>(begin), static_cast<int64_t>(count));
    std::vector<DinoRegionDescriptor> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const auto meta = dinoDecodeMeta(bytes.data() + index * kDinoMetaRecordBytes);
        if (meta.reserved1 != 0U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region meta slot contains a non-region record");
        }
        result.push_back(decodeRegionMeta(meta));
    }
    return result;
}

std::vector<DinoLocalLeaf> DinoIndexReader::readLocalMeta(const size_t begin, const size_t count) const
{
    const auto bytes = local_meta_->readBytes(static_cast<int64_t>(begin), static_cast<int64_t>(count));
    std::vector<DinoLocalLeaf> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const auto meta = dinoDecodeMeta(bytes.data() + index * kDinoMetaRecordBytes);
        if (meta.reserved1 != 1U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local meta slot contains a non-local record");
        }
        result.push_back(decodeLocalMeta(meta));
    }
    return result;
}

DinoRegionDescriptor DinoIndexReader::regionMetaAt(const size_t index) const
{
    if (index >= region_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region meta index is out of bounds");
    }
    const auto bytes = region_meta_->readBytes(static_cast<int64_t>(index), 1);
    const auto meta  = dinoDecodeMeta(bytes.data());
    if (meta.reserved1 != 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region meta slot contains a non-region record");
    }
    return decodeRegionMeta(meta);
}

DinoLocalLeaf DinoIndexReader::localMetaAt(const size_t index) const
{
    if (index >= local_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local meta index is out of bounds");
    }
    const auto bytes = local_meta_->readBytes(static_cast<int64_t>(index), 1);
    const auto meta  = dinoDecodeMeta(bytes.data());
    if (meta.reserved1 != 1U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local meta slot contains a non-local record");
    }
    return decodeLocalMeta(meta);
}

int DinoIndexReader::viewOfRegionDescriptor(const size_t index) const
{
    if (index >= region_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Region descriptor index is out of bounds");
    }
    // 描述按 view 连续排列，偏移表单调不减，可直接二分定位所属视图。
    const auto begin = offsets_.begin();
    const auto end   = begin + static_cast<std::ptrdiff_t>(views_.size()) + 1;
    const auto it    = std::upper_bound(begin, end, static_cast<int64_t>(index));
    const auto view  = static_cast<size_t>(std::distance(begin, it)) - 1U;
    return static_cast<int>(view);
}

int DinoIndexReader::viewOfLocalDescriptor(const size_t index) const
{
    if (index >= local_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Local descriptor index is out of bounds");
    }
    const auto base  = offsets_.begin() + static_cast<std::ptrdiff_t>(views_.size()) + 1;
    const auto end   = offsets_.begin() + static_cast<std::ptrdiff_t>(2U * (views_.size() + 1U));
    const auto it    = std::upper_bound(base, end, static_cast<int64_t>(index));
    const auto view  = static_cast<size_t>(std::distance(base, it)) - 1U;
    return static_cast<int>(view);
}

int DinoIndexReader::imageIndexById(const int64_t image_id) const
{
    for (size_t index = 0; index < images_.size(); ++index)
    {
        if (images_[index].image_id == image_id)
        {
            return static_cast<int>(index);
        }
    }
    return -1;
}


} // namespace irt::features::priv
