/**
 * @file ImageSearchFaissIndex.hpp
 * @brief CPU 磁盘 IVF+Flat 索引的构建、落盘与内存映射加载。
 */

#include "ImageSearchFaissIndex.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/features/ImageSearch.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#pragma warning(pop)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    ifdef ERROR_INVALID_ARGUMENT
#        undef ERROR_INVALID_ARGUMENT
#    endif
#    ifdef ERROR_NOT_IMPLEMENTED
#        undef ERROR_NOT_IMPLEMENTED
#    endif
#endif

namespace irt::features::priv {

inline size_t checkedU64ToSize(uint64_t value, const char *what)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s exceeds size_t range", what ? what : "Value");
    }
    return static_cast<size_t>(value);
}

inline uint64_t checkedSizeToU64(size_t value, const char *what)
{
    if (value > static_cast<size_t>(std::numeric_limits<uint64_t>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s exceeds uint64 range", what ? what : "Value");
    }
    return static_cast<uint64_t>(value);
}

inline faiss::idx_t checkedSizeToFaissIdx(size_t value, const char *what)
{
    if (value > static_cast<size_t>(std::numeric_limits<faiss::idx_t>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s exceeds Faiss index range",
                             what ? what : "Value");
    }
    return static_cast<faiss::idx_t>(value);
}

/**
 * @brief 向调用方报告索引构建进度。
 * @param progress_callback 可为空的进度回调。
 * @param stage 当前阶段。
 * @param batch_index 当前阶段内从 0 开始的批次编号。
 * @param batch_begin 当前批次对应的图库起始下标。
 * @param batch_count 当前批次处理的图库图片数量。
 * @param processed_count 当前阶段已完成的工作单元数。
 * @param total_count 当前阶段总工作单元数；不可度量时为 0。
 */
void reportBuildProgress(const BuildProgressCallback &progress_callback, ImageSearchBuildStage stage,
                         size_t batch_index, size_t batch_begin, size_t batch_count, size_t processed_count,
                         size_t total_count)
{
    if (!progress_callback)
    {
        return;
    }

    progress_callback(
        ImageSearchBuildProgress{stage, batch_index, batch_begin, batch_count, processed_count, total_count});
}

/**
 * @brief 生成 CPU 磁盘 IVF 倒排列表的伴生数据文件路径。
 * @param index_path Faiss 索引文件路径（``.faiss``）。
 * @return ``<index_path>.ivfdata``。
 */
std::filesystem::path cpuOnDiskIvfDataPath(const std::filesystem::path &index_path)
{
    return index_path.string() + ".ivfdata";
}

/**
 * @brief 只读映射磁盘文件。
 *
 * Windows 下使用 ``CreateFileMapping`` / ``MapViewOfFile`` 零拷贝映射；
 * 其他平台将整个文件读入内存后提供连续字节视图。
 */
class MappedFile
{
public:
    /**
     * @brief 打开并映射指定路径的二进制文件。
     * @param path 待映射文件路径。
     * @throws irt::Exception 打开、获取大小或映射失败时抛出。
     */
    explicit MappedFile(const std::filesystem::path &path)
    {
#ifdef _WIN32
        file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open Faiss on-disk data: %s",
                                 path.string().c_str());
        }

        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart <= 0)
        {
            close();
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid Faiss on-disk data file: %s",
                                 path.string().c_str());
        }
        if (static_cast<unsigned long long>(size.QuadPart)
            > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
        {
            close();
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Faiss on-disk data file is too large for this platform: %s", path.string().c_str());
        }
        size_ = static_cast<size_t>(size.QuadPart);

        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_)
        {
            close();
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to map Faiss on-disk data: %s",
                                 path.string().c_str());
        }

        data_ = static_cast<const uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!data_)
        {
            close();
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to view Faiss on-disk data: %s",
                                 path.string().c_str());
        }
#else
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open Faiss on-disk data: %s",
                                 path.string().c_str());
        }
        owned_data_ = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        if (owned_data_.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid Faiss on-disk data file: %s",
                                 path.string().c_str());
        }
        data_ = owned_data_.data();
        size_ = owned_data_.size();
#endif
    }

    /** @brief 释放映射资源（仅 Windows 需要显式清理）。 */
    ~MappedFile()
    {
#ifdef _WIN32
        close();
#endif
    }

    /** @brief 禁止拷贝构造。 */
    MappedFile(const MappedFile &) = delete;
    /** @brief 禁止拷贝赋值。 */
    MappedFile &operator=(const MappedFile &) = delete;

    /**
     * @brief 获取映射数据的起始指针。
     * @return 只读字节指针。
     */
    const uint8_t *data() const noexcept
    {
        return data_;
    }

    /**
     * @brief 获取映射数据的总字节数。
     * @return 文件大小。
     */
    size_t size() const noexcept
    {
        return size_;
    }

private:
#ifdef _WIN32
    /** @brief 解除文件视图并关闭句柄。 */
    void close() noexcept
    {
        if (data_)
        {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_)
        {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE)
        {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
    }
#endif

    const uint8_t *data_{nullptr}; ///< 映射数据的只读视图。
    size_t         size_{0};       ///< 映射数据字节数。
#ifdef _WIN32
    HANDLE file_{INVALID_HANDLE_VALUE}; ///< 源文件句柄。
    HANDLE mapping_{nullptr};           ///< 文件映射对象句柄。
#else
    std::vector<uint8_t> owned_data_; ///< 非 Windows 平台持有的完整文件副本。
#endif
};

/**
 * @brief CPU 磁盘 IVF 倒排列表文件的文件头。
 *
 * 魔数 ``0x46545649`` 对应 ASCII ``IVFT``，用于校验文件格式。
 */
struct CpuOnDiskIvfHeader
{
    uint32_t magic{0x46545649}; ///< 魔数，固定为 ``IVFT``。
    uint32_t version{1};        ///< 格式版本号。
    uint64_t nlist{0};          ///< IVF 聚类（倒排列表）数量。
    uint64_t code_size{0};      ///< 每条向量的 PQ/Flat 编码字节数。
    uint64_t ntotal{0};         ///< 索引中的向量总数。
};

/**
 * @brief 单个 IVF 倒排列表在 ``.ivfdata`` 文件中的布局元数据。
 */
/**
 * @brief 将数值向上对齐到指定边界。
 * @param value 待对齐的值。
 * @param alignment 对齐字节数（须为正）。
 * @return 对齐后的值。
 */
inline size_t alignUp(size_t value, size_t alignment)
{
    if (alignment == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Alignment must be positive");
    }
    const auto rounded = irt::checkedSizeAdd(value, alignment - 1, "Aligned size");
    return irt::checkedSizeMul(rounded / alignment, alignment, "Aligned size");
}

/**
 * @brief 从映射文件中按偏移读取定长 POD 值。
 * @tparam T 目标类型，须为平凡可复制类型。
 * @param mapped_file 已映射的文件。
 * @param offset 字节偏移。
 * @return 读取到的值。
 * @throws irt::Exception 越界时抛出。
 */
template<typename T>
T readMappedValue(const MappedFile &mapped_file, size_t offset)
{
    const auto end = irt::checkedSizeAdd(offset, sizeof(T), "Mapped value offset");
    if (end > mapped_file.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Corrupt Faiss on-disk data header");
    }

    T value{};
    std::memcpy(&value, mapped_file.data() + offset, sizeof(T));
    return value;
}

/**
 * @brief 基于内存映射的 Faiss 只读倒排列表实现。
 *
 * 将 ``.ivfdata`` 侧车文件中的 ID 与编码数据直接暴露给 Faiss IVF 索引，
 * 避免在加载时将全部倒排列表载入 RAM。
 */
class InferRtOnDiskInvertedLists : public faiss::ReadOnlyInvertedLists
{
public:
    /**
     * @brief 解析并映射倒排列表数据文件。
     * @param path ``.ivfdata`` 文件路径。
     * @throws irt::Exception 文件头或元数据校验失败时抛出。
     */
    explicit InferRtOnDiskInvertedLists(const std::filesystem::path &path)
        : faiss::ReadOnlyInvertedLists(0, 0)
        , mapped_file_(path)
    {
        const auto header = readMappedValue<CpuOnDiskIvfHeader>(mapped_file_, 0);
        if (header.magic != CpuOnDiskIvfHeader{}.magic || header.version != CpuOnDiskIvfHeader{}.version
            || header.nlist == 0 || header.code_size == 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid Faiss on-disk inverted lists: %s",
                                 path.string().c_str());
        }

        nlist        = checkedU64ToSize(header.nlist, "Faiss nlist");
        code_size    = checkedU64ToSize(header.code_size, "Faiss code size");
        entry_count_ = checkedU64ToSize(header.ntotal, "Faiss entry count");

        const size_t meta_offset = sizeof(CpuOnDiskIvfHeader);
        const size_t meta_bytes = irt::checkedSizeMul(nlist, sizeof(CpuOnDiskIvfListMeta), "Faiss metadata size");
        const size_t meta_end   = irt::checkedSizeAdd(meta_offset, meta_bytes, "Faiss metadata end");
        if (meta_end > mapped_file_.size())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Corrupt Faiss on-disk inverted lists: %s",
                                 path.string().c_str());
        }

        list_meta_.resize(nlist);
        std::memcpy(list_meta_.data(), mapped_file_.data() + meta_offset, meta_bytes);
        for (const auto &meta : list_meta_)
        {
            const auto ids_offset   = checkedU64ToSize(meta.ids_offset, "Faiss ids offset");
            const auto codes_offset = checkedU64ToSize(meta.codes_offset, "Faiss codes offset");
            const auto list_size    = checkedU64ToSize(meta.size, "Faiss list size");
            const auto ids_bytes    = irt::checkedSizeMul(list_size, sizeof(faiss::idx_t), "Faiss ids size");
            const auto codes_bytes  = irt::checkedSizeMul(list_size, code_size, "Faiss codes size");
            const auto ids_end      = irt::checkedSizeAdd(ids_offset, ids_bytes, "Faiss ids end");
            const auto codes_end    = irt::checkedSizeAdd(codes_offset, codes_bytes, "Faiss codes end");
            if (ids_end > mapped_file_.size() || codes_end > mapped_file_.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Corrupt Faiss on-disk inverted lists: %s",
                                     path.string().c_str());
            }
        }
    }

    /**
     * @brief 获取索引中的向量总数。
     * @return 与 Faiss ``ntotal`` 一致的条目数。
     */
    size_t entryCount() const noexcept
    {
        return entry_count_;
    }

    /**
     * @brief 获取指定倒排列表的向量条数。
     * @param list_no 列表编号。
     * @return 该列表中的向量数量。
     */
    size_t list_size(size_t list_no) const override
    {
        return list_meta_.at(list_no).size;
    }

    /**
     * @brief 获取指定倒排列表的编码数据指针。
     * @param list_no 列表编号。
     * @return 指向映射文件中编码区的只读指针。
     */
    const uint8_t *get_codes(size_t list_no) const override
    {
        return mapped_file_.data() + list_meta_.at(list_no).codes_offset;
    }

    /**
     * @brief 获取指定倒排列表的 ID 数组指针。
     * @param list_no 列表编号。
     * @return 指向映射文件中 ID 区的只读指针。
     */
    const faiss::idx_t *get_ids(size_t list_no) const override
    {
        return reinterpret_cast<const faiss::idx_t *>(mapped_file_.data() + list_meta_.at(list_no).ids_offset);
    }

private:
    MappedFile                        mapped_file_;    ///< 倒排列表数据文件的映射视图。
    std::vector<CpuOnDiskIvfListMeta> list_meta_;      ///< 各倒排列表的偏移与大小元数据。
    size_t                            entry_count_{0}; ///< 向量总数缓存。
};

/**
 * @brief 向 ``.ivfdata`` 输出流的指定偏移写入一段二进制数据。
 * @param output 已打开的二进制读写流。
 * @param offset 文件内字节偏移。
 * @param data 待写入数据起始地址。
 * @param size 待写入字节数。
 * @throws irt::Exception seek 或 write 失败时抛出。
 */
inline void writeAt(std::fstream &output, uint64_t offset, const void *data, size_t size)
{
    if (size == 0)
    {
        return;
    }

    const auto stream_offset = irt::checkedSizeToStreamoff(checkedU64ToSize(offset, "Faiss file offset"),
                                                           "Faiss file offset");
    const auto stream_size   = irt::checkedSizeToStreamsize(size, "Faiss write size");
    output.seekp(stream_offset, std::ios::beg);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to seek Faiss on-disk data");
    }
    output.write(static_cast<const char *>(data), stream_size);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write Faiss on-disk data");
    }
}

/**
 * @brief 按批从回调加载连续图库特征并拼接为平面数组。
 * @param begin 图库起始下标。
 * @param count 本批向量条数。
 * @param feature_dim 单条特征维度。
 * @param load_feature 按下标返回单条特征的回调。
 * @return 长度为 ``count * feature_dim`` 的拼接特征。
 */
std::vector<float> loadFeatureBatch(size_t begin, size_t count, int feature_dim,
                                    const LoadFeatureCallback &load_feature)
{
    if (feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature dimension must be positive");
    }
    const auto dim      = static_cast<size_t>(feature_dim);
    const auto expected = irt::checkedSizeMul(count, dim, "ImageSearch feature batch");
    std::vector<float> features;
    features.reserve(expected);
    for (size_t i = 0; i < count; ++i)
    {
        auto feature = load_feature(irt::checkedSizeAdd(begin, i, "ImageSearch feature index"));
        if (feature.size() != dim)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature size");
        }
        features.insert(features.end(), feature.begin(), feature.end());
    }
    return features;
}

/**
 * @brief 通过批量回调加载连续图像特征，并校验返回的扁平化尺寸。
 */
std::vector<float> loadFeatureBatch(size_t begin, size_t count, int feature_dim,
                                    const LoadFeatureBatchCallback &load_features)
{
    if (feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature dimension must be positive");
    }
    const auto expected = irt::checkedSizeMul(count, static_cast<size_t>(feature_dim), "ImageSearch feature batch");
    auto features = load_features(begin, count);
    if (features.size() != expected)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature batch size");
    }
    return features;
}

/**
 * @brief 判断一组图库下标是否构成连续区间。
 * @param indices 按采样顺序排列的图库下标。
 * @return 空列表或连续递增列表返回 true。
 */
inline bool isContiguousIndexBatch(const std::vector<size_t> &indices)
{
    if (indices.empty())
    {
        return true;
    }

    const size_t begin = indices.front();
    for (size_t i = 1; i < indices.size(); ++i)
    {
        if (indices[i] != irt::checkedSizeAdd(begin, i, "ImageSearch contiguous index"))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief 根据下标形态选择最高效的批量特征加载方式。
 *
 * 连续下标优先走 ``LoadFeatureBatchCallback``；非连续下标优先走 indexed batch 回调；
 * 两种批量回调均不可用时回退到逐条 ``LoadFeatureCallback``。
 *
 * @param first_index ``indices`` 第一项，用于连续区间批量加载。
 * @param indices 需要加载的图库下标列表。
 * @param feature_dim 单条特征维度。
 * @param load_feature 单条特征加载回调。
 * @param load_feature_batch 连续区间批量加载回调。
 * @param load_feature_index_batch 任意下标批量加载回调。
 * @return 扁平化批量特征。
 */
inline std::vector<float> loadFeatureIndexedBatch(size_t first_index, const std::vector<size_t> &indices,
                                                  int feature_dim, const LoadFeatureCallback &load_feature,
                                                  const LoadFeatureBatchCallback        &load_feature_batch,
                                                  const LoadFeatureIndexedBatchCallback &load_feature_index_batch)
{
    if (indices.empty())
    {
        return {};
    }
    if (feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature dimension must be positive");
    }

    if (load_feature_batch && isContiguousIndexBatch(indices))
    {
        return loadFeatureBatch(first_index, indices.size(), feature_dim, load_feature_batch);
    }

    if (load_feature_index_batch)
    {
        auto features = load_feature_index_batch(indices);
        const auto expected
            = irt::checkedSizeMul(indices.size(), static_cast<size_t>(feature_dim), "ImageSearch indexed batch");
        if (features.size() != expected)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Unexpected ImageSearch indexed feature batch size");
        }
        return features;
    }

    std::vector<float> features;
    const auto dim      = static_cast<size_t>(feature_dim);
    const auto expected = irt::checkedSizeMul(indices.size(), dim, "ImageSearch indexed batch");
    features.reserve(expected);
    for (const auto index : indices)
    {
        auto feature = load_feature(index);
        if (feature.size() != dim)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature size");
        }
        features.insert(features.end(), feature.begin(), feature.end());
    }
    return features;
}

/**
 * @brief Faiss 训练阶段采样出的特征集合。
 */
/**
 * @brief 按 stride 抽样并批量加载 Faiss 训练特征。
 *
 * 该函数只负责训练样本收集，不直接训练 Faiss；进度单位仍是训练向量数量，批次大小来自
 * ``ImageSearchConfig::model_batch_size``。
 *
 * @param vector_count 图库向量总数。
 * @param feature_dim 单条特征维度。
 * @param training_count 需要采样的训练向量数量。
 * @param stride 采样步长。
 * @param batch_size 特征提取批量。
 * @param load_feature 单条特征加载回调。
 * @param load_feature_batch 连续区间批量加载回调。
 * @param load_feature_index_batch 任意下标批量加载回调。
 * @param progress_callback 进度回调。
 * @return 训练特征及其图库下标。
 */
TrainingFeatureSample loadTrainingFeatures(size_t vector_count, int feature_dim, size_t training_count,
                                           size_t stride, size_t batch_size,
                                           const LoadFeatureCallback &load_feature,
                                           const LoadFeatureBatchCallback &load_feature_batch,
                                           const LoadFeatureIndexedBatchCallback &load_feature_index_batch)
{
    batch_size = std::max<size_t>(1, batch_size);

    TrainingFeatureSample sample;
    sample.features.reserve(
        irt::checkedSizeMul(training_count, static_cast<size_t>(feature_dim), "ImageSearch training features"));
    sample.indices.reserve(training_count);
    size_t              index_in_gallery = 0;
    std::vector<size_t> batch_indices;
    batch_indices.reserve(batch_size);
    while (index_in_gallery < vector_count && sample.count < training_count)
    {
        batch_indices.clear();
        const size_t batch_begin = index_in_gallery;
        while (index_in_gallery < vector_count && sample.count + batch_indices.size() < training_count
               && batch_indices.size() < batch_size)
        {
            batch_indices.push_back(index_in_gallery);
            index_in_gallery = irt::checkedSizeAdd(index_in_gallery, stride, "ImageSearch sampling index");
        }

        auto features = loadFeatureIndexedBatch(batch_begin, batch_indices, feature_dim, load_feature,
                                                load_feature_batch, load_feature_index_batch);
        sample.features.insert(sample.features.end(), features.begin(), features.end());
        sample.indices.insert(sample.indices.end(), batch_indices.begin(), batch_indices.end());
        sample.count += batch_indices.size();
    }
    return sample;
}

/**
 * @brief 将一批特征按内积最近邻分配到 IVF 聚类中心。
 * @param features 平面特征数组，布局为 ``count × feature_dim``。
 * @param count 向量条数。
 * @param feature_dim 特征维度。
 * @param centroids 聚类中心，布局为 ``nlist × feature_dim``。
 * @param nlist IVF 列表数量。
 * @return 每条特征对应的列表编号。
 */
inline std::vector<faiss::idx_t> assignFeatureBatchToCentroids(const std::vector<float> &features, size_t count,
                                                               int feature_dim, const std::vector<float> &centroids,
                                                               size_t nlist)
{
    if (feature_dim <= 0 || nlist == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageSearch centroid assignment requires positive dimensions");
    }
    const auto dim             = static_cast<size_t>(feature_dim);
    const auto expected_values = irt::checkedSizeMul(count, dim, "ImageSearch feature values");
    const auto expected_centroids = irt::checkedSizeMul(nlist, dim, "ImageSearch centroid values");
    if (features.size() != expected_values || centroids.size() != expected_centroids)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature/centroid buffer size mismatch");
    }
    std::vector<faiss::idx_t> assignments(count);
    for (size_t row = 0; row < count; ++row)
    {
        const float *feature = features.data() + irt::checkedSizeMul(row, dim, "ImageSearch feature offset");
        float        best    = std::numeric_limits<float>::lowest();
        size_t       best_no = 0;
        for (size_t list_no = 0; list_no < nlist; ++list_no)
        {
            const float *centroid = centroids.data() + irt::checkedSizeMul(list_no, dim, "ImageSearch centroid offset");
            float        score    = 0.0f;
            for (size_t col = 0; col < dim; ++col)
            {
                score += feature[col] * centroid[col];
            }
            if (score > best)
            {
                best    = score;
                best_no = list_no;
            }
        }
        assignments[row] = static_cast<faiss::idx_t>(best_no);
    }
    return assignments;
}

/**
 * @brief 根据各倒排列表的向量条数计算 ``.ivfdata`` 中的 ID/编码区偏移布局。
 * @param list_sizes 每个 IVF 列表将容纳的向量数量。
 * @param code_size 单条向量的编码字节数。
 * @return 与 ``list_sizes`` 等长的元数据数组。
 */
std::vector<CpuOnDiskIvfListMeta> makeCpuOnDiskIvfListMeta(const std::vector<uint64_t> &list_sizes,
                                                           size_t code_size)
{
    std::vector<CpuOnDiskIvfListMeta> list_meta(list_sizes.size());
    size_t offset = irt::checkedSizeAdd(
        sizeof(CpuOnDiskIvfHeader),
        irt::checkedSizeMul(list_meta.size(), sizeof(CpuOnDiskIvfListMeta), "Faiss metadata size"),
        "Faiss metadata offset");
    offset        = alignUp(offset, alignof(faiss::idx_t));

    for (size_t list_no = 0; list_no < list_sizes.size(); ++list_no)
    {
        auto &meta       = list_meta[list_no];
        const auto count = checkedU64ToSize(list_sizes[list_no], "Faiss list size");
        meta.size        = list_sizes[list_no];
        meta.ids_offset  = checkedSizeToU64(offset, "Faiss ids offset");
        offset            = irt::checkedSizeAdd(
            offset, irt::checkedSizeMul(count, sizeof(faiss::idx_t), "Faiss ids size"), "Faiss ids end");
        offset            = alignUp(offset, alignof(float));
        meta.codes_offset = checkedSizeToU64(offset, "Faiss codes offset");
        offset            = irt::checkedSizeAdd(offset, irt::checkedSizeMul(count, code_size, "Faiss codes size"),
                                                "Faiss codes end");
        offset = alignUp(offset, alignof(faiss::idx_t));
    }

    return list_meta;
}

/**
 * @brief 根据列表元数据计算 ``.ivfdata`` 文件所需的总字节数。
 */
inline size_t cpuOnDiskIvfDataFileSize(const std::vector<CpuOnDiskIvfListMeta> &list_meta, size_t code_size)
{
    size_t file_size = irt::checkedSizeAdd(
        sizeof(CpuOnDiskIvfHeader),
        irt::checkedSizeMul(list_meta.size(), sizeof(CpuOnDiskIvfListMeta), "Faiss metadata size"),
        "Faiss file size");
    for (const auto &meta : list_meta)
    {
        const auto count       = checkedU64ToSize(meta.size, "Faiss list size");
        const auto ids_offset  = checkedU64ToSize(meta.ids_offset, "Faiss ids offset");
        const auto codes_offset = checkedU64ToSize(meta.codes_offset, "Faiss codes offset");
        const auto ids_end = irt::checkedSizeAdd(ids_offset,
                                                 irt::checkedSizeMul(count, sizeof(faiss::idx_t), "Faiss ids size"),
                                                 "Faiss ids end");
        const auto codes_end = irt::checkedSizeAdd(codes_offset,
                                                   irt::checkedSizeMul(count, code_size, "Faiss codes size"),
                                                   "Faiss codes end");
        file_size = std::max(file_size, ids_end);
        file_size = std::max(file_size, codes_end);
    }
    return alignUp(file_size, alignof(faiss::idx_t));
}

/**
 * @brief 预分配并写入 ``.ivfdata`` 文件头与列表元数据区。
 */
inline void initializeCpuOnDiskIvfDataFile(const std::filesystem::path &data_path, const CpuOnDiskIvfHeader &header,
                                           const std::vector<CpuOnDiskIvfListMeta> &list_meta, size_t file_size)
{
    std::ofstream output(data_path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write Faiss on-disk data: %s",
                             data_path.string().c_str());
    }

    output.write(reinterpret_cast<const char *>(&header), sizeof(header));
    const auto metadata_bytes
        = irt::checkedSizeMul(list_meta.size(), sizeof(CpuOnDiskIvfListMeta), "Faiss metadata size");
    output.write(reinterpret_cast<const char *>(list_meta.data()),
                 irt::checkedSizeToStreamsize(metadata_bytes, "Faiss metadata size"));
    if (file_size > 0)
    {
        output.seekp(irt::checkedSizeToStreamoff(file_size - 1, "Faiss file size"), std::ios::beg);
        static constexpr char kZero = 0;
        output.write(&kZero, 1);
    }
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to initialize Faiss on-disk data: %s",
                             data_path.string().c_str());
    }
}

/** @brief 前向声明：按内存上限裁剪 Faiss 建库批大小。 */
size_t chooseFaissIndexBuildBatchSize(size_t requested_batch_size, size_t vector_count, int feature_dim);

/**
 * @brief 分两趟扫描图库：先统计各 IVF 列表大小，再分批写入 ID 与编码到 ``.ivfdata``。
 */
inline void writeCpuOnDiskIvfDataFileBatched(const faiss::IndexIVF &index, size_t vector_count, int feature_dim,
                                             const std::filesystem::path &data_path,
                                             const std::vector<float> &centroids, size_t batch_size,
                                             const LoadFeatureCallback      &load_feature,
                                             const LoadFeatureBatchCallback &load_feature_batch = {},
                                             const BuildProgressCallback    &progress_callback  = {})
{
    if (batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Faiss on-disk batch size must be positive");
    }
    if (feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature dimension must be positive");
    }
    const auto feature_bytes = irt::checkedSizeMul(static_cast<size_t>(feature_dim), sizeof(float),
                                                   "ImageSearch feature bytes");
    if (index.code_size != feature_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "CPU on-disk ImageSearch currently requires IVF Flat float codes");
    }
    const auto centroid_values
        = irt::checkedSizeMul(static_cast<size_t>(index.nlist), static_cast<size_t>(feature_dim),
                              "ImageSearch centroid values");
    if (centroids.size() != centroid_values)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid IVF centroid buffer size");
    }

    batch_size = chooseFaissIndexBuildBatchSize(batch_size, vector_count, feature_dim);
    const size_t progress_total = irt::checkedSizeMul(vector_count, 2, "ImageSearch progress count");

    std::vector<uint64_t> list_sizes(index.nlist, 0);
    size_t                assign_batch_index = 0;
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count       = std::min(batch_size, vector_count - begin);
        const auto   features    = load_feature_batch ? loadFeatureBatch(begin, count, feature_dim, load_feature_batch)
                                                      : loadFeatureBatch(begin, count, feature_dim, load_feature);
        const auto   assignments = assignFeatureBatchToCentroids(features, count, feature_dim, centroids, index.nlist);
        for (const auto list_no : assignments)
        {
            ++list_sizes[static_cast<size_t>(list_no)];
        }
        reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, assign_batch_index++, begin, count,
                            std::min(vector_count, irt::checkedSizeAdd(begin, count, "ImageSearch progress index")),
                            progress_total);
    }

    CpuOnDiskIvfHeader header;
    header.nlist     = static_cast<uint64_t>(index.nlist);
    header.code_size = static_cast<uint64_t>(index.code_size);
    header.ntotal    = static_cast<uint64_t>(vector_count);

    const auto list_meta = makeCpuOnDiskIvfListMeta(list_sizes, index.code_size);
    initializeCpuOnDiskIvfDataFile(data_path, header, list_meta, cpuOnDiskIvfDataFileSize(list_meta, index.code_size));

    std::fstream output(data_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open Faiss on-disk data for update: %s",
                             data_path.string().c_str());
    }

    std::vector<uint64_t> list_offsets(index.nlist, 0);
    size_t                add_batch_index = 0;
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count       = std::min(batch_size, vector_count - begin);
        const auto   features    = load_feature_batch ? loadFeatureBatch(begin, count, feature_dim, load_feature_batch)
                                                      : loadFeatureBatch(begin, count, feature_dim, load_feature);
        const auto   assignments = assignFeatureBatchToCentroids(features, count, feature_dim, centroids, index.nlist);

        std::vector<std::pair<faiss::idx_t, size_t>> ordered;
        ordered.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            ordered.emplace_back(assignments[i], i);
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

        for (size_t group_begin = 0; group_begin < ordered.size();)
        {
            const auto list_no   = ordered[group_begin].first;
            size_t     group_end = group_begin + 1;
            while (group_end < ordered.size() && ordered[group_end].first == list_no)
            {
                ++group_end;
            }

            const size_t group_count = group_end - group_begin;
            const size_t list_index  = static_cast<size_t>(list_no);
            const auto  &meta        = list_meta[list_index];
            const auto   entry_start = list_offsets[list_index];

            std::vector<faiss::idx_t> ids(group_count);
            std::vector<float> codes(irt::checkedSizeMul(group_count, static_cast<size_t>(feature_dim),
                                                         "ImageSearch code batch"));
            for (size_t i = 0; i < group_count; ++i)
            {
                const size_t batch_index = ordered[group_begin + i].second;
                ids[i] = static_cast<faiss::idx_t>(irt::checkedSizeAdd(begin, batch_index, "ImageSearch vector index"));
                const auto source_offset
                    = irt::checkedSizeMul(batch_index, static_cast<size_t>(feature_dim), "ImageSearch source offset");
                const auto destination_offset
                    = irt::checkedSizeMul(i, static_cast<size_t>(feature_dim), "ImageSearch destination offset");
                std::copy_n(features.data() + source_offset, static_cast<size_t>(feature_dim),
                            codes.data() + destination_offset);
            }

            const auto ids_offset = checkedSizeToU64(
                irt::checkedSizeAdd(checkedU64ToSize(meta.ids_offset, "Faiss ids offset"),
                                    irt::checkedSizeMul(checkedU64ToSize(entry_start, "Faiss entry offset"),
                                                        sizeof(faiss::idx_t), "Faiss ids entry offset"),
                                    "Faiss ids write offset"),
                "Faiss ids write offset");
            const auto codes_offset = checkedSizeToU64(
                irt::checkedSizeAdd(checkedU64ToSize(meta.codes_offset, "Faiss codes offset"),
                                    irt::checkedSizeMul(checkedU64ToSize(entry_start, "Faiss entry offset"),
                                                        index.code_size, "Faiss codes entry offset"),
                                    "Faiss codes write offset"),
                "Faiss codes write offset");
            writeAt(output, ids_offset, ids.data(),
                    irt::checkedSizeMul(ids.size(), sizeof(faiss::idx_t), "Faiss ids write size"));
            writeAt(output, codes_offset, codes.data(),
                    irt::checkedSizeMul(codes.size(), sizeof(float), "Faiss codes write size"));

            const auto current_offset = checkedU64ToSize(list_offsets[list_index], "Faiss list entry offset");
            list_offsets[list_index]
                = checkedSizeToU64(irt::checkedSizeAdd(current_offset, group_count, "Faiss list entry offset"),
                                   "Faiss list entry offset");
            group_begin = group_end;
        }

        reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, add_batch_index++, begin, count,
                            irt::checkedSizeAdd(vector_count,
                                                std::min(vector_count,
                                                         irt::checkedSizeAdd(begin, count, "ImageSearch progress index")),
                                                "ImageSearch progress count"),
                            progress_total);
    }
}

/**
 * @brief 将磁盘倒排列表挂接到已加载的 IVF 索引。
 *
 * 用 ``InferRtOnDiskInvertedLists`` 替换索引内原有的倒排列表，并同步 ``ntotal``。
 *
 * @param index 从 ``.faiss`` 读入的 Faiss 索引，须为 ``IndexIVF`` 类型。
 * @param data_path 倒排列表侧车文件路径。
 * @throws irt::Exception 索引类型不匹配或 ``nlist``/``code_size`` 不一致时抛出。
 */
inline void attachCpuOnDiskInvertedLists(faiss::Index &index, const std::filesystem::path &data_path)
{
    auto *ivf_index = dynamic_cast<faiss::IndexIVF *>(&index);
    if (!ivf_index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Faiss disk index must be an IVF index with on-disk inverted lists");
    }

    auto on_disk_invlists = std::make_unique<InferRtOnDiskInvertedLists>(data_path);
    if (on_disk_invlists->nlist != ivf_index->nlist || on_disk_invlists->code_size != ivf_index->code_size)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Faiss on-disk data does not match IVF index");
    }

    const auto entry_count = on_disk_invlists->entryCount();
    if (entry_count > static_cast<size_t>(std::numeric_limits<faiss::idx_t>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Faiss entry count exceeds idx_t range");
    }
    ivf_index->replace_invlists(on_disk_invlists.release(), true);
    index.ntotal = static_cast<faiss::idx_t>(entry_count);
}

/**
 * @brief 根据图库规模启发式选择 IVF 聚类数 ``nlist``。
 * @param vector_count 图库向量数量。
 * @return 介于 1 与 ``min(vector_count, 4096)`` 之间的聚类数。
 */
size_t chooseCpuOnDiskIvfListCount(size_t vector_count)
{
    if (vector_count == 0)
    {
        return 1;
    }

    const auto by_sqrt          = static_cast<size_t>(std::sqrt(static_cast<double>(vector_count)));
    const auto by_training_size = std::max<size_t>(1, vector_count / 64);
    return std::max<size_t>(1, std::min<size_t>({vector_count, by_sqrt, by_training_size, 4096}));
}

/**
 * @brief 在给定字节预算下，计算可一次性缓冲的向量条数上限。
 */
inline size_t maxCpuOnDiskIvfBufferedVectorCount(int feature_dim, size_t byte_limit)
{
    if (feature_dim <= 0 || byte_limit == 0)
    {
        return 1;
    }

    const auto dim = static_cast<size_t>(feature_dim);
    const auto bytes_per_vector = irt::checkedSizeMul(dim, sizeof(float), "Faiss feature bytes");
    if (bytes_per_vector > byte_limit)
    {
        return 1;
    }
    return std::max<size_t>(1, byte_limit / bytes_per_vector);
}

/**
 * @brief 综合图库规模与内存约束选择 IVF 聚类数 ``nlist``。
 */
size_t chooseCpuOnDiskIvfListCount(size_t vector_count, int feature_dim)
{
    const auto by_gallery = chooseCpuOnDiskIvfListCount(vector_count);
    const auto by_memory  = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    return std::max<size_t>(1, std::min<size_t>({vector_count, by_gallery, by_memory}));
}

/**
 * @brief 选择 IVF 训练采样向量条数（不超过图库总量与内存上限）。
 */
size_t chooseCpuOnDiskIvfTrainingCount(size_t vector_count, int feature_dim, size_t nlist)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    const auto cap       = std::min(kCpuOnDiskIvfMaxTrainingVectors, by_memory);
    return std::min(vector_count, std::max(nlist, cap));
}

/**
 * @brief 将用户请求的构建批大小限制在图库规模与批缓冲内存上限内。
 */
size_t chooseFaissIndexBuildBatchSize(size_t requested_batch_size, size_t vector_count, int feature_dim)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kFaissIndexBuildMaxBatchBytes);
    return std::max<size_t>(1, std::min<size_t>({requested_batch_size, vector_count, by_memory}));
}

std::unique_ptr<faiss::Index> buildRamFlatIndex(
    size_t vector_count, int feature_dim, size_t requested_batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const BuildProgressCallback &progress_callback)
{
    if (vector_count == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Flat Faiss index requires non-empty features");
    }

    auto index = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    const auto batch_size = chooseFaissIndexBuildBatchSize(requested_batch_size, vector_count, feature_dim);
    size_t batch_index = 0;
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count = std::min(batch_size, vector_count - begin);
        const auto features
            = load_feature_batch ? loadFeatureBatch(begin, count, feature_dim, load_feature_batch)
                                  : loadFeatureBatch(begin, count, feature_dim, load_feature);
        index->add(checkedSizeToFaissIdx(count, "Faiss add batch count"), features.data());
        reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, batch_index++, begin, count,
                            irt::checkedSizeAdd(begin, count, "ImageSearch progress index"), vector_count);
    }
    return index;
}

/**
 * @brief 从训练特征中均匀采样生成 IVF 量化器初始聚类中心。
 */
inline std::vector<float> makeCpuOnDiskIvfCentroids(const std::vector<float> &training_features, size_t training_count,
                                                    int feature_dim, size_t nlist)
{
    const auto dim = static_cast<size_t>(feature_dim);
    std::vector<float> centroids(irt::checkedSizeMul(nlist, dim, "Faiss centroid values"));
    for (size_t list_index = 0; list_index < nlist; ++list_index)
    {
        const size_t source_index = nlist == 1
                                         ? 0
                                         : irt::checkedSizeMul(list_index, training_count - 1,
                                                               "Faiss training sample offset")
                                               / (nlist - 1);
        const auto source_offset      = irt::checkedSizeMul(source_index, dim, "Faiss training offset");
        const auto destination_offset = irt::checkedSizeMul(list_index, dim, "Faiss centroid offset");
        std::copy_n(training_features.data() + source_offset, dim, centroids.data() + destination_offset);
    }
    return centroids;
}

inline constexpr size_t kRamIvfPqMaxSubQuantizers = 64;
inline constexpr size_t kRamIvfPqBitsPerCode      = 8;

/**
 * @brief 为 RAM IVF-PQ 选择 PQ 子量化器数量。
 *
 * Faiss PQ 要求特征维度能被子量化器数量整除；这里从上限向下查找可整除值，尽量保留压缩效率。
 *
 * @param feature_dim 单条特征维度。
 * @return 可整除 ``feature_dim`` 的子量化器数量，最小为 1。
 */
inline size_t chooseRamIvfPqSubQuantizerCount(int feature_dim)
{
    if (feature_dim <= 0)
    {
        return 1;
    }

    const auto dim = static_cast<size_t>(feature_dim);
    for (size_t candidate = std::min(kRamIvfPqMaxSubQuantizers, dim); candidate > 1; --candidate)
    {
        if (dim % candidate == 0)
        {
            return candidate;
        }
    }
    return 1;
}

/**
 * @brief 选择 RAM IVF-PQ 训练样本数量。
 * @param vector_count 图库向量总数。
 * @param feature_dim 单条特征维度。
 * @param nlist IVF 聚类数量。
 * @return 不超过图库规模和内存预算的训练向量数量。
 */
inline size_t chooseRamIvfPqTrainingCount(size_t vector_count, int feature_dim, size_t nlist)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    return std::min(vector_count, std::max<size_t>(nlist, by_memory));
}

/**
 * @brief 根据训练样本数选择 PQ code 位宽。
 *
 * GPU Faiss 当前固定使用 8-bit PQ code；CPU 路径在训练样本较少时降低位宽，避免训练阶段要求过多样本。
 *
 * @param training_count 实际训练向量数量。
 * @param require_gpu_compatible 是否要求后续可迁移到 GPU Faiss。
 * @return 每个 PQ 子码本的 bit 数。
 */
inline size_t chooseRamIvfPqBitsPerCode(size_t training_count, bool require_gpu_compatible = false)
{
    if (require_gpu_compatible)
    {
        return kRamIvfPqBitsPerCode;
    }

    size_t bits = 0;
    while (bits < kRamIvfPqBitsPerCode && (size_t{1} << (bits + 1)) <= training_count)
    {
        ++bits;
    }
    return std::max<size_t>(1, bits);
}

/**
 * @brief 根据 ``nlist`` 选择 IVF 检索时的探测列表数 ``nprobe``。
 * @param nlist IVF 聚类数量。
 * @return 介于 1 与 ``min(nlist, 16)`` 之间的 ``nprobe``。
 */
inline size_t chooseCpuOnDiskIvfProbeCount(size_t nlist)
{
    return std::max<size_t>(1, std::min<size_t>(nlist, 16));
}

/**
 * @brief 为 CPU 磁盘 IVF 索引配置默认检索参数。
 * @param index 待配置的 Faiss 索引；若为 IVF 类型则设置 ``nprobe``。
 */
inline void configureCpuOnDiskIvfSearch(faiss::Index &index)
{
    if (auto *ivf_index = dynamic_cast<faiss::IndexIVF *>(&index))
    {
        ivf_index->nprobe = chooseCpuOnDiskIvfProbeCount(ivf_index->nlist);
    }
}

/**
 * @brief 加载 CPU 磁盘 IVF+Flat 索引并挂接磁盘倒排列表。
 *
 * 读取 ``index_path`` 对应的 Faiss 索引文件，再从 ``.ivfdata`` 映射倒排列表，
 * 并应用默认 ``nprobe`` 配置。
 *
 * @param index_path Faiss 索引文件路径。
 * @return 可搜索的 Faiss 索引；倒排列表驻留磁盘。
 * @throws irt::Exception 加载失败、非 IVF 类型或侧车数据不匹配时抛出。
 */
std::unique_ptr<faiss::Index> loadCpuOnDiskIvfFlatIndex(const std::filesystem::path &index_path)
{
    auto index = std::unique_ptr<faiss::Index>(faiss::read_index(index_path.string().c_str()));
    if (!index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load Faiss on-disk index: %s",
                             index_path.string().c_str());
    }
    if (!dynamic_cast<faiss::IndexIVF *>(index.get()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Faiss disk index must be an IVF index with on-disk inverted lists");
    }

    attachCpuOnDiskInvertedLists(*index, cpuOnDiskIvfDataPath(index_path));
    configureCpuOnDiskIvfSearch(*index);
    return index;
}

/**
 * @brief 构建 CPU 磁盘 IVF+Flat 内积索引并落盘。
 *
 * 流程：按图库规模创建 ``IVF{nlist},Flat`` 索引 → 采样训练 → 写入骨架
 * ``.faiss`` → 分批统计 IVF list 大小 → 分批写入 ``.ivfdata`` → 重新加载为磁盘倒排列表索引。
 *
 * @param vector_count 图库向量数量。
 * @param feature_dim 特征维度。
 * @param index_path 输出索引文件路径。
 * @param batch_size 特征提取与磁盘倒排列表构建批量。
 * @param load_feature 按图库下标加载单条归一化特征的回调。
 * @param load_feature_batch 按连续下标区间批量加载特征的回调。
 * @param load_feature_index_batch 按任意下标列表批量加载训练特征的回调。
 * @param progress_callback 构建进度回调。
 * @return 已挂接磁盘倒排列表、可直接用于检索的 Faiss 索引。
 * @throws irt::Exception 参数非法、训练/添加失败或落盘失败时抛出。
 */
std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const LoadFeatureIndexedBatchCallback &load_feature_index_batch,
    const BuildProgressCallback &progress_callback);

std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(size_t vector_count, int feature_dim,
                                                         const std::filesystem::path &index_path, size_t batch_size,
                                                         const LoadFeatureCallback &load_feature,
                                                         const BuildProgressCallback &progress_callback)
{
    return buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, batch_size, load_feature, {}, {},
                                      progress_callback);
}

std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(size_t vector_count, int feature_dim,
                                                         const std::filesystem::path &index_path, size_t batch_size,
                                                         const LoadFeatureCallback &load_feature,
                                                         const LoadFeatureBatchCallback &load_feature_batch,
                                                         const BuildProgressCallback &progress_callback)
{
    return buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, batch_size, load_feature,
                                      load_feature_batch, {}, progress_callback);
}

std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const LoadFeatureIndexedBatchCallback &load_feature_index_batch, const BuildProgressCallback &progress_callback)
{
    if (vector_count == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "On-disk Faiss IVF index requires non-empty features");
    }

    const size_t nlist          = chooseCpuOnDiskIvfListCount(vector_count, feature_dim);
    const size_t training_count = chooseCpuOnDiskIvfTrainingCount(vector_count, feature_dim, nlist);
    const size_t stride         = std::max<size_t>(1, vector_count / training_count);

    const auto training
        = loadTrainingFeatures(vector_count, feature_dim, training_count, stride, batch_size, load_feature,
                               load_feature_batch, load_feature_index_batch);
    const size_t actual_training_count = training.count;
    if (actual_training_count < nlist)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Not enough training features for IVF index");
    }

    auto quantizer = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    auto centroids = makeCpuOnDiskIvfCentroids(training.features, actual_training_count, feature_dim, nlist);
    quantizer->add(checkedSizeToFaissIdx(nlist, "Faiss nlist"), centroids.data());

    auto  index = std::make_unique<faiss::IndexIVFFlat>(quantizer.release(), static_cast<size_t>(feature_dim), nlist,
                                                        faiss::METRIC_INNER_PRODUCT);
    auto *ivf_index       = index.get();
    ivf_index->own_fields = true;
    ivf_index->is_trained = true;
    ivf_index->nprobe     = chooseCpuOnDiskIvfProbeCount(nlist);

    faiss::write_index(index.get(), index_path.string().c_str());

    writeCpuOnDiskIvfDataFileBatched(*ivf_index, vector_count, feature_dim, cpuOnDiskIvfDataPath(index_path), centroids,
                                     batch_size, load_feature, load_feature_batch, progress_callback);
    return loadCpuOnDiskIvfFlatIndex(index_path);
}

/**
 * @brief 构建 RAM 常驻的 IVF-PQ 内积索引。
 *
 * 训练样本和图库向量添加都按 ``batch_size`` 分块；
 * 若 ``require_gpu_compatible`` 为 true，则选择 GPU Faiss 可接受的 PQ code 位宽。
 *
 * @param vector_count 图库向量总数。
 * @param feature_dim 单条特征维度。
 * @param batch_size 特征提取与图库向量添加批量。
 * @param load_feature 按图库下标加载单条归一化特征的回调。
 * @param load_feature_batch 按连续下标区间批量加载特征的回调。
 * @param load_feature_index_batch 按任意下标列表批量加载训练特征的回调。
 * @param progress_callback 构建进度回调。
 * @param require_gpu_compatible 是否生成可迁移到 GPU Faiss 的 PQ 配置。
 * @return 已训练并添加图库向量的 CPU Faiss 索引。
 */
std::unique_ptr<faiss::Index> buildRamIvfPqIndex(size_t vector_count, int feature_dim, size_t batch_size,
                                                 const LoadFeatureCallback &load_feature,
                                                 const LoadFeatureBatchCallback &load_feature_batch,
                                                 const LoadFeatureIndexedBatchCallback &load_feature_index_batch,
                                                 const BuildProgressCallback &progress_callback,
                                                 bool require_gpu_compatible);

std::unique_ptr<faiss::Index> buildRamIvfPqIndex(size_t vector_count, int feature_dim, size_t batch_size,
                                                 const LoadFeatureCallback &load_feature,
                                                 const BuildProgressCallback &progress_callback,
                                                 bool require_gpu_compatible)
{
    return buildRamIvfPqIndex(vector_count, feature_dim, batch_size, load_feature, {}, {}, progress_callback,
                              require_gpu_compatible);
}

std::unique_ptr<faiss::Index> buildRamIvfPqIndex(size_t vector_count, int feature_dim, size_t batch_size,
                                                 const LoadFeatureCallback &load_feature,
                                                 const LoadFeatureBatchCallback &load_feature_batch,
                                                 const BuildProgressCallback &progress_callback,
                                                 bool require_gpu_compatible)
{
    return buildRamIvfPqIndex(vector_count, feature_dim, batch_size, load_feature, load_feature_batch, {},
                              progress_callback, require_gpu_compatible);
}

std::unique_ptr<faiss::Index> buildRamIvfPqIndex(size_t vector_count, int feature_dim, size_t batch_size,
                                                 const LoadFeatureCallback &load_feature,
                                                 const LoadFeatureBatchCallback &load_feature_batch,
                                                 const LoadFeatureIndexedBatchCallback &load_feature_index_batch,
                                                 const BuildProgressCallback &progress_callback,
                                                 bool require_gpu_compatible)
{
    if (vector_count == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "In-memory Faiss IVF-PQ index requires non-empty features");
    }
    if (batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Faiss IVF-PQ build batch size must be positive");
    }

    const size_t nlist          = chooseCpuOnDiskIvfListCount(vector_count, feature_dim);
    const size_t training_count = chooseRamIvfPqTrainingCount(vector_count, feature_dim, nlist);
    const size_t stride         = std::max<size_t>(1, vector_count / training_count);

    auto training = loadTrainingFeatures(vector_count, feature_dim, training_count, stride, batch_size, load_feature,
                                         load_feature_batch, load_feature_index_batch);
    const size_t actual_training_count = training.count;
    if (actual_training_count < nlist)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Not enough training features for IVF-PQ index");
    }

    const size_t sub_quantizers = chooseRamIvfPqSubQuantizerCount(feature_dim);
    const size_t bits_per_code  = chooseRamIvfPqBitsPerCode(actual_training_count, require_gpu_compatible);
    const size_t pq_centroids   = size_t{1} << bits_per_code;

    const float       *faiss_training_data  = training.features.data();
    faiss::idx_t       faiss_training_count = checkedSizeToFaissIdx(actual_training_count, "Faiss training count");
    std::vector<float> padded_training_features;
    if (actual_training_count < pq_centroids)
    {
        const auto dim = static_cast<size_t>(feature_dim);
        padded_training_features.reserve(irt::checkedSizeMul(pq_centroids, dim, "Faiss padded training features"));
        for (size_t i = 0; i < pq_centroids; ++i)
        {
            const size_t source_index = i % actual_training_count;
            const auto source_offset = irt::checkedSizeMul(source_index, dim, "Faiss training offset");
            const auto source_end    = irt::checkedSizeAdd(source_offset, dim, "Faiss training offset");
            padded_training_features.insert(
                padded_training_features.end(),
                training.features.begin() + static_cast<std::ptrdiff_t>(source_offset),
                training.features.begin() + static_cast<std::ptrdiff_t>(source_end));
        }
        faiss_training_data  = padded_training_features.data();
        faiss_training_count = checkedSizeToFaissIdx(pq_centroids, "Faiss padded training count");
    }

    auto coarse_quantizer = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    auto index = std::make_unique<faiss::IndexIVFPQ>(coarse_quantizer.release(), static_cast<size_t>(feature_dim),
                                                     nlist, sub_quantizers, bits_per_code, faiss::METRIC_INNER_PRODUCT);
    auto *ivfpq_index                 = index.get();
    ivfpq_index->own_fields           = true;
    const auto points_per_pq_centroid = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(std::numeric_limits<int>::max()),
                         irt::checkedSizeAdd(static_cast<size_t>(faiss_training_count), pq_centroids - 1,
                                             "Faiss PQ points")
                             / pq_centroids));
    ivfpq_index->pq.cp.max_points_per_centroid
        = std::max(ivfpq_index->pq.cp.max_points_per_centroid, points_per_pq_centroid);
    ivfpq_index->train(faiss_training_count, faiss_training_data);
    ivfpq_index->nprobe = chooseCpuOnDiskIvfProbeCount(nlist);
    std::vector<float>().swap(training.features);
    std::vector<size_t>().swap(training.indices);
    std::vector<float>().swap(padded_training_features);

    batch_size = chooseFaissIndexBuildBatchSize(batch_size, vector_count, feature_dim);
    size_t add_batch_index = 0;
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count    = std::min(batch_size, vector_count - begin);
        const auto   features = load_feature_batch ? loadFeatureBatch(begin, count, feature_dim, load_feature_batch)
                                                   : loadFeatureBatch(begin, count, feature_dim, load_feature);
        ivfpq_index->add(checkedSizeToFaissIdx(count, "Faiss add batch count"), features.data());
        reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, add_batch_index++, begin, count,
                            std::min(vector_count, irt::checkedSizeAdd(begin, count, "ImageSearch progress index")),
                            vector_count);
    }

    return index;
}

} // namespace irt::features::priv
