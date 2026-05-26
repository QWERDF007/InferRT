#pragma once

/**
 * @file ImageSearchFaissIndex.hpp
 * @brief CPU 磁盘 IVF+Flat 索引的构建、落盘与内存映射加载。
 */

#include <inferrt/core/Exception.hpp>

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
#    ifdef ERROR_INVALID_OPERATION
#        undef ERROR_INVALID_OPERATION
#    endif
#    ifdef ERROR_NOT_IMPLEMENTED
#        undef ERROR_NOT_IMPLEMENTED
#    endif
#endif

namespace irt::features::priv {

/**
 * @brief 生成 CPU 磁盘 IVF 倒排列表的伴生数据文件路径。
 * @param index_path Faiss 索引文件路径（``.faiss``）。
 * @return ``<index_path>.ivfdata``。
 */
inline std::filesystem::path cpuOnDiskIvfDataPath(const std::filesystem::path &index_path)
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
struct CpuOnDiskIvfListMeta
{
    uint64_t size{0};         ///< 该列表中的向量条数。
    uint64_t ids_offset{0};   ///< ``idx_t`` ID 数组在文件中的字节偏移。
    uint64_t codes_offset{0}; ///< 编码向量数据在文件中的字节偏移。
};

/**
 * @brief 将数值向上对齐到指定边界。
 * @param value 待对齐的值。
 * @param alignment 对齐字节数（须为正）。
 * @return 对齐后的值。
 */
inline size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
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
    if (offset + sizeof(T) > mapped_file.size())
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

        nlist        = static_cast<size_t>(header.nlist);
        code_size    = static_cast<size_t>(header.code_size);
        entry_count_ = static_cast<size_t>(header.ntotal);

        const size_t meta_offset = sizeof(CpuOnDiskIvfHeader);
        const size_t meta_bytes  = nlist * sizeof(CpuOnDiskIvfListMeta);
        if (meta_offset + meta_bytes > mapped_file_.size())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Corrupt Faiss on-disk inverted lists: %s",
                                 path.string().c_str());
        }

        list_meta_.resize(nlist);
        std::memcpy(list_meta_.data(), mapped_file_.data() + meta_offset, meta_bytes);
        for (const auto &meta : list_meta_)
        {
            const auto ids_end   = meta.ids_offset + meta.size * sizeof(faiss::idx_t);
            const auto codes_end = meta.codes_offset + meta.size * code_size;
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
 * @brief 向输出流写入指定数量的零字节填充。
 * @param output 目标二进制输出流。
 * @param count 填充字节数。
 */
inline void writeAt(std::fstream &output, uint64_t offset, const void *data, size_t size)
{
    if (size == 0)
    {
        return;
    }

    output.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to seek Faiss on-disk data");
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
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
inline std::vector<float> loadFeatureBatch(size_t begin, size_t count, int feature_dim,
                                           const std::function<std::vector<float>(size_t)> &load_feature)
{
    std::vector<float> features;
    features.reserve(count * static_cast<size_t>(feature_dim));
    for (size_t i = 0; i < count; ++i)
    {
        auto feature = load_feature(begin + i);
        if (feature.size() != static_cast<size_t>(feature_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature size");
        }
        features.insert(features.end(), feature.begin(), feature.end());
    }
    return features;
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
    std::vector<faiss::idx_t> assignments(count);
    const auto                dim = static_cast<size_t>(feature_dim);
    for (size_t row = 0; row < count; ++row)
    {
        const float *feature = features.data() + row * dim;
        float        best    = std::numeric_limits<float>::lowest();
        size_t       best_no = 0;
        for (size_t list_no = 0; list_no < nlist; ++list_no)
        {
            const float *centroid = centroids.data() + list_no * dim;
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
inline std::vector<CpuOnDiskIvfListMeta> makeCpuOnDiskIvfListMeta(const std::vector<uint64_t> &list_sizes,
                                                                  size_t                       code_size)
{
    std::vector<CpuOnDiskIvfListMeta> list_meta(list_sizes.size());
    size_t offset = sizeof(CpuOnDiskIvfHeader) + list_meta.size() * sizeof(CpuOnDiskIvfListMeta);
    offset        = alignUp(offset, alignof(faiss::idx_t));

    for (size_t list_no = 0; list_no < list_sizes.size(); ++list_no)
    {
        auto &meta      = list_meta[list_no];
        meta.size       = list_sizes[list_no];
        meta.ids_offset = static_cast<uint64_t>(offset);
        offset += static_cast<size_t>(meta.size) * sizeof(faiss::idx_t);
        offset            = alignUp(offset, alignof(float));
        meta.codes_offset = static_cast<uint64_t>(offset);
        offset += static_cast<size_t>(meta.size) * code_size;
        offset = alignUp(offset, alignof(faiss::idx_t));
    }

    return list_meta;
}

/**
 * @brief 根据列表元数据计算 ``.ivfdata`` 文件所需的总字节数。
 */
inline size_t cpuOnDiskIvfDataFileSize(const std::vector<CpuOnDiskIvfListMeta> &list_meta, size_t code_size)
{
    size_t file_size = sizeof(CpuOnDiskIvfHeader) + list_meta.size() * sizeof(CpuOnDiskIvfListMeta);
    for (const auto &meta : list_meta)
    {
        file_size = std::max(file_size, static_cast<size_t>(meta.ids_offset + meta.size * sizeof(faiss::idx_t)));
        file_size = std::max(file_size, static_cast<size_t>(meta.codes_offset + meta.size * code_size));
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
    output.write(reinterpret_cast<const char *>(list_meta.data()),
                 static_cast<std::streamsize>(list_meta.size() * sizeof(CpuOnDiskIvfListMeta)));
    if (file_size > 0)
    {
        output.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
        static constexpr char kZero = 0;
        output.write(&kZero, 1);
    }
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to initialize Faiss on-disk data: %s",
                             data_path.string().c_str());
    }
}

/** @brief 前向声明：按内存上限裁剪磁盘构建批大小。 */
inline size_t chooseCpuOnDiskIvfBuildBatchSize(size_t requested_batch_size, size_t vector_count, int feature_dim);

/**
 * @brief 分两趟扫描图库：先统计各 IVF 列表大小，再分批写入 ID 与编码到 ``.ivfdata``。
 */
inline void writeCpuOnDiskIvfDataFileBatched(const faiss::IndexIVF &index, size_t vector_count, int feature_dim,
                                             const std::filesystem::path &data_path,
                                             const std::vector<float> &centroids, size_t batch_size,
                                             const std::function<std::vector<float>(size_t)> &load_feature)
{
    if (batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Faiss on-disk build batch size must be positive");
    }
    if (index.code_size != static_cast<size_t>(feature_dim) * sizeof(float))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "CPU on-disk ImageSearch currently requires IVF Flat float codes");
    }
    if (centroids.size() != index.nlist * static_cast<size_t>(feature_dim))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid IVF centroid buffer size");
    }

    batch_size = chooseCpuOnDiskIvfBuildBatchSize(batch_size, vector_count, feature_dim);

    std::vector<uint64_t> list_sizes(index.nlist, 0);
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count       = std::min(batch_size, vector_count - begin);
        const auto   features    = loadFeatureBatch(begin, count, feature_dim, load_feature);
        const auto   assignments = assignFeatureBatchToCentroids(features, count, feature_dim, centroids, index.nlist);
        for (const auto list_no : assignments)
        {
            ++list_sizes[static_cast<size_t>(list_no)];
        }
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
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count       = std::min(batch_size, vector_count - begin);
        const auto   features    = loadFeatureBatch(begin, count, feature_dim, load_feature);
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
            std::vector<float>        codes(group_count * static_cast<size_t>(feature_dim));
            for (size_t i = 0; i < group_count; ++i)
            {
                const size_t batch_index = ordered[group_begin + i].second;
                ids[i]                   = static_cast<faiss::idx_t>(begin + batch_index);
                std::copy_n(features.data() + batch_index * static_cast<size_t>(feature_dim),
                            static_cast<size_t>(feature_dim), codes.data() + i * static_cast<size_t>(feature_dim));
            }

            writeAt(output, meta.ids_offset + entry_start * sizeof(faiss::idx_t), ids.data(),
                    ids.size() * sizeof(faiss::idx_t));
            writeAt(output, meta.codes_offset + entry_start * index.code_size, codes.data(),
                    codes.size() * sizeof(float));

            list_offsets[list_index] += group_count;
            group_begin = group_end;
        }
    }
}

/**
 * @brief 将 IVF 索引的倒排列表序列化到 ``.ivfdata`` 侧车文件。
 *
 * 布局为：文件头 → 各列表元数据 → 按对齐规则排列的 ID 与编码块。
 *
 * @param index 已训练且已添加向量的 IVF 索引。
 * @param data_path 输出路径，通常为 ``cpuOnDiskIvfDataPath(index_path)``。
 * @throws irt::Exception 倒排列表为空或写入失败时抛出。
 */
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
    ivf_index->replace_invlists(on_disk_invlists.release(), true);
    index.ntotal = static_cast<faiss::idx_t>(entry_count);
}

/**
 * @brief 根据图库规模启发式选择 IVF 聚类数 ``nlist``。
 * @param vector_count 图库向量数量。
 * @return 介于 1 与 ``min(vector_count, 4096)`` 之间的聚类数。
 */
inline size_t chooseCpuOnDiskIvfListCount(size_t vector_count)
{
    if (vector_count == 0)
    {
        return 1;
    }

    const auto by_sqrt          = static_cast<size_t>(std::sqrt(static_cast<double>(vector_count)));
    const auto by_training_size = std::max<size_t>(1, vector_count / 64);
    return std::max<size_t>(1, std::min<size_t>({vector_count, by_sqrt, by_training_size, 4096}));
}

/// IVF 训练阶段最多采样的向量条数。
inline constexpr size_t kCpuOnDiskIvfMaxTrainingVectors = 8192;
/// IVF 训练特征缓冲区的最大字节数（512 MiB）。
inline constexpr size_t kCpuOnDiskIvfMaxTrainingBytes = 512ULL * 1024ULL * 1024ULL;
/// 磁盘构建批特征缓冲区的最大字节数（256 MiB）。
inline constexpr size_t kCpuOnDiskIvfMaxBatchBytes = 256ULL * 1024ULL * 1024ULL;

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
    if (dim > byte_limit / sizeof(float))
    {
        return 1;
    }
    return std::max<size_t>(1, byte_limit / (dim * sizeof(float)));
}

/**
 * @brief 综合图库规模与内存约束选择 IVF 聚类数 ``nlist``。
 */
inline size_t chooseCpuOnDiskIvfListCount(size_t vector_count, int feature_dim)
{
    const auto by_gallery = chooseCpuOnDiskIvfListCount(vector_count);
    const auto by_memory  = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    return std::max<size_t>(1, std::min<size_t>({vector_count, by_gallery, by_memory}));
}

/**
 * @brief 选择 IVF 训练采样向量条数（不超过图库总量与内存上限）。
 */
inline size_t chooseCpuOnDiskIvfTrainingCount(size_t vector_count, int feature_dim, size_t nlist)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    const auto cap       = std::min(kCpuOnDiskIvfMaxTrainingVectors, by_memory);
    return std::min(vector_count, std::max(nlist, cap));
}

/**
 * @brief 将用户请求的构建批大小限制在图库规模与批缓冲内存上限内。
 */
inline size_t chooseCpuOnDiskIvfBuildBatchSize(size_t requested_batch_size, size_t vector_count, int feature_dim)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxBatchBytes);
    return std::max<size_t>(1, std::min<size_t>({requested_batch_size, vector_count, by_memory}));
}

/**
 * @brief 从训练特征中均匀采样生成 IVF 量化器初始聚类中心。
 */
inline std::vector<float> makeCpuOnDiskIvfCentroids(const std::vector<float> &training_features, size_t training_count,
                                                    int feature_dim, size_t nlist)
{
    std::vector<float> centroids(nlist * static_cast<size_t>(feature_dim));
    for (size_t list_index = 0; list_index < nlist; ++list_index)
    {
        const size_t source_index = nlist == 1 ? 0 : (list_index * (training_count - 1)) / (nlist - 1);
        std::copy_n(training_features.data() + source_index * static_cast<size_t>(feature_dim),
                    static_cast<size_t>(feature_dim), centroids.data() + list_index * static_cast<size_t>(feature_dim));
    }
    return centroids;
}

inline constexpr size_t kRamIvfPqMaxSubQuantizers = 64;
inline constexpr size_t kRamIvfPqBitsPerCode      = 8;

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

inline size_t chooseRamIvfPqTrainingCount(size_t vector_count, int feature_dim, size_t nlist)
{
    const auto by_memory = maxCpuOnDiskIvfBufferedVectorCount(feature_dim, kCpuOnDiskIvfMaxTrainingBytes);
    return std::min(vector_count, std::max<size_t>(nlist, by_memory));
}

inline size_t chooseRamIvfPqBitsPerCode(size_t training_count)
{
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
inline std::unique_ptr<faiss::Index> loadCpuOnDiskIvfFlatIndex(const std::filesystem::path &index_path)
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
 * @param batch_size 磁盘倒排列表构建时的特征批量。
 * @param load_feature 按图库下标加载单条归一化特征的回调。
 * @return 已挂接磁盘倒排列表、可直接用于检索的 Faiss 索引。
 * @throws irt::Exception 参数非法、训练/添加失败或落盘失败时抛出。
 */
inline std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const std::function<std::vector<float>(size_t)> &load_feature)
{
    if (vector_count == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "On-disk Faiss IVF index requires non-empty features");
    }

    const size_t nlist          = chooseCpuOnDiskIvfListCount(vector_count, feature_dim);
    const size_t training_count = chooseCpuOnDiskIvfTrainingCount(vector_count, feature_dim, nlist);
    const size_t stride         = std::max<size_t>(1, vector_count / training_count);

    std::vector<float> training_features;
    training_features.reserve(training_count * static_cast<size_t>(feature_dim));
    for (size_t index_in_gallery = 0; index_in_gallery < vector_count
                                      && training_features.size() / static_cast<size_t>(feature_dim) < training_count;
         index_in_gallery += stride)
    {
        auto feature = load_feature(index_in_gallery);
        if (feature.size() != static_cast<size_t>(feature_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature size");
        }
        training_features.insert(training_features.end(), feature.begin(), feature.end());
    }

    const size_t actual_training_count = training_features.size() / static_cast<size_t>(feature_dim);
    if (actual_training_count < nlist)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Not enough training features for IVF index");
    }

    auto quantizer = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    auto centroids = makeCpuOnDiskIvfCentroids(training_features, actual_training_count, feature_dim, nlist);
    quantizer->add(static_cast<faiss::idx_t>(nlist), centroids.data());

    auto  index = std::make_unique<faiss::IndexIVFFlat>(quantizer.release(), static_cast<size_t>(feature_dim), nlist,
                                                        faiss::METRIC_INNER_PRODUCT);
    auto *ivf_index       = index.get();
    ivf_index->own_fields = true;
    ivf_index->is_trained = true;
    ivf_index->nprobe     = chooseCpuOnDiskIvfProbeCount(nlist);

    faiss::write_index(index.get(), index_path.string().c_str());

    writeCpuOnDiskIvfDataFileBatched(*ivf_index, vector_count, feature_dim, cpuOnDiskIvfDataPath(index_path), centroids,
                                     batch_size, load_feature);
    return loadCpuOnDiskIvfFlatIndex(index_path);
}

inline std::unique_ptr<faiss::Index> buildRamIvfPqIndex(size_t vector_count, int feature_dim, size_t batch_size,
                                                        const std::function<std::vector<float>(size_t)> &load_feature)
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

    std::vector<float>  training_features;
    std::vector<size_t> training_indices;
    training_features.reserve(training_count * static_cast<size_t>(feature_dim));
    training_indices.reserve(training_count);
    for (size_t index_in_gallery = 0; index_in_gallery < vector_count
                                      && training_features.size() / static_cast<size_t>(feature_dim) < training_count;
         index_in_gallery += stride)
    {
        auto feature = load_feature(index_in_gallery);
        if (feature.size() != static_cast<size_t>(feature_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unexpected ImageSearch feature size");
        }
        training_features.insert(training_features.end(), feature.begin(), feature.end());
        training_indices.push_back(index_in_gallery);
    }

    const size_t actual_training_count = training_features.size() / static_cast<size_t>(feature_dim);
    if (actual_training_count < nlist)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Not enough training features for IVF-PQ index");
    }

    const size_t sub_quantizers = chooseRamIvfPqSubQuantizerCount(feature_dim);
    const size_t bits_per_code  = chooseRamIvfPqBitsPerCode(actual_training_count);
    const size_t pq_centroids   = size_t{1} << bits_per_code;

    const float  *faiss_training_data  = training_features.data();
    faiss::idx_t  faiss_training_count = static_cast<faiss::idx_t>(actual_training_count);
    std::vector<float> padded_training_features;
    if (actual_training_count < pq_centroids)
    {
        padded_training_features.reserve(pq_centroids * static_cast<size_t>(feature_dim));
        for (size_t i = 0; i < pq_centroids; ++i)
        {
            const size_t source_index = i % actual_training_count;
            padded_training_features.insert(padded_training_features.end(),
                                            training_features.begin()
                                                + static_cast<std::ptrdiff_t>(source_index
                                                                              * static_cast<size_t>(feature_dim)),
                                            training_features.begin()
                                                + static_cast<std::ptrdiff_t>((source_index + 1)
                                                                              * static_cast<size_t>(feature_dim)));
        }
        faiss_training_data  = padded_training_features.data();
        faiss_training_count = static_cast<faiss::idx_t>(pq_centroids);
    }

    auto coarse_quantizer = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    auto         index
        = std::make_unique<faiss::IndexIVFPQ>(coarse_quantizer.release(), static_cast<size_t>(feature_dim), nlist,
                                              sub_quantizers, bits_per_code, faiss::METRIC_INNER_PRODUCT);
    auto *ivfpq_index       = index.get();
    ivfpq_index->own_fields = true;
    const auto points_per_pq_centroid =
        static_cast<int>(std::min<size_t>(static_cast<size_t>(std::numeric_limits<int>::max()),
                                          (static_cast<size_t>(faiss_training_count) + pq_centroids - 1)
                                              / pq_centroids));
    ivfpq_index->pq.cp.max_points_per_centroid =
        std::max(ivfpq_index->pq.cp.max_points_per_centroid, points_per_pq_centroid);
    ivfpq_index->train(faiss_training_count, faiss_training_data);
    ivfpq_index->nprobe = chooseCpuOnDiskIvfProbeCount(nlist);

    auto load_cached_feature = [&](size_t index_in_gallery) {
        const auto found = std::lower_bound(training_indices.begin(), training_indices.end(), index_in_gallery);
        if (found != training_indices.end() && *found == index_in_gallery)
        {
            const auto  training_index = static_cast<size_t>(std::distance(training_indices.begin(), found));
            const auto *begin = training_features.data() + training_index * static_cast<size_t>(feature_dim);
            return std::vector<float>(begin, begin + static_cast<size_t>(feature_dim));
        }
        return load_feature(index_in_gallery);
    };

    batch_size = chooseCpuOnDiskIvfBuildBatchSize(batch_size, vector_count, feature_dim);
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count    = std::min(batch_size, vector_count - begin);
        const auto   features = loadFeatureBatch(begin, count, feature_dim, load_cached_feature);
        ivfpq_index->add(static_cast<faiss::idx_t>(count), features.data());
    }

    return index;
}

} // namespace irt::features::priv
