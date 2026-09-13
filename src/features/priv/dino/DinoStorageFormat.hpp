#pragma once

/**
 * @file DinoStorageFormat.hpp
 * @brief 紧凑存储的底层格式：小端字节序、npy 容器与 32 字节 packed meta。
 */

#include "DinoTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace irt::features::priv {


/** @brief packed meta 记录的字节宽度。 */
constexpr size_t kDinoMetaRecordBytes = 32U;

/** @brief 一条区域描述 meta 的字段视图。 */
struct DinoPackedMeta
{
    uint32_t view_id{0};
    uint16_t x0{0};
    uint16_t y0{0};
    uint16_t x1{0};
    uint16_t y1{0};
    uint16_t representative_x{0};
    uint16_t representative_y{0};
    uint16_t member_count{0};
    uint16_t flags{0};
    float    radius{0.0F};
    uint32_t reserved0{0};
    uint32_t reserved1{0};
};

/** @brief 把 meta 编码为 32 字节小端记录。 */
void dinoEncodeMeta(const DinoPackedMeta &meta, uint8_t *out);

/** @brief 解码 32 字节小端记录。 */
DinoPackedMeta dinoDecodeMeta(const uint8_t *in);

/** @brief 读取整个文本文件。 */
std::string dinoReadTextFile(const std::filesystem::path &path);

/** @brief 文件字节长度。 */
uint64_t dinoFileSize(const std::filesystem::path &path);


/**
 * @brief 顺序写入 npy 数组文件。
 *
 * 头部区固定预留 256 字节，结束写入时回填真实 shape，因此调用方不必预先知道行数。
 * 头部字段固定为 ``fortran_order=False``，外部工具可直接读取与内存映射。
 */
class DinoNpyWriter
{
public:
    DinoNpyWriter(const std::filesystem::path &path, const std::string &descr, size_t element_bytes);
    ~DinoNpyWriter();

    DinoNpyWriter(const DinoNpyWriter &)            = delete;
    DinoNpyWriter &operator=(const DinoNpyWriter &) = delete;

    /** @brief 追加一段原始字节；长度必须是元素字节数的整数倍。 */
    void append(const void *data, size_t bytes);

    /** @brief 回填形状并结束写入。 */
    void finish(const std::vector<int64_t> &shape);

private:
    std::filesystem::path path_{};
    std::string           descr_{};
    size_t                element_bytes_{0};
    uint64_t              written_{0};
    std::ofstream         stream_{};
};

/** @brief npy 容器魔数；容器识别与写入共用同一份常量。 */
constexpr const char *kDinoNpyMagic = "\x93NUMPY";

/** @brief 预留的 npy 头部区字节数。 */
constexpr size_t kDinoNpyHeaderBytes = 256U;

/** @brief npy 数组的只读访问器。 */
class DinoNpyReader
{
public:
    DinoNpyReader(const std::filesystem::path &path, const std::string &expected_descr);

    /** @brief 读取指定行区间的原始字节。 */
    std::vector<uint8_t> readBytes(int64_t begin_row, int64_t row_count) const;

    /** @brief 读取整个数组。 */
    std::vector<uint8_t> readAll() const;

    const std::vector<int64_t> &shape() const noexcept
    {
        return shape_;
    }

    size_t elementBytes() const noexcept
    {
        return element_bytes_;
    }

    int64_t rowCount() const noexcept
    {
        return rows_;
    }

    uint64_t dataOffset() const noexcept
    {
        return data_offset_;
    }

private:
    std::filesystem::path path_{};
    mutable std::ifstream stream_{};
    std::vector<int64_t>  shape_{};
    size_t                element_bytes_{0};
    size_t                row_bytes_{0};
    int64_t               rows_{0};
    uint64_t              data_offset_{0};
};

/** @brief 小端标量读写。 */
void dinoWriteU16(std::vector<uint8_t> &out, uint16_t value);
void dinoWriteU32(std::vector<uint8_t> &out, uint32_t value);
void dinoWriteF32(std::vector<uint8_t> &out, float value);
uint16_t dinoReadU16(const uint8_t *data);
uint32_t dinoReadU32(const uint8_t *data);
float    dinoReadF32(const uint8_t *data);

} // namespace irt::features::priv
