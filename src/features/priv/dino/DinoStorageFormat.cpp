/**
 * @file DinoStorageFormat.cpp
 * @brief 紧凑存储底层格式实现。
 */

#include "DinoStorageFormat.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>


namespace irt::features::priv {

namespace {

std::string shapeText(const std::vector<int64_t> &shape)
{
    std::string text = "(";
    for (size_t index = 0; index < shape.size(); ++index)
    {
        text += std::to_string(shape[index]);
        text += index + 1U == shape.size() ? ",)" : ", ";
    }
    if (shape.empty())
    {
        text = "()";
    }
    return text;
}

std::string buildNpyHeader(const std::string &descr, const std::vector<int64_t> &shape)
{
    for (const auto dimension : shape)
    {
        if (dimension < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy shape dimensions must be non-negative");
        }
    }
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeText(shape) + ", }";
    const size_t preamble = 6U + 2U + 2U;
    if (preamble + dict.size() + 1U > kDinoNpyHeaderBytes)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header exceeds the reserved header area");
    }
    dict.append(kDinoNpyHeaderBytes - preamble - dict.size() - 1U, ' ');
    dict.push_back('\n');

    std::string header;
    header.append(kDinoNpyMagic, 6);
    header.push_back(1);
    header.push_back(0);
    const auto length = static_cast<uint16_t>(dict.size());
    header.push_back(static_cast<char>(length & 0xFFU));
    header.push_back(static_cast<char>((length >> 8U) & 0xFFU));
    header += dict;
    return header;
}

size_t elementBytesFromDescr(const std::string &descr)
{
    if (descr.size() != 3U || (descr[0] != '<' && descr[0] != '>' && descr[0] != '|' && descr[0] != '='))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported npy dtype descriptor: %s",
                             descr.c_str());
    }
    const char kind       = descr[1];
    const int  size_digit = descr[2] - '0';
    if (size_digit < 1 || size_digit > 8)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported npy dtype size: %s", descr.c_str());
    }
    switch (kind)
    {
    case 'f':
    case 'i':
    case 'u':
        return static_cast<size_t>(size_digit);
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported npy dtype kind: %s", descr.c_str());
    }
}


} // namespace

void dinoWriteU16(std::vector<uint8_t> &out, const uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void dinoWriteU32(std::vector<uint8_t> &out, const uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
}

void dinoWriteF32(std::vector<uint8_t> &out, const float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    dinoWriteU32(out, bits);
}

uint16_t dinoReadU16(const uint8_t *data)
{
    return static_cast<uint16_t>(static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8U));
}

uint32_t dinoReadU32(const uint8_t *data)
{
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8U)
         | (static_cast<uint32_t>(data[2]) << 16U) | (static_cast<uint32_t>(data[3]) << 24U);
}

float dinoReadF32(const uint8_t *data)
{
    const uint32_t bits = dinoReadU32(data);
    float          value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void dinoEncodeMeta(const DinoPackedMeta &meta, uint8_t *out)
{
    std::memset(out, 0, kDinoMetaRecordBytes);
    const auto write_u32_at = [&](const size_t offset, const uint32_t value)
    {
        out[offset]     = static_cast<uint8_t>(value & 0xFFU);
        out[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
        out[offset + 2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
        out[offset + 3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
    };
    const auto write_u16_at = [&](const size_t offset, const uint16_t value)
    {
        out[offset]     = static_cast<uint8_t>(value & 0xFFU);
        out[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    };
    const auto write_f32_at = [&](const size_t offset, const float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32_at(offset, bits);
    };

    write_u32_at(0, meta.view_id);
    write_u16_at(4, meta.x0);
    write_u16_at(6, meta.y0);
    write_u16_at(8, meta.x1);
    write_u16_at(10, meta.y1);
    write_u16_at(12, meta.representative_x);
    write_u16_at(14, meta.representative_y);
    write_u16_at(16, meta.member_count);
    write_u16_at(18, meta.flags);
    write_f32_at(20, meta.radius);
    write_u32_at(24, meta.reserved0);
    write_u32_at(28, meta.reserved1);
}

DinoPackedMeta dinoDecodeMeta(const uint8_t *in)
{
    DinoPackedMeta meta;
    meta.view_id          = dinoReadU32(in);
    meta.x0               = dinoReadU16(in + 4);
    meta.y0               = dinoReadU16(in + 6);
    meta.x1               = dinoReadU16(in + 8);
    meta.y1               = dinoReadU16(in + 10);
    meta.representative_x = dinoReadU16(in + 12);
    meta.representative_y = dinoReadU16(in + 14);
    meta.member_count     = dinoReadU16(in + 16);
    meta.flags            = dinoReadU16(in + 18);
    meta.radius           = dinoReadF32(in + 20);
    meta.reserved0        = dinoReadU32(in + 24);
    meta.reserved1        = dinoReadU32(in + 28);
    return meta;
}


uint64_t dinoFileSize(const std::filesystem::path &path)
{
    std::error_code error;
    const auto      size = std::filesystem::file_size(path, error);
    if (error)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read file size: %s",
                             path.string().c_str());
    }
    return static_cast<uint64_t>(size);
}

std::string dinoReadTextFile(const std::filesystem::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open file: %s", path.string().c_str());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}


DinoNpyWriter::DinoNpyWriter(const std::filesystem::path &path, const std::string &descr, const size_t element_bytes)
    : path_(path)
    , descr_(descr)
    , element_bytes_(element_bytes)
{
    if (element_bytes_ == 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy element size must be positive");
    }
    stream_.open(path_, std::ios::binary | std::ios::trunc);
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create npy file: %s",
                             path_.string().c_str());
    }
    // 先写入占位头部，结束写入时回填真实形状。
    const std::string placeholder = buildNpyHeader(descr_, {0});
    stream_.write(placeholder.data(), static_cast<std::streamsize>(placeholder.size()));
}

DinoNpyWriter::~DinoNpyWriter() = default;

void DinoNpyWriter::append(const void *data, const size_t bytes)
{
    if (bytes == 0U)
    {
        return;
    }
    if (bytes % element_bytes_ != 0U)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy append size is not element aligned");
    }
    stream_.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write npy file: %s",
                             path_.string().c_str());
    }
    written_ += bytes;
}


void DinoNpyWriter::finish(const std::vector<int64_t> &shape)
{
    uint64_t element_count = 1U;
    for (const auto dimension : shape)
    {
        if (dimension < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy shape dimensions must be non-negative");
        }
        if (dimension == 0)
        {
            element_count = 0U;
            continue;
        }
        if (element_count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dimension))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy shape element count overflows uint64");
        }
        element_count *= static_cast<uint64_t>(dimension);
    }
    if (element_count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(element_bytes_)
        || element_count * static_cast<uint64_t>(element_bytes_) != written_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "npy shape payload mismatch: shape expects a different byte length");
    }

    const std::string header = buildNpyHeader(descr_, shape);
    stream_.seekp(0, std::ios::beg);
    stream_.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream_.flush();
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to finalize npy header: %s",
                             path_.string().c_str());
    }
    stream_.close();

}

DinoNpyReader::DinoNpyReader(const std::filesystem::path &path, const std::string &expected_descr)
    : path_(path)
{
    stream_.open(path_, std::ios::binary);
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open npy file: %s",
                             path_.string().c_str());
    }

    char magic[6]{};
    stream_.read(magic, 6);
    if (!stream_ || std::memcmp(magic, kDinoNpyMagic, 6) != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid npy magic in %s", path_.string().c_str());
    }
    char version[2]{};
    stream_.read(version, 2);
    if (!stream_ || version[0] != 1 || version[1] != 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported npy version in %s",
                             path_.string().c_str());
    }
    char length_bytes[2]{};
    stream_.read(length_bytes, 2);
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Truncated npy header in %s", path_.string().c_str());
    }
    const auto header_length = static_cast<size_t>(static_cast<uint8_t>(length_bytes[0]))
                             | (static_cast<size_t>(static_cast<uint8_t>(length_bytes[1])) << 8U);
    std::string header(header_length, '\0');
    stream_.read(header.data(), static_cast<std::streamsize>(header_length));
    if (!stream_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Truncated npy header in %s",
                             path_.string().c_str());
    }

    const auto descr_position = header.find("'descr':");
    if (descr_position == std::string::npos)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header misses descr in %s",
                             path_.string().c_str());
    }
    const auto quote_start = header.find('\'', descr_position + 8U);
    const auto quote_end   = quote_start == std::string::npos ? std::string::npos
                                                                : header.find('\'', quote_start + 1U);
    if (quote_start == std::string::npos || quote_end == std::string::npos)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header has an invalid descr in %s",
                             path_.string().c_str());
    }
    const auto descr = header.substr(quote_start + 1U, quote_end - quote_start - 1U);
    if (descr != expected_descr)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy dtype mismatch in %s: expected %s, got %s",
                             path_.string().c_str(), expected_descr.c_str(), descr.c_str());
    }

    const auto shape_position = header.find("'shape':");
    if (shape_position == std::string::npos)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header misses shape in %s",
                             path_.string().c_str());
    }
    const auto open_paren  = header.find('(', shape_position);
    const auto close_paren = open_paren == std::string::npos ? std::string::npos
                                                               : header.find(')', open_paren + 1U);
    if (open_paren == std::string::npos || close_paren == std::string::npos)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header misses shape in %s",
                             path_.string().c_str());
    }
    std::string shape_text = header.substr(open_paren + 1U, close_paren - open_paren - 1U);
    std::replace(shape_text.begin(), shape_text.end(), ',', ' ');
    std::istringstream shape_stream(shape_text);
    int64_t            value = 0;
    while (shape_stream >> value)
    {
        if (value < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy shape dimensions must be non-negative");
        }
        shape_.push_back(value);
    }
    shape_stream >> std::ws;
    if (!shape_stream.eof())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy header has an invalid shape in %s",
                             path_.string().c_str());
    }

    element_bytes_ = elementBytesFromDescr(expected_descr);
    rows_          = shape_.empty() ? 1 : shape_.front();
    row_bytes_     = element_bytes_;
    for (size_t index = 1; index < shape_.size(); ++index)
    {
        if (shape_[index] == 0)
        {
            row_bytes_ = 0U;
            continue;
        }
        if (static_cast<uint64_t>(shape_[index]) > std::numeric_limits<size_t>::max()
            || row_bytes_ > std::numeric_limits<size_t>::max() / static_cast<size_t>(shape_[index]))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy row byte count overflows size_t");
        }
        row_bytes_ *= static_cast<size_t>(shape_[index]);
    }
    if (shape_.empty())
    {
        row_bytes_ = element_bytes_;
    }
    const auto data_offset = static_cast<uint64_t>(6U + 2U + 2U + header_length);
    if (static_cast<uint64_t>(rows_) > 0U
        && row_bytes_ > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(rows_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy payload byte count overflows uint64");
    }
    const auto payload_bytes = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(row_bytes_);
    if (data_offset > std::numeric_limits<uint64_t>::max() - payload_bytes
        || dinoFileSize(path_) != data_offset + payload_bytes)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy file size does not match its shape in %s",
                             path_.string().c_str());
    }
    data_offset_ = data_offset;
}

std::vector<uint8_t> DinoNpyReader::readBytes(const int64_t begin_row, const int64_t row_count) const
{
    if (begin_row < 0 || row_count < 0 || begin_row > rows_ || row_count > rows_ - begin_row)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy read range is out of bounds");
    }
    const auto rows = static_cast<size_t>(row_count);
    if (row_bytes_ != 0U && rows > std::numeric_limits<size_t>::max() / row_bytes_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy read size overflows size_t");
    }
    std::vector<uint8_t> buffer(rows * row_bytes_);
    if (buffer.empty())
    {
        return buffer;
    }
    auto &mutable_stream = stream_;
    mutable_stream.clear();
    mutable_stream.seekg(static_cast<std::streamoff>(data_offset_ + static_cast<uint64_t>(begin_row) * row_bytes_),
                         std::ios::beg);
    if (!mutable_stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy seek failed in %s", path_.string().c_str());
    }
    mutable_stream.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    if (mutable_stream.gcount() != static_cast<std::streamsize>(buffer.size()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "npy short read in %s", path_.string().c_str());
    }
    return buffer;
}

std::vector<uint8_t> DinoNpyReader::readAll() const
{
    return readBytes(0, rows_);
}

} // namespace irt::features::priv
