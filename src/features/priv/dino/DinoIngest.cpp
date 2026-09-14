/**
 * @file DinoIngest.cpp
 * @brief 图像接入实现。
 */

#include "DinoIngest.hpp"
#include "DinoPaths.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/features/ImageSearch.hpp>

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <cctype>
#include <limits>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features::priv {

namespace {


inline bool isTiffFile(const fs::path &path)
{
    const auto extension = path.extension().string();
    return extension == ".tif" || extension == ".tiff" || extension == ".TIF" || extension == ".TIFF";
}

/**
 * @brief 统计 TIFF 的目录（页）数量。
 *
 * 首版标准模式不接受多页 TIFF，也不允许静默取第一页；这里只遍历 IFD 链，不解析标签。
 */
int countTiffDirectories(const fs::path &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open TIFF file: %s",
                             path.string().c_str());
    }

    uint8_t header[8]{};
    stream.read(reinterpret_cast<char *>(header), 8);
    if (stream.gcount() != 8)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "UNSUPPORTED_IMAGE: truncated TIFF header");
    }

    const bool little_endian = header[0] == 'I' && header[1] == 'I';
    const bool big_endian    = header[0] == 'M' && header[1] == 'M';
    if (!little_endian && !big_endian)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "UNSUPPORTED_IMAGE: invalid TIFF byte order");
    }
    const bool swap = little_endian;

    const auto readUInt32 = [&](const uint8_t *bytes)
    {
        if (swap)
        {
            return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U)
                 | (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
        }
        return (static_cast<uint32_t>(bytes[0]) << 24U) | (static_cast<uint32_t>(bytes[1]) << 16U)
             | (static_cast<uint32_t>(bytes[2]) << 8U) | static_cast<uint32_t>(bytes[3]);
    };
    const auto readUInt16 = [&](const uint8_t *bytes)
    {
        return swap ? static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8U))
                    : static_cast<uint16_t>((static_cast<uint16_t>(bytes[0]) << 8U) | bytes[1]);
    };

    uint32_t directory_offset = readUInt32(header + 4);
    int      directory_count  = 0;
    while (directory_offset != 0U && directory_count < 1024)
    {
        stream.seekg(static_cast<std::streamoff>(directory_offset), std::ios::beg);
        uint8_t count_bytes[2]{};
        stream.read(reinterpret_cast<char *>(count_bytes), 2);
        if (stream.gcount() != 2)
        {
            break;
        }
        const auto entry_count = readUInt16(count_bytes);
        const auto next_offset_position
            = static_cast<std::streamoff>(directory_offset) + 2 + static_cast<std::streamoff>(entry_count) * 12;
        stream.seekg(next_offset_position, std::ios::beg);
        uint8_t next_bytes[4]{};
        stream.read(reinterpret_cast<char *>(next_bytes), 4);
        if (stream.gcount() != 4)
        {
            break;
        }
        directory_offset = readUInt32(next_bytes);
        ++directory_count;
    }
    return std::max(directory_count, 1);
}

cv::Mat toCanonicalBgr(const cv::Mat &decoded)
{
    if (decoded.depth() != CV_8U)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "UNSUPPORTED_IMAGE: only 8-bit imagery is supported in the standard profile (depth=%d)",
                             decoded.depth());
    }

    if (decoded.channels() == 3)
    {
        return decoded.clone();
    }
    if (decoded.channels() == 1)
    {
        cv::Mat bgr;
        cv::cvtColor(decoded, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (decoded.channels() == 4)
    {
        // 固定白色背景合成：alpha 为 0 处等于白，避免透明像素污染特征。
        std::vector<cv::Mat> planes;
        cv::split(decoded, planes);
        cv::Mat alpha;
        planes[3].convertTo(alpha, CV_32F, 1.0 / 255.0);
        cv::Mat inverse = cv::Scalar(1.0) - alpha;
        std::vector<cv::Mat> merged(3);
        for (int channel = 0; channel < 3; ++channel)
        {
            cv::Mat values;
            planes[channel].convertTo(values, CV_32F);
            merged[static_cast<size_t>(channel)] = values.mul(alpha) + inverse * 255.0;
        }
        cv::Mat bgr;
        cv::merge(merged, bgr);
        bgr.convertTo(bgr, CV_8U);
        return bgr;
    }

    throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "UNSUPPORTED_IMAGE: unsupported channel count %d",
                         decoded.channels());
}

std::string decodePipelineName()
{
    return std::string("opencv-") + CV_VERSION + "|bgr8|exif-applied";
}

} // namespace

DinoImageCache::DinoImageCache(const uint64_t budget_bytes)
    : budget_bytes_(budget_bytes)
{
}

std::shared_ptr<const DinoCanonicalImage> DinoImageCache::find(const std::string &key)
{
    std::lock_guard lock(mutex_);
    const auto      it = entries_.find(key);
    if (it == entries_.end())
    {
        ++misses_;
        return nullptr;
    }
    order_.erase(it->second.order);
    order_.push_front(key);
    it->second.order = order_.begin();
    ++hits_;
    return it->second.image;
}

size_t DinoImageCache::hits() const noexcept
{
    std::lock_guard lock(mutex_);
    return hits_;
}

size_t DinoImageCache::misses() const noexcept
{
    std::lock_guard lock(mutex_);
    return misses_;
}

uint64_t DinoImageCache::bytes() const noexcept
{
    std::lock_guard lock(mutex_);
    return bytes_;
}

void DinoImageCache::insert(const std::string &key, std::shared_ptr<const DinoCanonicalImage> image)
{
    if (image == nullptr)
    {
        return;
    }
    const auto *mat = &image->image;
    const auto  rows = static_cast<uint64_t>(std::max(mat->rows, 0));
    const auto  step = static_cast<uint64_t>(mat->step);
    const auto  bytes = step > 0U && rows > std::numeric_limits<uint64_t>::max() / step
                          ? std::numeric_limits<uint64_t>::max()
                          : rows * step;
    if (bytes > budget_bytes_)
    {
        return;
    }

    std::lock_guard lock(mutex_);
    const auto existing = entries_.find(key);
    if (existing != entries_.end())
    {
        order_.erase(existing->second.order);
        bytes_ -= existing->second.bytes;
        entries_.erase(existing);
    }

    order_.push_front(key);
    entries_.emplace(key, Entry{std::move(image), bytes, order_.begin()});
    bytes_ += bytes;
    evict();
}

void DinoImageCache::evict()
{
    while (bytes_ > budget_bytes_ && !order_.empty())
    {
        const auto &victim = order_.back();
        const auto  it     = entries_.find(victim);
        if (it != entries_.end())
        {
            bytes_ -= it->second.bytes;
            entries_.erase(it);
        }
        order_.pop_back();
    }
}

bool DinoImageLoader::isSupportedFile(const fs::path &path)
{
    return irt::features::ImageSearch::isImageFile(path);
}

std::vector<fs::path> DinoImageLoader::collectGalleryImages(const fs::path &gallery_root)
{
    return irt::features::ImageSearch::collectGalleryImages(gallery_root);
}

DinoImageIdentity DinoImageLoader::statIdentity(const fs::path &path)
{
    DinoImageIdentity record;
    record.source_path = dinoPathToUtf8(fs::absolute(path).lexically_normal());
    record.image_id = record.source_path;
#ifdef _WIN32
    std::transform(record.image_id.begin(), record.image_id.end(), record.image_id.begin(),
                   [](const unsigned char value) { return static_cast<char>(std::tolower(value)); });
#endif
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (!error) record.file_size = static_cast<int64_t>(size);
    error.clear();
    const auto write_time = fs::last_write_time(path, error);
    if (!error) record.mtime_ns = static_cast<int64_t>(write_time.time_since_epoch().count());
    return record;
}

std::string cacheIdentity(const DinoImageIdentity &record)
{
    return record.source_path + "|" + std::to_string(record.file_size) + "|" + std::to_string(record.mtime_ns);
}

DinoCanonicalImage DinoImageLoader::load(const fs::path &path)
{
    if (!fs::exists(path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image file does not exist: %s",
                             path.string().c_str());
    }
    if (!isSupportedFile(path))
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "UNSUPPORTED_IMAGE: unsupported file extension: %s",
                             path.string().c_str());
    }
    if (isTiffFile(path) && countTiffDirectories(path) > 1)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "UNSUPPORTED_IMAGE: multi-page TIFF is not supported in the standard profile: %s",
                             path.string().c_str());
    }

    DinoCanonicalImage result;
    result.record = statIdentity(path);

    // OpenCV 的 imread 在 Windows 上按窄字符打开文件，非 ASCII 路径会直接解码失败；
    // 因此先按 fs::path 读取原始字节（MSVC 的 ifstream(path) 走宽字符接口），再交给 imdecode。
    // imdecode 与 imread 共用同一解码路径，IMREAD_ANYDEPTH|IMREAD_ANYCOLOR 保留位深与通道信息，
    // 并且同样按 EXIF 方向旋转。
    std::vector<uint8_t> encoded;
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open image file: %s",
                                 path.string().c_str());
        }
        const auto size = stream.tellg();
        if (size <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image file is empty: %s",
                                 path.string().c_str());
        }
        encoded.resize(static_cast<size_t>(size));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
        if (stream.gcount() != static_cast<std::streamsize>(encoded.size()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read the whole image file: %s",
                                 path.string().c_str());
        }
    }

    const cv::Mat encoded_mat(1, static_cast<int>(encoded.size()), CV_8UC1, encoded.data());
    const cv::Mat decoded = cv::imdecode(encoded_mat, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (decoded.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to decode image: %s",
                             path.string().c_str());
    }

    result.image = toCanonicalBgr(decoded);
    result.record.width  = result.image.cols;
    result.record.height = result.image.rows;
    result.record.exif_orientation_applied = true;
    result.record.decode_pipeline          = decodePipelineName();

    return result;
}

std::shared_ptr<const DinoCanonicalImage> DinoImageLoader::loadCached(const fs::path &path, DinoImageCache &cache)
{
    const auto record = statIdentity(path);
    const auto key = cacheIdentity(record);
    if (auto cached = cache.find(key))
    {
        return cached;
    }
    auto loaded = std::make_shared<DinoCanonicalImage>(load(path));
    cache.insert(key, loaded);
    return loaded;
}

} // namespace irt::features::priv
