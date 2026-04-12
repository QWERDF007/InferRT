/**
 * @file HashMD5.cpp
 * @brief MD5 哈希计算实现
 * 
 * 使用 OpenSSL 的 EVP 接口实现 MD5 哈希计算功能。
 * 该实现支持增量更新和重复使用哈希上下文。
 */

#include "HashMD5.hpp"

#include <openssl/evp.h>

#include <cassert>

namespace vtest {

/**
 * @brief HashMD5 的私有实现结构
 * 
 * 使用 Pimpl 模式隐藏 OpenSSL 的实现细节
 */
struct HashMD5::Impl
{
    EVP_MD_CTX *ctx; ///< OpenSSL 的消息摘要上下文
};

/**
 * @brief 构造函数
 * 
 * 创建并初始化 MD5 哈希上下文
 */
HashMD5::HashMD5()
    : pimpl{std::make_unique<Impl>()}
{
    // 创建消息摘要上下文
    pimpl->ctx = EVP_MD_CTX_create();
    assert(pimpl->ctx != nullptr);

    // 初始化 MD5 摘要算法
    int ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    assert(ret == 1);
}

/**
 * @brief 析构函数
 * 
 * 销毁 MD5 哈希上下文，释放资源
 */
HashMD5::~HashMD5()
{
    EVP_MD_CTX_destroy(pimpl->ctx);
}

/**
 * @brief 更新哈希值（添加数据）
 * 
 * 将数据添加到当前的哈希计算中。可以多次调用以增量方式计算哈希。
 * 
 * @param data 要添加的数据指针
 * @param lenBytes 数据长度（字节）
 */
void HashMD5::operator()(const void *data, size_t lenBytes)
{
    // 更新摘要计算，添加新数据
    int ret = EVP_DigestUpdate(pimpl->ctx, data, lenBytes);
    assert(ret == 1);
}

/**
 * @brief 获取最终哈希值并重置上下文
 * 
 * 完成当前的哈希计算，返回 128 位（16 字节）的 MD5 哈希值，
 * 并自动重置上下文以便进行下一次哈希计算。
 * 
 * @return 16 字节的 MD5 哈希值
 */
std::array<uint8_t, 16> HashMD5::getHashAndReset()
{
    unsigned char buf[EVP_MAX_MD_SIZE];
    unsigned int  nwritten = sizeof(buf);

    // 完成摘要计算并获取结果（此操作也会重置上下文）
    int ret = EVP_DigestFinal(pimpl->ctx, buf, &nwritten);
    assert(ret == 1);

    // 重新初始化上下文，准备下一次计算
    ret = EVP_DigestInit_ex(pimpl->ctx, EVP_md5(), NULL);
    assert(ret == 1);

    // MD5 哈希值固定为 16 字节
    assert(nwritten == 16);
    std::array<uint8_t, 16> hash;
    memcpy(&hash[0], buf, sizeof(hash));
    return hash;
}

/**
 * @brief 更新 C 风格字符串的哈希值
 * 
 * 特殊处理 C 风格字符串：
 * - 如果指针非空，将其作为字符串视图处理
 * - 如果指针为空，使用特殊的魔数值 -28374 代替
 * 
 * @param hash MD5 哈希对象
 * @param value C 风格字符串指针（可以为 nullptr）
 */
void Update(HashMD5 &hash, const char *value)
{
    if (value)
    {
        // 非空指针：作为字符串处理
        Update(hash, std::string_view(value));
    }
    else
    {
        // 空指针：使用魔数值代替
        Update(hash, -28374);
    }
}

} // namespace vtest
