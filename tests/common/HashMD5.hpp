/**
 * @file HashMD5.hpp
 * @brief MD5 哈希计算工具类
 * 
 * 提供了一个灵活的 MD5 哈希计算接口，支持多种数据类型：
 * - 基本类型（整数、浮点数等）
 * - 字符串和字符串视图
 * - 容器和范围
 * - 元组和可选类型
 * - 自定义类型
 * 
 * 使用 OpenSSL 的 EVP 接口实现底层哈希计算。
 */

#pragma once

#include "Ranges.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

namespace vtest {

/**
 * @brief MD5 哈希计算类
 * 
 * 提供增量式 MD5 哈希计算功能。可以多次添加数据，
 * 最后获取完整的哈希值。
 * 
 * 特性：
 * - 支持增量更新（可以分多次添加数据）
 * - 自动处理多种数据类型
 * - 使用 Pimpl 模式隐藏实现细节
 * - 不可拷贝（避免上下文状态混乱）
 * 
 * 使用示例：
 * @code
 * HashMD5 hash;
 * hash(42);
 * hash("hello");
 * auto result = hash.getHashAndReset();
 * @endcode
 */
class HashMD5
{
public:
    /**
     * @brief 构造函数
     * 
     * 创建并初始化 MD5 哈希上下文
     */
    HashMD5();

    /**
     * @brief 禁用拷贝构造
     * 
     * 哈希上下文包含状态，不应该被拷贝
     */
    HashMD5(const HashMD5 &) = delete;

    /**
     * @brief 析构函数
     * 
     * 清理 MD5 哈希上下文
     */
    ~HashMD5();

    /**
     * @brief 添加原始数据到哈希计算
     * 
     * @param data 数据指针
     * @param lenBytes 数据长度（字节）
     */
    void operator()(const void *data, size_t lenBytes);

    /**
     * @brief 获取最终哈希值并重置上下文
     * 
     * @return 16 字节的 MD5 哈希值
     */
    std::array<uint8_t, 16> getHashAndReset();

    /**
     * @brief 添加具有唯一对象表示的类型到哈希计算
     * 
     * 只有具有唯一对象表示的类型才能安全地进行字节级哈希。
     * 这排除了包含填充位或未初始化内存的类型。
     * 
     * @tparam T 数据类型（必须具有唯一对象表示）
     * @param value 要哈希的值
     */
    template<class T>
    void operator()(const T &value)
    {
        static_assert(std::has_unique_object_representations_v<T>, "Can't hash this type");
        this->operator()(&value, sizeof(value));
    }

private:
    struct Impl;                 ///< 前向声明的实现结构
    std::unique_ptr<Impl> pimpl; ///< Pimpl 指针
};

/**
 * @brief 更新具有唯一对象表示的非范围类型的哈希值
 * 
 * 对于具有唯一对象表示且不是范围类型的值，直接进行字节级哈希
 * 
 * @tparam T 值类型
 * @param hash MD5 哈希对象
 * @param value 要哈希的值
 */
template<class T>
std::enable_if_t<std::has_unique_object_representations_v<T> && !ranges::IsRange<T>> Update(HashMD5 &hash,
                                                                                            const T &value)
{
    hash(value);
}

/**
 * @brief 禁止对指针进行哈希
 * 
 * 对指针进行哈希通常是错误的，因为指针值本身没有意义。
 * 应该对指针指向的内容进行哈希。
 * 
 * @tparam T 指针指向的类型
 */
template<class T>
void Update(HashMD5 &hash, const T *value)
{
    static_assert(sizeof(T) == 0, "Won't do md5 of a pointer");
}

/**
 * @brief 更新 C 风格字符串的哈希值
 * 
 * C 风格字符串是指针的特殊情况，需要特殊处理
 * 
 * @param hash MD5 哈希对象
 * @param value C 风格字符串指针
 */
void Update(HashMD5 &hash, const char *value);

/**
 * @brief 更新范围类型的哈希值
 * 
 * 对于范围类型（容器、数组等），先哈希元素数量，然后哈希每个元素。
 * 
 * 优化策略：
 * - 如果是随机访问范围且元素具有唯一对象表示，使用连续内存哈希（更快）
 * - 否则，逐个元素进行哈希
 * 
 * @tparam R 范围类型
 * @param hash MD5 哈希对象
 * @param r 要哈希的范围
 */
template<class R, std::enable_if_t<ranges::IsRange<R>, int> = 0>
void Update(vtest::HashMD5 &hash, const R &r)
{
    // 先哈希元素数量，确保不同大小的范围产生不同的哈希
    Update(hash, ranges::Size(r));

    // 在 C++20 中应该使用 std::ranges::contiguous_range
    if constexpr (ranges::IsRandomAccessRange<R> && std::has_unique_object_representations_v<ranges::RangeValue<R>>)
    {
        // 对于连续内存且元素具有唯一对象表示的范围，可以直接哈希整块内存（更快）
        hash(ranges::Data(r), ranges::Size(r) * sizeof(ranges::RangeValue<R>));
    }
    else
    {
        // 否则必须逐个元素哈希
        for (auto &v : r)
        {
            Update(hash, v);
        }
    }
}

/**
 * @brief 更新多个值的哈希值
 * 
 * 按顺序哈希多个值
 * 
 * @tparam T1 第一个值的类型
 * @tparam T2 第二个值的类型
 * @tparam TT 剩余值的类型包
 * @param hash MD5 哈希对象
 * @param v1 第一个值
 * @param v2 第二个值
 * @param v 剩余的值
 */
template<class T1, class T2, class... TT>
void Update(HashMD5 &hash, const T1 &v1, const T2 &v2, const TT &...v)
{
    Update(hash, v1);
    Update(hash, v2);

    // 使用折叠表达式哈希剩余的值
    (..., Update(hash, v));
}

/**
 * @brief 更新浮点数的哈希值
 * 
 * 浮点数不能直接进行字节级哈希（因为 NaN 和 ±0 的问题），
 * 所以使用 std::hash 进行哈希
 * 
 * @tparam T 浮点数类型
 * @param hash MD5 哈希对象
 * @param value 浮点数值
 */
template<class T>
std::enable_if_t<std::is_floating_point_v<T>> Update(HashMD5 &hash, const T &value)
{
    hash(std::hash<T>()(value));
}

} // namespace vtest

/**
 * @brief std 命名空间中的 Update 函数重载
 * 
 * 为标准库类型提供哈希更新函数
 */
namespace std {

/**
 * @brief 更新 tuple 的哈希值
 * 
 * 如果 tuple 具有唯一对象表示，直接进行字节级哈希；
 * 否则，递归地哈希每个元素。
 * 
 * @tparam TT tuple 中的类型参数包
 * @param hash MD5 哈希对象
 * @param t 要哈希的 tuple
 */
template<typename... TT>
void Update(vtest::HashMD5 &hash, const tuple<TT...> &t)
{
    if constexpr (has_unique_object_representations_v<tuple<TT...>>)
    {
        // tuple 具有唯一对象表示，可以直接哈希
        return hash(t);
    }

    // 将哈希对象和 tuple 元素组合，然后应用 Update 函数
    auto th = forward_as_tuple(hash);
    apply(vtest::Update<TT...>, tuple_cat(th, t));
};

/**
 * @brief 更新 std::string 的哈希值
 * 
 * @param hash MD5 哈希对象
 * @param s 要哈希的字符串
 */
inline void Update(vtest::HashMD5 &hash, const string &s)
{
    return hash(s.data(), s.size());
}

/**
 * @brief 更新 std::string_view 的哈希值
 * 
 * @param hash MD5 哈希对象
 * @param s 要哈希的字符串视图
 */
inline void Update(vtest::HashMD5 &hash, const string_view &s)
{
    return hash(s.data(), s.size());
}

/**
 * @brief 更新 std::type_info 的哈希值
 * 
 * 使用类型信息的哈希码
 * 
 * @param hash MD5 哈希对象
 * @param t 类型信息对象
 */
inline void Update(vtest::HashMD5 &hash, const std::type_info &t)
{
    return hash(t.hash_code());
}

/**
 * @brief 更新 std::optional 的哈希值
 * 
 * 不能依赖 std::hash<T> 来处理 optional，因为它要求 T 有有效的哈希特化。
 * 由于我们的类型使用 HashValue 重载，所以需要特殊处理。
 * 
 * @tparam T optional 包含的类型
 * @param hash MD5 哈希对象
 * @param o 要哈希的 optional 对象
 */
template<class T>
void Update(vtest::HashMD5 &hash, const optional<T> &o)
{
    using vtest::Update;

    if (o)
    {
        // 如果 optional 有值，哈希其内容
        return Update(hash, *o);
    }
    else
    {
        // 如果 optional 为空，使用 nullopt 的哈希值
        return Update(hash, std::hash<optional<int>>()(nullopt));
    }
}

} // namespace std
