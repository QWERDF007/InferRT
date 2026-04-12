/**
 * @file Ranges.hpp
 * @brief 范围（Range）类型特征和工具函数
 * 
 * 提供了一组用于检测和操作范围类型的工具。
 * 范围是指任何可以迭代的类型，如容器、数组等。
 * 
 * 这是 C++20 ranges 库的简化版本，用于 C++17 环境。
 */

#pragma once

#include <iterator>
#include <type_traits>

namespace vtest::ranges {

namespace detail {

/**
 * @brief 默认的 begin 函数（匹配所有类型）
 * 
 * 返回 void，用于 SFINAE 检测
 */
void begin(...);

/**
 * @brief 数组的 begin 函数
 * 
 * @tparam R 数组元素类型
 * @tparam N 数组大小
 * @param r 数组右值引用
 * @return 指向数组首元素的指针
 */
template<class R, int N>
R *begin(R (&&r)[N]);

/**
 * @brief 数组的 end 函数
 * 
 * @tparam R 数组元素类型
 * @tparam N 数组大小
 * @param r 数组右值引用
 * @return 指向数组末尾的指针
 */
template<class R, int N>
R *end(R (&&r)[N]);

/**
 * @brief 检测类型是否有 begin 函数
 * 
 * 使用 SFINAE 技术检测类型 T 是否支持 begin() 操作
 * 
 * @tparam T 要检测的类型
 * @return 如果类型有 begin 则返回 true
 */
template<class T>
constexpr bool HasBegin()
{
    using detail::begin;
    using std::begin;
    return !std::is_same_v<decltype(begin(std::declval<T>())), void>;
}

/**
 * @brief 默认的 end 函数（匹配所有类型）
 * 
 * 返回 void，用于 SFINAE 检测
 */
void end(...);

/**
 * @brief 检测类型是否有 end 函数
 * 
 * 使用 SFINAE 技术检测类型 T 是否支持 end() 操作
 * 
 * @tparam T 要检测的类型
 * @return 如果类型有 end 则返回 true
 */
template<class T>
constexpr bool HasEnd()
{
    using detail::end;
    using std::end;
    return !std::is_same_v<decltype(end(std::declval<T>())), void>;
}

} // namespace detail

/**
 * @brief 编译期常量：判断类型是否为范围
 * 
 * 如果类型同时具有 begin 和 end 函数，则认为它是一个范围
 * 
 * @tparam T 要检测的类型
 */
template<class T>
constexpr bool IsRange = detail::HasBegin<T>() && detail::HasEnd<T>();

/**
 * @brief 获取 C 数组的起始迭代器
 * 
 * @tparam R 数组元素类型
 * @tparam N 数组大小
 * @param r 数组引用
 * @return 指向数组首元素的指针
 */
template<class R, int N>
auto Begin(R (&r)[N])
{
    return r;
}

/**
 * @brief 获取 C 数组的结束迭代器
 * 
 * @tparam R 数组元素类型
 * @tparam N 数组大小
 * @param r 数组引用
 * @return 指向数组末尾的指针
 */
template<class R, int N>
auto End(R (&r)[N])
{
    return r + N;
}

/**
 * @brief 获取范围的起始迭代器
 * 
 * 使用 ADL（参数依赖查找）查找合适的 begin 函数
 * 
 * @tparam R 范围类型
 * @param r 范围对象
 * @return 起始迭代器
 */
template<class R>
auto Begin(R &&r)
{
    using std::begin;
    return begin(r);
}

/**
 * @brief 获取范围的结束迭代器
 * 
 * 使用 ADL（参数依赖查找）查找合适的 end 函数
 * 
 * @tparam R 范围类型
 * @param r 范围对象
 * @return 结束迭代器
 */
template<class R>
auto End(R &&r)
{
    using std::end;
    return end(r);
}

/**
 * @brief 获取范围的数据指针
 * 
 * 返回指向范围第一个元素的指针
 * 
 * @tparam R 范围类型
 * @param r 范围对象
 * @return 数据指针
 */
template<class R>
auto Data(R &&r)
{
    return &*Begin(r);
}

/**
 * @brief 获取范围的大小
 * 
 * 计算范围中元素的数量
 * 
 * @tparam R 范围类型
 * @param r 范围对象
 * @return 元素数量
 */
template<class R>
auto Size(const R &r)
{
    using std::distance;
    return distance(Begin(r), End(r));
}

/**
 * @brief 范围的值类型
 * 
 * 提取范围中元素的类型（去除引用）
 * 
 * @tparam T 范围类型
 */
template<class T>
using RangeValue = std::remove_reference_t<decltype(*Begin(std::declval<T>()))>;

namespace detail {
/**
 * @brief 检测类型是否为随机访问范围
 * 
 * 随机访问范围支持 O(1) 时间复杂度的随机访问操作
 * 
 * @tparam T 要检测的类型
 * @return 如果是随机访问范围则返回 true
 */
template<class T>
constexpr bool IsRandomAccessRange()
{
    if constexpr (IsRange<T>)
    {
        // 检查迭代器类别是否为随机访问迭代器
        return std::is_same_v<typename std::iterator_traits<
                                  std::remove_reference_t<decltype(Begin(std::declval<T>()))>>::iterator_category,
                              std::random_access_iterator_tag>;
    }
    else
    {
        return false;
    }
}
} // namespace detail

/**
 * @brief 编译期常量：判断类型是否为随机访问范围
 * 
 * @tparam T 要检测的类型
 */
template<class T>
constexpr bool IsRandomAccessRange = detail::IsRandomAccessRange<T>();

} // namespace vtest::ranges
