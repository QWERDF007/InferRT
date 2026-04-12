/**
 * @file ValueList.hpp
 * @brief 用于 Google Test 参数化测试的值列表工具类
 * 
 * 该文件提供了一个灵活的 ValueList 类模板，用于简化 Google Test 中的参数化测试。
 * 它支持单值、元组和混合类型的参数列表，并提供了丰富的操作接口。
 */

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <list>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace vtest {

namespace detail {

/**
 * @brief 类型特征：判断类型 T 是否为 std::tuple
 * 
 * 基础模板：对于非 tuple 类型，继承自 std::false_type
 */
template<class T>
struct IsTuple : std::false_type
{
};

/**
 * @brief 类型特征：判断类型 T 是否为 std::tuple（特化版本）
 * 
 * 对于 std::tuple<TT...> 类型，继承自 std::true_type
 * 
 * @tparam TT tuple 中包含的类型参数包
 */
template<class... TT>
struct IsTuple<std::tuple<TT...>> : std::true_type
{
};

/**
 * @brief 类型特征：判断左值引用类型是否为 tuple（特化版本）
 * 
 * 如果 T 是一个引用类型，先使用 std::decay_t 去除引用和 cv 限定符，
 * 然后再判断去掉引用后的类型是否是 std::tuple 类型。
 * 
 * std::decay_t 会移除类型的：
 * - cv 限定符（const 和 volatile）
 * - 引用（& 和 &&）
 * - 数组到指针的转换
 * - 函数到函数指针的转换
 * 
 * @tparam T 待判断的左值引用类型
 */
template<class T>
struct IsTuple<T &> : IsTuple<std::decay_t<T>>
{
};

/**
 * @brief 类型特征：判断右值引用类型是否为 tuple（特化版本）
 * 
 * 如果 T 是 std::tuple 类型的右值引用，先去除引用后再判断
 * 
 * @tparam T 待判断的右值引用类型
 */
template<class T>
struct IsTuple<T &&> : IsTuple<std::decay_t<T>>
{
};

/**
 * @brief 合并 tuple 的递归终止条件
 * 
 * 当没有参数时，返回一个空的 tuple
 * 
 * @return 空的 std::tuple<>
 */
inline auto JoinTuple()
{
    return std::tuple<>();
}

/**
 * @brief 合并多个值或 tuple 为一个 tuple（前向声明）
 * 
 * 递归函数，将多个 tuple 或普通值合并成一个大的 tuple
 * 
 * @tparam T 第一个参数的类型
 * @tparam TAIL 剩余参数的类型包
 * @param a0 第一个参数
 * @param as 剩余参数
 * @return 合并后的 tuple
 */
template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as);

/**
 * @brief 合并多个 tuple（特化版本：第一个参数是 tuple）
 * 
 * 当第一个参数是 tuple 时，使用 std::tuple_cat 将其与递归处理后的剩余参数合并
 * 
 * @tparam TT 第一个 tuple 中的类型参数包
 * @tparam TAIL 剩余参数的类型包
 * @param a0 第一个 tuple
 * @param as 剩余参数
 * @return 合并后的 tuple
 */
template<class... TT, class... TAIL>
auto JoinTuple(std::tuple<TT...> a0, TAIL &&...as)
{
    auto rest = JoinTuple(std::forward<TAIL>(as)...);
    return std::tuple_cat(std::move(a0), std::move(rest));
}

/**
 * @brief 合并多个值或 tuple（通用版本：第一个参数不是 tuple）
 * 
 * 当第一个参数不是 tuple 时，先将其包装成 tuple，然后与递归处理后的剩余参数合并
 * 
 * @tparam T 第一个参数的类型
 * @tparam TAIL 剩余参数的类型包
 * @param a0 第一个参数
 * @param as 剩余参数
 * @return 合并后的 tuple
 */
template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as)
{
    return std::tuple_cat(std::make_tuple(std::move(a0)), JoinTuple(std::forward<TAIL>(as)...));
}

/**
 * @brief 获取 tuple 的第一个元素（头部）
 * 
 * @tparam T 第一个元素的类型
 * @tparam TT 剩余元素的类型包
 * @param t 输入的 tuple
 * @return tuple 的第一个元素
 */
template<class T, class... TT>
auto Head(const std::tuple<T, TT...> &t)
{
    return std::get<0>(t);
}

/**
 * @brief 获取 tuple 尾部的实现函数
 * 
 * 使用索引序列来提取 tuple 中除第一个元素外的所有元素
 * 
 * @tparam IDX 索引序列
 * @tparam TT tuple 中的类型参数包
 * @param 索引序列（编译期参数）
 * @param t 输入的 tuple
 * @return 包含除第一个元素外所有元素的新 tuple
 */
template<size_t... IDX, class... TT>
auto TailImpl(std::index_sequence<IDX...>, const std::tuple<TT...> &t)
{
    return std::make_tuple(std::get<IDX + 1>(t)...);
}

/**
 * @brief 获取 tuple 的尾部（除第一个元素外的所有元素）
 * 
 * @tparam T 第一个元素的类型
 * @tparam TT 剩余元素的类型包
 * @param t 输入的 tuple
 * @return 包含除第一个元素外所有元素的新 tuple
 */
template<class T, class... TT>
auto Tail(const std::tuple<T, TT...> &t)
{
    return TailImpl(std::make_index_sequence<sizeof...(TT)>(), t);
}

/**
 * @brief 类型恒等元（Identity）元函数
 * 
 * 用于在模板元编程中保持类型不变，常用于条件类型选择
 * 
 * @tparam T 输入类型
 */
template<class T>
struct Identity
{
    using type = T;
};

/**
 * @brief 从 tuple 中提取指定索引的元素（递归终止条件）
 * 
 * 当没有指定索引时，返回空 tuple
 * 
 * @tparam TT tuple 中的类型参数包
 * @param t 输入的 tuple
 * @return 空 tuple
 */
template<class... TT>
inline std::tuple<> ExtractTuple(std::tuple<TT...> t)
{
    return {};
}

/**
 * @brief 从 tuple 中提取指定索引的元素
 * 
 * 根据提供的索引序列 N, TAIL...，从 tuple 中提取对应位置的元素并组成新的 tuple
 * 
 * @tparam N 第一个要提取的索引
 * @tparam TAIL 剩余要提取的索引
 * @tparam TT tuple 中的类型参数包
 * @param t 输入的 tuple
 * @return 包含提取元素的新 tuple
 */
template<int N, int... TAIL, class... TT>
auto ExtractTuple(std::tuple<TT...> t)
{
    return JoinTuple(std::get<N>(t), ExtractTuple<TAIL...>(t));
}

/**
 * @brief 从非 tuple 类型中提取元素（单索引版本）
 * 
 * 对于非 tuple 类型，只能提取索引 0（即元素本身）
 * 
 * @tparam N 索引（必须为 0）
 * @tparam T 元素类型
 * @param t 输入元素
 * @return 包含该元素的 tuple
 */
template<int N, class T>
auto ExtractTuple(T t)
{
    static_assert(N == 0, "Index out of range");

    return std::make_tuple(t);
}

/**
 * @brief 从非 tuple 类型中提取元素（无索引版本）
 * 
 * 当没有指定索引时，返回空 tuple
 * 
 * @tparam T 元素类型
 * @param t 输入元素
 * @return 空 tuple
 */
template<class T>
auto ExtractTuple(T t)
{
    return std::tuple<>();
}

/**
 * @brief 可选地从值中提取 tuple 元素
 * 
 * 如果提供了索引序列 NN...，则提取对应索引的元素；
 * 否则，将整个值包装成 tuple
 * 
 * @tparam NN 可选的索引序列
 * @tparam T 输入类型
 * @param t 输入值
 * @return 提取或包装后的 tuple
 */
template<int... NN, class T>
auto MaybeExtractTuple(T t)
{
    if constexpr (sizeof...(NN) == 0)
    {
        return JoinTuple(t);
    }
    else
    {
        return ExtractTuple<NN...>(std::move(t));
    }
}

/**
 * @brief 默认值占位符类型
 * 
 * 这个结构体用作参数的默认值占位符。在参数化测试中，
 * 可以使用 Default 类型来表示某个参数应该使用其默认值。
 * 
 * 提供了比较运算符的实现：
 * - operator<: 总是返回 false（所有 Default 实例相等）
 * - operator==: 总是返回 true（所有 Default 实例相等）
 */
struct Default
{
    bool operator<(const Default &that) const
    {
        return false;
    }

    bool operator==(const Default &that) const
    {
        return true;
    }
};

/**
 * @brief 替换单个索引位置的默认值（实现函数）
 * 
 * 如果输入 tuple 在索引 IDX 处的类型不是 Default，
 * 则将该值转换为目标类型并赋值给输出 tuple
 * 
 * @tparam IDX 要处理的索引位置
 * @tparam T 输入 tuple 类型
 * @tparam U 输出 tuple 类型
 * @param out 输出 tuple（引用）
 * @param in 输入 tuple（常量引用）
 */
template<int IDX, class T, class U>
void ReplaceDefaultsImpl(U &out, const T &in)
{
    using SRC = typename std::tuple_element<IDX, T>::type;
    using DST = typename std::tuple_element<IDX, U>::type;

    if constexpr (!std::is_same_v<SRC, Default>)
    {
        std::get<IDX>(out) = static_cast<DST>(std::get<IDX>(in));
    }
}

/**
 * @brief 替换所有索引位置的默认值（实现函数）
 * 
 * 使用索引序列遍历 tuple 的所有元素，对每个位置调用单索引版本的替换函数
 * 
 * @tparam T 输入 tuple 类型
 * @tparam U 输出 tuple 类型
 * @tparam IDX 索引序列
 * @param out 输出 tuple（引用）
 * @param in 输入 tuple（常量引用）
 * @param 索引序列（编译期参数）
 */
template<class T, class U, size_t... IDX>
void ReplaceDefaultsImpl(U &out, const T &in, std::index_sequence<IDX...>)
{
    (ReplaceDefaultsImpl<IDX>(out, in), ...);
}

/**
 * @brief 替换 tuple 中的默认值（可默认构造版本）
 * 
 * 将输入 tuple 中的 Default 类型替换为目标类型的默认构造值。
 * 如果输入中没有 Default，则直接转换类型。
 * 
 * 此版本用于目标 tuple 类型可默认构造的情况。
 * 
 * @tparam UU 目标 tuple 的类型参数包
 * @tparam TT 输入 tuple 的类型参数包
 * @param in 输入 tuple
 * @return 替换默认值后的 tuple
 */
template<class... UU, class... TT>
std::enable_if_t<std::is_default_constructible_v<std::tuple<UU...>>, std::tuple<UU...>> ReplaceDefaults(
    const std::tuple<TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    std::tuple<UU...> out;
    ReplaceDefaultsImpl(out, in, std::index_sequence_for<TT...>());
    return out;
}

/**
 * @brief 替换 tuple 中的默认值（不可默认构造版本）
 * 
 * 将输入 tuple 中的 Default 类型替换为目标类型的默认构造值。
 * 
 * 此版本用于目标 tuple 类型不可默认构造的情况，采用递归方式处理：
 * - 如果第一个元素是 Default，则使用目标类型的默认构造
 * - 否则，使用输入值构造目标类型
 * 
 * @tparam U 目标 tuple 第一个元素的类型
 * @tparam UU 目标 tuple 剩余元素的类型参数包
 * @tparam T 输入 tuple 第一个元素的类型
 * @tparam TT 输入 tuple 剩余元素的类型参数包
 * @param in 输入 tuple
 * @return 替换默认值后的 tuple
 */
template<class U, class... UU, class T, class... TT>
std::enable_if_t<!std::is_default_constructible_v<std::tuple<U, UU...>>, std::tuple<U, UU...>> ReplaceDefaults(
    const std::tuple<T, TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    if constexpr (std::is_same_v<T, Default>)
    {
        static_assert(std::is_default_constructible_v<U>,
                      "Param type must have an explicit default value as it's not default constructible");

        return JoinTuple(U{}, ReplaceDefaults<UU...>(Tail(in)));
    }
    else
    {
        return JoinTuple(U{std::get<0>(in)}, ReplaceDefaults<UU...>(Tail(in)));
    }
}

} // namespace detail

/**
 * @brief 参数化测试的值列表容器
 * 
 * ValueList 是一个灵活的容器类，用于存储参数化测试的测试参数。
 * 它可以处理单个值、tuple 或混合类型的参数列表。
 * 
 * 主要特性：
 * - 支持单值和 tuple 类型的参数
 * - 自动处理类型转换和默认值替换
 * - 提供丰富的容器操作接口（插入、删除、遍历等）
 * - 与 Google Test 的参数化测试框架无缝集成
 * 
 * @tparam TT 值列表中元素的类型参数包
 */
template<class... TT>
class ValueList
{
public:
    /**
     * @brief tuple 值类型
     * 
     * 将所有模板参数 TT... 合并后的 tuple 类型
     */
    using tuple_value_type = decltype(detail::JoinTuple(std::declval<TT>()...));

    /**
     * @brief 实际存储的值类型
     * 
     * 如果 tuple_value_type 只有一个元素，则 value_type 为该元素类型；
     * 否则为 tuple_value_type 本身
     */
    using value_type =
        typename std::conditional_t<std::tuple_size_v<tuple_value_type> == 1, std::tuple_element<0, tuple_value_type>,
                                    detail::Identity<tuple_value_type>>::type;

    /**
     * @brief 底层容器类型
     * 
     * 使用 std::list 作为底层存储容器
     */
    using list_type = std::list<value_type>;

    /**
     * @brief 迭代器类型
     */
    using iterator = typename list_type::iterator;

    /**
     * @brief 常量迭代器类型
     */
    using const_iterator = typename list_type::const_iterator;

    /**
     * @brief 默认构造函数
     */
    ValueList() = default;

    /**
     * @brief 从初始化列表构造
     * 
     * @param initList 初始化列表
     */
    ValueList(std::initializer_list<value_type> initList)
        : m_list(std::move(initList))
    {
    }

    /**
     * @brief 从另一个不同类型的 ValueList 构造（转换构造函数）
     * 
     * 支持从不同类型参数的 ValueList 进行转换构造，
     * 会自动处理类型转换和默认值替换。
     * 
     * @tparam UU 源 ValueList 的类型参数包
     * @param that 源 ValueList
     */
    template<class... UU, std::enable_if_t<!std::is_same_v<tuple_value_type, std::tuple<UU...>>, int> = 0>
    explicit ValueList(const ValueList<UU...> &that)
    {
        // 遍历源列表中的每个元素
        for (auto &v : that)
        {
            // 如果目标类型是 tuple
            if constexpr (detail::IsTuple<value_type>::value)
            {
                // 将值合并成 tuple，替换默认值，然后添加到列表中
                m_list.emplace_back(detail::ReplaceDefaults<TT...>(detail::JoinTuple(v)));
            }
            else
            {
                // 将值合并成 tuple，提取第一个元素，替换默认值，然后添加到列表中
                m_list.emplace_back(std::get<0>(detail::ReplaceDefaults<TT...>(detail::JoinTuple(v))));
            }
        }
    }

    /**
     * @brief 从 vector 构造
     * 
     * @param v 包含初始值的 vector
     */
    ValueList(const std::vector<value_type> &v)
    {
        m_list.insert(m_list.end(), v.begin(), v.end());
    }

    /**
     * @brief 获取起始迭代器
     * 
     * @return 指向列表开始的迭代器
     */
    auto begin()
    {
        return m_list.begin();
    }

    /**
     * @brief 获取结束迭代器
     * 
     * @return 指向列表末尾的迭代器
     */
    auto end()
    {
        return m_list.end();
    }

    /**
     * @brief 获取常量起始迭代器
     * 
     * @return 指向列表开始的常量迭代器
     */
    auto cbegin() const
    {
        return m_list.cbegin();
    }

    /**
     * @brief 获取常量结束迭代器
     * 
     * @return 指向列表末尾的常量迭代器
     */
    auto cend() const
    {
        return m_list.cend();
    }

    /**
     * @brief 获取常量起始迭代器（const 版本）
     * 
     * @return 指向列表开始的常量迭代器
     */
    auto begin() const
    {
        return m_list.begin();
    }

    /**
     * @brief 获取常量结束迭代器（const 版本）
     * 
     * @return 指向列表末尾的常量迭代器
     */
    auto end() const
    {
        return m_list.end();
    }

    /**
     * @brief 在列表末尾就地构造元素
     * 
     * 接受多个参数，将它们合并成 tuple 后添加到列表末尾
     * 
     * @tparam A 第一个参数的类型
     * @tparam AA 剩余参数的类型包
     * @param a0 第一个参数
     * @param args 剩余参数
     */
    template<class A, class... AA>
    void emplace_back(A &&a0, AA &&...args)
    {
        m_list.emplace_back(detail::JoinTuple(std::forward<A>(a0), std::forward<AA>(args)...));
    }

    /**
     * @brief 在指定位置插入元素
     * 
     * @param it 插入位置的迭代器
     * @param v 要插入的值
     * @return 指向插入元素的迭代器
     */
    auto insert(const_iterator it, value_type v)
    {
        return m_list.insert(it, std::move(v));
    }

    /**
     * @brief 在列表头部添加元素
     * 
     * @param v 要添加的值
     * @return 对新添加元素的引用
     */
    auto push_front(value_type v)
    {
        return m_list.emplace_front(std::move(v));
    }

    /**
     * @brief 在列表尾部添加元素
     * 
     * @param v 要添加的值
     * @return 对新添加元素的引用
     */
    auto push_back(value_type v)
    {
        return m_list.emplace_back(std::move(v));
    }

    /**
     * @brief 在列表尾部添加 tuple 元素（特化版本）
     * 
     * 当 tuple_value_type 与 value_type 不同时启用，
     * 使用 std::apply 展开 tuple 并添加到列表
     * 
     * @tparam X 用于 SFINAE 的辅助模板参数
     * @param v 要添加的 tuple
     * @return 对新添加元素的引用
     */
    template<class X = void, std::enable_if_t<sizeof(X) != 0 && !std::is_same_v<tuple_value_type, value_type>, int> = 0>
    auto push_back(tuple_value_type v)
    {
        return std::apply([this](auto &...args) { m_list.emplace_back(args...); }, v);
    }

    /**
     * @brief 连接另一个 ValueList
     * 
     * 将另一个 ValueList 的所有元素移动到当前列表的末尾
     * 
     * @param other 要连接的 ValueList（右值引用）
     */
    void concat(ValueList &&other)
    {
        m_list.splice(m_list.end(), std::move(other.m_list));
    }

    /**
     * @brief 删除指定位置的元素
     * 
     * @param it 要删除元素的迭代器
     */
    void erase(iterator it)
    {
        m_list.erase(it);
    }

    /**
     * @brief 删除指定范围的元素
     * 
     * @param itbeg 范围起始迭代器
     * @param itend 范围结束迭代器
     */
    void erase(iterator itbeg, iterator itend)
    {
        m_list.erase(itbeg, itend);
    }

    /**
     * @brief 删除所有等于指定值的元素
     * 
     * @param v 要删除的值
     * @return 如果至少删除了一个元素则返回 true，否则返回 false
     */
    bool erase(value_type v)
    {
        bool removedAtLeastOne = false;

        for (auto it = m_list.begin(); it != m_list.end();)
        {
            it = std::find(it, m_list.end(), v);
            if (it != m_list.end())
            {
                m_list.erase(it++);
                removedAtLeastOne = true;
            }
        }

        return removedAtLeastOne;
    }

    /**
     * @brief 相等比较运算符
     * 
     * @param that 要比较的另一个 ValueList
     * @return 如果两个列表相等则返回 true
     */
    bool operator==(const ValueList<TT...> &that) const
    {
        return m_list == that.m_list;
    }

    /**
     * @brief 不等比较运算符
     * 
     * @param that 要比较的另一个 ValueList
     * @return 如果两个列表不相等则返回 true
     */
    bool operator!=(const ValueList<TT...> &that) const
    {
        return m_list != that.m_list;
    }

    /**
     * @brief 检查值是否存在于列表中
     * 
     * @param v 要查找的值
     * @return 如果值存在则返回 true，否则返回 false
     */
    bool exists(const value_type &v) const
    {
        return std::find(m_list.begin(), m_list.end(), v) != m_list.end();
    }

    /**
     * @brief UniqueSort 函数的友元声明
     * 
     * 允许 UniqueSort 函数访问私有成员
     */
    template<int... NN, class F, class... UU>
    friend ValueList<UU...> UniqueSort(F extractor, ValueList<UU...> a);

    /**
     * @brief 获取列表大小
     * 
     * @return 列表中元素的数量
     */
    size_t size() const
    {
        return m_list.size();
    }

private:
    /**
     * @brief 底层存储容器
     * 
     * 使用 std::list 存储合并后的 tuple 或单值
     */
    list_type m_list;
};

/**
 * @brief 类型推导指南：从初始化列表推导 ValueList 类型
 * 
 * 允许从 std::initializer_list<T> 推导出 ValueList<T>
 */
template<class T>
ValueList(std::initializer_list<T>) -> ValueList<T>;

/**
 * @brief 类型推导指南：从 tuple 初始化列表推导 ValueList 类型
 * 
 * 允许从 std::initializer_list<std::tuple<TT...>> 推导出 ValueList<TT...>
 */
template<class... TT>
ValueList(std::initializer_list<std::tuple<TT...>>) -> ValueList<TT...>;

/**
 * @brief 类型推导指南：从 vector 推导 ValueList 类型
 * 
 * 允许从 std::vector<T> 推导出 ValueList<T>
 */
template<class T>
ValueList(const std::vector<T> &) -> ValueList<T>;

/**
 * @brief 类型推导指南：从 tuple vector 推导 ValueList 类型
 * 
 * 允许从 std::vector<std::tuple<TT...>> 推导出 ValueList<TT...>
 */
template<class... TT>
ValueList(const std::vector<std::tuple<TT...>> &) -> ValueList<TT...>;

/**
 * @brief 创建包含单个值的 ValueList
 * 
 * 便捷函数，用于创建只包含一个值的 ValueList
 * 
 * @tparam T 值的类型
 * @param v 要包含的值
 * @return 包含该值的 ValueList
 */
template<class T>
ValueList<T> Value(T v)
{
    return {v};
}

/**
 * @brief 创建包含默认值的 ValueList
 * 
 * 便捷函数，用于创建包含 Default 占位符的 ValueList
 * 
 * @return 包含 Default 值的 ValueList
 */
inline ValueList<detail::Default> ValueDefault()
{
    return {detail::Default{}};
}

namespace detail {

/**
 * @brief 规范化 ValueList 类型（基础模板）
 * 
 * 用于将 ValueList 类型转换为其规范形式
 */
template<class T>
struct NormalizeValueList;

/**
 * @brief 规范化 ValueList 类型（ValueList 特化）
 * 
 * 对于 ValueList<TT...>，保持类型不变
 */
template<class... TT>
struct NormalizeValueList<ValueList<TT...>>
{
    using type = ValueList<TT...>;
};

/**
 * @brief 规范化 ValueList 类型（ValueList<tuple> 特化）
 * 
 * 对于 ValueList<std::tuple<TT...>>，展开为 ValueList<TT...>
 */
template<class... TT>
struct NormalizeValueList<ValueList<std::tuple<TT...>>>
{
    using type = ValueList<TT...>;
};

/**
 * @brief 类型特征：判断类型是否为 ValueList（基础模板）
 * 
 * 对于非 ValueList 类型，继承自 std::false_type
 */
template<class T>
struct IsValueList : std::false_type
{
};

/**
 * @brief 类型特征：判断类型是否为 ValueList（特化版本）
 * 
 * 对于 ValueList<TT...> 类型，继承自 std::true_type
 */
template<class... TT>
struct IsValueList<ValueList<TT...>> : std::true_type
{
};

} // namespace detail

/**
 * @brief 编译期常量：判断类型 T 是否为 ValueList
 * 
 * 使用示例：
 * @code
 * if constexpr (IsValueList<T>) {
 *     // T 是 ValueList 类型
 * }
 * @endcode
 */
template<class T>
constexpr bool IsValueList = detail::IsValueList<T>::value;

// ============================================================================
// UniqueSort - 排序并去重
// ============================================================================

/**
 * @brief 对 ValueList 进行排序并去除重复元素（自定义提取器版本）
 * 
 * 该函数首先根据提取器函数对列表进行排序，然后移除相邻的重复元素。
 * 
 * 排序规则：
 * 1. 首先使用提取器函数提取比较键
 * 2. 如果提取的键相等，则使用元素本身进行比较
 * 3. 否则使用提取的键进行比较
 * 
 * 去重规则：
 * 移除提取器函数返回相同值的相邻元素
 * 
 * @tparam NN 可选的索引序列，用于从 tuple 中提取特定元素
 * @tparam F 提取器函数类型
 * @tparam TT ValueList 的类型参数包
 * @param extractor 提取器函数，用于从元素中提取比较键
 * @param a 要处理的 ValueList
 * @return 排序并去重后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 按第一个元素排序并去重
 * auto result = UniqueSort<0>(
 *     [](int x) { return x; },
 *     ValueList{std::make_tuple(3, "a"), std::make_tuple(1, "b"), std::make_tuple(1, "c")}
 * );
 * @endcode
 */
template<int... NN, class F, class... TT>
ValueList<TT...> UniqueSort(F extractor, ValueList<TT...> a)
{
    // 使用自定义比较器对列表进行排序
    a.m_list.sort(
        [&extractor](const auto &a, const auto &b)
        {
            // 使用提取器函数提取比较键
            auto ta = apply(extractor, detail::MaybeExtractTuple<NN...>(a));
            auto tb = apply(extractor, detail::MaybeExtractTuple<NN...>(b));

            // 如果提取的键相等，则使用元素本身比较
            if (ta == tb)
            {
                return a < b;
            }
            else
            {
                return ta < tb;
            }
        });

    // 移除相邻的重复元素（根据提取器函数判断是否重复）
    a.m_list.unique(
        [&extractor](const auto &a, const auto &b)
        {
            return apply(extractor, detail::MaybeExtractTuple<NN...>(a))
                == apply(extractor, detail::MaybeExtractTuple<NN...>(b));
        });

    return a;
}

/**
 * @brief 对 ValueList 进行排序并去除重复元素（默认提取器版本）
 * 
 * 使用默认的提取器（将所有元素组成 tuple）进行排序和去重
 * 
 * @tparam NN 可选的索引序列，用于从 tuple 中提取特定元素
 * @tparam TT ValueList 的类型参数包
 * @param a 要处理的 ValueList
 * @return 排序并去重后的 ValueList
 * 
 * 使用示例：
 * @code
 * auto result = UniqueSort(ValueList{3, 1, 2, 1, 3});
 * // 结果: {1, 2, 3}
 * @endcode
 */
template<int... NN, class... TT>
ValueList<TT...> UniqueSort(ValueList<TT...> a)
{
    return UniqueSort<NN...>([](auto... v) { return std::make_tuple(v...); }, std::move(a));
};

// ============================================================================
// Concat - 连接多个值或列表
// ============================================================================

/**
 * @brief 连接单个值（递归终止条件）
 * 
 * 将单个值包装成 ValueList
 * 
 * @tparam T 值的类型
 * @param v 要包装的值
 * @return 包含该值的 ValueList
 */
template<class T>
ValueList<T> Concat(T v)
{
    return {v};
}

/**
 * @brief 连接单个 ValueList（递归终止条件）
 * 
 * 直接返回 ValueList 本身
 * 
 * @tparam TT ValueList 的类型参数包
 * @param v 要返回的 ValueList
 * @return 原 ValueList
 */
template<class... TT>
auto Concat(ValueList<TT...> v)
{
    return v;
}

/**
 * @brief 连接单个 tuple（递归终止条件）
 * 
 * 将 tuple 转换为 ValueList
 * 
 * @tparam TT tuple 的类型参数包
 * @param v 要转换的 tuple
 * @return 包含该 tuple 的 ValueList
 */
template<class... TT>
ValueList<TT...> Concat(std::tuple<TT...> v)
{
    return v;
}

/**
 * @brief 连接多个 ValueList
 * 
 * 递归地将多个 ValueList 连接成一个大的 ValueList
 * 
 * @tparam TT 第一个 ValueList 的类型参数包
 * @tparam TAIL 剩余参数的类型包
 * @param head 第一个 ValueList
 * @param tail 剩余的 ValueList 或值
 * @return 连接后的 ValueList
 * 
 * 使用示例：
 * @code
 * auto result = Concat(
 *     ValueList{1, 2, 3},
 *     ValueList{4, 5},
 *     6
 * );
 * // 结果: {1, 2, 3, 4, 5, 6}
 * @endcode
 */
template<class... TT, class... TAIL>
ValueList<TT...> Concat(ValueList<TT...> head, TAIL &&...tail)
{
    typename detail::NormalizeValueList<ValueList<TT...>>::type rest = Concat(std::forward<TAIL>(tail)...);
    head.concat(std::move(rest));
    return head;
}

/**
 * @brief 连接单个值和其他参数
 * 
 * 将单个值添加到递归连接的结果前面
 * 
 * @tparam T 值的类型
 * @tparam TAIL 剩余参数的类型包
 * @param v 要添加的值
 * @param tail 剩余的值或 ValueList
 * @return 连接后的 ValueList
 */
template<class T, class... TAIL>
ValueList<T> Concat(T v, TAIL &&...tail)
{
    ValueList<T> rest = Concat(std::forward<TAIL>(tail)...);

    rest.push_front(v);
    return rest;
}

// ============================================================================
// Difference - 集合差集
// ============================================================================

/**
 * @brief 计算两个 ValueList 的差集（A - B）
 * 
 * 返回在 a 中但不在 b 中的所有元素。
 * 
 * 算法：
 * 1. 对 b 进行排序和去重
 * 2. 遍历 a，移除所有在 b 中存在的元素
 * 
 * @tparam TT ValueList 的类型参数包
 * @param a 第一个 ValueList（被减数）
 * @param b 第二个 ValueList（减数）
 * @return 差集结果
 * 
 * 使用示例：
 * @code
 * auto result = Difference(
 *     ValueList{1, 2, 3, 4},
 *     ValueList{2, 4, 6}
 * );
 * // 结果: {1, 3}
 * @endcode
 */
template<class... TT>
ValueList<TT...> Difference(ValueList<TT...> a, ValueList<TT...> b)
{
    // 对 b 进行排序和去重，以便使用二分查找
    b = UniqueSort(b);

    // 遍历 a，移除所有在 b 中存在的元素
    for (auto it = a.begin(); it != a.end();)
    {
        if (binary_search(b.begin(), b.end(), *it))
        {
            a.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return a;
}

/**
 * @brief 计算 ValueList 与单个值的差集
 * 
 * 从 ValueList 中移除指定的单个值
 * 
 * @tparam T 值的类型
 * @param a ValueList
 * @param b 要移除的值
 * @return 差集结果
 */
template<class T>
ValueList<T> Difference(ValueList<T> a, T b)
{
    return Difference(a, ValueList<T>{b});
}

/**
 * @brief 计算单个值与 ValueList 的差集
 * 
 * 如果单个值不在 ValueList 中，则返回包含该值的列表；否则返回空列表
 * 
 * @tparam T 值的类型
 * @param a 单个值
 * @param b ValueList
 * @return 差集结果
 */
template<class T>
ValueList<T> Difference(T a, ValueList<T> b)
{
    return Difference(ValueList<T>{a}, b);
}

// ============================================================================
// SymmetricDifference - 集合对称差
// ============================================================================

/**
 * @brief 计算两个 ValueList 的对称差集
 * 
 * 对称差集包含所有只在 a 或只在 b 中出现的元素，
 * 即 (A - B) ∪ (B - A)
 * 
 * @tparam TT ValueList 的类型参数包
 * @param a 第一个 ValueList
 * @param b 第二个 ValueList
 * @return 对称差集结果
 * 
 * 使用示例：
 * @code
 * auto result = SymmetricDifference(
 *     ValueList{1, 2, 3, 4},
 *     ValueList{3, 4, 5, 6}
 * );
 * // 结果: {1, 2, 5, 6}
 * @endcode
 */
template<class... TT>
ValueList<TT...> SymmetricDifference(ValueList<TT...> a, ValueList<TT...> b)
{
    return Concat(Difference(a, b), Difference(b, a));
}

/**
 * @brief 计算 ValueList 与单个值的对称差集
 * 
 * @tparam T 值的类型
 * @param a ValueList
 * @param b 单个值
 * @return 对称差集结果
 */
template<class T>
ValueList<T> SymmetricDifference(ValueList<T> a, T b)
{
    return SymmetricDifference(a, ValueList<T>{b});
}

/**
 * @brief 计算单个值与 ValueList 的对称差集
 * 
 * @tparam T 值的类型
 * @param a 单个值
 * @param b ValueList
 * @return 对称差集结果
 */
template<class T>
ValueList<T> SymmetricDifference(T a, ValueList<T> b)
{
    return SymmetricDifference(ValueList<T>{a}, b);
}

// ============================================================================
// Intersection - 集合交集
// ============================================================================

/**
 * @brief 计算多个 ValueList 的交集
 * 
 * 返回所有输入列表中都存在的元素。
 * 支持计算两个或多个列表的交集。
 * 
 * 算法：
 * 1. 如果有多个参数，先递归计算后续参数的交集
 * 2. 对结果进行排序和去重
 * 3. 遍历第一个列表，保留所有在结果中存在的元素
 * 
 * @tparam TT ValueList 的类型参数包
 * @tparam TAIL 剩余参数的类型包
 * @param a 第一个 ValueList
 * @param b 第二个 ValueList
 * @param tail 可选的其他 ValueList
 * @return 交集结果
 * 
 * 使用示例：
 * @code
 * auto result = Intersection(
 *     ValueList{1, 2, 3, 4},
 *     ValueList{2, 3, 4, 5},
 *     ValueList{3, 4, 5, 6}
 * );
 * // 结果: {3, 4}
 * @endcode
 */
template<class... TT, class... TAIL>
ValueList<TT...> Intersection(ValueList<TT...> a, ValueList<TT...> b, TAIL &&...tail)
{
    ValueList<TT...> tmp;

    // 如果有多个参数，先递归计算后续参数的交集
    if constexpr (sizeof...(TAIL) > 0)
    {
        tmp = Intersection(std::move(b), std::forward<TAIL>(tail)...);
    }
    else
    {
        tmp = std::move(b);
    }

    // 对结果进行排序和去重，以便使用二分查找
    tmp = UniqueSort(tmp);

    // 遍历第一个列表，保留所有在结果中存在的元素
    for (auto it = a.begin(); it != a.end();)
    {
        if (binary_search(tmp.begin(), tmp.end(), *it))
        {
            ++it;
        }
        else
        {
            a.erase(it++);
        }
    }

    return a;
}

/**
 * @brief 计算 ValueList 与单个值的交集
 * 
 * 如果值在 ValueList 中存在，返回包含该值的列表；否则返回空列表
 * 
 * @tparam T 值的类型
 * @param a ValueList
 * @param b 单个值
 * @return 交集结果
 */
template<class T>
ValueList<T> Intersection(ValueList<T> a, T b)
{
    return Intersection(a, ValueList<T>{b});
}

/**
 * @brief 计算单个值与 ValueList 的交集
 * 
 * 如果值在 ValueList 中存在，返回包含该值的列表；否则返回空列表
 * 
 * @tparam T 值的类型
 * @param a 单个值
 * @param b ValueList
 * @return 交集结果
 */
template<class T>
ValueList<T> Intersection(T a, ValueList<T> b)
{
    return Intersection(ValueList<T>{a}, b);
}

// ============================================================================
// Transform - 转换函数
// ============================================================================

/**
 * @brief 对 ValueList 中的每个元素应用转换函数
 * 
 * 转换函数可以返回：
 * 1. 单个参数（tuple 或普通值）- 将被添加到输出列表
 * 2. ValueList（包含多个参数）- 所有元素都会被添加到输出列表
 * 
 * @tparam TT 输入 ValueList 的类型参数包
 * @tparam F 转换函数类型
 * @param in 输入的 ValueList
 * @param xform 转换函数
 * @return 转换后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 将每个数字转换为其平方
 * auto result = Transform(ValueList{1, 2, 3}, 
 *     [](int x) { return x * x; });
 * // 结果: {1, 4, 9}
 * 
 * // 将每个数字展开为多个值
 * auto result2 = Transform(ValueList{1, 2}, 
 *     [](int x) { return ValueList{x, x * 2}; });
 * // 结果: {1, 2, 2, 4}
 * @endcode
 */
template<class... TT, class F>
auto Transform(ValueList<TT...> in, F xform)
{
    using R = std::invoke_result_t<F, TT...>;

    typename detail::NormalizeValueList<std::conditional_t<IsValueList<R>, R, ValueList<R>>>::type out;
    for (const auto &v : in)
    {
        if constexpr (IsValueList<R>)
        {
            // 如果转换函数返回 ValueList，则将其所有元素添加到输出
            for (const auto &vv : std::apply(xform, detail::JoinTuple(v)))
            {
                out.push_back(vv);
            }
        }
        else
        {
            // 如果转换函数返回单个值，则直接添加到输出
            out.push_back(std::apply(xform, detail::JoinTuple(v)));
        }
    }

    return out;
}

// ============================================================================
// Make - 构造新类型的 ValueList
// ============================================================================

namespace detail {

/**
 * @brief 类型特征：检查类型 T 是否有 value_type 成员类型（基础模板）
 * 
 * 对于没有 value_type 的类型，继承自 std::false_type
 */
template<class T, class EN = void>
struct HasValueType : std::false_type
{
};

/**
 * @brief 类型特征：检查类型 T 是否有 value_type 成员类型（特化版本）
 * 
 * 对于有 value_type 的类型（如 std::optional），继承自 std::true_type
 */
template<class T>
struct HasValueType<T, std::enable_if_t<sizeof(typename T::value_type) != 0>> : std::true_type
{
};

} // namespace detail

/**
 * @brief 将 ValueList 中的元素转换为新类型
 * 
 * 根据目标类型 OUT 的构造方式，使用不同的策略：
 * 1. 如果 OUT 可以从 TT... 直接构造，使用 std::make_from_tuple
 * 2. 如果 OUT 支持 std::in_place 构造（如 std::optional），使用 in_place 构造
 * 3. 如果 OUT 有 value_type（如 std::optional），使用 value_type 构造
 * 4. 否则，使用聚合初始化
 * 
 * @tparam OUT 目标类型
 * @tparam TT 输入 ValueList 的类型参数包
 * @param in 输入的 ValueList
 * @return 包含新类型元素的 ValueList
 * 
 * 使用示例：
 * @code
 * struct Point { int x, y; };
 * auto points = Make<Point>(ValueList{
 *     std::make_tuple(1, 2),
 *     std::make_tuple(3, 4)
 * });
 * // 结果: ValueList<Point>{{1,2}, {3,4}}
 * @endcode
 */
template<class OUT, class... TT>
ValueList<OUT> Make(const ValueList<TT...> &in)
{
    ValueList<OUT> out;

    for (auto &v : in)
    {
        if constexpr (std::is_constructible_v<OUT, TT...>)
        {
            // 直接构造
            out.push_back(std::make_from_tuple<OUT>(detail::JoinTuple(v)));
        }
        else if constexpr (std::is_constructible_v<OUT, std::in_place_t, TT...>)
        {
            // in_place 构造，用于 std::optional、std::any、std::variant 等
            out.push_back(std::apply([](TT &&...aa) { return OUT{std::in_place, aa...}; }, detail::JoinTuple(v)));
        }
        else if constexpr (detail::HasValueType<OUT>::value)
        {
            // 对于有 value_type 的类型（如 std::optional），但没有显式构造函数
            out.push_back(
                std::apply([](TT &&...aa) { return OUT{typename OUT::value_type{aa...}}; }, detail::JoinTuple(v)));
        }
        else
        {
            // 聚合初始化
            out.push_back(std::apply([](TT &&...aa) { return OUT{aa...}; }, detail::JoinTuple(v)));
        }
    }

    return out;
}

// ============================================================================
// Combine - 笛卡尔积组合
// ============================================================================

/**
 * @brief 组合单个 ValueList（递归终止条件）
 * 
 * @tparam TT ValueList 的类型参数包
 * @param a 要返回的 ValueList
 * @return 原 ValueList
 */
template<class... TT>
auto Combine(ValueList<TT...> a)
{
    return a;
}

/**
 * @brief 组合单个值（递归终止条件）
 * 
 * @tparam T 值的类型
 * @param v 要包装的值
 * @return 包含该值的 ValueList
 */
template<class T>
ValueList<std::decay_t<T>> Combine(T &&v)
{
    return {v};
}

/**
 * @brief 组合多个单独的值（递归终止条件）
 * 
 * 将多个值组合成一个 tuple 并包装成 ValueList
 * 
 * @tparam TT 值的类型参数包
 * @param v 要组合的值
 * @return 包含 tuple 的 ValueList
 */
template<class... TT>
ValueList<std::decay_t<TT>...> Combine(TT &&...v)
{
    return {std::make_tuple(v...)};
}

/**
 * @brief 计算多个 ValueList 的笛卡尔积
 * 
 * 生成所有可能的参数组合。对于输入列表 A 和 B，
 * 结果包含 A 中每个元素与 B 中每个元素的所有组合。
 * 
 * @tparam TT 第一个 ValueList 的类型参数包
 * @tparam TAIL 剩余参数的类型包
 * @param a 第一个 ValueList
 * @param tail 剩余的 ValueList 或值
 * @return 笛卡尔积结果
 * 
 * 使用示例：
 * @code
 * auto result = Combine(
 *     ValueList{1, 2},
 *     ValueList{"a", "b"}
 * );
 * // 结果: {(1,"a"), (1,"b"), (2,"a"), (2,"b")}
 * @endcode
 */
template<class... TT, class... TAIL>
auto Combine(ValueList<TT...> a, TAIL &&...tail)
{
    auto rest = Combine(std::forward<TAIL>(tail)...);

    typename detail::NormalizeValueList<
        ValueList<decltype(tuple_cat(std::tuple<TT...>(), typename decltype(rest)::tuple_value_type()))>>::type r;

    // 嵌套循环生成笛卡尔积
    for (auto ita = a.begin(); ita != a.end(); ++ita)
    {
        for (auto itr = rest.begin(); itr != rest.end(); ++itr)
        {
            r.push_back(detail::JoinTuple(*ita, *itr));
        }
    }

    return r;
}

/**
 * @brief 组合单个值和其他参数
 * 
 * 将单个值转换为 ValueList 后进行组合
 * 
 * @tparam T 值的类型
 * @tparam TAIL 剩余参数的类型包
 * @param a 单个值
 * @param tail 剩余的 ValueList 或值
 * @return 组合结果
 */
template<class T, class... TAIL>
auto Combine(T &&a, TAIL &&...tail)
{
    return Combine(ValueList<std::decay_t<T>>{a}, std::forward<TAIL>(tail)...);
}

// ============================================================================
// Zip - 拉链组合（对应位置配对）
// ============================================================================

/**
 * @brief 拉链单个 ValueList（递归终止条件）
 * 
 * @tparam TT ValueList 的类型参数包
 * @param a 要返回的 ValueList
 * @return 原 ValueList
 */
template<class... TT>
auto Zip(ValueList<TT...> a)
{
    return a;
}

/**
 * @brief 拉链单个值（递归终止条件）
 * 
 * @tparam T 值的类型
 * @param v 要包装的值
 * @return 包含该值的 ValueList
 */
template<class T>
ValueList<std::decay_t<T>> Zip(T &&v)
{
    return {v};
}

/**
 * @brief 拉链多个单独的值（递归终止条件）
 * 
 * @tparam TT 值的类型参数包
 * @param v 要组合的值
 * @return 包含 tuple 的 ValueList
 */
template<class... TT>
ValueList<std::decay_t<TT>...> Zip(TT &&...v)
{
    return {std::make_tuple(v...)};
}

/**
 * @brief 将多个 ValueList 按对应位置配对
 * 
 * 类似于拉链，将多个列表的对应位置元素组合在一起。
 * 所有输入列表必须具有相同的大小，否则抛出异常。
 * 
 * @tparam TT 第一个 ValueList 的类型参数包
 * @tparam TAIL 剩余参数的类型包
 * @param a 第一个 ValueList
 * @param tail 剩余的 ValueList 或值
 * @return 拉链组合结果
 * @throws std::logic_error 如果列表大小不同
 * 
 * 使用示例：
 * @code
 * auto result = Zip(
 *     ValueList{1, 2, 3},
 *     ValueList{"a", "b", "c"}
 * );
 * // 结果: {(1,"a"), (2,"b"), (3,"c")}
 * @endcode
 */
template<class... TT, class... TAIL>
auto Zip(ValueList<TT...> a, TAIL &&...tail)
{
    auto rest = Zip(std::forward<TAIL>(tail)...);

    typename detail::NormalizeValueList<
        ValueList<decltype(tuple_cat(std::tuple<TT...>(), typename decltype(rest)::tuple_value_type()))>>::type r;

    // 检查列表大小是否相同
    if (a.size() == rest.size())
    {
        // 按对应位置配对
        for (auto ita = a.begin(), itr = rest.begin(); ita != a.end(); ++ita, ++itr)
        {
            r.push_back(detail::JoinTuple(*ita, *itr));
        }
    }
    else
    {
        throw std::logic_error("Zip: value lists can't have different sizes");
    }

    return r;
}

/**
 * @brief 拉链单个值和其他参数
 * 
 * @tparam T 值的类型
 * @tparam TAIL 剩余参数的类型包
 * @param a 单个值
 * @param tail 剩余的 ValueList 或值
 * @return 拉链组合结果
 */
template<class T, class... TAIL>
auto Zip(T &&a, TAIL &&...tail)
{
    return Zip(ValueList<T>{a}, std::forward<TAIL>(tail)...);
}

// ============================================================================
// IsSameArgs - 判断所有参数是否相同
// ============================================================================

/**
 * @brief 判断所有参数是否相同的函数对象
 * 
 * 用于检查传入的多个参数是否都相等
 */
struct IsSameArgsFunctor
{
private:
    /**
     * @brief 单个参数的情况（递归终止条件）
     * 
     * 单个参数无法比较，返回 false
     */
    template<class T>
    static bool isSameArgs(T &&)
    {
        return false;
    }

    /**
     * @brief 两个参数的情况（递归终止条件）
     * 
     * 比较两个参数是否相等
     */
    template<class T, class U>
    static bool isSameArgs(T &&a, U &&b)
    {
        return a == b;
    }

    /**
     * @brief 多个参数的情况（递归版本）
     * 
     * 递归比较所有参数是否相等
     */
    template<class T, class U, class... TAIL>
    static bool isSameArgs(T &&a, U &&b, TAIL &&...tail)
    {
        if (a == b)
        {
            return isSameArgs(b, tail...);
        }
        else
        {
            return false;
        }
    }

public:
    /**
     * @brief 函数调用运算符
     * 
     * @tparam TT 参数类型包
     * @param args 要比较的参数
     * @return 如果所有参数都相等则返回 true
     */
    template<class... TT>
    bool operator()(TT &&...args) const
    {
        return isSameArgs(std::forward<TT>(args)...);
    }
};

/**
 * @brief IsSameArgsFunctor 的全局实例
 * 
 * 使用示例：
 * @code
 * bool result = IsSameArgs(1, 1, 1);  // true
 * bool result2 = IsSameArgs(1, 2, 1); // false
 * @endcode
 */
inline const IsSameArgsFunctor IsSameArgs;

// ============================================================================
// Match - 匹配辅助工具
// ============================================================================

/**
 * @brief 匹配辅助类模板
 * 
 * 用于检查元素是否匹配指定的模式。
 * 支持部分匹配（只匹配指定索引的元素）和完全匹配。
 * 
 * @tparam MATCH_ALL 是否要求匹配所有元素
 * @tparam SEQ 要匹配的索引序列
 * @tparam TT 匹配模式的类型参数包
 */
template<bool MATCH_ALL, class SEQ, class... TT>
struct MatchHelper
{
    static_assert(SEQ::size() == sizeof...(TT));

private:
    /**
     * @brief 匹配终止条件
     * 
     * 当所有索引都已检查完毕，返回 true
     */
    template<int IDX, class T, class U>
    static bool match(std::integer_sequence<int>, const T &item, const U &needle)
    {
        static_assert(IDX == SEQ::size());
        return true;
    }

    /**
     * @brief 递归匹配指定索引的元素
     * 
     * 检查 item 在索引 N 处的元素是否等于 needle 的对应元素
     */
    template<int IDX, int N, int... NTAIL, class T, class U>
    bool match(std::integer_sequence<int, N, NTAIL...>, const T &item, const U &needle) const
    {
        return std::get<IDX>(needle) == std::get<N>(detail::JoinTuple(item))
            && match<IDX + 1>(std::integer_sequence<int, NTAIL...>(), item, needle);
    }

    ValueList<TT...> m_needle; ///< 匹配模式列表

public:
    /**
     * @brief 从单个值构造匹配器
     * 
     * @param needle 要匹配的值
     */
    MatchHelper(TT &&...needle)
        : m_needle({typename ValueList<TT...>::value_type{std::forward<TT>(needle)...}})
    {
    }

    /**
     * @brief 从 ValueList 构造匹配器
     * 
     * 可以匹配列表中的任意一个值
     * 
     * @param needle 要匹配的值列表
     */
    MatchHelper(ValueList<TT...> needle)
        : m_needle(std::move(needle))
    {
    }

    /**
     * @brief 检查参数是否匹配模式
     * 
     * @tparam UU 参数类型包
     * @param args 要检查的参数
     * @return 如果匹配则返回 true
     */
    template<class... UU>
    bool operator()(UU &&...args) const
    {
        static_assert(!MATCH_ALL || sizeof...(UU) == sizeof...(TT), "Match arity must match list arity");

        if constexpr (SEQ::size() >= 1)
        {
            // 检查是否匹配列表中的任意一个模式
            for (const auto &m : m_needle)
            {
                if (match<0>(SEQ(), std::make_tuple(std::forward<UU>(args)...), detail::JoinTuple(m)))
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            return false;
        }
    }
};

/**
 * @brief 创建部分匹配器（匹配指定索引的元素）
 * 
 * @tparam N 第一个要匹配的索引
 * @tparam NN 其他要匹配的索引
 * @tparam TT 匹配值的类型参数包
 * @param args 要匹配的值
 * @return 匹配器对象
 * 
 * 使用示例：
 * @code
 * // 匹配第 0 个元素为 1 的所有项
 * auto filtered = RemoveIf(Match<0>(1), myList);
 * @endcode
 */
template<int N, int... NN, class... TT>
auto Match(TT &&...args)
{
    return MatchHelper<false, std::integer_sequence<int, N, NN...>, TT...>(std::forward<TT>(args)...);
}

/**
 * @brief 创建部分匹配器（从 ValueList 匹配指定索引的元素）
 * 
 * @tparam N 第一个要匹配的索引
 * @tparam NN 其他要匹配的索引
 * @tparam TT ValueList 的类型参数包
 * @param list 要匹配的值列表
 * @return 匹配器对象
 */
template<int N, int... NN, class... TT>
auto Match(ValueList<TT...> list)
{
    return MatchHelper<false, std::integer_sequence<int, N, NN...>, TT...>(std::move(list));
}

/**
 * @brief 创建完全匹配器（从 ValueList 匹配所有元素）
 * 
 * @tparam TT ValueList 的类型参数包
 * @param list 要匹配的值列表
 * @return 匹配器对象
 */
template<class... TT>
auto Match(ValueList<TT...> list)
{
    return MatchHelper<true, std::make_integer_sequence<int, sizeof...(TT)>, TT...>(std::move(list));
}

/**
 * @brief 创建完全匹配器（匹配所有元素）
 * 
 * @tparam TT 匹配值的类型参数包
 * @param args 要匹配的值
 * @return 匹配器对象
 * 
 * 使用示例：
 * @code
 * // 匹配 (1, "hello") 的所有项
 * auto filtered = RemoveIf(Match(1, "hello"), myList);
 * @endcode
 */
template<class... TT>
auto Match(TT &&...args)
{
    return MatchHelper<true, std::make_integer_sequence<int, sizeof...(TT)>, TT...>(std::forward<TT>(args)...);
}

// ============================================================================
// RemoveIf - 条件移除函数
// ============================================================================

/**
 * @brief 根据指定索引的元素是否满足条件来移除列表项
 * 
 * 只检查指定索引位置的元素是否满足条件
 * 
 * @tparam I 第一个要检查的索引
 * @tparam II 其他要检查的索引
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 移除满足条件的元素后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 移除第 0 个元素大于 5 的所有项
 * auto result = RemoveIf<0>([](int x) { return x > 5; }, myList);
 * @endcode
 */
template<int I, int... II, class... TT, class F>
ValueList<TT...> RemoveIf(F criteria, vtest::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;

        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            // 提取指定索引的元素并应用条件函数
            mustRemove = std::apply(criteria, detail::ExtractTuple<I, II...>(*it));
        }
        else
        {
            // 对于非 tuple 类型，直接应用条件函数
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

/**
 * @brief 根据元素是否满足条件来移除列表项
 * 
 * 检查整个元素（所有字段）是否满足条件
 * 
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 移除满足条件的元素后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 移除所有偶数
 * auto result = RemoveIf([](int x) { return x % 2 == 0; }, ValueList{1, 2, 3, 4});
 * // 结果: {1, 3}
 * @endcode
 */
template<class... TT, class F>
ValueList<TT...> RemoveIf(F criteria, vtest::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;

        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            // 对于 tuple 类型，展开所有元素并应用条件函数
            mustRemove = std::apply(criteria, *it);
        }
        else
        {
            // 对于非 tuple 类型，直接应用条件函数
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// ============================================================================
// RemoveIfAny - 移除任意字段满足条件的元素
// ============================================================================

namespace detail {
/**
 * @brief 检查 tuple 中是否有任意元素满足条件
 * 
 * @tparam II 索引序列
 * @tparam F 条件函数类型
 * @tparam TT tuple 的类型参数包
 * @param 索引序列（编译期参数）
 * @param criteria 条件函数
 * @param v 要检查的 tuple
 * @return 如果任意元素满足条件则返回 true
 */
template<int... II, class F, class... TT>
bool MustRemoveAny(std::integer_sequence<int, II...>, const F &criteria, const std::tuple<TT...> &v)
{
    static_assert(sizeof...(II) == sizeof...(TT));

    return (false || ... || criteria(std::get<II>(v)));
}
} // namespace detail

/**
 * @brief 移除任意字段满足条件的元素
 * 
 * 如果元素的任意一个字段满足条件，则移除该元素
 * 
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 移除后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 移除任意字段为负数的元素
 * auto result = RemoveIfAny([](int x) { return x < 0; }, 
 *     ValueList{std::make_tuple(1, 2), std::make_tuple(-1, 3)});
 * // 结果: {(1, 2)}
 * @endcode
 */
template<class... TT, class F>
ValueList<TT...> RemoveIfAny(const F &criteria, vtest::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;
        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = detail::MustRemoveAny(std::make_integer_sequence<int, sizeof...(TT)>(), criteria, *it);
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// ============================================================================
// RemoveIfAll - 移除所有字段都满足条件的元素
// ============================================================================

namespace detail {
/**
 * @brief 检查 tuple 中是否所有元素都满足条件
 * 
 * @tparam II 索引序列
 * @tparam F 条件函数类型
 * @tparam TT tuple 的类型参数包
 * @param 索引序列（编译期参数）
 * @param criteria 条件函数
 * @param v 要检查的 tuple
 * @return 如果所有元素都满足条件则返回 true
 */
template<int... II, class F, class... TT>
bool MustRemoveAll(std::integer_sequence<int, II...>, const F &criteria, const std::tuple<TT...> &v)
{
    static_assert(sizeof...(II) == sizeof...(TT));

    return (true && ... && criteria(std::get<II>(v)));
}
} // namespace detail

/**
 * @brief 移除所有字段都满足条件的元素
 * 
 * 只有当元素的所有字段都满足条件时，才移除该元素
 * 
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 移除后的 ValueList
 * 
 * 使用示例：
 * @code
 * // 移除所有字段都为正数的元素
 * auto result = RemoveIfAll([](int x) { return x > 0; }, 
 *     ValueList{std::make_tuple(1, 2), std::make_tuple(-1, 3)});
 * // 结果: {(-1, 3)}
 * @endcode
 */
template<class... TT, class F>
ValueList<TT...> RemoveIfAll(const F &criteria, vtest::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;
        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = detail::MustRemoveAll(std::make_integer_sequence<int, sizeof...(TT)>(), criteria, *it);
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// ============================================================================
// SelectIf 系列 - 条件选择函数（RemoveIf 的反向操作）
// ============================================================================

/**
 * @brief 选择满足条件的元素
 * 
 * 保留满足条件的元素，移除不满足条件的元素（RemoveIf 的反向操作）
 * 
 * @tparam II 可选的索引序列
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 选择后的 ValueList
 */
template<int... II, class... TT, class F>
ValueList<TT...> SelectIf(const F &criteria, vtest::ValueList<TT...> list)
{
    return RemoveIf<II...>(std::not_fn(criteria), std::move(list));
}

/**
 * @brief 选择任意字段满足条件的元素
 * 
 * 保留至少有一个字段满足条件的元素
 * 
 * @tparam II 可选的索引序列
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 选择后的 ValueList
 */
template<int... II, class... TT, class F>
ValueList<TT...> SelectIfAny(const F &criteria, vtest::ValueList<TT...> list)
{
    return RemoveIfAll<II...>(std::not_fn(criteria), std::move(list));
}

/**
 * @brief 选择所有字段都满足条件的元素
 * 
 * 保留所有字段都满足条件的元素
 * 
 * @tparam II 可选的索引序列
 * @tparam TT ValueList 的类型参数包
 * @tparam F 条件函数类型
 * @param criteria 条件函数
 * @param list 要处理的 ValueList
 * @return 选择后的 ValueList
 */
template<int... II, class... TT, class F>
ValueList<TT...> SelectIfAll(const F &criteria, vtest::ValueList<TT...> list)
{
    return RemoveIfAny<II...>(std::not_fn(criteria), std::move(list));
}

// ============================================================================
// Extract - 提取指定索引的元素
// ============================================================================

/**
 * @brief 从 ValueList 中提取指定索引的元素
 * 
 * 创建一个新的 ValueList，只包含原列表中指定索引位置的元素
 * 
 * @tparam NN 要提取的索引序列
 * @tparam TT ValueList 的类型参数包
 * @param v 输入的 ValueList
 * @return 包含提取元素的新 ValueList
 * 
 * 使用示例：
 * @code
 * auto list = ValueList{
 *     std::make_tuple(1, "a", 3.14),
 *     std::make_tuple(2, "b", 2.71)
 * };
 * auto result = Extract<0, 2>(list);
 * // 结果: {(1, 3.14), (2, 2.71)}
 * @endcode
 */
template<int... NN, class... TT>
auto Extract(const ValueList<TT...> &v)
{
    using DestTuple = decltype(detail::ExtractTuple<NN...>(detail::JoinTuple(*v.begin())));

    typename detail::NormalizeValueList<ValueList<DestTuple>>::type out;

    for (auto &src : v)
    {
        if constexpr (sizeof...(NN) == 1)
        {
            // 如果只提取一个索引，直接添加该元素（不包装成 tuple）
            out.push_back(std::get<0>(detail::ExtractTuple<NN...>(detail::JoinTuple(src))));
        }
        else
        {
            // 如果提取多个索引，添加包含这些元素的 tuple
            out.push_back(detail::ExtractTuple<NN...>(detail::JoinTuple(src)));
        }
    }

    return out;
}

// ============================================================================
// 逻辑运算辅助函数
// ============================================================================

/**
 * @brief 创建逻辑非函数对象
 * 
 * 对给定的函数对象取反
 * 
 * @tparam F 函数类型
 * @param fn 要取反的函数
 * @return 取反后的函数对象
 */
template<class F>
auto Not(F fn)
{
    return std::not_fn(std::move(fn));
}

namespace detail {
/**
 * @brief 逻辑或函数对象
 * 
 * 组合两个函数对象，返回它们的逻辑或结果
 * 
 * @tparam LHS 左侧函数类型
 * @tparam RHS 右侧函数类型
 */
template<class LHS, class RHS>
struct Or
{
    LHS lhs; ///< 左侧函数对象
    RHS rhs; ///< 右侧函数对象

    /// 左值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) & noexcept(noexcept(std::invoke(lhs, args...)
                                                                  || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...);
    }

    /// 常量左值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) const & noexcept(
        noexcept(std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...);
    }

    /// 右值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) && noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }

    /// 常量右值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) const && noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }
};
} // namespace detail

/**
 * @brief 创建逻辑或函数对象
 * 
 * 组合两个函数对象，返回它们的逻辑或
 * 
 * @tparam LHS 左侧函数类型
 * @tparam RHS 右侧函数类型
 * @param lhs 左侧函数
 * @param rhs 右侧函数
 * @return 逻辑或函数对象
 * 
 * 使用示例：
 * @code
 * auto isNegativeOrLarge = Or(
 *     [](int x) { return x < 0; },
 *     [](int x) { return x > 100; }
 * );
 * @endcode
 */
template<class LHS, class RHS>
constexpr detail::Or<std::decay_t<LHS>, std::decay_t<RHS>> Or(LHS &&lhs, RHS &&rhs)
{
    return {std::forward<LHS>(lhs), std::forward<RHS>(rhs)};
}

namespace detail {
/**
 * @brief 逻辑与函数对象
 * 
 * 组合两个函数对象，返回它们的逻辑与结果
 * 
 * @tparam LHS 左侧函数类型
 * @tparam RHS 右侧函数类型
 */
template<class LHS, class RHS>
struct And
{
    LHS lhs; ///< 左侧函数对象
    RHS rhs; ///< 右侧函数对象

    /// 左值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) & noexcept(noexcept(std::invoke(lhs, args...)
                                                                  || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) && std::invoke(rhs, std::forward<Args>(args)...);
    }

    /// 常量左值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) const & noexcept(
        noexcept(std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) && std::invoke(rhs, std::forward<Args>(args)...);
    }

    /// 右值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) && noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) && std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }

    /// 常量右值引用版本
    template<class... Args>
    constexpr auto operator()(Args &&...args) const && noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) && std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }
};
} // namespace detail

/**
 * @brief 创建逻辑与函数对象
 * 
 * 组合两个函数对象，返回它们的逻辑与
 * 
 * @tparam LHS 左侧函数类型
 * @tparam RHS 右侧函数类型
 * @param lhs 左侧函数
 * @param rhs 右侧函数
 * @return 逻辑与函数对象
 * 
 * 使用示例：
 * @code
 * auto isPositiveAndSmall = And(
 *     [](int x) { return x > 0; },
 *     [](int x) { return x < 100; }
 * );
 * @endcode
 */
template<class LHS, class RHS>
constexpr detail::And<std::decay_t<LHS>, std::decay_t<RHS>> And(LHS &&lhs, RHS &&rhs)
{
    return {std::forward<LHS>(lhs), std::forward<RHS>(rhs)};
}

// ============================================================================
// 运算符重载 - 提供更简洁的语法
// ============================================================================

/**
 * @brief 差集运算符（ValueList - ValueList）
 * 
 * @return Difference(a, b)
 */
template<class... TT>
ValueList<TT...> operator-(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Difference(a, b);
}

/**
 * @brief 差集运算符（ValueList - 单值）
 * 
 * @return Difference(a, b)
 */
template<class T>
ValueList<T> operator-(const ValueList<T> &a, const T &b)
{
    return Difference(a, b);
}

template<class T>
ValueList<T> operator-(const T &a, const ValueList<T> &b)
{
    return Difference(a, b);
}

/**
 * @brief 差集运算符（单值 - ValueList）
 * 
 * @return Difference(a, b)
 */
template<class T>
ValueList<T> operator-(const T &a, const ValueList<T> &b)
{
    return Difference(a, b);
}

/**
 * @brief 条件移除运算符（ValueList - 函数）
 * 
 * 使用函数作为条件移除元素
 * 
 * @return RemoveIf(fn, a)
 */
template<class... TT, class F>
ValueList<TT...> operator-(const ValueList<TT...> &a, const F &fn)
{
    return RemoveIf(fn, a);
}

/**
 * @brief 连接运算符（ValueList | ValueList）
 * 
 * @return Concat(a, b)
 */
template<class... TT>
ValueList<TT...> operator|(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Concat(a, b);
}

/**
 * @brief 连接运算符（ValueList | 单值）
 * 
 * @return Concat(a, b)
 */
template<class T>
ValueList<T> operator|(const ValueList<T> &a, const T &b)
{
    return Concat(a, b);
}

/**
 * @brief 连接运算符（单值 | ValueList）
 * 
 * @return Concat(a, b)
 */
template<class T>
ValueList<T> operator|(const T &a, const ValueList<T> &b)
{
    return Concat(a, b);
}

/**
 * @brief 交集运算符（ValueList & ValueList）
 * 
 * @return Intersection(a, b)
 */
template<class... TT>
ValueList<TT...> operator&(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Intersection(a, b);
}

/**
 * @brief 交集运算符（ValueList & 单值）
 * 
 * @return Intersection(a, ValueList{b})
 */
template<class T>
ValueList<T> operator&(const ValueList<T> &a, const T &b)
{
    return Intersection(a, ValueList<T>{b});
}

/**
 * @brief 交集运算符（单值 & ValueList）
 * 
 * @return Intersection(ValueList{a}, b)
 */
template<class T>
ValueList<T> operator&(const T &a, const ValueList<T> &b)
{
    return Intersection(ValueList<T>{a}, b);
}

/**
 * @brief 笛卡尔积运算符（ValueList * ValueList）
 * 
 * @return Combine(a, b)
 */
template<class... TT, class... UU>
auto operator*(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return Combine(a, b);
}

/**
 * @brief 笛卡尔积运算符（ValueList * 单值）
 * 
 * @return Combine(a, b)
 */
template<class... TT, class U>
auto operator*(const ValueList<TT...> &a, const U &b)
{
    return Combine(a, b);
}

/**
 * @brief 笛卡尔积运算符（单值 * ValueList）
 * 
 * @return Combine(a, b)
 */
template<class T, class... UU>
auto operator*(const T &a, const ValueList<UU...> &b)
{
    return Combine(a, b);
}

/**
 * @brief 拉链运算符（ValueList % ValueList）
 * 
 * 按对应位置配对元素
 * 
 * @return Zip(a, b)
 */
template<class... TT, class... UU>
auto operator%(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return Zip(a, b);
}

/**
 * @brief 对称差集运算符（ValueList ^ ValueList）
 * 
 * @return SymmetricDifference(a, b)
 */
template<class... TT, class... UU>
auto operator^(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return SymmetricDifference(a, b);
}

/**
 * @brief 对称差集运算符（ValueList ^ 单值）
 * 
 * @return SymmetricDifference(a, b)
 */
template<class... TT, class U>
auto operator^(const ValueList<TT...> &a, const U &b)
{
    return SymmetricDifference(a, b);
}

/**
 * @brief 对称差集运算符（单值 ^ ValueList）
 * 
 * @return SymmetricDifference(a, b)
 */
template<class T, class... UU>
auto operator^(const T &a, const ValueList<UU...> &b)
{
    return SymmetricDifference(a, b);
}

// ============================================================================
// Dup - 复制元素
// ============================================================================

/**
 * @brief 复制实现函数
 * 
 * 使用索引序列将每个元素复制 N 次
 * 
 * @tparam IDX 索引序列
 * @tparam T 元素类型
 * @param v 输入的 ValueList
 * @param 索引序列（编译期参数）
 * @return 复制后的 ValueList
 */
template<size_t... IDX, class T>
auto DupImpl(const ValueList<T> &v, std::index_sequence<IDX...>)
{
    return Extract<(IDX * 0)...>(v);
}

/**
 * @brief 将 ValueList 中的每个元素复制 N 次
 * 
 * @tparam N 复制次数
 * @tparam T 元素类型
 * @param v 输入的 ValueList
 * @return 复制后的 ValueList
 * 
 * 使用示例：
 * @code
 * auto result = Dup<3>(ValueList{1});
 * // 结果: {(1, 1, 1)}
 * @endcode
 */
template<int N, class T>
auto Dup(const ValueList<T> &v)
{
    return DupImpl(v, std::make_index_sequence<N>());
}

} // namespace vtest
