/**
 * @file ValueTests.hpp
 * @brief 参数化测试的辅助工具
 * 
 * 该文件提供了用于 Google Test 参数化测试的辅助类和宏。
 * 主要功能包括：
 * - 命名参数类型（Param）
 * - 测试参数哈希生成（用于生成唯一的测试名称）
 * - 测试套件宏定义
 */

#pragma once

#include "HashMD5.hpp"
#include "ValueList.hpp"

namespace vtest {

/**
 * @brief 字符串字面量包装类
 * 
 * 用于在编译期存储字符串字面量，可以作为模板参数使用。
 * 这是 C++20 之前实现字符串作为非类型模板参数的一种方法。
 * 
 * @tparam N 字符串长度（包括结尾的 '\0'）
 */
template<size_t N>
struct StringLiteral
{
    /**
     * @brief 从字符数组构造
     * 
     * @param str 字符数组字面量
     */
    constexpr StringLiteral(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }

    char value[N]; ///< 存储的字符串内容

    /**
     * @brief 输出流运算符重载
     * 
     * 允许将 StringLiteral 直接输出到流
     */
    friend std::ostream &operator<<(std::ostream &out, const StringLiteral &p)
    {
        return out << p.value;
    };
};

/**
 * @brief 命名测试参数类
 * 
 * 用于定义带有名称的测试参数。在参数化测试中，可以为每个参数指定一个名称，
 * 使测试输出更加清晰易读。
 * 
 * 特性：
 * - 支持指定默认值（可选）
 * - 如果类型不可默认构造，必须提供默认值才能使用 ValueDefault()
 * - 自动生成友好的输出格式
 * 
 * @tparam NAME 参数名称（字符串字面量）
 * @tparam T 参数值的类型
 * @tparam DEFAULT 可选的默认值（最多一个）
 * 
 * 使用示例：
 * @code
 * using Width = Param<"width", int, 640>;
 * using Height = Param<"height", int, 480>;
 * 
 * auto params = ValueList{
 *     std::make_tuple(Width{1920}, Height{1080}),
 *     std::make_tuple(Width{1280}, Height{720})
 * };
 * @endcode
 */
template<StringLiteral NAME, class T, T... DEFAULT>
class Param
{
    static_assert(sizeof...(DEFAULT) <= 1);

public:
    /**
     * @brief 默认构造函数（有默认值版本）
     * 
     * 当提供了默认值时，使用该默认值初始化
     */
    template<class U = void *, std::enable_if_t<sizeof(U) * 0 + sizeof...(DEFAULT) == 1, int> = 0>
    constexpr Param()
        : m_value(DEFAULT...)
    {
    }

    /**
     * @brief 默认构造函数（类型可默认构造版本）
     * 
     * 当类型可默认构造且没有提供默认值时，使用类型的默认构造函数
     */
    template<class U = void *,
             std::enable_if_t<std::is_default_constructible_v<T> && sizeof(U) * 0 + sizeof...(DEFAULT) == 0, int> = 0>
    constexpr Param()
        : m_value(T{})
    {
    }

    /**
     * @brief 从值构造
     * 
     * @param value 参数值
     */
    constexpr Param(T value)
        : m_value(value)
    {
    }

    /**
     * @brief 隐式转换为底层类型
     * 
     * 允许 Param 对象在需要时自动转换为其包装的值类型
     */
    constexpr operator T() const
    {
        return m_value;
    }

    /**
     * @brief 输出流运算符重载
     * 
     * 输出格式：NAME(value)
     * 例如：width(1920)
     */
    friend std::ostream &operator<<(std::ostream &out, Param p)
    {
        out << NAME << std::boolalpha;
        out << '(' << p.m_value << ')';
        out << std::noboolalpha;
        return out;
    };

    /**
     * @brief 相等比较运算符
     */
    constexpr bool operator==(const Param &that) const
    {
        return m_value == that.m_value;
    }

    /**
     * @brief 不等比较运算符
     */
    constexpr bool operator!=(const Param &that) const
    {
        return !(*this == that);
    }

    /**
     * @brief 小于比较运算符
     * 
     * 用于排序和集合操作
     */
    constexpr bool operator<(const Param &that) const
    {
        return m_value < that.m_value;
    }

private:
    T m_value; ///< 存储的参数值
};

/**
 * @brief 更新 Param 类型的哈希值
 * 
 * 将 Param 对象的值添加到 MD5 哈希计算中
 * 
 * @tparam NAME 参数名称
 * @tparam T 参数类型
 * @tparam DEFAULT 默认值
 * @param hash MD5 哈希对象
 * @param p Param 对象
 */
template<StringLiteral NAME, class T, T... DEFAULT>
void Update(vtest::HashMD5 &hash, const Param<NAME, T, DEFAULT...> &p)
{
    Update(hash, static_cast<T>(p));
}

namespace detail {

/**
 * @brief 生成测试参数的哈希字符串（辅助函数）
 * 
 * 使用 MD5 哈希算法为测试参数生成唯一的标识符。
 * 这个标识符用于生成测试用例的后缀名称。
 * 
 * 算法：
 * 1. 计算参数的 MD5 哈希（128 位）
 * 2. 将 128 位哈希折叠为 32 位（通过异或操作）
 * 3. 转换为十六进制字符串
 * 
 * 使用 32 位而不是 64 位的原因：
 * - 32 位足以提供足够的唯一性
 * - 生成的测试名称后缀更短，更易读
 * 
 * @tparam P 参数类型
 * @param info 测试参数
 * @return 8 位十六进制字符串（32 位哈希值）
 */
template<class P>
std::string GetTestParamHashHelper(const P &info)
{
    // 使用参数集的哈希作为索引
    vtest::HashMD5 hash;
    Update(hash, info);

    // 不需要 64 位的变化范围，32 位足够且能生成更短的后缀

    union Cast
    {
        uint8_t  array[16]; ///< 128 位哈希的字节数组表示
        uint64_t value[2];  ///< 128 位哈希的两个 64 位整数表示
    };

    static_assert(sizeof(hash.getHashAndReset()) == sizeof(Cast::array));

    Cast caster;
    memcpy(caster.array, &hash.getHashAndReset()[0], sizeof(caster.array));

    // 将 128 位折叠为 32 位
    uint64_t code64 = caster.value[0] ^ caster.value[1];
    uint32_t code32 = (code64 & UINT32_MAX) ^ (code64 >> 32);

    // 转换为十六进制字符串
    std::ostringstream out;
    out << std::hex << std::setw(sizeof(code32) * 2) << std::setfill('0') << code32;
    return out.str();
}

} // namespace detail

/**
 * @brief 获取测试参数的哈希字符串
 * 
 * 为 Google Test 的参数化测试生成唯一的测试名称后缀
 * 
 * @tparam P 参数类型
 * @param info 测试参数接口
 * @return 哈希字符串
 */
template<class P>
std::string GetTestParamHash(const ::testing::WithParamInterface<P> &info)
{
    return detail::GetTestParamHashHelper(info.GetParam());
}

/**
 * @brief 测试后缀打印器
 * 
 * 用于 Google Test 参数化测试的自定义名称生成器。
 * 
 * 为什么不使用 Google Test 的默认后缀生成器（递增数字）：
 * - 默认的数字后缀与测试参数没有绑定关系
 * - 如果某些平台不支持特定的测试参数，数字会指向不同的参数
 * - 我们需要整个测试名称与相同的测试实例关联，无论在什么平台上
 * 
 * 使用哈希的优点：
 * - 测试名称与参数内容绑定
 * - 跨平台一致性
 * - 易于追踪和比较测试结果
 */
struct TestSuffixPrinter
{
    /**
     * @brief 生成测试参数的后缀名称
     * 
     * @tparam P 参数类型
     * @param info 测试参数信息
     * @return 基于参数哈希的后缀字符串
     */
    template<class P>
    std::string operator()(const ::testing::TestParamInfo<P> &info) const
    {
        return detail::GetTestParamHashHelper(info.param);
    }
};

} // namespace vtest

/**
 * @brief 实例化参数化测试套件的内部宏
 * 
 * 这是一个内部宏，用于实例化 Google Test 的参数化测试套件。
 * 它会对参数列表进行排序和去重，并使用自定义的测试名称生成器。
 * 
 * @param GROUP 测试组名称（通常使用 _ 作为默认组）
 * @param TEST 测试类名称
 * @param ... 测试参数列表（ValueList）
 * 
 * 工作流程：
 * 1. 规范化 ValueList 类型
 * 2. 对参数列表进行排序和去重（UniqueSort）
 * 3. 使用 ValuesIn 将参数传递给 Google Test
 * 4. 使用 TestSuffixPrinter 生成基于哈希的测试名称后缀
 */
#define _INSTANTIATE_TEST_SUITE_P(                                                                                   \
    GROUP, TEST,                                                                                                     \
    ::testing::ValuesIn(                                                                                             \
        UniqueSort(typename ::vtest::detail::NormalizeValueList<::vtest::ValueList<typename TEST::ParamType>>::type( \
            __VA_ARGS__))),                                                                                          \
    ::vtest::TestSuffixPrinter())

/**
 * @brief 定义参数化测试套件的宏
 * 
 * 这是用户使用的主要宏，用于定义参数化测试套件。
 * 它会创建一个测试类，并自动实例化测试套件。
 * 
 * @param TEST 测试类名称
 * @param ... 测试参数列表（ValueList）
 * 
 * 功能：
 * 1. 创建全局参数列表变量 g_<TEST>_Params，并对参数进行排序去重
 * 2. 定义测试类，继承自 ::testing::TestWithParam
 * 3. 提供 GetParamValue<I>() 辅助方法，用于从 tuple 参数中提取指定索引的值
 * 4. 自动实例化测试套件
 * 
 * 使用示例：
 * @code
 * _TEST_SUITE_P(MyTest, 
 *     ValueList{1, 2, 3} * ValueList{"a", "b"}
 * );
 * 
 * TEST_P(MyTest, TestCase) {
 *     int value = GetParamValue<0>();      // 获取第一个参数
 *     const char* str = GetParamValue<1>(); // 获取第二个参数
 *     // 测试逻辑...
 * }
 * @endcode
 */
#define _TEST_SUITE_P(TEST, ...)                                                          \
    static ::vtest::ValueList g_##TEST##_Params = ::vtest::UniqueSort(__VA_ARGS__);       \
    class TEST : public ::testing::TestWithParam<decltype(g_##TEST##_Params)::value_type> \
    {                                                                                     \
    protected:                                                                            \
        template<int I>                                                                   \
        auto GetParamValue() const                                                        \
        {                                                                                 \
            return std::get<I>(GetParam());                                               \
        }                                                                                 \
    };                                                                                    \
    _INSTANTIATE_TEST_SUITE_P(_, TEST, g_##TEST##_Params)

// clang-format off
// clang-format on
