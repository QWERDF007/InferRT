#pragma once

namespace irt::cvcuda::priv {

// 使用 type trait 映射
template<typename CT>
struct make_vector2
{
    using type = float2; // 默认
};

template<>
struct make_vector2<unsigned char>
{
    using type = uchar2;
};

template<>
struct make_vector2<int>
{
    using type = int2;
};

template<>
struct make_vector2<double>
{
    using type = double2;
};

template<typename CT>
using make_vector2_t = typename make_vector2<CT>::type;

template<typename CT>
struct make_vector4
{
    using type = float4; // 默认
};

template<>
struct make_vector4<unsigned char>
{
    using type = uchar4;
};

template<>
struct make_vector4<int>
{
    using type = int4;
};

template<>
struct make_vector4<double>
{
    using type = double4;
};

template<typename CT>
using make_vector4_t = typename make_vector4<CT>::type;

} // namespace irt::cvcuda::priv