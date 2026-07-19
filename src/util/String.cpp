#include <inferrt/util/String.hpp>

#include <algorithm>
#include <cctype>
#include <utility>

namespace irt::util {

std::string trim(std::string value)
{
    const auto not_space = [](unsigned char ch)
    {
        return !std::isspace(ch);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::vector<std::string> split(std::string_view value, std::string_view delimiters)
{
    std::vector<std::string> result;
    size_t                   start = 0;
    while (start <= value.size())
    {
        const size_t end = delimiters.empty() ? std::string_view::npos : value.find_first_of(delimiters, start);
        auto token = trim(std::string(value.substr(start, end == std::string_view::npos ? std::string_view::npos
                                                                     : end - start)));
        if (!token.empty())
        {
            result.push_back(std::move(token));
        }
        if (end == std::string_view::npos)
        {
            break;
        }
        start = end + 1;
    }
    return result;
}

std::string sanitizeFileStem(std::string_view value, std::string_view fallback)
{
    std::string stem;
    stem.reserve(value.size());
    for (const char character : value)
    {
        const auto ch = static_cast<unsigned char>(character);
        stem.push_back(std::isalnum(ch) ? character : '_');
    }
    return stem.empty() ? std::string(fallback) : stem;
}

} // namespace irt::util
