#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace irt::test {

struct ClusterTestData
{
    std::vector<float>   samples;
    std::vector<int64_t> expected_labels;
    int64_t              num_samples{0};
    int64_t              num_features{3};
};

inline std::filesystem::path findRepoRoot(const char *source_file)
{
    std::filesystem::path path = std::filesystem::absolute(std::filesystem::path(source_file));
    while (!path.empty())
    {
        if (std::filesystem::exists(path / "assets" / "dbscan_testdata.txt"))
        {
            return path;
        }
        path = path.parent_path();
    }
    return {};
}

inline ClusterTestData loadClusterTestData(const char *source_file)
{
    const auto    data_path = findRepoRoot(source_file) / "assets" / "dbscan_testdata.txt";
    std::ifstream input(data_path);
    if (!input)
    {
        throw std::runtime_error("Failed to open " + data_path.string());
    }

    ClusterTestData data;
    std::string     line;
    std::getline(input, line);
    data.num_samples = std::stoll(line);
    data.samples.reserve(static_cast<size_t>(data.num_samples * data.num_features));
    data.expected_labels.reserve(static_cast<size_t>(data.num_samples));

    while (std::getline(input, line))
    {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        float              x     = 0.0f;
        float              y     = 0.0f;
        float              z     = 0.0f;
        int64_t            label = 0;
        stream >> x >> y >> z >> label;
        data.samples.insert(data.samples.end(), {x, y, z});
        data.expected_labels.push_back(label);
    }
    return data;
}

inline void expectSamePartition(const std::vector<int64_t> &actual, const std::vector<int64_t> &expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t lhs = 0; lhs < actual.size(); ++lhs)
    {
        for (size_t rhs = 0; rhs < actual.size(); ++rhs)
        {
            EXPECT_EQ(actual[lhs] == actual[rhs], expected[lhs] == expected[rhs]) << "lhs=" << lhs << " rhs=" << rhs;
        }
    }
}

inline int64_t countClusters(const std::vector<int64_t> &labels)
{
    std::vector<int64_t> clusters;
    for (const int64_t label : labels)
    {
        if (label >= 0 && std::find(clusters.begin(), clusters.end(), label) == clusters.end())
        {
            clusters.push_back(label);
        }
    }
    return static_cast<int64_t>(clusters.size());
}

} // namespace irt::test
