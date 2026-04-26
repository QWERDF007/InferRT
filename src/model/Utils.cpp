#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <fstream>

namespace irt::model {

WeightsMap loadWeights(const std::string &file)
{
    WeightsMap weights_map;

    std::ifstream input(file);
    if (!input.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open weights file: %s", file.c_str());
    }

    int32_t count;
    input >> count;
    if (count <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to read valid count of weights from file: %s",
                             file.c_str());
    }

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> wt.count;

        // Load blob
        auto *val = new uint32_t[wt.count];
        input >> std::hex;
        for (auto x = 0ll; x < wt.count; ++x)
        {
            input >> val[x];
        }
        wt.values         = val;
        weights_map[name] = wt;
    }

    return weights_map;
}

std::vector<std::string> readImagenetLabels(const std::string &label_file)
{
    std::vector<std::string> labels(1000);
    std::ifstream            file(label_file);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open weights file: %s", label_file.c_str());
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t colon_pos = line.find(": ");
        if (colon_pos != std::string::npos)
        {
            int         idx   = std::stoi(line.substr(0, colon_pos));
            std::string label = line.substr(colon_pos + 2);
            if (label.size() >= 2 && label.front() == '\'' && label.back() == ',')
            {
                label = label.substr(1, label.size() - 3);
            }
            if (idx >= 0 && idx < 1000)
            {
                labels[idx] = label;
            }
        }
    }

    return labels;
}

} // namespace irt::model