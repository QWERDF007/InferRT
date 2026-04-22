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

} // namespace irt::model