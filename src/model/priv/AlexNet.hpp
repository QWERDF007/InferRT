#pragma once

#include <inferrt/model/IModel.hpp>

namespace irt::model {

class AlexNet : public IModel
{
public:
    explicit AlexNet() = default;
    ~AlexNet()         = default;

    void build();
};

} // namespace irt::model