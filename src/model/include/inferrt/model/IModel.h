#pragma once

#include "IModel.hpp"

#include <map>
#include <memory>
#include <string>

namespace irt::model {

using ModelCreator = std::unique_ptr<IModel> (*)();

class INFERRT_MODEL_API ModelRegistrar
{
public:
    ModelRegistrar(const std::string &name, ModelCreator creator);
};

INFERRT_MODEL_API bool RegisterModel(const std::string &name, ModelCreator creator);

INFERRT_MODEL_API std::unique_ptr<IModel> CreateModel(const std::string &name);

} // namespace irt::model

#define INFERRT_REGISTER_MODEL(MODEL_CLASS)                              \
    namespace {                                                          \
    std::unique_ptr<::irt::model::IModel> Create##MODEL_CLASS()          \
    {                                                                    \
        return std::make_unique<::irt::model::MODEL_CLASS>();            \
    }                                                                    \
    [[maybe_unused]] const ::irt::model::ModelRegistrar                  \
        registered_##MODEL_CLASS(::irt::model::MODEL_CLASS::key(),       \
                                 &Create##MODEL_CLASS);                  \
    }
