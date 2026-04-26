#pragma once

#include "ModelFactory.hpp"

#define INFERRT_REGISTER_MODEL(MODEL_CLASS)                                                                        \
    namespace {                                                                                                    \
    std::unique_ptr<::irt::model::priv::IModelImpl> Create##MODEL_CLASS()                                          \
    {                                                                                                              \
        return std::make_unique<::irt::model::MODEL_CLASS>();                                                      \
    }                                                                                                              \
    [[maybe_unused]] const ::irt::model::ModelRegistrar registered_##MODEL_CLASS(::irt::model::MODEL_CLASS::key(), \
                                                                                 &Create##MODEL_CLASS);            \
    }
