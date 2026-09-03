#include <inferrt/core/Version.h>

#include <iostream>
#include <string_view>

#if defined(INFERRT_COMPONENT_core)
#    include <inferrt/core/Tensor.hpp>
#elif defined(INFERRT_COMPONENT_util)
#    include <inferrt/util/Device.hpp>
#elif defined(INFERRT_COMPONENT_ops)
#    include <inferrt/ops/DBSCAN.hpp>
#elif defined(INFERRT_COMPONENT_cvcuda)
#    include <inferrt/cvcuda/OpNMS.hpp>
#elif defined(INFERRT_COMPONENT_model)
#    include <inferrt/model/ModelFactory.hpp>
#elif defined(INFERRT_COMPONENT_engine)
#    include <inferrt/engine/EngineConfig.hpp>
#elif defined(INFERRT_COMPONENT_features)
#    include <inferrt/features/ImageSearch.hpp>
#else
#    error "Unsupported InferRT component consumer"
#endif

int main()
{
    std::cout << "InferRT component " << INFERRT_COMPONENT_NAME << " " << irt::GetVersionString() << '\n';

#if defined(INFERRT_COMPONENT_core)
    irt::Shape shape{1, 3, 2, 2};
    return shape.elementCount() == 12U ? 0 : 1;
#elif defined(INFERRT_COMPONENT_util)
    return irt::util::getCPUDeviceName().empty() ? 1 : 0;
#elif defined(INFERRT_COMPONENT_ops)
    const float samples[] = {0.0F, 0.0F, 0.1F, 0.1F};
    irt::ops::DBSCANConfig config;
    config.eps         = 1.0F;
    config.min_samples = 1;
    const auto result  = irt::ops::dbscan(samples, 2, 2, config);
    return result.labels.size() == 2U ? 0 : 1;
#elif defined(INFERRT_COMPONENT_cvcuda)
    irt::cvcuda::NMS op;
    return op.handle() == nullptr ? 1 : 0;
#elif defined(INFERRT_COMPONENT_model)
    auto model = irt::model::CreateModel("resnet18");
    return model && model->isValid() ? 0 : 1;
#elif defined(INFERRT_COMPONENT_engine)
    irt::engine::EngineConfig config;
    config.validate();
    return 0;
#elif defined(INFERRT_COMPONENT_features)
    irt::features::ImageSearchConfig config;
    return config.model_name.empty() ? 1 : 0;
#endif
}
