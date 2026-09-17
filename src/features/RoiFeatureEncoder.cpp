#include <inferrt/features/RoiFeatureEncoder.hpp>
#include "priv/RoiFeatureExtractor.hpp"
namespace irt::features {
class RoiFeatureEncoder::Impl {
public:
    Impl(const RoiFeatureConfig& config,const std::filesystem::path& weights):extractor(config,weights){}
    priv::RoiFeatureExtractor extractor;
};
RoiFeatureEncoder::RoiFeatureEncoder(const RoiFeatureConfig& config,const std::filesystem::path& weights)
    :impl_(std::make_unique<Impl>(config,weights)){}
RoiFeatureEncoder::~RoiFeatureEncoder()=default;
int RoiFeatureEncoder::featureDim()const{return impl_->extractor.featureDim();}
std::vector<float> RoiFeatureEncoder::extract(const std::vector<RoiFeatureItem>& items){return impl_->extractor.extractAll(items);}
}
