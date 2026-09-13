#include "DinoSearchCache.hpp"

#include <mutex>

namespace irt::features::priv {

namespace {

class DinoSearchCacheRegistry final
{
public:
    std::shared_ptr<DinoSearchCaches> acquire(const std::string &extractor_signature,
                                              const uint64_t image_budget_bytes,
                                              const uint64_t feature_budget_bytes)
    {
        std::lock_guard lock(mutex_);
        if (active_ != nullptr && extractor_signature_ == extractor_signature
            && image_budget_bytes_ == image_budget_bytes && feature_budget_bytes_ == feature_budget_bytes)
        {
            return active_;
        }
        extractor_signature_  = extractor_signature;
        image_budget_bytes_   = image_budget_bytes;
        feature_budget_bytes_ = feature_budget_bytes;
        active_               = std::make_shared<DinoSearchCaches>(image_budget_bytes, feature_budget_bytes);
        return active_;
    }

private:
    std::mutex                        mutex_{};
    std::string                       extractor_signature_{};
    uint64_t                          image_budget_bytes_{0};
    uint64_t                          feature_budget_bytes_{0};
    std::shared_ptr<DinoSearchCaches> active_{};
};

DinoSearchCacheRegistry &registry()
{
    static DinoSearchCacheRegistry instance;
    return instance;
}

} // namespace

std::shared_ptr<DinoSearchCaches> dinoAcquireSearchCaches(const std::string &extractor_signature,
                                                          const uint64_t image_budget_bytes,
                                                          const uint64_t feature_budget_bytes)
{
    return registry().acquire(extractor_signature, image_budget_bytes, feature_budget_bytes);
}

} // namespace irt::features::priv
