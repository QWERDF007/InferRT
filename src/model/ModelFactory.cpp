#include <inferrt/core/Exception.hpp>
#include <inferrt/model/ModelFactory.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace irt::model {

namespace {

struct ModelRegistration
{
    ModelCreator creator;
    PatchTokenMetadata patch_tokens;
};

using ModelRegistry = std::map<std::string, ModelRegistration>;

struct RegistryHolder
{
    std::shared_mutex mutex;
    ModelRegistry     registry;
};

RegistryHolder &GetRegistryHolder()
{
    static RegistryHolder holder;
    return holder;
}

std::string normalizeModelName(const std::string &name)
{
    const auto first = name.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
    {
        return "";
    }
    const auto last = name.find_last_not_of(" \t\n\r");
    std::string trimmed = name.substr(first, last - first + 1);
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return trimmed;
}

} // namespace

ModelRegistrar::ModelRegistrar(const std::string &name, ModelCreator creator, PatchTokenMetadata patch_tokens)
{
    RegisterModel(name, creator, patch_tokens);
}

bool RegisterModel(const std::string &name, ModelCreator creator, PatchTokenMetadata patch_tokens)
{
    if (name.empty() || creator == nullptr)
    {
        return false;
    }

    const std::string normalized = normalizeModelName(name);
    if (normalized.empty())
    {
        return false;
    }

    auto &holder = GetRegistryHolder();
    std::unique_lock<std::shared_mutex> lock(holder.mutex);
    return holder.registry.emplace(normalized, ModelRegistration{creator, patch_tokens}).second;
}

bool isSupportedModel(const std::string &name)
{
    if (name.empty())
    {
        return false;
    }

    const std::string normalized = normalizeModelName(name);
    auto &holder = GetRegistryHolder();
    std::shared_lock<std::shared_mutex> lock(holder.mutex);
    return holder.registry.find(normalized) != holder.registry.end();
}

std::vector<std::string> getRegisteredModelNames()
{
    auto &holder = GetRegistryHolder();
    std::shared_lock<std::shared_mutex> lock(holder.mutex);

    std::vector<std::string> names;
    names.reserve(holder.registry.size());
    for (const auto &[name, creator] : holder.registry)
    {
        (void)creator;
        names.push_back(name);
    }
    return names;
}

PatchTokenMetadata describePatchTokens(const std::string &name)
{
    const std::string normalized = normalizeModelName(name);
    auto &holder = GetRegistryHolder();
    std::shared_lock<std::shared_mutex> lock(holder.mutex);
    const auto it = holder.registry.find(normalized);
    if (it == holder.registry.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown backbone model: %s", name.c_str());
    }
    const auto metadata = it->second.patch_tokens;
    if (metadata.patch_size <= 0 || metadata.channels <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Model '%s' does not declare patch-token metadata", name.c_str());
    }
    return metadata;
}

std::unique_ptr<IModel> CreateModel(const std::string &name, std::unique_ptr<IModelConfig> config)
{
    if (name.empty())
    {
        return nullptr;
    }

    const std::string normalized = normalizeModelName(name);
    ModelCreator creator = nullptr;

    {
        auto &holder = GetRegistryHolder();
        std::shared_lock<std::shared_mutex> lock(holder.mutex);
        const auto it = holder.registry.find(normalized);
        if (it == holder.registry.end())
        {
            return nullptr;
        }
        creator = it->second.creator;
    }

    if (creator == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Model creator for '%s' is null", name.c_str());
    }

    auto model = creator();
    if (model == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Model creator for '%s' returned nullptr", name.c_str());
    }

    if (config)
    {
        model->setModelConfig(std::move(config));
    }
    return model;
}

} // namespace irt::model
