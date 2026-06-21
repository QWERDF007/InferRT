#include <inferrt/model/ModelFeatures.hpp>

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace irt::model {

namespace {

// ============================================================================
// 模型族特征名称定义
// ============================================================================

// 判断 model_key 是否以 prefix 开头（大小写不敏感）
bool keyStartsWith(const std::string &model_key, const char *prefix)
{
    const auto len = std::strlen(prefix);
    if (model_key.size() < len)
    {
        return false;
    }
    for (size_t i = 0; i < len; ++i)
    {
        if (std::tolower(static_cast<unsigned char>(model_key[i]))
            != std::tolower(static_cast<unsigned char>(prefix[i])))
        {
            return false;
        }
    }
    return true;
}

// 判断 model_key 是否含有关键词
bool keyContains(const std::string &model_key, const char *word)
{
    auto lower = model_key;
    for (auto &c : lower)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower.find(word) != std::string::npos;
}

// -- ResNet 族 ---------------------------------------------------------------

std::vector<std::string> resnetFeatures()
{
    return {
        "layer1", "layer2", "layer3", "layer4", "avgpool",
    };
}

// -- MobileNet ---------------------------------------------------------------

std::vector<std::string> mobilenetV2Features()
{
    return {
        "features.5",
        "features.11",
        "features.16",
    };
}

std::vector<std::string> mobilenetV3Features()
{
    return {
        "features.5",
        "features.11",
        "features.16",
    };
}

// -- ViT 族 ------------------------------------------------------------------

std::vector<std::string> vitFeatures()
{
    return {"norm", "cls"};
}

// -- DINO 族 -----------------------------------------------------------------

std::vector<std::string> dinoV2Features()
{
    return {
        "x_norm_clstoken",
        "x_norm_regtokens",
        "x_norm_patchtokens",
        "x_prenorm",
    };
}

std::vector<std::string> dinoV3Features()
{
    return {
        "x_prenorm",
        "x_norm_clstoken",
        "x_storage_tokens",
        "x_norm_patchtokens",
    };
}

} // namespace

// ============================================================================
// 公共接口
// ============================================================================

std::vector<std::string> modelFeatureNames(const std::string &model_key)
{
    // 按模型族名称前缀匹配

    if (keyStartsWith(model_key, "resnet") || keyStartsWith(model_key, "wide_resnet"))
    {
        return resnetFeatures();
    }
    if (keyStartsWith(model_key, "mobilenet_v2") || keyContains(model_key, "mobilenetv2"))
    {
        return mobilenetV2Features();
    }
    if (keyStartsWith(model_key, "mobilenet_v3") || keyContains(model_key, "mobilenetv3"))
    {
        return mobilenetV3Features();
    }

    // ViT timm 标准
    if (keyStartsWith(model_key, "vit_") || keyStartsWith(model_key, "vit"))
    {
        // 过滤掉 "vit" → 即 ViTBasePatch16_224 的别名
        if (keyContains(model_key, "tiny") || keyContains(model_key, "small") || keyContains(model_key, "base")
            || keyContains(model_key, "large") || keyContains(model_key, "huge") || keyContains(model_key, "giant")
            || keyContains(model_key, "gigantic"))
        {
            return vitFeatures();
        }
    }
    // "vit" 特指 ViTBasePatch16_224 别名
    if (model_key == "vit")
    {
        return vitFeatures();
    }

    // DINO v2
    if (keyStartsWith(model_key, "dinov2_") || keyStartsWith(model_key, "timmdinov2_"))
    {
        return dinoV2Features();
    }
    // DINO v3
    if (keyStartsWith(model_key, "dinov3_") || keyStartsWith(model_key, "timmdinov3_"))
    {
        return dinoV3Features();
    }

    return {};
}

} // namespace irt::model
