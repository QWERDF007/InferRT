#include "LingBotVision.hpp"

#include <inferrt/model/ModelFactory.h>

namespace irt::model {
namespace {

#define INFERRT_LINGBOT_ALIAS_CLASS(CLASS_NAME, BASE_CLASS, KEY_LITERAL) \
    class CLASS_NAME : public BASE_CLASS                                  \
    {                                                                     \
    public:                                                               \
        static const char *key() noexcept                                 \
        {                                                                 \
            return KEY_LITERAL;                                           \
        }                                                                 \
    }

INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTS16NamedAlias, LingBotVisionViTS16, "lingbot_vision_vit_small");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTB16NamedAlias, LingBotVisionViTB16, "lingbot_vision_vit_base");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTL16NamedAlias, LingBotVisionViTL16, "lingbot_vision_vit_large");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTG16NamedAlias, LingBotVisionViTG16, "lingbot_vision_vit_giant");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTS16HyphenAlias, LingBotVisionViTS16, "lingbot-vision-vit-small");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTB16HyphenAlias, LingBotVisionViTB16, "lingbot-vision-vit-base");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTL16HyphenAlias, LingBotVisionViTL16, "lingbot-vision-vit-large");
INFERRT_LINGBOT_ALIAS_CLASS(LingBotVisionViTG16HyphenAlias, LingBotVisionViTG16, "lingbot-vision-vit-giant");

} // namespace
} // namespace irt::model

INFERRT_REGISTER_MODEL(LingBotVisionViTS16)
INFERRT_REGISTER_MODEL(LingBotVisionViTB16)
INFERRT_REGISTER_MODEL(LingBotVisionViTL16)
INFERRT_REGISTER_MODEL(LingBotVisionViTG16)
INFERRT_REGISTER_MODEL(LingBotVisionViTS16NamedAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTB16NamedAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTL16NamedAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTG16NamedAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTS16HyphenAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTB16HyphenAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTL16HyphenAlias)
INFERRT_REGISTER_MODEL(LingBotVisionViTG16HyphenAlias)
