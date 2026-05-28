"""分类样例 ``model_zoo`` 工具函数的单元测试。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from model_zoo import (  # noqa: E402
    convert_transformers_dinov3_state_dict,
    create_model,
    export_model_state_dict,
    list_supported_models,
    normalize_image_size,
    parse_image_size,
    preprocess,
    resolve_input_size,
)


class DummyPatchEmbedModel:
    """模拟 timm ViT，通过 ``patch_embed.img_size`` 暴露真实输入尺寸。"""

    patch_embed = SimpleNamespace(img_size=(384, 384))
    default_cfg = {"input_size": (3, 224, 224)}


class DummyConfigModel:
    """模拟仅通过配置暴露输入尺寸的 timm/torchvision 模型。"""

    pretrained_cfg = {"input_size": (3, 320, 256)}


def test_resolve_input_size_prefers_patch_embed_size() -> None:
    """ViT 的实际 patch 输入尺寸应优先于默认配置。"""

    assert resolve_input_size(DummyPatchEmbedModel()) == (384, 384)


def test_resolve_input_size_reads_model_config() -> None:
    """当模型没有 ``patch_embed`` 时，应从配置中的 ``input_size`` 解析 H/W。"""

    assert resolve_input_size(DummyConfigModel()) == (320, 256)


def test_resolve_input_size_falls_back_to_default() -> None:
    """模型未提供尺寸元信息时应回退到 ImageNet 默认 224。"""

    assert resolve_input_size(object()) == (224, 224)


def test_normalize_image_size_rejects_invalid_shape() -> None:
    """非法尺寸应尽早报错，避免 OpenCV resize 阶段出现难读异常。"""

    with pytest.raises(ValueError):
        normalize_image_size((1, 3, 224, 224, 1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("384", (384, 384)),
        ("384x256", (384, 256)),
        ("3x320x256", (320, 256)),
        ("1,3,640,480", (640, 480)),
    ],
)
def test_parse_image_size_accepts_cli_formats(raw: str, expected: tuple[int, int]) -> None:
    """命令行尺寸解析应兼容常见 H/W、CHW 和 NCHW 写法。"""

    assert parse_image_size(raw) == expected


def test_parse_image_size_rejects_invalid_text() -> None:
    """非数字尺寸字符串应在进入预处理前报错。"""

    with pytest.raises(ValueError):
        parse_image_size("abc")


def test_preprocess_uses_custom_hw_size() -> None:
    """预处理应按调用方传入的 H/W 输出 NCHW 张量。"""

    image = np.zeros((12, 20, 3), dtype=np.uint8)
    tensor = preprocess(image, image_size=(384, 256))

    assert tuple(tensor.shape) == (1, 3, 384, 256)
    assert str(tensor.dtype) == "torch.float32"


def test_list_supported_timm_models_queries_dino_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    """timm 模型枚举应显式覆盖 DINOv2/DINOv3，避免只依赖 ``vit*`` 前缀。"""

    calls: list[str] = []

    def fake_list_models(pattern: str) -> list[str]:
        """记录 timm 查询模式并返回伪造模型名。

        Args:
            pattern: ``model_zoo`` 传入的 timm 通配符。

        Returns:
            包含该通配符的伪造模型列表。
        """

        calls.append(pattern)
        return [f"{pattern}_model"]

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(list_models=fake_list_models))

    names = list_supported_models("timm")

    assert "*dinov2*" in calls
    assert "*dinov3*" in calls
    assert len(names) == len(dict.fromkeys(names))


def test_list_supported_torchhub_models_includes_official_dinov2_keys() -> None:
    """torchhub 后端应列出官方 DINOv2 backbone key，避免依赖网络枚举。"""

    names = list_supported_models("torchhub")

    assert "dinov2_vits14" in names
    assert "dinov2_vits14_reg" in names
    assert "dinov2_vitg14_reg" in names
    assert "dinov3_vitb16" not in names


def test_list_supported_transformers_models_includes_official_dinov3_keys() -> None:
    """transformers 后端应列出 Hugging Face DINOv3 backbone key。"""

    names = list_supported_models("transformers")

    assert "dinov3_vits16" in names
    assert "dinov3_vitb16" in names
    assert "dinov3_vit7b16" in names


def test_create_timm_dinov3_uses_cls_token_pooling(monkeypatch: pytest.MonkeyPatch) -> None:
    """导出 DINOv3 timm 权重时应使用 CLS-token pooling，与官方 DINOv3 前向保持一致。"""

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, **kwargs: object) -> object:
        """捕获 timm.create_model 调用参数。

        Args:
            model_name: 传入 timm 的模型名。
            **kwargs: 创建模型时传入的关键字参数。

        Returns:
            用于占位的模型对象。
        """

        captured["model_name"] = model_name
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=fake_create_model))

    create_model("vit_base_patch16_dinov3", "timm")

    assert captured["model_name"] == "vit_base_patch16_dinov3"
    assert captured["pretrained"] is True
    assert captured["global_pool"] == "token"


def test_create_torchhub_dinov2_uses_official_repo_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """torchhub 后端默认应等价于加载 ``facebookresearch/dinov2`` 官方仓库。"""

    captured: dict[str, object] = {}

    def fake_hub_load(repo_or_dir: str, model: str, **kwargs: object) -> object:
        """捕获 torch.hub.load 的仓库、模型名和参数。

        Args:
            repo_or_dir: torchhub 仓库或本地目录。
            model: torchhub 模型 key。
            **kwargs: 额外加载参数。

        Returns:
            用于占位的模型对象。
        """

        captured["repo_or_dir"] = repo_or_dir
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("torch.hub.load", fake_hub_load)

    create_model("dinov2_vits14", "torchhub")

    assert captured["repo_or_dir"] == "facebookresearch/dinov2"
    assert captured["model"] == "dinov2_vits14"
    assert captured["source"] == "github"
    assert captured["pretrained"] is True
    assert captured["trust_repo"] is True
    assert captured["skip_validation"] is True


def test_create_transformers_dinov3_uses_huggingface_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """DINOv3 transformers 后端应通过 Hugging Face pipeline 创建。"""

    captured: dict[str, object] = {}

    class FakeHFModel(torch.nn.Module):
        """模拟 Hugging Face DINOv3 模型对象。"""

        config = SimpleNamespace(
            image_size=224,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=0,
            num_register_tokens=4,
            rope_theta=100.0,
        )

        def forward(self, pixel_values: torch.Tensor) -> object:
            """返回包含 pooler 和 token 输出的伪造前向结果。

            Args:
                pixel_values: 输入图像张量。

            Returns:
                带 ``pooler_output`` 和 ``last_hidden_state`` 的对象。
            """

            return SimpleNamespace(
                pooler_output=torch.zeros((pixel_values.shape[0], 768)),
                last_hidden_state=torch.zeros((pixel_values.shape[0], 1 + 4 + 196, 768)),
            )

    class FakePipeline:
        """模拟 transformers pipeline 返回对象。"""

        model = FakeHFModel()
        image_processor = SimpleNamespace(size={"height": 224, "width": 224})

    def fake_pipeline(**kwargs: object) -> object:
        """捕获 transformers.pipeline 调用参数。

        Args:
            **kwargs: pipeline 创建参数。

        Returns:
            伪造的 pipeline 对象。
        """

        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))

    model = create_model("dinov3_vitb16", "transformers")

    assert captured["model"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert captured["task"] == "image-feature-extraction"
    assert captured["local_files_only"] is False
    assert model.model_id == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert resolve_input_size(model) == (224, 224)


def test_create_torchhub_dinov2_supports_local_repo_and_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """torchhub 后端应支持本地 DINOv2 仓库和显式权重路径，便于离线导出。"""

    captured: dict[str, object] = {}

    def fake_hub_load(repo_or_dir: str, model: str, **kwargs: object) -> object:
        """捕获离线 torchhub 加载参数。

        Args:
            repo_or_dir: 本地 DINOv2 仓库路径。
            model: DINOv2 模型 key。
            **kwargs: torchhub 加载参数。

        Returns:
            用于占位的模型对象。
        """

        captured["repo_or_dir"] = repo_or_dir
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("torch.hub.load", fake_hub_load)

    local_repo = "offline_dinov2_repo"
    local_weights = "offline_dinov2_vits14_pretrain.pth"
    create_model(
        "dinov2_vits14",
        "torchhub",
        hub_repo=local_repo,
        hub_source="local",
        pretrained=True,
        hub_weights=local_weights,
    )

    assert captured["repo_or_dir"] == local_repo
    assert captured["model"] == "dinov2_vits14"
    assert captured["source"] == "local"
    assert captured["pretrained"] is True
    assert captured["weights"] == local_weights
    assert "trust_repo" not in captured
    assert "skip_validation" not in captured


def test_create_transformers_dinov3_supports_model_id_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """transformers 后端应允许显式覆盖 Hugging Face 模型 id 并限制只读本地缓存。"""

    captured: dict[str, object] = {}

    class FakeHFModel(torch.nn.Module):
        """模拟带配置的 Hugging Face DINOv3 模型。"""

        config = SimpleNamespace(
            image_size=224,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=0,
            num_register_tokens=4,
            rope_theta=100.0,
        )

    class FakePipeline:
        """模拟 transformers pipeline 返回对象。"""

        model = FakeHFModel()
        image_processor = SimpleNamespace(size={"height": 224, "width": 224})

    def fake_pipeline(**kwargs: object) -> object:
        """捕获 transformers.pipeline 的本地模型参数。

        Args:
            **kwargs: pipeline 创建参数。

        Returns:
            伪造的 pipeline 对象。
        """

        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))

    create_model(
        "dinov3_vitb16",
        "transformers",
        hf_model_id="local-dinov3-vitb16",
        local_files_only=True,
    )

    assert captured["model"] == "local-dinov3-vitb16"
    assert captured["task"] == "image-feature-extraction"
    assert captured["local_files_only"] is True


def _make_transformers_dinov3_state_dict(*, gated_mlp: bool = False) -> dict[str, torch.Tensor]:
    """构造最小 HF DINOv3 权重表，复用标准 MLP 与拆分 SwiGLU 转换测试数据。

    Args:
        gated_mlp: 为 true 时生成 DINOv3 Plus/7B 使用的 ``gate/up/down`` 三分支 MLP。

    Returns:
        可传入 ``convert_transformers_dinov3_state_dict`` 的伪 HF 权重表。
    """

    state_dict = {
        "embeddings.cls_token": torch.arange(4.0).reshape(1, 1, 4),
        "embeddings.register_tokens": torch.arange(16.0).reshape(1, 4, 4),
        "embeddings.patch_embeddings.weight": torch.zeros((4, 3, 16, 16)),
        "embeddings.patch_embeddings.bias": torch.zeros(4),
        "norm.weight": torch.ones(4),
        "norm.bias": torch.zeros(4),
        "model.layer.0.norm1.weight": torch.ones(4),
        "model.layer.0.norm1.bias": torch.zeros(4),
        "model.layer.0.norm2.weight": torch.ones(4),
        "model.layer.0.norm2.bias": torch.zeros(4),
        "model.layer.0.attention.q_proj.weight": torch.full((4, 4), 1.0),
        "model.layer.0.attention.k_proj.weight": torch.full((4, 4), 2.0),
        "model.layer.0.attention.v_proj.weight": torch.full((4, 4), 3.0),
        "model.layer.0.attention.q_proj.bias": torch.full((4,), 4.0),
        "model.layer.0.attention.v_proj.bias": torch.full((4,), 5.0),
        "model.layer.0.attention.o_proj.weight": torch.full((4, 4), 6.0),
        "model.layer.0.attention.o_proj.bias": torch.full((4,), 7.0),
        "model.layer.0.layer_scale1.lambda1": torch.full((4,), 8.0),
        "model.layer.0.layer_scale2.lambda1": torch.full((4,), 9.0),
    }
    if gated_mlp:
        state_dict.update(
            {
                "model.layer.0.mlp.gate_proj.weight": torch.full((8, 4), 14.0),
                "model.layer.0.mlp.gate_proj.bias": torch.full((8,), 15.0),
                "model.layer.0.mlp.up_proj.weight": torch.full((8, 4), 10.0),
                "model.layer.0.mlp.up_proj.bias": torch.full((8,), 11.0),
                "model.layer.0.mlp.down_proj.weight": torch.full((4, 8), 12.0),
                "model.layer.0.mlp.down_proj.bias": torch.full((4,), 13.0),
            }
        )
    else:
        state_dict.update(
            {
                "model.layer.0.mlp.up_proj.weight": torch.full((8, 4), 10.0),
                "model.layer.0.mlp.up_proj.bias": torch.full((8,), 11.0),
                "model.layer.0.mlp.down_proj.weight": torch.full((4, 8), 12.0),
                "model.layer.0.mlp.down_proj.bias": torch.full((4,), 13.0),
            }
        )
    return state_dict


def test_transformers_dinov3_export_state_dict_packs_hf_keys() -> None:
    """HF DINOv3 拆分权重应转换为 TensorRT DINO 构建器使用的 qkv/blocks 命名。"""

    config = SimpleNamespace(hidden_size=4, num_attention_heads=2, num_hidden_layers=1, rope_theta=100.0)
    state_dict = _make_transformers_dinov3_state_dict()

    converted = convert_transformers_dinov3_state_dict(state_dict, config)

    assert torch.equal(
        converted["blocks.0.attn.qkv.weight"],
        torch.cat(
            (
                state_dict["model.layer.0.attention.q_proj.weight"],
                state_dict["model.layer.0.attention.k_proj.weight"],
                state_dict["model.layer.0.attention.v_proj.weight"],
            ),
            dim=0,
        ),
    )
    assert torch.equal(
        converted["blocks.0.attn.qkv.bias"],
        torch.cat((torch.full((4,), 4.0), torch.zeros(4), torch.full((4,), 5.0)), dim=0),
    )
    assert converted["patch_embed.proj.weight"].shape == (4, 3, 16, 16)
    assert torch.equal(converted["blocks.0.mlp.fc1.weight"], state_dict["model.layer.0.mlp.up_proj.weight"])
    assert torch.equal(converted["blocks.0.ls1.gamma"], state_dict["model.layer.0.layer_scale1.lambda1"])


def test_transformers_dinov3_export_state_dict_accepts_unwrapped_layer_keys() -> None:
    """Transformers DINOv3 新版 state_dict 使用无 ``model.`` 前缀的 ``layer.*`` 键名。"""

    config = SimpleNamespace(hidden_size=4, num_attention_heads=2, num_hidden_layers=1, rope_theta=100.0)
    legacy_state_dict = _make_transformers_dinov3_state_dict()
    state_dict = {
        key.removeprefix("model."): value for key, value in legacy_state_dict.items()
    }

    converted = convert_transformers_dinov3_state_dict(state_dict, config)

    assert torch.equal(converted["blocks.0.attn.proj.weight"], state_dict["layer.0.attention.o_proj.weight"])
    assert torch.equal(converted["blocks.0.mlp.fc1.weight"], state_dict["layer.0.mlp.up_proj.weight"])
    assert torch.equal(converted["blocks.0.ls1.gamma"], state_dict["layer.0.layer_scale1.lambda1"])


def test_transformers_dinov3_export_state_dict_maps_gated_mlp() -> None:
    """DINOv3 Plus/7B 的拆分 SwiGLU MLP 应映射到 C++ 侧的 ``w1/w2/w3`` 命名。"""

    config = SimpleNamespace(hidden_size=4, num_attention_heads=2, num_hidden_layers=1, rope_theta=100.0)
    state_dict = _make_transformers_dinov3_state_dict(gated_mlp=True)

    converted = convert_transformers_dinov3_state_dict(state_dict, config)

    assert torch.equal(converted["blocks.0.mlp.w1.weight"], state_dict["model.layer.0.mlp.gate_proj.weight"])
    assert torch.equal(converted["blocks.0.mlp.w2.weight"], state_dict["model.layer.0.mlp.up_proj.weight"])
    assert torch.equal(converted["blocks.0.mlp.w3.weight"], state_dict["model.layer.0.mlp.down_proj.weight"])
    assert "blocks.0.mlp.fc1.weight" not in converted


def test_export_model_state_dict_uses_adapter_export() -> None:
    """``gen_wts.py`` 应通过通用导出入口读取适配器转换后的权重。"""

    class FakeExportModel(torch.nn.Module):
        """模拟提供自定义导出 state_dict 的模型。"""

        def export_state_dict(self) -> dict[str, torch.Tensor]:
            """返回转换后的伪造权重表。

            Returns:
                仅包含一个权重项的 state_dict。
            """

            return {"converted.weight": torch.ones(1)}

    exported = export_model_state_dict(FakeExportModel())

    assert list(exported.keys()) == ["converted.weight"]


def test_torchhub_rejects_dinov3_models() -> None:
    """DINOv3 已切换到 transformers pipeline，torchhub 后端应给出明确错误。"""

    with pytest.raises(ValueError, match="Unsupported torchhub model"):
        create_model("dinov3_vitb16", "torchhub")
