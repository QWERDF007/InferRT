"""RF-DETR 上游源码导入兼容辅助。"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any


def patch_transformers_backbone_api() -> None:
    """兼容本地 ``transformers`` 版本与上游 RF-DETR 期望的 backbone API。

    上游 RF-DETR 当前按 ``transformers>=5.1`` 的方式从包根导入
    ``BackboneConfigMixin`` / ``BackboneMixin``，并调用无参
    ``BackboneMixin._init_transformers_backbone()``。本地环境若仍是过渡版本，
    只补齐这两个窄接口，不改变模型计算逻辑。
    """

    try:
        import transformers
        from transformers.utils.backbone_utils import BackboneConfigMixin, BackboneMixin
    except ImportError:
        return

    if not hasattr(transformers, "BackboneConfigMixin"):
        transformers.BackboneConfigMixin = BackboneConfigMixin
    if not hasattr(transformers, "BackboneMixin"):
        transformers.BackboneMixin = BackboneMixin

    original = getattr(BackboneMixin, "_init_transformers_backbone", None)
    if original is None or getattr(original, "_inferrt_accepts_optional_config", False):
        return

    signature = inspect.signature(original)
    config_param = signature.parameters.get("config")
    if config_param is None or config_param.default is not inspect.Parameter.empty:
        return

    def _init_transformers_backbone(self: Any, config: Any | None = None) -> Any:
        """把旧版 ``(self, config)`` 签名适配为新版可无参调用。"""

        if config is None:
            config = getattr(self, "config")
        return original(self, config)

    _init_transformers_backbone._inferrt_accepts_optional_config = True  # type: ignore[attr-defined]
    BackboneMixin._init_transformers_backbone = _init_transformers_backbone


def patch_deprecate_api() -> None:
    """补齐上游 RF-DETR 只用于弃用提示的 ``deprecated_class`` 装饰器。"""

    try:
        import deprecate
    except ImportError:
        return

    if hasattr(deprecate, "deprecated_class"):
        return

    def deprecated_class(*args: Any, **kwargs: Any) -> Any:
        """返回原类的轻量兼容装饰器。"""

        if args and isinstance(args[0], type):
            return args[0]

        def decorator(cls: type) -> type:
            return cls

        return decorator

    deprecate.deprecated_class = deprecated_class


def configure_rfdetr_import(rfdetr_root: Path | None) -> None:
    """配置上游 RF-DETR 导入路径，并应用本地依赖兼容补丁。

    Args:
        rfdetr_root: 上游 ``rf-detr`` 仓库根目录；为空时使用已安装包。

    Raises:
        FileNotFoundError: 指定仓库不存在时抛出。
    """

    if rfdetr_root is not None:
        root = rfdetr_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"RF-DETR root does not exist: {root}")
        src = root / "src"
        sys.path.insert(0, str(src if src.exists() else root))
    patch_transformers_backbone_api()
    patch_deprecate_api()
