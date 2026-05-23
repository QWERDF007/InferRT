import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
CLASSIFICATION_DIR = THIS_DIR.parent / "classification"
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

from model_zoo import create_model, list_supported_models, preprocess, resolve_input_size


@dataclass
class TensorSpec:
    name: str
    dtype: str
    dims: tuple[int, ...]
    file_name: str


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def parse_manifest(manifest_path: Path) -> tuple[dict[str, str], dict[str, TensorSpec]]:
    metadata: dict[str, str] = {}
    tensors: dict[str, TensorSpec] = {}

    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("tensor|"):
                _, name, dtype, dims, file_name = line.split("|")
                tensors[name] = TensorSpec(name=name, dtype=dtype, dims=parse_dims(dims), file_name=file_name)
                continue
            key, value = line.split("=", 1)
            metadata[key] = value

    return metadata, tensors


def dtype_to_numpy(dtype: str):
    mapping = {
        "float32": np.float32,
        "float16": np.float16,
        "int8": np.int8,
        "uint8": np.uint8,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype in manifest: {dtype}")
    return mapping[dtype]


def load_tensor_from_dump(dump_dir: Path, spec: TensorSpec) -> np.ndarray:
    tensor_path = dump_dir / spec.file_name
    values = np.fromfile(tensor_path, dtype=dtype_to_numpy(spec.dtype))
    if spec.dims:
        values = values.reshape(spec.dims)
    return values


def summarize_diff(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual - reference)
    ref_abs = np.abs(reference)
    rel_mask = ref_abs > 1e-3
    rel = np.zeros_like(diff, dtype=np.float32)
    if np.any(rel_mask):
        rel[rel_mask] = diff[rel_mask] / ref_abs[rel_mask]
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
        "ref_abs_max": float(ref_abs.max()),
    }


def model_family(model_name: str) -> str:
    lower_name = model_name.lower()
    if "dinov2" in lower_name or "dinov3" in lower_name:
        return "dino"
    if model_name == "alexnet":
        return "alexnet"
    if model_name.startswith("resnet") or model_name.startswith("wide_resnet"):
        return "resnet"
    if model_name.startswith("vgg"):
        return "vgg"
    if model_name == "mobilenet_v2":
        return "mobilenet_v2"
    if model_name in {"mobilenet_v3_large", "mobilenet_v3_small"}:
        return "mobilenet_v3"
    raise ValueError(f"Unsupported model for feature comparison: {model_name}")


def default_backend_for_model(model_name: str, backend: str) -> str:
    """@brief 根据模型名推断 PyTorch 参考后端。
    @param model_name InferRT 模型注册名。
    @param backend 命令行显式传入的后端；非空时直接返回。
    @return 可传给 ``model_zoo.create_model`` 的后端名称。
    """

    if backend:
        return backend

    lower_name = model_name.lower()
    if "dinov3" in lower_name:
        return "transformers"
    if "dinov2" in lower_name:
        return "torchhub"
    return "torchvision"


def image_size_from_manifest(tensors: dict[str, TensorSpec]) -> tuple[int, int] | None:
    """@brief 从 C++ dump manifest 的 input 张量形状中解析预处理尺寸。
    @param tensors ``parse_manifest`` 返回的张量描述表。
    @return ``(height, width)``；manifest 缺少 input 或形状不完整时返回 ``None``。
    """

    input_spec = tensors.get("input")
    if input_spec is None or len(input_spec.dims) != 4:
        return None
    return input_spec.dims[2], input_spec.dims[3]


def capture_resnet_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}

    x = model.conv1(x)
    features["stem.conv1"] = x.detach().cpu()

    x = model.bn1(x)
    relu = getattr(model, "relu", None)
    if relu is None:
        relu = getattr(model, "act1")
    x = relu(x)
    features["stem.relu"] = x.detach().cpu()

    x = model.maxpool(x)
    features["stem.pool"] = x.detach().cpu()

    x = model.layer1(x)
    features["layer1"] = x.detach().cpu()
    x = model.layer2(x)
    features["layer2"] = x.detach().cpu()
    x = model.layer3(x)
    features["layer3"] = x.detach().cpu()
    x = model.layer4(x)
    features["layer4"] = x.detach().cpu()

    avgpool = getattr(model, "avgpool", None)
    if avgpool is None:
        avgpool = getattr(model, "global_pool")
    x = avgpool(x)
    features["avgpool"] = x.detach().cpu()

    x = torch.flatten(x, 1)
    features["flatten"] = x.detach().cpu()

    x = model.fc(x)
    features["logits"] = x.detach().cpu()
    return features


def capture_alexnet_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}

    x = model.features[0](x)
    features["conv1"] = x.detach().cpu()
    x = model.features[1](x)
    x = model.features[2](x)
    features["pool1"] = x.detach().cpu()

    x = model.features[3](x)
    features["conv2"] = x.detach().cpu()
    x = model.features[4](x)
    x = model.features[5](x)
    features["pool2"] = x.detach().cpu()

    x = model.features[6](x)
    features["conv3"] = x.detach().cpu()
    x = model.features[7](x)

    x = model.features[8](x)
    features["conv4"] = x.detach().cpu()
    x = model.features[9](x)

    x = model.features[10](x)
    features["conv5"] = x.detach().cpu()
    x = model.features[11](x)
    x = model.features[12](x)
    features["pool3"] = x.detach().cpu()

    x = model.avgpool(x)
    features["avgpool"] = x.detach().cpu()

    x = torch.flatten(x, 1)
    features["flatten"] = x.detach().cpu()

    x = model.classifier[1](x)
    features["fc1"] = x.detach().cpu()
    x = model.classifier[2](x)

    x = model.classifier[4](x)
    features["fc2"] = x.detach().cpu()
    x = model.classifier[5](x)

    x = model.classifier[6](x)
    features["logits"] = x.detach().cpu()
    return features


def vgg_block_depths(model_name: str) -> list[int]:
    mapping = {
        "vgg11": [1, 1, 2, 2, 2],
        "vgg13": [2, 2, 2, 2, 2],
        "vgg16": [2, 2, 3, 3, 3],
        "vgg19": [2, 2, 4, 4, 4],
    }
    return mapping[model_name]


def capture_vgg_features(model_name: str, model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    layer_index = 0

    for block_index, depth in enumerate(vgg_block_depths(model_name), start=1):
        for _ in range(depth):
            x = model.features[layer_index](x)
            layer_index += 1
            x = model.features[layer_index](x)
            layer_index += 1
        x = model.features[layer_index](x)
        layer_index += 1
        features[f"block{block_index}"] = x.detach().cpu()

    x = model.avgpool(x)
    features["avgpool"] = x.detach().cpu()

    x = torch.flatten(x, 1)
    features["flatten"] = x.detach().cpu()

    x = model.classifier[0](x)
    x = model.classifier[1](x)
    features["fc1"] = x.detach().cpu()

    x = model.classifier[3](x)
    x = model.classifier[4](x)
    features["fc2"] = x.detach().cpu()

    x = model.classifier[6](x)
    features["logits"] = x.detach().cpu()
    return features


def capture_mobilenet_v2_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}

    x = model.features[0](x)
    features["stem"] = x.detach().cpu()

    for index in range(1, len(model.features)):
        x = model.features[index](x)
        features[f"features.{index}"] = x.detach().cpu()

    x = F.adaptive_avg_pool2d(x, output_size=1)
    x = torch.flatten(x, 1)
    features["flatten"] = x.detach().cpu()

    x = model.classifier(x)
    features["logits"] = x.detach().cpu()
    return features


def capture_mobilenet_v3_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}

    x = model.features[0](x)
    features["stem"] = x.detach().cpu()

    for index in range(1, len(model.features)):
        x = model.features[index](x)
        features[f"features.{index}"] = x.detach().cpu()

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    features["flatten"] = x.detach().cpu()

    x = model.classifier[0](x)
    x = model.classifier[1](x)
    features["classifier.0"] = x.detach().cpu()

    x = model.classifier[2](x)
    x = model.classifier[3](x)
    features["logits"] = x.detach().cpu()
    return features


def capture_dino_features(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """@brief 调用官方 DINO ``forward_features`` 并转换为 CPU 张量字典。
    @param model DINOv2 torch.hub 模型或 DINOv3 Transformers 适配器。
    @param x 预处理后的 NCHW 输入张量。
    @return 特征名到 PyTorch 张量的映射。
    @exception TypeError 当官方模型未返回字典时抛出，避免静默比较错误张量。
    """

    outputs = model.forward_features(x)
    if not isinstance(outputs, dict):
        raise TypeError(f"DINO forward_features() must return a dict, got {type(outputs)!r}")
    return {name: value.detach().cpu() for name, value in outputs.items()}


@torch.no_grad()
def extract_features(model_name: str, model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    family = model_family(model_name)
    if family == "alexnet":
        return capture_alexnet_features(model, x)
    if family == "resnet":
        return capture_resnet_features(model, x)
    if family == "vgg":
        return capture_vgg_features(model_name, model, x)
    if family == "mobilenet_v2":
        return capture_mobilenet_v2_features(model, x)
    if family == "mobilenet_v3":
        return capture_mobilenet_v3_features(model, x)
    if family == "dino":
        return capture_dino_features(model, x)
    raise ValueError(f"Unsupported model family: {family}")


def write_manifest(output_dir: Path, metadata: dict[str, str], tensors: dict[str, np.ndarray]) -> None:
    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}={value}\n")
        for name, values in tensors.items():
            dims = ",".join(str(dim) for dim in values.shape)
            file_name = f"{name.replace('/', '_').replace('.', '_')}.bin"
            f.write(f"tensor|{name}|float32|{dims}|{file_name}\n")
            values.astype(np.float32).tofile(output_dir / file_name)


def default_image_path() -> Path:
    return (THIS_DIR / "../../../assets/pics/dog.jpg").resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare InferRT feature dumps against PyTorch reference tensors")
    parser.add_argument("-m", "--model", type=str, default="", help="Model name")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=("torchvision", "timm", "torchhub", "transformers"),
        default="",
        help="Model provider backend. If omitted, DINOv2 uses torchhub and DINOv3 uses transformers.",
    )
    parser.add_argument("-i", "--img_path", type=str, default="", help="Path to the input image")
    parser.add_argument(
        "-f",
        "--feature_names",
        type=str,
        default="",
        help="Comma-separated feature names. If omitted, use names from --compare_dir when available.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="torch.hub repo or local directory, used when --backend torchhub",
    )
    parser.add_argument(
        "--hub-source",
        type=str,
        choices=("github", "local"),
        default="github",
        help="torch.hub source, used when --backend torchhub",
    )
    parser.add_argument(
        "--hub-weights",
        type=str,
        default=None,
        help="Optional DINO torch.hub weights path or URL, used when --backend torchhub",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model id, used when --backend transformers",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models from the local cache only, used when --backend transformers",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Create model without pretrained weights when the selected backend supports it",
    )
    parser.add_argument(
        "--compare_dir",
        type=str,
        default="",
        help="Directory produced by inferrt_sample_feature_extract",
    )
    parser.add_argument("--dump_dir", type=str, default="", help="Optional output directory for Python-side dumps")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1.2e-1, help="Absolute tolerance")
    parser.add_argument("-l", "--list_model", action="store_true")
    parser.set_defaults(pretrained=True)
    return parser


@torch.no_grad()
def main(args) -> None:
    if args.list_model:
        backend = args.backend or "torchvision"
        models = list_supported_models(backend)
        print(f"{backend} models:", len(models))
        print(models)
        return

    compare_dir = Path(args.compare_dir).resolve() if args.compare_dir else None
    manifest_metadata: dict[str, str] = {}
    cpp_specs: dict[str, TensorSpec] = {}
    if compare_dir is not None:
        manifest_metadata, cpp_specs = parse_manifest(compare_dir / "manifest.txt")

    model_name = args.model or manifest_metadata.get("model_name", "")
    if not model_name:
        raise ValueError("Model name is required")

    feature_names = parse_csv(args.feature_names) if args.feature_names else parse_csv(
        manifest_metadata.get("feature_tensor_names", "")
    )
    if not feature_names:
        raise ValueError("Feature names are required")

    image_path = Path(args.img_path).resolve() if args.img_path else (
        Path(manifest_metadata["image_path"]).resolve() if "image_path" in manifest_metadata else default_image_path()
    )

    backend = default_backend_for_model(model_name, args.backend)
    model = create_model(
        model_name,
        backend,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        pretrained=args.pretrained,
        hub_weights=args.hub_weights,
        hf_model_id=args.hf_model_id,
        local_files_only=args.local_files_only,
    )
    model.eval()

    image_size = image_size_from_manifest(cpp_specs) or resolve_input_size(model)
    print(f"Loading image: {image_path}")
    print(f"Reference backend: {backend}, input size: {image_size[0]}x{image_size[1]}")
    image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    input_tensor = preprocess(image, image_size=image_size).float()
    all_features = extract_features(model_name, model, input_tensor)
    selected = {name: all_features[name].numpy().astype(np.float32, copy=False) for name in feature_names}
    selected["input"] = input_tensor.numpy().astype(np.float32, copy=False)

    if args.dump_dir:
        dump_dir = Path(args.dump_dir).resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            dump_dir,
            {
                "version": "1",
                "backend": f"pytorch:{backend}",
                "model_name": model_name,
                "image_path": image_path.as_posix(),
                "feature_tensor_names": ",".join(feature_names),
            },
            selected,
        )
        print(f"Saved Python dump to: {dump_dir}")

    if compare_dir is None:
        return

    cpp_input = load_tensor_from_dump(compare_dir, cpp_specs["input"]).astype(np.float32, copy=False)
    py_input = selected["input"]
    input_close = np.allclose(cpp_input, py_input, rtol=args.rtol, atol=args.atol)
    input_stats = summarize_diff(py_input, cpp_input)
    print(
        f"input: match={input_close} max_abs={input_stats['max_abs']:.6g} "
        f"mean_abs={input_stats['mean_abs']:.6g} max_rel={input_stats['max_rel']:.6g} "
        f"mean_rel={input_stats['mean_rel']:.6g} ref_abs_max={input_stats['ref_abs_max']:.6g}"
    )

    all_ok = input_close
    for name in feature_names:
        if name not in cpp_specs:
            raise KeyError(f"Tensor '{name}' is missing from compare_dir manifest")

        cpp_values = load_tensor_from_dump(compare_dir, cpp_specs[name]).astype(np.float32, copy=False)
        py_values = selected[name]

        if cpp_values.shape != py_values.shape:
            raise ValueError(f"Shape mismatch for {name}: cpp={cpp_values.shape}, py={py_values.shape}")

        is_close = np.allclose(cpp_values, py_values, rtol=args.rtol, atol=args.atol)
        stats = summarize_diff(py_values, cpp_values)
        all_ok = all_ok and is_close
        print(
            f"{name}: match={is_close} shape={cpp_values.shape} max_abs={stats['max_abs']:.6g} "
            f"mean_abs={stats['mean_abs']:.6g} max_rel={stats['max_rel']:.6g} "
            f"mean_rel={stats['mean_rel']:.6g} ref_abs_max={stats['ref_abs_max']:.6g}"
        )

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
