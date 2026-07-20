"""PyTorch 与 InferRT 的 DINO/LingBot-Vision 性能基准。

默认覆盖 DINOv2、DINOv3 和 LingBot-Vision 的 ViT-S/ViT-B，分别运行
batch=1/2/4/8 与 FP32/FP16。PyTorch 与 InferRT 都使用 GPU 常驻输入；
这样测量的是模型前向本身，不把 NumPy 分配和主机拷贝混入结果。

示例（先只跑小模型，缩短首次 engine 构建时间）：

    D:\\Software\\anaconda3\\envs\\py312\\python.exe \\
        benchmark\\python\\benchmark_vision.py \\
        --models dinov2_vits14,lingbot_vision_vits16 \\
        --benchmark_min_time=0.05s \\
        --benchmark_format=json \\
        --benchmark_out=benchmark\\python\\results\\vision.json

LingBot-Vision 的 ``.pt`` 会自动转换为 ``.wts``，转换结果写入
``<build-dir>/benchmark_vision_artifacts``，C++ benchmark 默认也会搜索该目录。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

import google_benchmark as benchmark
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = Path(os.environ.get("INFERRT_BUILD_DIR", REPO_ROOT / "build")).resolve()
DEFAULT_MODEL_ROOT = Path(os.environ.get("INFERRT_MODEL_ROOT", "F:/models")).resolve()
DEFAULT_DINO_ROOT = Path(os.environ.get("INFERRT_DINO_ROOT", "F:/Github/DINO-based")).resolve()
DEFAULT_LINGBOT_ROOT = Path(os.environ.get("INFERRT_LINGBOT_VISION_REPO", "F:/Github/lingbot-vision")).resolve()
DEFAULT_LINGBOT_MODEL_ROOT = Path(
    os.environ.get("INFERRT_LINGBOT_VISION_ROOT", DEFAULT_MODEL_ROOT / "lingbot-vision")
).resolve()
DEFAULT_ARTIFACT_DIR = Path(
    os.environ.get("INFERRT_BENCHMARK_ARTIFACT_ROOT", DEFAULT_BUILD_DIR / "benchmark_vision_artifacts")
).resolve()

_DLL_DIRECTORY_HANDLES: list[object] = []
_SINK: Any = None
_ACTIVE_KEY: tuple[str, str, str] | None = None
_ACTIVE_RUNNER: Any = None


@dataclass(frozen=True)
class VisionSpec:
    """描述一个待测试的视觉 backbone。"""

    name: str
    family: str
    input_size: int
    checkpoint: Path
    config_name: str | None = None
    wts_names: tuple[str, ...] = ()


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示 benchmark 参数并退出。")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dino-root", type=Path, default=DEFAULT_DINO_ROOT)
    parser.add_argument("--lingbot-root", type=Path, default=DEFAULT_LINGBOT_ROOT)
    parser.add_argument("--lingbot-model-root", type=Path, default=DEFAULT_LINGBOT_MODEL_ROOT)
    parser.add_argument(
        "--models",
        default="dinov2_vits14,dinov2_vitb14,dinov3_vits16,dinov3_vitb16,"
        "lingbot_vision_vits16,lingbot_vision_vitb16",
        help="模型名 CSV，也可使用 all、dino 或 lingbot。",
    )
    parser.add_argument("--batches", default="1,2,4,8", help="动态 batch CSV。")
    parser.add_argument("--precisions", default="fp32,fp16", help="fp32/fp16 CSV。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=10, help="每个 runtime/batch 的 warmup 次数。")
    args, benchmark_tail = parser.parse_known_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args, [sys.argv[0], *benchmark_tail]


ARGS, BENCHMARK_ARGV = _parse_args(sys.argv[1:])


def _read_cmake_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#") or line.startswith("//") or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0].strip()] = value.strip()
    return values


def _add_runtime_dir(path: Path, path_entries: list[str]) -> None:
    if not path.exists():
        return
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    if text not in path_entries:
        path_entries.insert(0, text)
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(text))


def _configure_inferrt_runtime(build_dir: Path) -> None:
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        raise FileNotFoundError(f"Build directory not found: {build_dir}")

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for path in (build_dir / "bin", build_dir / "lib"):
        _add_runtime_dir(path, path_entries)

    cache = _read_cmake_cache(build_dir / "CMakeCache.txt")
    trt_root = cache.get("TRT_ROOT", "")
    if trt_root:
        _add_runtime_dir(Path(trt_root) / "bin", path_entries)
        _add_runtime_dir(Path(trt_root) / "lib", path_entries)

    opencv_dir = cache.get("OpenCV_DIR", "")
    if not opencv_dir:
        details = cache.get("FIND_PACKAGE_MESSAGE_DETAILS_OpenCV", "")
        match = re.search(r"\[([A-Za-z]:/.+?)\]", details)
        if match:
            opencv_dir = match.group(1)
    if opencv_dir:
        opencv_root = Path(opencv_dir)
        for path in (opencv_root / "bin", opencv_root / "x64" / "vc16" / "bin", opencv_root.parent / "bin"):
            _add_runtime_dir(path, path_entries)

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        _add_runtime_dir(Path(cuda_path) / "bin", path_entries)
    for raw in os.environ.get("INFERRT_DLL_DIRS", "").split(";"):
        if raw.strip():
            _add_runtime_dir(Path(raw.strip()), path_entries)
    os.environ["PATH"] = os.pathsep.join(path_entries)

    # CMake places the Python extension in ``bin``. Older build trees may still
    # contain a stale copy in ``lib``; keep the current runtime directory first
    # so the benchmark cannot silently import that copy.
    module_dir = str(build_dir / "bin")
    sys.path[:] = [entry for entry in sys.path if entry != module_dir]
    sys.path.insert(0, module_dir)


def _configure_import_paths() -> None:
    entries = [
        REPO_ROOT / "samples" / "model" / "python",
        ARGS.dino_root / "dinov2",
        ARGS.dino_root / "dinov3",
        ARGS.lingbot_root,
    ]
    for entry in entries:
        if entry.exists() and str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = sorted({int(item) for item in _parse_csv(value)})
    except ValueError as exc:
        raise ValueError(f"Invalid --batches value: {value}") from exc
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("--batches must contain positive integers")
    return tuple(batches)


def _parse_precisions(value: str) -> tuple[str, ...]:
    precisions: list[str] = []
    for item in _parse_csv(value):
        normalized = item.lower()
        if normalized in ("float32", "fp32"):
            normalized = "fp32"
        elif normalized in ("float16", "fp16"):
            normalized = "fp16"
        else:
            raise ValueError(f"Invalid precision '{item}', expected fp32 or fp16")
        if normalized not in precisions:
            precisions.append(normalized)
    if not precisions:
        raise ValueError("--precisions must not be empty")
    return tuple(precisions)


def _find_file(root: Path, names: tuple[str, ...]) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Model root not found: {root}")
    lower_names = {name.lower() for name in names}
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in lower_names:
            return path
    raise FileNotFoundError(f"Could not find one of {names} under {root}")


def _find_huggingface_snapshot(root: Path, model_fragment: str) -> Path:
    fragment = model_fragment.lower()
    candidates = []
    for config_path in root.rglob("config.json"):
        snapshot = config_path.parent
        if fragment in str(snapshot).lower() and (snapshot / "model.safetensors").is_file():
            candidates.append(snapshot)
    if not candidates:
        raise FileNotFoundError(f"Could not find local Hugging Face snapshot for {model_fragment} under {root}")
    return sorted(candidates)[0]


def _resolve_specs() -> dict[str, VisionSpec]:
    model_root = ARGS.model_root.resolve()
    lingbot_model_root = ARGS.lingbot_model_root.resolve()
    specs = {
        "dinov2_vits14": VisionSpec(
            "dinov2_vits14",
            "dinov2",
            518,
            _find_file(model_root / "dinov2", ("dinov2_vits14_pretrain.pth",)),
            wts_names=("dinov2_vits14.wts",),
        ),
        "dinov2_vitb14": VisionSpec(
            "dinov2_vitb14",
            "dinov2",
            518,
            _find_file(model_root / "dinov2", ("dinov2_vitb14_pretrain.pth",)),
            wts_names=("dinov2_vitb14.wts",),
        ),
        "dinov3_vits16": VisionSpec(
            "dinov3_vits16",
            "dinov3",
            224,
            _find_huggingface_snapshot(model_root / "dinov3", "dinov3-vits16-pretrain-lvd1689m"),
            wts_names=("dinov3_vits16.wts",),
        ),
        "dinov3_vitb16": VisionSpec(
            "dinov3_vitb16",
            "dinov3",
            224,
            _find_huggingface_snapshot(model_root / "dinov3", "dinov3-vitb16-pretrain-lvd1689m"),
            wts_names=("dinov3_vitb16.wts",),
        ),
        "lingbot_vision_vits16": VisionSpec(
            "lingbot_vision_vits16",
            "lingbot",
            512,
            _find_file(lingbot_model_root, ("lingbot-vision-vit-small.pt", "lbotv_vit_small.pt")),
            config_name="lbot_vision_vits.yaml",
            wts_names=("lingbot_vision_vits16.wts", "lingbot-vision-vit-small.wts"),
        ),
        "lingbot_vision_vitb16": VisionSpec(
            "lingbot_vision_vitb16",
            "lingbot",
            512,
            _find_file(lingbot_model_root, ("lingbot-vision-vit-base.pt", "lbotv_vit_base.pt")),
            config_name="lbot_vision_vitb.yaml",
            wts_names=("lingbot_vision_vitb16.wts", "lingbot-vision-vit-base.wts"),
        ),
    }
    return specs


def _expand_model_names(value: str) -> tuple[str, ...]:
    all_names = (
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov3_vits16",
        "dinov3_vitb16",
        "lingbot_vision_vits16",
        "lingbot_vision_vitb16",
    )
    names: list[str] = []
    for item in _parse_csv(value):
        normalized = item.lower()
        expanded = all_names if normalized == "all" else all_names[:4] if normalized == "dino" else all_names[4:]
        if normalized in ("all", "dino", "lingbot", "lingbot-vision"):
            for name in expanded:
                if name not in names:
                    names.append(name)
        elif item not in all_names:
            raise ValueError(f"Unsupported --models entry: {item}")
        elif item not in names:
            names.append(item)
    if not names:
        raise ValueError("--models must not be empty")
    return tuple(names)


def _source_files(spec: VisionSpec) -> list[Path]:
    files = [spec.checkpoint]
    files.extend(
        [
            REPO_ROOT / "samples" / "model" / "python" / "dino_gen_wts.py",
            REPO_ROOT / "samples" / "model" / "python" / "dino_model_zoo.py",
            REPO_ROOT / "samples" / "model" / "python" / "classification_model_zoo.py",
        ]
    )
    if spec.family == "lingbot":
        files.extend(
            [
                ARGS.lingbot_root / "lingbot_vision" / "loader.py",
                ARGS.lingbot_root / "lingbot_vision" / "build.py",
                ARGS.lingbot_root / "lingbot_vision" / "vit.py",
                ARGS.lingbot_root / "lingbot_vision" / "configs" / str(spec.config_name),
            ]
        )
    return files


def _is_fresh(output: Path, sources: list[Path]) -> bool:
    if not output.is_file() or output.stat().st_size <= 0:
        return False
    output_time = output.stat().st_mtime
    return all(not source.exists() or output_time >= source.stat().st_mtime for source in sources)


def _find_existing_wts(spec: VisionSpec) -> Path | None:
    roots = [ARGS.model_root, ARGS.lingbot_model_root]
    wanted = {name.lower() for name in spec.wts_names}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.wts")):
            if path.name.lower() in wanted:
                return path
    return None


def _load_reference_model(spec: VisionSpec) -> Any:
    torch = _torch()
    _configure_import_paths()
    if spec.family == "dinov2":
        from dino_model_zoo import create_model

        model = create_model(
            spec.name,
            "torchhub",
            hub_repo=str(ARGS.dino_root / "dinov2"),
            hub_source="local",
            pretrained=True,
            hub_weights=str(spec.checkpoint),
        )
    elif spec.family == "dinov3":
        from dino_model_zoo import create_model

        model = create_model(
            spec.name,
            "transformers",
            hf_model_id=str(spec.checkpoint),
            local_files_only=True,
        )
    else:
        if str(ARGS.lingbot_root) not in sys.path:
            sys.path.insert(0, str(ARGS.lingbot_root))
        from lingbot_vision import load_backbone, load_backbone_state, load_config

        config_path = ARGS.lingbot_root / "lingbot_vision" / "configs" / str(spec.config_name)
        config = load_config(config_path)
        model, _ = load_backbone(
            config,
            load_backbone_state(spec.checkpoint),
            device="cpu",
            dtype=torch.float32,
            verbose=False,
        )
    return model.eval()


def _ensure_wts(spec: VisionSpec, model: Any | None = None) -> Path:
    existing = _find_existing_wts(spec)
    if existing is not None:
        return existing

    output = ARGS.artifact_root.resolve() / f"{spec.name}.wts"
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = _source_files(spec)
    if _is_fresh(output, sources):
        return output

    _configure_import_paths()
    from dino_gen_wts import write_wts

    owned_model = model is None
    model = model if model is not None else _load_reference_model(spec)
    write_wts(model, str(output), verbose=False)
    if owned_model:
        del model
        gc.collect()
    return output


def _torch() -> Any:
    import torch

    return torch


def _resolve_device() -> Any:
    torch = _torch()
    if ARGS.device == "cuda" or (ARGS.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def _precision_dtype(precision: str) -> Any:
    torch = _torch()
    return torch.float16 if precision == "fp16" else torch.float32


def _forward_reference(model: Any, spec: VisionSpec, inputs: Any) -> Any:
    if spec.family == "lingbot":
        return model.forward_features(inputs)["x_norm_clstoken"]
    return model(inputs)


class PyTorchRunner:
    """运行一个模型/精度组合的 PyTorch GPU 前向。"""

    def __init__(self, spec: VisionSpec, precision: str) -> None:
        torch = _torch()
        self.spec = spec
        self.precision = precision
        self.device = _resolve_device()
        self.dtype = _precision_dtype(precision)
        model = _load_reference_model(spec)
        _ensure_wts(spec, model)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.inputs: dict[int, Any] = {}
        self.torch = torch

    def _input(self, batch: int) -> Any:
        if batch not in self.inputs:
            count = batch * 3 * self.spec.input_size * self.spec.input_size
            self.inputs[batch] = self.torch.linspace(
                -1.0,
                1.0,
                steps=count,
                device=self.device,
                dtype=self.dtype,
            ).reshape(batch, 3, self.spec.input_size, self.spec.input_size)
        return self.inputs[batch]

    def run(self, state: benchmark.State) -> None:
        inputs = self._input(int(state.range(0)))
        with self.torch.inference_mode():
            for _ in range(ARGS.warmup):
                _store(_forward_reference(self.model, self.spec, inputs))
            _synchronize(self.device, self.torch)
            while state:
                start = time.perf_counter()
                _store(_forward_reference(self.model, self.spec, inputs))
                _synchronize(self.device, self.torch)
                state.counters["last_call_ms"] = (time.perf_counter() - start) * 1000.0
        state.items_processed = state.iterations * int(state.range(0))
        state.counters["batch"] = int(state.range(0))


def _numpy_dtype(name: str) -> Any:
    return np.float16 if name == "float16" else np.float32


def _torch_dtype(torch: Any, name: str) -> Any:
    return torch.float16 if name == "float16" else torch.float32


class InferRTRunner:
    """运行一个模型/精度组合的 InferRT Python 绑定前向。"""

    def __init__(self, spec: VisionSpec, precision: str) -> None:
        _configure_inferrt_runtime(ARGS.build_dir)
        import inferrt_model_py as irt

        self.spec = spec
        self.precision = precision
        self.irt = irt
        self.torch = _torch()
        self.device = _resolve_device()
        self.weights = _ensure_wts(spec)
        config = irt.ModelConfig()
        config.input_shape = [1, 3, spec.input_size, spec.input_size]
        config.dynamic_batch_range = [1, max(_parse_batches(ARGS.batches)), max(_parse_batches(ARGS.batches))]
        config.precision = irt.ModelPrecision.FP16 if precision == "fp16" else irt.ModelPrecision.FP32
        self.model = irt.create_model(spec.name, config=config)
        self.model.build_or_load(str(self.weights))
        input_names = list(self.model.input_tensor_names())
        if len(input_names) != 1:
            raise RuntimeError(f"InferRT benchmark expects one input, got {input_names}")
        self.input_name = input_names[0]
        self.output_names = list(self.model.output_tensor_names())
        if len(self.output_names) != 1:
            raise RuntimeError(f"InferRT vision benchmark expects one output, got {self.output_names}")

    def _prepare(self, batch: int) -> tuple[Any, Any]:
        shape = [batch, 3, self.spec.input_size, self.spec.input_size]
        self.model.set_tensor_shape(self.input_name, shape)
        input_dtype_name = self.model.tensor_dtype(self.input_name)
        output_name = self.output_names[0]
        output_shape = tuple(int(value) for value in self.model.tensor_shape(output_name))
        output_dtype_name = self.model.tensor_dtype(output_name)
        count = batch * 3 * self.spec.input_size * self.spec.input_size
        input_host = np.linspace(-1.0, 1.0, num=count, dtype=np.float32).reshape(shape)

        if self.device.type == "cuda":
            input_tensor = self.torch.from_numpy(np.ascontiguousarray(input_host)).to(self.device)
            if input_dtype_name == "float16":
                input_tensor = input_tensor.half()
            output_tensor = self.torch.empty(
                output_shape,
                dtype=_torch_dtype(self.torch, output_dtype_name),
                device=self.device,
            )
            return input_tensor, output_tensor

        input_array = np.ascontiguousarray(input_host, dtype=_numpy_dtype(input_dtype_name))
        output_array = np.empty(output_shape, dtype=_numpy_dtype(output_dtype_name))
        return input_array, output_array

    def _infer_once(self, inputs: Any, outputs: Any) -> Any:
        if self.device.type == "cuda":
            return self.model.infer_v2(inputs, outputs, None, False)
        return self.model.infer(inputs, outputs)

    def run(self, state: benchmark.State) -> None:
        inputs, outputs = self._prepare(int(state.range(0)))
        for _ in range(ARGS.warmup):
            _store(self._infer_once(inputs, outputs))
        _synchronize(self.device, self.torch)
        while state:
            start = time.perf_counter()
            _store(self._infer_once(inputs, outputs))
            _synchronize(self.device, self.torch)
            state.counters["last_call_ms"] = (time.perf_counter() - start) * 1000.0
        state.items_processed = state.iterations * int(state.range(0))
        state.counters["batch"] = int(state.range(0))


def _synchronize(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _store(value: Any) -> None:
    global _SINK
    _SINK = value


def _active_runner(runtime: str, spec: VisionSpec, precision: str) -> Any:
    global _ACTIVE_KEY, _ACTIVE_RUNNER, _SINK
    key = (runtime, spec.name, precision)
    if _ACTIVE_KEY != key:
        old_torch = getattr(_ACTIVE_RUNNER, "torch", None)
        _ACTIVE_RUNNER = None
        _SINK = None
        gc.collect()
        if old_torch is not None and old_torch.cuda.is_available():
            old_torch.cuda.empty_cache()
        if runtime == "pytorch":
            _ACTIVE_RUNNER = PyTorchRunner(spec, precision)
        else:
            _ACTIVE_RUNNER = InferRTRunner(spec, precision)
        _ACTIVE_KEY = key
    return _ACTIVE_RUNNER


def _register(name: str, callback: Callable[[benchmark.State], None], batch: int) -> None:
    options = callback
    options = benchmark.option.args((batch,))(options)
    options = benchmark.option.arg_name("batch")(options)
    options = benchmark.option.unit(benchmark.kMillisecond)(options)
    options = benchmark.option.use_real_time()(options)
    benchmark.register(options, name=name)


def _register_benchmarks(specs: dict[str, VisionSpec], models: tuple[str, ...], batches: tuple[int, ...], precisions: tuple[str, ...]) -> None:
    runtime_labels = {"pytorch": "PyTorch", "inferrt": "InferRT"}
    for name in models:
        spec = specs[name]
        for precision in precisions:
            for runtime in ("pytorch", "inferrt"):
                for batch in batches:
                    benchmark_name = f"{runtime_labels[runtime]}/{spec.name}/{precision}/batch_{batch}"

                    def callback(state: benchmark.State, *, runtime=runtime, spec=spec, precision=precision) -> None:
                        try:
                            _active_runner(runtime, spec, precision).run(state)
                        except Exception as exc:  # noqa: BLE001 - benchmark should report a failed case cleanly
                            state.skip_with_error(str(exc))

                    _register(benchmark_name, callback, batch)


def main() -> None:
    _configure_inferrt_runtime(ARGS.build_dir)
    _configure_import_paths()
    specs = _resolve_specs()
    models = _expand_model_names(ARGS.models)
    batches = _parse_batches(ARGS.batches)
    precisions = _parse_precisions(ARGS.precisions)
    _register_benchmarks(specs, models, batches, precisions)

    benchmark.add_custom_context("build_dir", str(ARGS.build_dir.resolve()))
    benchmark.add_custom_context("model_root", str(ARGS.model_root.resolve()))
    benchmark.add_custom_context("device", str(_resolve_device()))
    benchmark.add_custom_context("torch", _torch().__version__)
    benchmark.main(BENCHMARK_ARGV)


if __name__ == "__main__":
    main()
