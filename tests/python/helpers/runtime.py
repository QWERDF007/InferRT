"""定位 InferRT 构建产物、启动 C++ sample 与 Python 绑定推理的测试运行时工具。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from util import allocate_output_tensors, preprocess_image


def project_root() -> Path:
    """推断仓库根目录。

    Returns:
        Path: ``tests/python/helpers`` 向上三级即为仓库根。
    """

    return Path(__file__).resolve().parents[3]


def default_build_dir(root: Path) -> Path:
    """解析默认 CMake 构建目录。

    Args:
        root: 仓库根目录。

    Returns:
        Path: ``INFERRT_BUILD_DIR`` 环境变量值，或 ``root / "build"`` 的绝对路径。
    """

    return Path(os.environ.get("INFERRT_BUILD_DIR", root / "build")).resolve()


def sample_executable(build_dir: Path, sample_name: str) -> Path:
    """在常见输出布局下查找 ``inferrt_sample_<name>`` 可执行文件。

    Args:
        build_dir: CMake 构建根目录。
        sample_name: 不含前缀的名称，如 ``classification``、``feature_extract``。

    Returns:
        Path: 找到的第一个存在的可执行文件路径。

    Raises:
        FileNotFoundError: 在 ``bin/``、``bin/Debug/``、``bin/Release/`` 下均未找到时。
    """

    exe_name = f"inferrt_sample_{sample_name}"
    if os.name == "nt":
        exe_name += ".exe"
    candidates = [
        build_dir / "bin" / exe_name,
        build_dir / "bin" / "Debug" / exe_name,
        build_dir / "bin" / "Release" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Sample executable '{exe_name}' not found under {build_dir}. "
        "Build targets inferrt_sample_classification and inferrt_sample_feature_extract first."
    )


def require_weights(root: Path, relative_weights: str) -> Path:
    """解析权重相对路径。

    Args:
        root: 仓库根目录。
        relative_weights: 相对 ``root`` 的 ``.wts`` 路径。

    Returns:
        Path: 权重文件绝对路径。

    Raises:
        FileNotFoundError: 文件不存在时，消息中含 ``gen_wts.py`` 生成提示。
    """

    weights_path = (root / relative_weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file missing: {weights_path}. "
            "Generate it with samples/model/classification/gen_wts.py."
        )
    return weights_path


def run_process(command: list[str], *, cwd: Path) -> None:
    """运行子进程并在非零退出码时抛出 ``RuntimeError``。

    Args:
        command:  argv 列表（不含 shell 展开）。
        cwd: 子进程工作目录。

    Raises:
        RuntimeError: 退出码非零时，附带完整命令与 stdout/stderr。
    """

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(command)}\n"
            f"  exit: {completed.returncode}\n"
            f"  stdout:\n{completed.stdout}\n"
            f"  stderr:\n{completed.stderr}"
        )


def run_cpp_classification_dump(
    *,
    build_dir: Path,
    model_name: str,
    weights_path: Path,
    image_path: Path,
    dump_dir: Path,
    cwd: Path,
) -> None:
    """调用 C++ 分类 sample，将输入/输出张量写入 ``dump_dir``。

    Args:
        build_dir: 构建目录。
        model_name: ``--model`` 参数。
        weights_path: ``--weights-file`` 参数（绝对路径）。
        image_path: ``--image-path`` 参数。
        dump_dir: ``--dump-dir`` 输出目录（含 ``manifest.txt``）。
        cwd: 子进程 ``cwd``，通常为 ``repo_root``。
    """

    executable = sample_executable(build_dir, "classification")
    command = [
        str(executable),
        "--model",
        model_name,
        "--weights-file",
        str(weights_path),
        "--image-path",
        str(image_path),
        "--dump-dir",
        str(dump_dir),
    ]
    run_process(command, cwd=cwd)


def run_cpp_feature_dump(
    *,
    build_dir: Path,
    model_name: str,
    weights_path: Path,
    image_path: Path,
    feature_names: list[str],
    dump_dir: Path,
    cwd: Path,
) -> None:
    """调用 C++ 特征提取 sample，dump 指定中间层张量。

    Args:
        build_dir: 构建目录。
        model_name: ``--model`` 参数。
        weights_path: ``--weights-file`` 参数。
        image_path: ``--image-path`` 参数。
        feature_names: ``--features`` 逗号分隔的中间层名列表。
        dump_dir: ``--output-dir`` 输出目录。
        cwd: 子进程工作目录。
    """

    executable = sample_executable(build_dir, "feature_extract")
    command = [
        str(executable),
        "--model",
        model_name,
        "--weights-file",
        str(weights_path),
        "--features",
        ",".join(feature_names),
        "--image-path",
        str(image_path),
        "--output-dir",
        str(dump_dir),
    ]
    run_process(command, cwd=cwd)


def run_torch_classification(model_name: str, input_tensor: np.ndarray) -> np.ndarray:
    """使用 torchvision 预训练模型执行分类前向（与 ``gen_wts.py`` 一致）。

    Args:
        model_name: ``model_zoo.TORCHVISION_MODEL_ZOO`` 中的模型名。
        input_tensor: 形状 ``(1, 3, 224, 224)`` 的 ``float32`` NCHW 张量。

    Returns:
        np.ndarray: PyTorch logits，形状通常为 ``(1, num_classes)``。

    Raises:
        ImportError: 未安装 ``torch`` / ``torchvision`` 时由调用方 ``importorskip`` 处理。
        ValueError: 不支持的模型名。
    """

    import torch
    from model_zoo import create_model

    model = create_model(model_name, "torchvision")
    model.eval()

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        logits = model(batch)
    return logits.detach().cpu().numpy()


def run_python_classification(
    irt_module: object,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """InferRT pybind11 绑定执行分类 ``infer()``。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
        model_name: ``create_model`` 使用的模型名。
        weights_path: ``build_or_load`` 的 ``.wts`` 路径。
        input_tensor: 单输入模型的 NCHW 张量。

    Returns:
        dict[str, np.ndarray]: 各 ``output_tensor_names()`` 输出张量（不含 ``input``）。

    Raises:
        RuntimeError: 模型输入张量数量不为 1 时。
    """

    model = irt_module.create_model(model_name)
    model.build_or_load(str(weights_path))

    input_names = model.input_tensor_names()
    output_names = model.output_tensor_names()
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input tensor, got: {input_names}")

    inputs = {input_names[0]: input_tensor}
    outputs = allocate_output_tensors(model, output_names)
    infer_result = model.infer(inputs, outputs)

    if isinstance(infer_result, dict):
        return {name: np.asarray(array) for name, array in infer_result.items()}
    if len(output_names) == 1:
        return {output_names[0]: np.asarray(infer_result)}
    raise RuntimeError(f"Unexpected infer() return type for outputs: {output_names}")


def run_torch_features(model_name: str, input_tensor: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    """使用 PyTorch 预训练模型提取中间层特征。

    通过 forward hook 捕获指定子模块的输出，实现与 InferRT ``forward_features()``
    的逐层对比。

    Args:
        model_name: ``model_zoo.TORCHVISION_MODEL_ZOO`` 中的模型名。
        input_tensor: 形状 ``(1, 3, 224, 224)`` 的 ``float32`` NCHW 张量。
        feature_names: 要捕获的 InferRT 特征名列表。

    Returns:
        dict[str, np.ndarray]: 特征名 -> 对应 PyTorch 子模块输出（已转为 NumPy）。

    Raises:
        ValueError: 无法在 PyTorch 模型中定位某个特征名对应的子模块。
    """

    import torch
    from model_zoo import create_model

    # InferRT 特征名 -> PyTorch ``named_modules`` 键的映射。
    # 同名直接命中的不需要列在这里；不同模型族同名特征需单独处理。
    _NAME_MAP: dict[str, str] = {
        # MobileNetV2/V3: stem 指 features 的第一个子模块
        "stem": "features.0",
        # VGG: blockN 为第 N 个 MaxPool2d 之后的输出
        "block1": "features.2",
        "block2": "features.5",
        "block3": "features.10",
        "block4": "features.15",
        "block5": "features.20",
        # AlexNet: poolN 为第 N 个 MaxPool2d 之后的输出
        "pool1": "features.2",
        "pool2": "features.5",
        "pool3": "features.12",
        # AlexNet fc2: classifier[4] Linear 的输出（不含后续 ReLU）
        "fc2": "classifier.4",
    }
    if model_name == "googlenet":
        _NAME_MAP = {
            "pool1": "maxpool1",
            "pool2": "maxpool2",
            "pool3": "maxpool3",
            "pool4": "maxpool4",
        }

    model = create_model(model_name, "torchvision")
    model.eval()
    named_modules = dict(model.named_modules())

    outputs: dict[str, np.ndarray] = {}
    hooks: list = []

    def _make_hook(name: str):
        def _hook(_module, _input, _output):
            outputs[name] = _output.detach().cpu().numpy()

        return _hook

    for feature_name in feature_names:
        module_name = _NAME_MAP.get(feature_name, feature_name)
        if module_name not in named_modules:
            available = sorted(named_modules.keys())
            raise ValueError(
                f"Cannot resolve feature '{feature_name}' → '{module_name}' "
                f"in model '{model_name}'. Available modules: {available}"
            )
        hooks.append(named_modules[module_name].register_forward_hook(_make_hook(feature_name)))

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        model(batch)

    for h in hooks:
        h.remove()

    return outputs


def run_python_features(
    irt_module: object,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """Python 绑定执行 ``forward_features()``。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
        model_name: ``create_model`` 使用的模型名。
        weights_path: 权重文件路径。
        input_tensor: 预处理后的输入张量。
        feature_names: 建网层 key；同时写入 ``output_tensor_names``（与 C++ sample 一致）。

    Returns:
        dict[str, np.ndarray]: 以 ``output_tensor_names`` 为键的特征张量；单向量时包装为单键字典。

    Raises:
        RuntimeError: 输入张量数量不为 1 时。
    """

    config = irt_module.ModelConfig()
    config.feature_tensor_names = feature_names
    config.output_tensor_names = feature_names
    config.feature_only = True

    model = irt_module.create_model(model_name, config)
    model.build_or_load(str(weights_path))

    input_names = model.input_tensor_names()
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input tensor, got: {input_names}")

    output_names = list(model.output_tensor_names())
    outputs = model.forward_features({input_names[0]: input_tensor})
    if isinstance(outputs, np.ndarray):
        output_name = output_names[0] if output_names else feature_names[0]
        output_dict = {output_name: outputs}
    else:
        output_dict = dict(outputs)

    return output_dict
