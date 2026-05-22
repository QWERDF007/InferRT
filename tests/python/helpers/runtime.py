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


def run_python_classification(
    irt_module: object,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """Python 绑定执行分类 ``infer()``。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
        model_name: ``create_model`` 使用的模型名。
        weights_path: ``build_or_load`` 的 ``.wts`` 路径。
        input_tensor: 单输入模型的 NCHW 张量。

    Returns:
        dict[str, np.ndarray]: 键 ``input`` 加各 ``output_tensor_names()`` 输出。

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
    model.infer(inputs, outputs)
    return {"input": input_tensor, **outputs}


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
        feature_names: ``ModelConfig.feature_tensor_names``；``feature_only=True``。

    Returns:
        dict[str, np.ndarray]: 键 ``input`` 加各特征输出；若绑定返回单向量则包装为单键字典。

    Raises:
        RuntimeError: 输入张量数量不为 1 时。
    """

    config = irt_module.ModelConfig()
    config.feature_tensor_names = feature_names
    config.feature_only = True

    model = irt_module.create_model(model_name, config)
    model.build_or_load(str(weights_path))

    input_names = model.input_tensor_names()
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input tensor, got: {input_names}")

    outputs = model.forward_features({input_names[0]: input_tensor})
    if isinstance(outputs, np.ndarray):
        feature_output_names = model.feature_output_tensor_names()
        output_name = feature_output_names[0] if feature_output_names else feature_names[0]
        output_dict = {output_name: outputs}
    else:
        output_dict = dict(outputs)

    return {"input": input_tensor, **output_dict}
