"""InferRT Python 绑定集成测试的 pytest 公共 fixture 与命令行选项。

命令行参数（由 ``pytest_addoption`` 注册）:
    --inferrt-build-dir: CMake 构建目录，含 ``inferrt_model_py`` 与 sample 可执行文件。
        默认空字符串，表示使用 ``INFERRT_BUILD_DIR`` 或 ``<repo>/build``。
    --inferrt-rtol: 分类/infer_v2 张量比对相对容差。默认 ``1e-6``。
    --inferrt-atol: 分类/infer_v2 张量比对绝对容差。默认 ``5e-2``。
    --inferrt-feature-rtol: 特征提取张量比对相对容差。默认 ``1e-4``。
    --inferrt-feature-atol: 特征提取张量比对绝对容差。默认 ``1.5e-1``。

环境变量:
    INFERRT_BUILD_DIR: 未传 ``--inferrt-build-dir`` 时使用的构建目录路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 仓库根目录：tests/python -> tests -> InferRT
ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PYTHON = ROOT / "samples" / "model" / "python"
CLASSIFICATION_SAMPLES = ROOT / "samples" / "model" / "classification"

if str(SAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(SAMPLES_PYTHON))
if str(CLASSIFICATION_SAMPLES) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_SAMPLES))

from util import ensure_module_path, preprocess_image  # noqa: E402

from helpers.runtime import default_build_dir, project_root, require_weights  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册 InferRT 测试专用的命令行参数。

    Args:
        parser: pytest 内置的参数解析器。

    注册的选项见本模块顶部 docstring；可通过 ``pytest.ini`` 的 ``addopts`` 设置默认值。
    """

    parser.addoption(
        "--inferrt-build-dir",
        action="store",
        default="",
        help="包含 inferrt_model_py 的 CMake 构建目录（特征 parity 另需 sample 可执行文件）",
    )
    parser.addoption(
        "--inferrt-rtol",
        action="store",
        type=float,
        default=1e-6,
        help="张量比对的相对容差（numpy.allclose 的 rtol）",
    )
    parser.addoption(
        "--inferrt-atol",
        action="store",
        type=float,
        default=5e-2,
        help="分类/infer_v2 张量比对的绝对容差（numpy.allclose 的 atol）",
    )
    parser.addoption(
        "--inferrt-feature-rtol",
        action="store",
        type=float,
        default=1e-4,
        help="特征提取张量比对的相对容差（numpy.allclose 的 rtol）",
    )
    parser.addoption(
        "--inferrt-feature-atol",
        action="store",
        type=float,
        default=1.5e-1,
        help="特征提取张量比对的绝对容差（numpy.allclose 的 atol）",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """仓库根目录路径（session 级，全测试共享）。

    Returns:
        Path: InferRT 仓库根目录的绝对路径。
    """

    return project_root()


@pytest.fixture(scope="session")
def build_dir(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    """CMake 构建目录。

    Args:
        repo_root: 仓库根目录，用于解析默认 ``<repo>/build``。
        pytestconfig: pytest 配置对象，读取 ``--inferrt-build-dir``。

    Returns:
        Path: 已解析的构建目录绝对路径。

    目录不存在时对依赖本 fixture 的测试执行 ``pytest.skip``。
    """

    override = pytestconfig.getoption("--inferrt-build-dir")
    path = Path(override).resolve() if override else default_build_dir(repo_root)
    if not path.exists():
        pytest.skip(f"Build directory not found: {path}")
    return path


@pytest.fixture(scope="session")
def irt_module(build_dir: Path):
    """已配置 DLL 搜索路径并导入的 ``inferrt_model_py`` 模块。

    Args:
        build_dir: 构建目录，用于 ``ensure_module_path`` 加载 ``.pyd``/``.so`` 及依赖。

    Returns:
        module: ``inferrt_model_py`` 模块对象（别名 ``irt``）。
    """

    ensure_module_path(build_dir)
    import inferrt_model_py as irt

    return irt


@pytest.fixture(scope="session")
def default_image(repo_root: Path) -> Path:
    """默认测试图片路径。

    Args:
        repo_root: 仓库根目录。

    Returns:
        Path: ``assets/pics/dog.jpg`` 的绝对路径。

    文件不存在时对依赖本 fixture 的测试执行 ``pytest.skip``。
    """

    image_path = repo_root / "assets" / "pics" / "dog.jpg"
    if not image_path.exists():
        pytest.skip(f"Default test image missing: {image_path}")
    return image_path


@pytest.fixture(scope="session")
def input_tensor(default_image: Path):
    """经 ImageNet 预处理后的输入张量。

    Args:
        default_image: 待预处理的图片路径。

    Returns:
        np.ndarray: 形状 ``(1, 3, 224, 224)`` 的 ``float32`` 张量（NCHW）。
    """

    return preprocess_image(default_image)


@pytest.fixture(scope="session")
def tolerances(pytestconfig: pytest.Config) -> tuple[float, float]:
    """分类/infer_v2 张量数值比对的容差对。

    Args:
        pytestconfig: pytest 配置对象，读取 ``--inferrt-rtol`` 与 ``--inferrt-atol``。

    Returns:
        tuple[float, float]: ``(rtol, atol)``，供 ``assert_tensors_close`` 使用。
    """

    return pytestconfig.getoption("--inferrt-rtol"), pytestconfig.getoption("--inferrt-atol")


@pytest.fixture(scope="session")
def feature_tolerances(pytestconfig: pytest.Config) -> tuple[float, float]:
    """特征提取张量数值比对的容差对（比分类更宽松，因跨框架中间层差异更大）。

    Args:
        pytestconfig: pytest 配置对象，读取 ``--inferrt-feature-rtol`` 与
            ``--inferrt-feature-atol``。

    Returns:
        tuple[float, float]: ``(rtol, atol)``。
    """

    return pytestconfig.getoption("--inferrt-feature-rtol"), pytestconfig.getoption("--inferrt-feature-atol")


@pytest.fixture
def weights_path(repo_root: Path):
    """解析权重文件路径的工厂 fixture。

    Args:
        repo_root: 仓库根目录，权重路径相对此目录解析。

    Returns:
        Callable[[str], Path]: 接受相对路径（如 ``samples/model/.../resnet18.wts``），
        返回绝对 ``Path``；文件缺失时 ``pytest.skip`` 而非抛错到测试体。
    """

    def _resolve(relative_path: str) -> Path:
        try:
            return require_weights(repo_root, relative_path)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    return _resolve
