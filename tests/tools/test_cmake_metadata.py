from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dependency_utils import load_dependencies


ROOT = Path(__file__).resolve().parents[2]


def _manifest_default(dependency_name: str) -> Path:
    dependencies = {
        str(dependency["name"]): dependency
        for dependency in load_dependencies(ROOT / "tools" / "dependencies.yaml")
    }
    return Path(str(dependencies[dependency_name]["default"]))


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.split(".")]
    return tuple((parts + [0, 0, 0])[:3])


def test_preset_minimum_matches_project_minimum() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    preset_version = presets["cmakeMinimumRequired"]
    preset = ".".join(str(preset_version[key]) for key in ("major", "minor", "patch"))

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"cmake_minimum_required\(VERSION\s+([0-9]+(?:\.[0-9]+){1,2})\)", cmake)
    assert match is not None

    assert _version_tuple(preset) == _version_tuple(match.group(1))


def test_package_config_uses_standard_prefix_layout() -> None:
    package = (ROOT / "cmake" / "ConfigCMakePackage.cmake").read_text(encoding="utf-8")
    assert 'set(INFERRT_INSTALL_CMAKEDIR "lib/cmake/${PROJECT_NAME}"' in package


def test_project_default_install_prefix_is_versioned(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the install prefix contract")

    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DINFERRT_ENABLE_CUDA=OFF",
            "-DINFERRT_BUILD_TENSORRT=OFF",
            "-DINFERRT_BUILD_TESTS=OFF",
            "-DINFERRT_BUILD_SAMPLES=OFF",
            "-DINFERRT_BUILD_BENCHMARK=OFF",
            "-DINFERRT_BUILD_PYTHON=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    cache_values = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(CMAKE_PROJECT_VERSION|CMAKE_INSTALL_PREFIX):[^=]+=(.*)$", line)
        if match:
            cache_values[match.group(1)] = match.group(2)

    assert cache_values["CMAKE_INSTALL_PREFIX"]
    assert Path(cache_values["CMAKE_INSTALL_PREFIX"]).resolve() == (
        ROOT / f"InferRT-{cache_values['CMAKE_PROJECT_VERSION']}"
    ).resolve()


def _write_install_prefix_probe(
    source: Path,
    version: str,
    stale_prefix: Path | None = None,
) -> None:
    module = (ROOT / "cmake" / "ConfigInstallPrefix.cmake").as_posix()
    source.mkdir(parents=True, exist_ok=True)
    stale_prefix_line = ""
    if stale_prefix is not None:
        stale_prefix_line = f'set(CMAKE_INSTALL_PREFIX "{stale_prefix.as_posix()}")\n'
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        f"project(InferRT VERSION {version} LANGUAGES NONE)\n"
        f"{stale_prefix_line}"
        f'include("{module}")\n'
        "include(GNUInstallDirs)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/install-prefix.txt"\n'
        '     "${PROJECT_VERSION}\n${CMAKE_INSTALL_PREFIX}\n")\n',
        encoding="utf-8",
    )


def _run_install_prefix_probe(
    source: Path,
    build: Path,
    explicit_prefix: Path | None = None,
) -> tuple[str, Path]:
    command = ["cmake", "-S", str(source), "-B", str(build)]
    if explicit_prefix is not None:
        command.append(f"-DCMAKE_INSTALL_PREFIX={explicit_prefix.as_posix()}")
    subprocess.run(command, check=True, capture_output=True, text=True)
    version, prefix = (build / "install-prefix.txt").read_text(encoding="utf-8").splitlines()
    return version, Path(prefix)


def test_default_install_prefix_follows_version_across_reconfigure(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the install prefix contract")

    source = tmp_path / "source"
    build = tmp_path / "build"
    _write_install_prefix_probe(source, "1.2.3")
    first_version, first_prefix = _run_install_prefix_probe(source, build)

    _write_install_prefix_probe(source, "1.2.4", first_prefix)
    second_version, second_prefix = _run_install_prefix_probe(source, build)

    assert first_version == "1.2.3"
    assert first_prefix.resolve() == (source / "InferRT-1.2.3").resolve()
    assert second_version == "1.2.4"
    assert second_prefix.resolve() == (source / "InferRT-1.2.4").resolve()
    cache_prefix = next(
        line.split("=", 1)[1]
        for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith("CMAKE_INSTALL_PREFIX:")
    )
    assert Path(cache_prefix).resolve() == (source / "InferRT-1.2.4").resolve()


def test_explicit_install_prefix_survives_reconfigure(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the install prefix contract")

    source = tmp_path / "source"
    build = tmp_path / "build"
    explicit_prefix = tmp_path / "custom-install"
    _write_install_prefix_probe(source, "1.2.3")
    _run_install_prefix_probe(source, build, explicit_prefix)

    _write_install_prefix_probe(source, "1.2.4")
    version, prefix = _run_install_prefix_probe(source, build)

    assert version == "1.2.4"
    assert prefix.resolve() == explicit_prefix.resolve()


def test_explicit_install_prefix_override_after_default_is_preserved(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the install prefix contract")

    source = tmp_path / "source"
    build = tmp_path / "build"
    explicit_prefix = tmp_path / "custom-install"
    _write_install_prefix_probe(source, "1.2.3")
    _run_install_prefix_probe(source, build)
    _run_install_prefix_probe(source, build, explicit_prefix)

    _write_install_prefix_probe(source, "1.2.4")
    version, prefix = _run_install_prefix_probe(source, build)

    assert version == "1.2.4"
    assert prefix.resolve() == explicit_prefix.resolve()


def test_dependency_config_defaults_are_manifest_backed() -> None:
    dependencies = {
        str(dependency["name"]): dependency
        for dependency in load_dependencies(ROOT / "tools" / "dependencies.yaml")
    }
    module_dependencies = {
        "ConfigFaiss.cmake": "faiss",
        "ConfigOpenCV.cmake": "opencv",
        "ConfigPython.cmake": "python-torch-zlib",
        "ConfigTensorRT.cmake": "tensorrt",
    }

    for module_name, dependency_name in module_dependencies.items():
        source = (ROOT / "cmake" / module_name).read_text(encoding="utf-8")
        assert re.search(
            rf"(?:inferrt_dependency_default\(\s*{re.escape(dependency_name)}\b|"
            rf"inferrt_dependency_resolve_path\([^)]*\b{re.escape(dependency_name)}\b)",
            source,
        )
        assert str(dependencies[dependency_name].get("default", "")).strip()
        assert "D:/Software" not in source


def test_cmake_default_reader_returns_manifest_values(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured dependency defaults are Windows-specific")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    defaults_module = (ROOT / "cmake" / "ConfigDependencyDefaults.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(DependencyDefaultsProbe NONE)\n"
        f'include("{defaults_module}")\n'
        "inferrt_dependency_default(faiss _faiss)\n"
        "inferrt_dependency_default(opencv _opencv)\n"
        "inferrt_dependency_default(python-torch-zlib _python)\n"
        "inferrt_dependency_default(tensorrt _tensorrt)\n"
        "file(WRITE \"${CMAKE_BINARY_DIR}/defaults.txt\" "
        "\"${_faiss}\\n${_opencv}\\n${_python}\\n${_tensorrt}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    actual = (build / "defaults.txt").read_text(encoding="utf-8").splitlines()
    expected = [
        _manifest_default("faiss"),
        _manifest_default("opencv"),
        _manifest_default("python-torch-zlib"),
        _manifest_default("tensorrt"),
    ]
    assert [Path(value).resolve() for value in actual] == [path.resolve() for path in expected]


def test_dependency_cache_provenance_honors_updated_d_value(tmp_path: Path) -> None:
    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for dependency cache provenance")

    default_root = tmp_path / "default-dependency"
    explicit_one = tmp_path / "explicit-one"
    explicit_two = tmp_path / "explicit-two"
    default_root.mkdir()
    explicit_one.mkdir()
    explicit_two.mkdir()

    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        "  - name: test-dependency\n"
        f"    default: {default_root.as_posix()}\n",
        encoding="utf-8",
    )

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    defaults_module = (ROOT / "cmake" / "ConfigDependencyDefaults.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(DependencyProvenanceProbe NONE)\n"
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest.as_posix()}")\n'
        f'include("{defaults_module}")\n'
        "inferrt_dependency_resolve_path(\n"
        "    _resolved _origin test-dependency\n"
        "    VARIABLES TEST_ROOT\n"
        ")\n"
        "if(NOT _resolved STREQUAL \"${TEST_ROOT}\")\n"
        "    message(FATAL_ERROR \"Explicit dependency root was not selected\")\n"
        "endif()\n"
        "inferrt_dependency_cache_set(\n"
        "    TEST_ROOT \"${_resolved}\" PATH \"Test dependency root\"\n"
        "    INFERRT_DEPENDENCY_TEST_ROOT \"${_origin}\"\n"
        ")\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "${TEST_ROOT}\\n")\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DTEST_ROOT={explicit_one.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DTEST_ROOT={explicit_two.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    )

    resolved = Path((build / "resolved.txt").read_text(encoding="utf-8").strip())
    assert resolved.resolve() == explicit_two.resolve()


def test_faiss_config_caches_manifest_mkl_default(tmp_path: Path, monkeypatch) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured Faiss default is Windows-specific")

    monkeypatch.delenv("MKL_ROOT", raising=False)
    monkeypatch.delenv("MKLROOT", raising=False)

    fake_faiss = tmp_path / "faiss"
    (fake_faiss / "include" / "faiss").mkdir(parents=True)
    (fake_faiss / "lib").mkdir()
    (fake_faiss / "bin").mkdir()
    (fake_faiss / "include" / "faiss" / "Index.h").write_text("", encoding="utf-8")
    (fake_faiss / "lib" / "faiss.lib").write_bytes(b"import")
    (fake_faiss / "bin" / "faiss.dll").write_bytes(b"runtime")
    stale_faiss = tmp_path / "stale-faiss"
    stale_faiss.mkdir()
    fake_mkl = tmp_path / "mkl"
    (fake_mkl / "lib").mkdir(parents=True)

    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        "  - name: faiss\n"
        f"    default: {fake_faiss.as_posix()}\n"
        "  - name: faiss-mkl-runtime\n"
        f"    default: {fake_mkl.as_posix()}\n",
        encoding="utf-8",
    )

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigFaiss.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(FaissDefaultProbe NONE)\n"
        f'set(Faiss_HOME "{stale_faiss.as_posix()}" CACHE PATH "Faiss installation directory")\n'
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest.as_posix()}")\n'
        f'include("{config_module}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/faiss-values.txt" "${Faiss_HOME}\n${MKL_ROOT}\n")\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    faiss_home, mkl_root = (build / "faiss-values.txt").read_text(encoding="utf-8").splitlines()
    assert Path(faiss_home).resolve() == fake_faiss.resolve()
    assert Path(mkl_root).resolve() == fake_mkl.resolve()
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    assert f"MKL_ROOT:PATH={fake_mkl.as_posix()}" in cache


def test_opencv_config_applies_manifest_default_before_find(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured dependency defaults are Windows-specific")

    fake_opencv = tmp_path / "opencv"
    (fake_opencv / "lib").mkdir(parents=True)
    (fake_opencv / "bin").mkdir()
    (fake_opencv / "lib" / "OpenCVConfig.cmake").write_text(
        "set(OpenCV_FOUND TRUE)\n"
        "set(OpenCV_VERSION 4.8.0)\n"
        "set(OpenCV_LIBS)\n"
        "set(OpenCV_INCLUDE_DIRS)\n",
        encoding="utf-8",
    )
    stale_opencv = tmp_path / "stale-opencv"
    stale_opencv.mkdir()

    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        "  - name: opencv\n"
        f"    default: {fake_opencv.as_posix()}\n",
        encoding="utf-8",
    )

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigOpenCV.cmake").as_posix()
    defaults_manifest = manifest.as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(OpenCVDefaultProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        "set(INFERRT_ENABLE_CUDA ON)\n"
        "set(INFERRT_BUILD_SAMPLES OFF)\n"
        "set(INFERRT_BUILD_BENCHMARK OFF)\n"
        f'set(OpenCV_DIR "{(stale_opencv / "lib").as_posix()}" CACHE PATH "OpenCV CMake package directory")\n'
        f'set(INFERRT_DEPENDENCY_MANIFEST "{defaults_manifest}")\n'
        "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH OFF)\n"
        "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)\n"
        "set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)\n"
        "set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)\n"
        "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)\n"
        f'include("{config_module}")\n'
        "if(NOT OpenCV_FOUND)\n"
        "    message(FATAL_ERROR \"OpenCV default package was not found\")\n"
        "endif()\n"
        "file(WRITE \"${CMAKE_BINARY_DIR}/opencv-values.txt\" "
        "\"${OpenCV_HOME}\\n${OpenCV_DIR}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    home, package_dir = (build / "opencv-values.txt").read_text(encoding="utf-8").splitlines()
    assert Path(home).resolve() == fake_opencv.resolve()
    assert Path(package_dir).resolve() == (fake_opencv / "lib").resolve()


def test_tensorrt_config_applies_manifest_default_before_stale_cache(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured TensorRT default is Windows-specific")

    fake_tensorrt = tmp_path / "tensorrt"
    (fake_tensorrt / "include").mkdir(parents=True)
    (fake_tensorrt / "lib").mkdir()
    (fake_tensorrt / "bin").mkdir()
    (fake_tensorrt / "include" / "NvInferVersion.h").write_text(
        "#define NV_TENSORRT_MAJOR 10\n"
        "#define NV_TENSORRT_MINOR 16\n"
        "#define NV_TENSORRT_PATCH 1\n"
        "#define NV_TENSORRT_BUILD 11\n",
        encoding="utf-8",
    )
    for library in (
        "nvinfer_10",
        "nvinfer_plugin_10",
        "nvinfer_vc_plugin_10",
        "nvinfer_dispatch_10",
        "nvinfer_lean_10",
        "nvonnxparser_10",
    ):
        (fake_tensorrt / "lib" / f"{library}.lib").write_bytes(b"import")
    stale_tensorrt = tmp_path / "stale-tensorrt"
    stale_tensorrt.mkdir()

    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        "  - name: tensorrt\n"
        f"    default: {fake_tensorrt.as_posix()}\n",
        encoding="utf-8",
    )

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigTensorRT.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(TensorRTDefaultProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        "set(INFERRT_ENABLE_CUDA ON)\n"
        f'set(TRT_ROOT "{stale_tensorrt.as_posix()}" CACHE PATH "TensorRT installation directory")\n'
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest.as_posix()}")\n'
        f'include("{config_module}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/tensorrt-values.txt" "${TRT_ROOT}\n${TRT_VERSION}\n")\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    root, version = (build / "tensorrt-values.txt").read_text(encoding="utf-8").splitlines()
    assert Path(root).resolve() == fake_tensorrt.resolve()
    assert version == "10.16.1.11"


def test_python_config_default_wins_over_parent_python3_hint(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured Python default is Windows-specific")

    default_root = _manifest_default("python-torch-zlib")
    base_executable = default_root.parents[1] / "python.exe"
    if not (default_root / "python.exe").is_file() or not base_executable.is_file():
        pytest.skip("The configured Windows Python installations are unavailable")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigPython.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(PythonProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        f'set(Python3_EXECUTABLE "{base_executable.as_posix()}")\n'
        f'include("{config_module}")\n'
        "file(WRITE \"${CMAKE_BINARY_DIR}/python-values.txt\" "
        "\"${Python_EXECUTABLE}\\n${Python_ROOT_DIR}\\n${Python_VERSION}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    executable, root, version = (
        (build / "python-values.txt").read_text(encoding="utf-8").splitlines()
    )
    assert Path(executable).resolve() == (default_root / "python.exe").resolve()
    assert Path(root).resolve() == default_root.resolve()
    assert version.startswith("3.12.")


def test_python_config_replaces_legacy_find_cache_without_d_override(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured Python default is Windows-specific")

    default_root = _manifest_default("python-torch-zlib")
    if not (default_root / "python.exe").is_file():
        pytest.skip("The configured Python default is unavailable")

    stale_root = tmp_path / "legacy-python"
    stale_root.mkdir()
    (stale_root / "python.exe").write_bytes(b"legacy")
    fake_find_module = tmp_path / "cmake"
    fake_find_module.mkdir()
    (fake_find_module / "FindPython.cmake").write_text(
        "set(Python_FOUND TRUE)\n"
        "set(Python_Interpreter_FOUND TRUE)\n"
        "set(Python_Development_FOUND TRUE)\n"
        "set(Python_VERSION 3.12.0)\n",
        encoding="utf-8",
    )

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    stale_executable = (stale_root / "python.exe").as_posix()
    stale_root_path = stale_root.as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(PythonProbe NONE)\n"
        f'set(Python_EXECUTABLE "{stale_executable}" CACHE FILEPATH "legacy executable")\n'
        f'set(Python_ROOT_DIR "{stale_root_path}" CACHE PATH "legacy root")\n',
        encoding="utf-8",
    )
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)

    config_module = (ROOT / "cmake" / "ConfigPython.cmake").as_posix()
    manifest = (ROOT / "tools" / "dependencies.yaml").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(PythonProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        f'list(APPEND CMAKE_MODULE_PATH "{fake_find_module.as_posix()}")\n'
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest}")\n'
        f'include("{config_module}")\n'
        "file(WRITE \"${CMAKE_BINARY_DIR}/python-values.txt\" "
        "\"${Python_EXECUTABLE}\\n${Python_ROOT_DIR}\\n${Python_VERSION}\\n\")\n",
        encoding="utf-8",
    )
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)

    executable, root, version = (build / "python-values.txt").read_text(encoding="utf-8").splitlines()
    assert Path(executable).resolve() == (default_root / "python.exe").resolve()
    assert Path(root).resolve() == default_root.resolve()
    assert version.startswith("3.12.")


def test_python_config_honors_explicit_executable(tmp_path: Path) -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("The configured Python default is Windows-specific")

    default_root = _manifest_default("python-torch-zlib")
    base_executable = default_root.parents[1] / "python.exe"
    if not base_executable.is_file():
        pytest.skip("The configured base Python installation is unavailable")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigPython.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(PythonProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        f'include("{config_module}")\n'
        "file(WRITE \"${CMAKE_BINARY_DIR}/python-values.txt\" "
        "\"${Python_EXECUTABLE}\\n${Python_ROOT_DIR}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DPython_EXECUTABLE={base_executable.as_posix()}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    executable, root = (
        (build / "python-values.txt").read_text(encoding="utf-8").splitlines()
    )
    assert Path(executable).resolve() == base_executable.resolve()
    assert Path(root).resolve() == base_executable.parent.resolve()


def test_opencv_runtime_root_follows_package_bin_layout(tmp_path: Path) -> None:
    fake_opencv = tmp_path / "opencv"
    (fake_opencv / "bin").mkdir(parents=True)
    (fake_opencv / "bin" / "opencv_world.dll").write_bytes(b"runtime")

    fake_find_module = tmp_path / "cmake"
    fake_find_module.mkdir()
    (fake_find_module / "FindOpenCV.cmake").write_text(
        "set(OpenCV_FOUND TRUE)\n"
        f"set(OpenCV_DIR \"{fake_opencv.as_posix()}\")\n"
        "set(OpenCV_VERSION 4.8.0)\n"
        "set(OpenCV_LIBS)\n"
        "set(OpenCV_INCLUDE_DIRS)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text("dependencies:\n", encoding="utf-8")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigOpenCV.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(OpenCVProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        "set(INFERRT_ENABLE_CUDA ON)\n"
        "set(INFERRT_BUILD_SAMPLES OFF)\n"
        "set(INFERRT_BUILD_BENCHMARK OFF)\n"
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest.as_posix()}")\n'
        f"list(APPEND CMAKE_MODULE_PATH \"{fake_find_module.as_posix()}\")\n"
        f"include(\"{config_module}\")\n"
        "file(WRITE \"${CMAKE_BINARY_DIR}/opencv-paths.txt\" "
        "\"${OpenCV_HOME}\\n${OpenCV_BIN_DIR}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    paths = (build / "opencv-paths.txt").read_text(encoding="utf-8").splitlines()
    assert paths == [fake_opencv.as_posix(), (fake_opencv / "bin").as_posix()]


def test_opencv_runtime_bin_follows_resolved_library_layout(tmp_path: Path) -> None:
    fake_opencv = tmp_path / "opencv"
    package_lib = fake_opencv / "x64" / "vc16" / "lib"
    runtime_bin = fake_opencv / "x64" / "vc16" / "bin"
    package_lib.mkdir(parents=True)
    (fake_opencv / "include").mkdir()
    runtime_bin.mkdir()
    (runtime_bin / "opencv_world.dll").write_bytes(b"runtime")

    fake_find_module = tmp_path / "cmake"
    fake_find_module.mkdir()
    (fake_find_module / "FindOpenCV.cmake").write_text(
        "set(OpenCV_FOUND TRUE)\n"
        f"set(OpenCV_DIR \"{fake_opencv.as_posix()}\")\n"
        f"set(OpenCV_LIB_PATH \"{package_lib.as_posix()}\")\n"
        "set(OpenCV_VERSION 4.8.0)\n"
        "set(OpenCV_LIBS)\n"
        "set(OpenCV_INCLUDE_DIRS)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text("dependencies:\n", encoding="utf-8")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    config_module = (ROOT / "cmake" / "ConfigOpenCV.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(OpenCVProbe NONE)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        "set(INFERRT_ENABLE_CUDA ON)\n"
        "set(INFERRT_BUILD_SAMPLES OFF)\n"
        "set(INFERRT_BUILD_BENCHMARK OFF)\n"
        f'set(INFERRT_DEPENDENCY_MANIFEST "{manifest.as_posix()}")\n'
        f"list(APPEND CMAKE_MODULE_PATH \"{fake_find_module.as_posix()}\")\n"
        f"include(\"{config_module}\")\n"
        "file(WRITE \"${CMAKE_BINARY_DIR}/opencv-paths.txt\" "
        "\"${OpenCV_HOME}\\n${OpenCV_BIN_DIR}\\n\")\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )

    paths = (build / "opencv-paths.txt").read_text(encoding="utf-8").splitlines()
    assert paths == [fake_opencv.as_posix(), runtime_bin.as_posix()]


def test_package_config_does_not_load_optional_dependencies_for_core_component(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    template = (ROOT / "cmake" / "InferRTConfig.cmake.in").read_text(encoding="utf-8")
    package_config = template.replace(
        "@PACKAGE_INIT@",
        f'set(PACKAGE_PREFIX_DIR "{package_dir.as_posix()}")\n'
        "function(check_required_components _NAME)\n"
        "    foreach(_component IN LISTS ${_NAME}_FIND_COMPONENTS)\n"
        "        if(NOT ${_NAME}_${_component}_FOUND)\n"
        "            set(${_NAME}_FOUND FALSE PARENT_SCOPE)\n"
        "        endif()\n"
        "    endforeach()\n"
        "endfunction()",
    )
    package_config = package_config.replace("@INFERRT_ENABLE_CUDA@", "ON")
    package_config = package_config.replace("@INFERRT_BUILD_TENSORRT@", "ON")
    package_config = package_config.replace("@INFERRT_BUILD_ONNX@", "ON")
    package_config = package_config.replace("@INFERRT_BUILD_OPENVINO@", "ON")
    (package_dir / "InferRTConfig.cmake").write_text(package_config, encoding="utf-8")
    (package_dir / "InferRT_coreTargets.cmake").write_text(
        "add_library(InferRT::core INTERFACE IMPORTED)\n",
        encoding="utf-8",
    )

    source = tmp_path / "consumer"
    build = tmp_path / "consumer-build"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(InferRTPackageProbe LANGUAGES CXX)\n"
        "find_package(InferRT CONFIG REQUIRED COMPONENTS core)\n"
        "foreach(_unexpected_target IN ITEMS CUDA::cudart opencv_core "
        "TensorRT::TensorRT ONNXRuntime::ONNXRuntime Faiss::faiss)\n"
        "    if(TARGET ${_unexpected_target})\n"
        "        message(FATAL_ERROR \"Core package unexpectedly loaded ${_unexpected_target}\")\n"
        "    endif()\n"
        "endforeach()\n"
        "add_executable(probe main.cpp)\n"
        "target_link_libraries(probe PRIVATE InferRT::core)\n",
        encoding="utf-8",
    )
    (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DCMAKE_PREFIX_PATH={package_dir.as_posix()}",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_sanitizer_option_reaches_target_compile_and_link_options(tmp_path: Path) -> None:
    generator = None
    if shutil.which("g++") is None:
        pytest.skip("GNU C++ compiler is required for the sanitizer contract")
    if shutil.which("ninja") is not None:
        generator = "Ninja"
    elif shutil.which("mingw32-make") is not None:
        generator = "MinGW Makefiles"
    else:
        pytest.skip("A GNU-compatible CMake generator is required for the sanitizer contract")

    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    compiler_module = (ROOT / "cmake" / "ConfigCompiler.cmake").as_posix()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(SanitizerProbe LANGUAGES C CXX)\n"
        "set(PROJECT_NAME_UPPER INFERRT)\n"
        "set(INFERRT_ENABLE_SANITIZER ON)\n"
        "set(INFERRT_ENABLE_CUDA OFF)\n"
        "set(INFERRT_BUILD_ONNX OFF)\n"
        "set(INFERRT_BUILD_OPENVINO OFF)\n"
        "set(CMAKE_CXX_COMPILE_FEATURES cxx_std_20)\n"
        "set(CMAKE_CXX20_COMPILE_FEATURES cxx_std_20)\n"
        f'include("{compiler_module}")\n'
        "add_executable(probe main.cpp)\n"
        "inferrt_apply_compile_options(probe)\n"
        "get_target_property(compile_options probe COMPILE_OPTIONS)\n"
        "get_target_property(link_options probe LINK_OPTIONS)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/options.txt" "${compile_options}\\n${link_options}\\n")\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-G", generator],
        check=True,
        capture_output=True,
        text=True,
    )
    options = (build / "options.txt").read_text(encoding="utf-8")

    assert "-fsanitize=address" in options
    assert options.count("-fsanitize=address") == 2
