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


def _write_install_prefix_probe(source: Path, version: str) -> None:
    module = (ROOT / "cmake" / "ConfigInstallPrefix.cmake").as_posix()
    source.mkdir(parents=True, exist_ok=True)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        f"project(InferRT VERSION {version} LANGUAGES NONE)\n"
        f'include("{module}")\n'
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

    _write_install_prefix_probe(source, "1.2.4")
    second_version, second_prefix = _run_install_prefix_probe(source, build)

    assert first_version == "1.2.3"
    assert first_prefix.resolve() == (source / "InferRT-1.2.3").resolve()
    assert second_version == "1.2.4"
    assert second_prefix.resolve() == (source / "InferRT-1.2.4").resolve()


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
            rf"inferrt_dependency_default\(\s*{re.escape(dependency_name)}\b",
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
