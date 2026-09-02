from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


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
