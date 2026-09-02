from __future__ import annotations

from pathlib import Path

from tools.quality_gates import (
    main,
    scan_cmake,
    scan_cuda_allocations,
    scan_installed_configs,
    scan_public_headers,
    scan_source_artifacts,
)


def test_public_header_gate_rejects_backend_leak(tmp_path: Path) -> None:
    header = tmp_path / "src/model/include/inferrt/model/Leaked.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("#include <NvInfer.h>\n", encoding="utf-8")

    findings = scan_public_headers(tmp_path)

    assert len(findings) == 1
    assert findings[0].gate == "public-header"


def test_install_config_gate_rejects_source_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    install = tmp_path / "install"
    install.mkdir()
    config = install / "lib/cmake/InferRT/InferRTTargets.cmake"
    config.parent.mkdir(parents=True)
    config.write_text(f'set(_source "{source.as_posix()}")\n', encoding="utf-8")

    findings = scan_installed_configs(install, source, build)

    assert len(findings) == 1
    assert findings[0].gate == "install-config"


def test_repository_cmake_passes_static_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    assert scan_cmake(root) == []


def test_cmake_runtime_copy_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text(
        "add_custom_command(TARGET app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different a.dll b.dll)\n",
        encoding="utf-8",
    )

    findings = scan_cmake(tmp_path)

    assert len(findings) == 1
    assert findings[0].gate == "cmake-runtime-copy"


def test_global_compile_definitions_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text(
        "add_compile_definitions(INFERRT_BUILD_ONNX=1)\n",
        encoding="utf-8",
    )

    findings = scan_cmake(tmp_path)

    assert len(findings) == 1
    assert findings[0].gate == "cmake-flags"


def test_cmake_runtime_packaging_invocation_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text(
        "add_custom_target(package_runtime\n"
        "    COMMAND ${Python_EXECUTABLE} tools/package_runtime_dlls.py\n"
        ")\n",
        encoding="utf-8",
    )

    findings = scan_cmake(tmp_path)

    assert len(findings) == 1
    assert findings[0].gate == "cmake-runtime-deploy"


def test_cuda_workspace_owner_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "src/cvcuda/priv/OpNMSImpl.cu"
    header = source.with_suffix(".hpp")
    header.parent.mkdir(parents=True)
    header.write_text("class Workspace { ~Workspace(); void reset(); };\n", encoding="utf-8")
    source.write_text(
        "void Workspace::reset() { cudaFreeAsync(nullptr, nullptr); }\n"
        "void allocate() { cudaMallocAsync(nullptr, 1, nullptr); }\n",
        encoding="utf-8",
    )

    assert scan_cuda_allocations(tmp_path) == []


def test_source_artifact_gate_rejects_tracked_build_output(tmp_path: Path) -> None:
    artifact = tmp_path / "samples/model/onnx/export.log"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("generated output\n", encoding="utf-8")

    findings = scan_source_artifacts(tmp_path, [artifact.relative_to(tmp_path)])

    assert len(findings) == 1
    assert findings[0].gate == "source-artifact"
    assert findings[0].path == "samples/model/onnx/export.log"


def test_json_output_creates_its_parent_directory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "nested" / "quality-gates.json"

    assert main(["--root", str(root), "--json", str(output)]) == 0
    assert output.is_file()
