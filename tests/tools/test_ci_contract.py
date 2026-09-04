from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _commands(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


def test_full_release_workflow_is_an_executable_release_gate() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = document["jobs"]
    job = jobs["full-release-gpu-windows"]

    labels = {str(label).lower() for label in job["runs-on"]}
    commands = _commands(job)
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    assert {"self-hosted", "windows", "x64", "gpu"} <= labels
    assert "cmake --preset full" in commands
    assert "Python_EXECUTABLE" in commands
    assert "cmake --build --preset full-release" in commands
    assert "--config Debug" not in workflow_text
    assert "package_runtime_dlls.py" in commands and "--strict" in commands
    assert "ctest" in commands and "-C Release" in commands
    assert "tests/python" in commands
    assert "--inferrt-compare-runtime=tensorrt,onnx,openvino" in commands
    assert "--inferrt-compare-devices=cpu,gpu" in commands
    assert "--inferrt-strict-skips" in commands
    assert "inferrt_benchmark_cuda_ops" in commands and "--benchmark_out=" in commands
    assert "inferrt_benchmark_engine" in commands
    assert "quality_gates.py" in commands
    assert any(step.get("uses", "").startswith("actions/upload-artifact@") for step in job["steps"])
    assert "Set-Content" not in workflow_text


def test_linux_cuda_workflow_is_an_executable_release_gate() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = document["jobs"]["cuda-linux"]

    labels = {str(label).lower() for label in job["runs-on"]}
    commands = _commands(job)

    assert {"self-hosted", "linux", "x64", "gpu"} <= labels
    assert "cmake --preset cuda" in commands
    assert "cmake --build --preset cuda-release" in commands
    assert "ctest --test-dir build/presets/cuda" in commands
    assert "--no-tests=error" in commands
    assert "--output-junit" in commands and "GITHUB_WORKSPACE" in commands
    assert any(step.get("uses", "").startswith("actions/upload-artifact@") for step in job["steps"])


def test_linux_cuda_workflow_runs_cuda_memory_checker() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = document["jobs"]["cuda-linux"]
    commands = _commands(job)

    assert "compute-sanitizer" in commands
    assert "--tool memcheck" in commands
    assert "--error-exitcode=1" in commands
    assert "inferrt_test_cvcuda" in commands


def test_ci_ctest_evidence_paths_are_workspace_absolute() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    ctest_lines = [line for line in workflow_text.splitlines() if "--output-junit" in line]

    assert ctest_lines
    assert all("GITHUB_WORKSPACE" in line for line in ctest_lines)
    assert all("--no-tests=error" in workflow_text for _ in [0])


def test_ci_validates_each_ctest_layer_and_requires_named_targets() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    expected_cases = {
        "core-only-linux": ("inferrt_test_core", "inferrt_package_relocation"),
        "cuda-linux": ("inferrt_test_cvcuda", "inferrt_package_relocation"),
        "core-only-windows": ("inferrt_test_core", "inferrt_package_relocation"),
        "install-downstream-windows": ("inferrt_package_relocation",),
        "quality-and-sanitizer-linux": ("inferrt_test_core", "inferrt_package_relocation"),
        "full-release-gpu-windows": ("inferrt_test_features", "inferrt_package_relocation"),
    }

    for job_name, cases in expected_cases.items():
        commands = _commands(document["jobs"][job_name])
        assert "validate_test_report.py" in commands, job_name
        for case in cases:
            assert f"--require-case {case}" in commands, (job_name, case)


def test_ci_python_gate_validates_required_skips_and_real_model_cases() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    commands = _commands(document["jobs"]["full-release-gpu-windows"])

    assert "pytest.xml" in commands
    assert "--allow-skip-regex" in commands
    assert "backend (was|is) disabled" in commands
    assert "--require-case \"test_model_root_checkpoints_export_and_match_pytorch\"" in commands
    assert "--require-case \"test_sam_v1_pybind_matches_official_pytorch_forward\"" in commands


def test_ci_quality_layer_publishes_tool_report_and_summary() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    commands = _commands(document["jobs"]["quality-and-sanitizer-linux"])

    assert "--junitxml=artifacts/tools/pytest.xml" in commands
    assert "artifacts/tools/pytest-summary.json" in commands
