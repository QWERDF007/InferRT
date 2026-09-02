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
