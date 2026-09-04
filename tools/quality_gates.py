"""Static release gates for the InferRT source and generated package.

The checks intentionally operate on the source tree and a caller-selected
install prefix.  They do not inspect build logs or silently turn missing
artifacts into a pass; a non-empty finding list exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PUBLIC_ROOTS = (
    Path("src/core/include"),
    Path("src/model/include"),
    Path("src/engine/include"),
    Path("src/cvcuda/include"),
    Path("src/features/include"),
    Path("src/ops/include"),
    Path("src/util/include"),
)
PUBLIC_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
SOURCE_SUFFIXES = {".cmake", ".txt", ".yml", ".yaml"}
CUDA_SUFFIXES = {".cu", ".cpp", ".cxx", ".cc"}
GENERATED_ARTIFACT_NAMES = {"CMakeCache.txt", "cmake_install.cmake", "install_manifest.txt"}
GENERATED_ARTIFACT_SUFFIXES = {
    ".a",
    ".dll",
    ".dylib",
    ".exe",
    ".exp",
    ".ilk",
    ".lib",
    ".log",
    ".o",
    ".obj",
    ".pdb",
    ".pyc",
    ".sln",
    ".so",
    ".vcxproj",
}
GENERATED_ARTIFACT_DIRECTORIES = {"__pycache__", "CMakeFiles"}


@dataclass(frozen=True)
class Finding:
    gate: str
    path: str
    line: int
    message: str


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _cmake_without_comments(path: Path) -> str:
    """Return CMake source with full-line comments removed.

    This keeps command detection useful for multi-line invocations while
    allowing explanatory comments to mention the deployment script.
    """

    return "\n".join(
        "" if line.lstrip().startswith("#") else line
        for line in _lines(path)
    )


def scan_public_headers(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    forbidden = re.compile(r"(?:NvInfer\.h|nvinfer1::|\bpriv::|::priv\b|/priv/|\\priv\\)")
    shape_matcher_public_root = root / "src/features/include/inferrt/features"
    for relative_root in PUBLIC_ROOTS:
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in PUBLIC_SUFFIXES:
                continue
            try:
                shape_matcher_relative = path.relative_to(shape_matcher_public_root)
            except ValueError:
                shape_matcher_relative = None
            if shape_matcher_relative is not None and (
                shape_matcher_relative.parts[:1] in (("v0",), ("v1",), ("v2",))
                or path.name == "ShapeTemplateMatcherBase.hpp"
            ):
                findings.append(
                    Finding(
                        "public-header",
                        str(path.relative_to(root)),
                        1,
                        "ISA-specific shape matcher headers are not part of the public interface",
                    )
                )
                continue
            for number, line in enumerate(_lines(path), 1):
                if forbidden.search(line):
                    findings.append(
                        Finding("public-header", str(path.relative_to(root)), number, "backend/private type leaked")
                    )
    return findings


def _git_tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve().as_posix()}",
            "-C",
            str(root),
            "ls-files",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value]


def scan_source_artifacts(root: Path, tracked_paths: Iterable[Path] | None = None) -> list[Finding]:
    """Reject generated build/runtime files that are part of the Git deliverable."""

    try:
        paths = list(tracked_paths) if tracked_paths is not None else _git_tracked_paths(root)
    except (OSError, subprocess.CalledProcessError) as error:
        return [Finding("source-artifact", str(root), 0, f"cannot enumerate Git tracked files: {error}")]

    findings: list[Finding] = []
    for relative in sorted(paths, key=lambda value: value.as_posix().lower()):
        normalized = Path(relative.as_posix())
        if not (root / normalized).is_file():
            continue
        if (
            normalized.name in GENERATED_ARTIFACT_NAMES
            or normalized.suffix.lower() in GENERATED_ARTIFACT_SUFFIXES
            or any(part in GENERATED_ARTIFACT_DIRECTORIES for part in normalized.parts)
        ):
            findings.append(
                Finding("source-artifact", normalized.as_posix(), 0, "generated build/runtime artifact is tracked")
            )
    return findings


def _cmake_files(root: Path) -> list[Path]:
    paths = [root / "CMakeLists.txt"]
    paths.append(root / "3rdparty" / "CMakeLists.txt")
    paths.extend(sorted((root / "cmake").rglob("*.cmake")))
    paths.extend(sorted((root / "src").rglob("CMakeLists.txt")))
    paths.extend(sorted((root / "tests").rglob("CMakeLists.txt")))
    paths.extend(sorted((root / "benchmark").rglob("CMakeLists.txt")))
    paths.extend(sorted((root / "samples").rglob("CMakeLists.txt")))
    return [path for path in paths if path.is_file()]


def scan_cmake(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    absolute_path = re.compile(r"(?i)(?:[A-Z]:[\\/](?:Projects|Users|Software)|/home/|/workspace/)")
    force = re.compile(r"CACHE\s+[^\n]*\sFORCE\b", re.IGNORECASE)
    global_flags = re.compile(r"\bCMAKE_(?:CXX|CUDA|EXE_LINKER|SHARED_LINKER|STATIC_LINKER)_FLAGS(?:_[A-Z]+)?\s*=")
    global_definitions = re.compile(r"\badd_(?:compile_)?definitions\s*\(", re.IGNORECASE)
    glob = re.compile(r"file\s*\(\s*GLOB(?:_RECURSE)?\b", re.IGNORECASE)
    runtime_copy = re.compile(
        r"(?:copy_if_different|copy_directory|file\s*\(\s*copy\b|cmake\s+-e\s+copy(?:_directory)?\b)",
        re.IGNORECASE,
    )
    for path in _cmake_files(root):
        relative = path.relative_to(root)
        lines = _lines(path)
        for number, line in enumerate(lines, 1):
            if absolute_path.search(line):
                findings.append(Finding("cmake-path", str(relative), number, "developer-machine absolute path"))
            if force.search(line):
                findings.append(Finding("cmake-cache", str(relative), number, "CACHE FORCE is not allowed"))
            if global_flags.search(line):
                findings.append(Finding("cmake-flags", str(relative), number, "global compiler/linker flags assignment"))
            if global_definitions.search(line):
                findings.append(Finding("cmake-flags", str(relative), number,
                                       "compile definitions must be attached to a target"))
            if glob.search(line) and "CONFIGURE_DEPENDS" not in line:
                # The relocation verifier intentionally enumerates generated
                # install files and is the one read-only exception.
                if relative.as_posix() != "tests/packaging/verify_relocation.cmake":
                    findings.append(Finding("cmake-glob", str(relative), number, "GLOB requires CONFIGURE_DEPENDS"))
            if runtime_copy.search(line):
                findings.append(
                    Finding(
                        "cmake-runtime-copy",
                        str(relative),
                        number,
                        "runtime files must be deployed by tools/package_runtime_dlls.py",
                    )
                )
        runtime_deploy = re.compile(
            r"\b(?:add_custom_(?:target|command)|execute_process)\s*\(.*?package_runtime_dlls\.py",
            re.IGNORECASE | re.DOTALL,
        )
        source_without_comments = _cmake_without_comments(path)
        for match in runtime_deploy.finditer(source_without_comments):
            line = source_without_comments.count("\n", 0, match.start()) + 1
            findings.append(
                Finding(
                    "cmake-runtime-deploy",
                    str(relative),
                    line,
                    "runtime deployment must be invoked explicitly by the user",
                )
            )
    return findings


def scan_cuda_allocations(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    allowed_fragments = ("Allocator", "Workspace", "TRTBuffers", "Buffers")
    for path in sorted((root / "src").rglob("*")):
        if path.suffix.lower() not in CUDA_SUFFIXES or not path.is_file():
            continue
        relative = path.relative_to(root)
        source_text = "\n".join(_lines(path))
        sibling_text = ""
        for sibling in (path.with_suffix(".hpp"), path.with_suffix(".h")):
            if sibling.is_file():
                sibling_text += "\n" + "\n".join(_lines(sibling))
        # Raw calls are allowed only inside a module that visibly owns the
        # allocation with an RAII object.  The owner itself remains covered by
        # the C++ type/destructor review and by the focused runtime tests.
        has_named_owner = any(fragment.lower() in path.name.lower() for fragment in allowed_fragments)
        has_explicit_owner = (
            "cudaMalloc" in source_text
            and (
                "std::unique_ptr" in source_text
                or "~DeviceBuffer" in source_text
                or "std::unique_ptr" in sibling_text
                or (
                    re.search(r"\bclass\s+Workspace\b", sibling_text)
                    and re.search(r"~Workspace\s*\(|\bWorkspace::reset\s*\(", source_text)
                )
            )
        )
        if has_named_owner or has_explicit_owner:
            continue
        for number, line in enumerate(_lines(path), 1):
            if re.search(r"\bcuda(?:Malloc|Free)(?:Async)?\s*\(", line):
                findings.append(Finding("cuda-allocation", str(relative), number, "allocation must use an RAII adapter"))
    return findings


def scan_installed_configs(install_dir: Path, source_dir: Path | None, build_dir: Path | None) -> list[Finding]:
    findings: list[Finding] = []
    if not install_dir.is_dir():
        return [Finding("install-config", str(install_dir), 0, "install directory does not exist")]
    forbidden_paths = [path for path in (source_dir, build_dir) if path is not None]
    normalized = [path.resolve().as_posix().lower() for path in forbidden_paths]
    for path in sorted(install_dir.rglob("*.cmake")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.replace("\\", "/").lower()
        for candidate in normalized:
            if candidate and candidate in lowered:
                findings.append(Finding("install-config", str(path), 0, f"contains build/source path: {candidate}"))
    return findings


def run_gates(root: Path, install_dir: Path | None = None, source_dir: Path | None = None,
              build_dir: Path | None = None) -> list[Finding]:
    findings = scan_source_artifacts(root)
    findings.extend(scan_public_headers(root))
    findings.extend(scan_cmake(root))
    findings.extend(scan_cuda_allocations(root))
    if install_dir is not None:
        findings.extend(scan_installed_configs(install_dir, source_dir, build_dir))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)

    findings = run_gates(args.root.resolve(), args.install_dir.resolve() if args.install_dir else None,
                         args.source_dir.resolve() if args.source_dir else None,
                         args.build_dir.resolve() if args.build_dir else None)
    payload = {"passed": not findings, "finding_count": len(findings),
               "findings": [asdict(finding) for finding in findings]}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
