"""Copy InferRT runtime files and YAML dependencies into the install bin directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dependency_utils import (
    build_dll_variant_sets,
    cmake_config_name,
    copy_file,
    dependency_matches_config,
    dependency_patterns,
    dll_matches_config,
    expand_dependency_pattern,
    load_dependencies,
    platform_key,
    read_cmake_cache_value,
    resolve_dependency_root,
    resolve_project_path,
    unique_paths,
    warn,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="Package InferRT runtime DLLs and Python modules.")
    parser.add_argument("--build-dir", "-BuildDir", default="build")
    parser.add_argument("--config", "-Config", default="Release")
    parser.add_argument("--install-dir", "-InstallDir", default="")
    parser.add_argument("--install-bin-dir", "-InstallBinDir", default="")
    parser.add_argument("--dependencies", default="tools/dependencies.yaml")
    parser.add_argument("--skip-dependencies", action="store_true")
    return parser.parse_args()


def default_install_dir(build_dir: Path) -> Path:
    """推断默认安装目录。

    Args:
        build_dir: CMake 构建目录。

    Returns:
        Path: ``CMAKE_INSTALL_PREFIX`` 或仓库下的 ``InferRT-0.0.1``。
    """

    install_prefix = read_cmake_cache_value(build_dir / "CMakeCache.txt", "CMAKE_INSTALL_PREFIX")
    if install_prefix:
        return resolve_project_path(install_prefix)
    return resolve_project_path("InferRT-0.0.1")


def resolve_install_bin_dir(build_dir: Path, install_dir_arg: str, install_bin_dir_arg: str) -> Path:
    """解析安装包 bin 目录。

    Args:
        build_dir: CMake 构建目录。
        install_dir_arg: 命令行传入的安装根目录。
        install_bin_dir_arg: 命令行传入的安装 bin 目录。

    Returns:
        Path: 运行时文件复制目标目录。
    """

    if install_bin_dir_arg:
        return resolve_project_path(install_bin_dir_arg)
    install_dir = resolve_project_path(install_dir_arg) if install_dir_arg else default_install_dir(build_dir)
    return install_dir / "bin"


def project_runtime_files(build_dir: Path, config: str) -> list[Path]:
    """收集 InferRT 自身运行时产物。

    Args:
        build_dir: CMake 构建目录。
        config: 构建配置。

    Returns:
        list[Path]: ``inferrt*.dll`` 和 ``inferrt*_py*.pyd`` 文件列表。
    """

    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        warn(f"skip project runtime files, missing {bin_dir}")
        return []

    candidates = list(bin_dir.glob("inferrt*.dll"))
    candidates.extend(bin_dir.glob("inferrt*_py*.pyd"))
    debug_names, release_names = build_dll_variant_sets(candidates)
    result: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.name.lower()):
        if path.suffix.lower() == ".dll" and not dll_matches_config(path, config, debug_names, release_names):
            continue
        result.append(path)
    return unique_paths(result)


def dependency_files(build_dir: Path, dependency_file: Path, config: str) -> list[Path]:
    """按 YAML 清单收集第三方运行时依赖文件。

    Args:
        build_dir: CMake 构建目录。
        dependency_file: 依赖清单路径。
        config: 构建配置。

    Returns:
        list[Path]: 去重后的依赖文件列表。
    """

    platform = platform_key()
    files: list[Path] = []
    for dep in load_dependencies(dependency_file):
        if not dependency_matches_config(dep, config):
            continue

        root = resolve_dependency_root(dep, build_dir)
        if root is None:
            warn(f"skip dependency {dep.get('name', '<unnamed>')}, root {dep.get('root')} was not found")
            continue
        if not root.exists():
            warn(f"skip dependency {dep.get('name', '<unnamed>')}, missing root: {root}")
            continue

        matched: list[Path] = []
        for pattern in dependency_patterns(dep, platform, config):
            matches = expand_dependency_pattern(root, pattern)
            if not matches and "*" not in pattern and "?" not in pattern:
                warn(f"dependency file was not found: {root / pattern}")
            matched.extend(matches)

        debug_names, release_names = build_dll_variant_sets(matched)
        for runtime in matched:
            if platform == "windows" and not dll_matches_config(runtime, config, debug_names, release_names):
                continue
            files.append(runtime)

    return unique_paths(files)


def copy_runtime_files(files: list[Path], destination: Path) -> int:
    """复制运行时文件到目标目录。

    Args:
        files: 源文件列表。
        destination: 目标目录。

    Returns:
        int: 复制或跳过的文件数量。
    """

    destination.mkdir(parents=True, exist_ok=True)
    processed = 0
    seen_names: set[str] = set()
    for source in files:
        key = source.name.lower()
        if key in seen_names:
            continue
        copy_file(source, destination / source.name)
        seen_names.add(key)
        processed += 1
    return processed


def main() -> int:
    """执行运行时文件打包流程。

    Returns:
        int: 进程退出码。
    """

    args = parse_args()
    build_dir = resolve_project_path(args.build_dir)
    config = cmake_config_name(args.config)
    dependency_file = resolve_project_path(args.dependencies)
    install_bin_dir = resolve_install_bin_dir(build_dir, args.install_dir, args.install_bin_dir)

    files = project_runtime_files(build_dir, config)
    if args.skip_dependencies:
        print("skip dependencies")
    elif dependency_file.is_file():
        files.extend(dependency_files(build_dir, dependency_file, config))
    else:
        warn(f"skip dependencies, missing {dependency_file}")

    processed = copy_runtime_files(unique_paths(files), install_bin_dir)
    print(f"runtime package complete. Files processed: {processed}")
    print(f"Destination: {install_bin_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
