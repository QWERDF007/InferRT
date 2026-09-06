"""Link InferRT runtime dependencies declared in tools/dependencies.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dependency_utils import (
    build_dll_variant_sets,
    dependency_destinations,
    dependency_enabled,
    dependency_matches_config,
    dependency_patterns,
    dll_matches_config,
    expand_dependency_pattern,
    link_file,
    load_dependencies,
    platform_key,
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

    parser = argparse.ArgumentParser(description="Link InferRT runtime dependencies.")
    parser.add_argument("--build-dir", "-BuildDir", default="build")
    parser.add_argument("--config", "-Config", default="Release")
    parser.add_argument("--dependencies", default="tools/dependencies.yaml")
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="Default creates symlink, then falls back to hardlink/copy.",
    )
    return parser.parse_args()


def _matched_files(dep: dict, root: Path, platform: str, config: str) -> list[Path]:
    """展开单个依赖条目的文件列表。

    Args:
        dep: 依赖条目。
        root: 依赖根目录。
        platform: 当前平台键名。
        config: 当前构建配置。

    Returns:
        list[Path]: 去重后的匹配文件列表。
    """

    matched: list[Path] = []
    for pattern in dependency_patterns(dep, platform, config):
        matches = expand_dependency_pattern(root, pattern)
        if not matches and "*" not in pattern and "?" not in pattern:
            warn(f"dependency file was not found: {root / pattern}")
        matched.extend(matches)
    return unique_paths(matched)


def link_dependencies(build_dir: Path, dependency_file: Path, config: str, mode: str) -> int:
    """按 YAML 清单把运行时依赖链接到目标目录。

    Args:
        build_dir: CMake 构建目录。
        dependency_file: 依赖清单路径。
        config: 构建配置。
        mode: 链接模式，支持 ``symlink``、``hardlink`` 和 ``copy``。

    Returns:
        int: 成功处理的目标文件数量。
    """

    platform = platform_key()
    processed = 0

    for dep in load_dependencies(dependency_file):
        if not dependency_enabled(dep, build_dir):
            continue
        if not dependency_matches_config(dep, config):
            continue

        destinations = [resolve_project_path(dest) for dest in dependency_destinations(dep)]
        if not destinations:
            warn(f"skip dependency {dep.get('name', '<unnamed>')}, no destinations configured")
            continue

        root = resolve_dependency_root(dep, build_dir, platform=platform)
        if root is None:
            warn(f"skip dependency {dep.get('name', '<unnamed>')}, root {dep.get('root')} was not found")
            continue
        if not root.exists():
            warn(f"skip dependency {dep.get('name', '<unnamed>')}, missing root: {root}")
            continue

        matched = _matched_files(dep, root, platform, config)
        debug_names, release_names = build_dll_variant_sets(matched)
        for runtime in matched:
            if platform == "windows" and not dll_matches_config(runtime, config, debug_names, release_names):
                continue
            for destination in destinations:
                link_file(runtime, destination / runtime.name, mode)
                processed += 1

    return processed


def main() -> int:
    """执行依赖链接流程。

    Returns:
        int: 进程退出码。
    """

    args = parse_args()
    build_dir = resolve_project_path(args.build_dir)
    dependency_file = resolve_project_path(args.dependencies)
    if not dependency_file.is_file():
        raise RuntimeError(f"dependency file does not exist: {dependency_file}")

    processed = link_dependencies(build_dir, dependency_file, args.config, args.mode)
    print(f"link runtime dependencies complete. Files processed: {processed}")
    print(f"Destination root: {build_dir / 'bin'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
