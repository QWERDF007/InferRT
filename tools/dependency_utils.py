"""Shared helpers for InferRT runtime dependency scripts."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR_NAMES = {"debug", "release", "relwithdebinfo", "minsizerel"}


def warn(message: str) -> None:
    """向标准错误输出警告信息。

    Args:
        message: 要输出的警告文本。
    """

    print(f"warning: {message}", file=sys.stderr)


def platform_key() -> str:
    """返回 dependencies.yaml 中使用的平台键名。

    Returns:
        str: ``windows``、``linux``、``macos`` 或当前 ``sys.platform``。
    """

    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def canonical_config(config: str) -> str:
    """规范化 CMake 构建配置名称。

    Args:
        config: 用户传入的构建配置名称。

    Returns:
        str: 小写后的配置名，例如 ``release`` 或 ``debug``。
    """

    return str(config or "Release").strip().lower()


def cmake_config_name(config: str) -> str:
    """返回常见 CMake 配置名称的标准大小写。

    Args:
        config: 用户传入的构建配置名称。

    Returns:
        str: 可用于路径模板的 CMake 配置名，例如 ``Release``。
    """

    names = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }
    return names.get(canonical_config(config), str(config or "Release"))


def resolve_project_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """把用户路径解析为绝对路径。

    Args:
        path: 绝对路径或相对仓库根目录的路径。
        repo_root: 仓库根目录，默认自动从脚本位置推断。

    Returns:
        Path: 解析后的绝对路径，路径不要求已经存在。
    """

    value = os.path.expandvars(os.path.expanduser(str(path)))
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (repo_root / raw).resolve(strict=False)


def normalize_list(value: Any) -> list[str]:
    """把 YAML 中的标量或列表统一转换为字符串列表。

    Args:
        value: YAML 字段值。

    Returns:
        list[str]: 规范化后的字符串列表。
    """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_yaml_file(path: Path) -> dict[str, Any]:
    """读取 YAML 文件并校验顶层对象类型。

    Args:
        path: YAML 文件路径。

    Returns:
        dict[str, Any]: YAML 顶层映射。

    Raises:
        RuntimeError: 当前 Python 环境没有安装 PyYAML。
        ValueError: YAML 顶层不是映射。
    """

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required. Run these scripts with "
            f"{Path(sys.executable)} or install pyyaml in your Python environment."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid YAML root in {path}")
    return data


def load_dependencies(path: Path) -> list[dict[str, Any]]:
    """读取依赖清单中的依赖条目列表。

    Args:
        path: ``dependencies.yaml`` 文件路径。

    Returns:
        list[dict[str, Any]]: 有效的依赖条目。
    """

    data = load_yaml_file(path)
    deps = data.get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError(f"dependencies must be a list in {path}")
    return [dep for dep in deps if isinstance(dep, dict)]


def dependency_matches_config(dep: dict[str, Any], config: str) -> bool:
    """判断依赖条目是否适用于当前构建配置。

    Args:
        dep: 依赖条目。
        config: 当前构建配置。

    Returns:
        bool: 适用时返回 ``True``。
    """

    dep_config = str(dep.get("config", "all")).strip().lower()
    return dep_config in {"", "all", "any"} or dep_config == canonical_config(config)


class _FormatVars(dict[str, str]):
    """路径模板变量字典，未知变量保持原样。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_dependency_value(value: str, config: str) -> str:
    """展开依赖路径模板中的少量内置变量。

    Args:
        value: YAML 中的路径模板。
        config: 当前构建配置。

    Returns:
        str: 展开后的路径字符串。
    """

    variables = _FormatVars(
        config=canonical_config(config),
        build_config=cmake_config_name(config),
        repo_root=str(REPO_ROOT).replace("\\", "/"),
    )
    return value.format_map(variables)


def dependency_patterns(dep: dict[str, Any], platform: str, config: str) -> list[str]:
    """取得当前平台需要处理的文件匹配模式。

    Args:
        dep: 依赖条目。
        platform: 当前平台键名。
        config: 当前构建配置。

    Returns:
        list[str]: 文件匹配模式列表。
    """

    patterns = normalize_list(dep.get("all")) + normalize_list(dep.get(platform))
    return [format_dependency_value(pattern, config) for pattern in patterns]


def dependency_destinations(dep: dict[str, Any]) -> list[str]:
    """取得链接阶段的目标目录列表。

    Args:
        dep: 依赖条目。

    Returns:
        list[str]: 目标目录列表。
    """

    return normalize_list(dep.get("destinations", dep.get("dest")))


def is_direct_root(value: str) -> bool:
    """判断 root 字段是否已经是路径而不是 CMake 变量名。

    Args:
        value: ``root`` 字段原始值。

    Returns:
        bool: 看起来像路径时返回 ``True``。
    """

    return (
        ":" in value
        or "/" in value
        or "\\" in value
        or value.startswith(".")
        or value.startswith("~")
    )


def read_cmake_cache_value(cache_file: Path, key: str) -> str | None:
    """从 CMakeCache.txt 读取变量值。

    Args:
        cache_file: ``CMakeCache.txt`` 路径。
        key: 变量名。

    Returns:
        str | None: 读取到的变量值；不存在或为 ``*-NOTFOUND`` 时返回 ``None``。
    """

    if not cache_file.exists():
        return None
    pattern = re.compile(rf"^{re.escape(key)}(?::[^=]+)?=(.*)$", re.IGNORECASE)
    for line in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            value = match.group(1).strip().strip('"')
            return None if value.endswith("-NOTFOUND") else value
    return None


def cmake_cache_bool(cache_file: Path, key: str) -> bool | None:
    """读取 CMakeCache.txt 中的 BOOL 选项。

    返回 ``None`` 表示缓存不存在或没有该选项；调用方可据此保留清单
    的默认启用行为。
    """

    value = read_cmake_cache_value(cache_file, key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"on", "true", "yes", "y", "1"}:
        return True
    if normalized in {"off", "false", "no", "n", "0"}:
        return False
    return None


def dependency_enabled(dep: dict[str, Any], build_dir: Path) -> bool:
    """判断依赖是否由当前构建配置启用。

    清单中的 ``requires`` 是 CMake BOOL 变量列表，所有变量都必须为 ON。
    没有缓存或没有对应变量时返回 ``True``，这样脚本也能用于尚未配置的
    构建目录，并继续通过根路径解析报告缺失依赖。
    """

    requirements = normalize_list(dep.get("requires"))
    if not requirements:
        return True
    cache_file = build_dir / "CMakeCache.txt"
    values = [cmake_cache_bool(cache_file, key) for key in requirements]
    if any(value is False for value in values):
        return False
    return True


def read_cmake_set(cmake_file: Path, name: str) -> str | None:
    """从简单的 CMake ``set(...)`` 语句读取变量值。

    Args:
        cmake_file: CMake 配置文件路径。
        name: 变量名。

    Returns:
        str | None: 变量值；未找到时返回 ``None``。
    """

    if not cmake_file.exists():
        return None
    pattern = re.compile(rf"^\s*set\s*\(\s*{re.escape(name)}\s+(.+?)\s*\)", re.IGNORECASE)
    for line in cmake_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rest = match.group(1).strip()
        if rest.startswith('"'):
            end = rest.find('"', 1)
            if end >= 0:
                return rest[1:end]
        return rest.split()[0].strip('"')
    return None


def read_cmake_set_expanded(cmake_file: Path, name: str) -> str | None:
    """读取 CMake 变量，并展开同文件内的简单 ``${VAR}`` 引用。

    Args:
        cmake_file: CMake 配置文件路径。
        name: 变量名。

    Returns:
        str | None: 展开后的变量值；未找到时返回 ``None``。
    """

    value = read_cmake_set(cmake_file, name)
    if not value:
        return None
    for _ in range(8):
        match = re.search(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
        if not match:
            break
        ref_value = read_cmake_set(cmake_file, match.group(1))
        if not ref_value:
            break
        value = value.replace(match.group(0), ref_value)
    return value


def _candidate_env_names(dep: dict[str, Any], root_spec: str) -> list[str]:
    """构造依赖 root 可尝试读取的环境变量名列表。

    Args:
        dep: 依赖条目。
        root_spec: ``root`` 字段中的变量名。

    Returns:
        list[str]: 去重后的环境变量名列表。
    """

    names = [root_spec]
    names.extend(normalize_list(dep.get("env")))
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.upper()
        if key not in seen:
            unique.append(name)
            seen.add(key)
    return unique


def resolve_dependency_root(
    dep: dict[str, Any],
    build_dir: Path,
    repo_root: Path = REPO_ROOT,
    platform: str | None = None,
) -> Path | None:
    """解析依赖条目的根目录。

    解析顺序为：直接路径、环境变量、YAML ``default`` 字段、
    ``CMakeCache.txt``、YAML 指定的 CMake 配置文件。设置 ``<platform>_root``
    时，该字段优先于通用 ``root``。清单默认值优先于无法静态展开的 CMake
    表达式。

    Args:
        dep: 依赖条目。
        build_dir: CMake 构建目录。
        repo_root: 仓库根目录。
        platform: 当前平台键名，用于选择平台专用 root 字段。

    Returns:
        Path | None: 解析到的根目录；无法解析时返回 ``None``。
    """

    platform_root = dep.get(f"{platform}_root") if platform else None
    root_spec = str(platform_root if platform_root not in (None, "") else dep.get("root", "")).strip()
    if not root_spec:
        return None

    if is_direct_root(root_spec) or not dep.get("cmake"):
        return resolve_project_path(root_spec, repo_root)

    for env_name in _candidate_env_names(dep, root_spec):
        value = os.environ.get(env_name)
        if value:
            return resolve_project_path(value, repo_root)

    # A platform-specific root describes a different filesystem location than
    # the dependency's generic installation root.  For example, OpenCV's
    # CMake package lives beside ``bin`` while Windows runtime DLLs live in
    # ``OpenCV_BIN_DIR``.  Once configure has resolved that location, the
    # cache value is the authoritative platform root; applying the generic
    # default here would point the runtime collector at the wrong directory.
    if platform_root not in (None, ""):
        cache_value = read_cmake_cache_value(build_dir / "CMakeCache.txt", root_spec)
        if cache_value:
            cache_root = resolve_project_path(cache_value, repo_root)
            if cache_root.exists():
                return cache_root

    default_value = dep.get("default")
    if default_value:
        candidate = resolve_project_path(str(default_value), repo_root)
        if candidate.exists():
            return candidate

    cache_value = read_cmake_cache_value(build_dir / "CMakeCache.txt", root_spec)
    if cache_value:
        return resolve_project_path(cache_value, repo_root)

    if default_value:
        return resolve_project_path(str(default_value), repo_root)

    cmake_file = resolve_project_path(str(dep["cmake"]), repo_root)
    cmake_value = read_cmake_set_expanded(cmake_file, root_spec)
    if cmake_value:
        return resolve_project_path(cmake_value, repo_root)

    return None


def expand_dependency_pattern(root: Path, pattern: str) -> list[Path]:
    """在依赖根目录下展开文件匹配模式。

    Args:
        root: 依赖根目录。
        pattern: 相对 ``root`` 的 glob 模式。

    Returns:
        list[Path]: 匹配到的普通文件列表。
    """

    expanded_pattern = os.path.expandvars(os.path.expanduser(pattern))
    matches = [Path(match).resolve(strict=False) for match in glob.glob(str(root / expanded_pattern))]
    matches = [path for path in matches if path.is_file()]
    if matches:
        return sorted(matches, key=lambda path: str(path).lower())
    if "*" not in expanded_pattern and "?" not in expanded_pattern:
        candidate = (root / expanded_pattern).resolve(strict=False)
        if candidate.is_file():
            return [candidate]
    return []


def build_dll_variant_sets(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    """根据候选 DLL 识别成对存在的 release/debug 文件名。

    只有 ``xxx.dll`` 和 ``xxxd.dll`` 同时存在时，才把 ``xxxd.dll`` 视为
    debug 变体，避免误判本身以 ``d`` 结尾的 release DLL。

    Args:
        paths: 候选文件列表。

    Returns:
        tuple[set[str], set[str]]: debug DLL 名称集合和对应 release DLL 名称集合。
    """

    names = {path.name.lower() for path in paths if path.suffix.lower() == ".dll"}
    debug_names: set[str] = set()
    release_names: set[str] = set()
    for name in names:
        if not name.endswith("d.dll"):
            continue
        release_name = f"{name[:-5]}.dll"
        if release_name in names:
            debug_names.add(name)
            release_names.add(release_name)
    return debug_names, release_names


def dll_matches_config(path: Path, config: str, debug_names: set[str], release_names: set[str]) -> bool:
    """判断 DLL 是否匹配目标构建配置。

    Args:
        path: 候选文件路径。
        config: 目标构建配置。
        debug_names: 已识别的 debug DLL 名称集合。
        release_names: 已识别的 release DLL 名称集合。

    Returns:
        bool: 文件应被处理时返回 ``True``。
    """

    name = path.name.lower()
    if not name.endswith(".dll"):
        return True
    if canonical_config(config) == "debug":
        return name not in release_names
    return name not in debug_names


def remove_existing_file(path: Path) -> None:
    """删除已存在的文件或文件链接，拒绝覆盖真实目录。

    Args:
        path: 要删除的目标路径。

    Raises:
        RuntimeError: 目标是非链接目录时抛出。
    """

    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"existing path is a directory, refusing to overwrite as file: {path}")
    path.unlink()


def copy_file(source: Path, destination: Path) -> None:
    """复制单个文件，并用真实文件替换目标文件链接。

    Args:
        source: 源文件路径。
        destination: 目标文件路径。
    """

    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_symlink() and source == destination.resolve(strict=False):
        print(f"skip existing {destination}")
        return
    remove_existing_file(destination)
    shutil.copy2(source, destination)
    print(f"copy {source} -> {destination}")


def link_file(source: Path, link: Path, mode: str = "symlink") -> None:
    """创建文件链接，失败时自动回退。

    Args:
        source: 源文件路径。
        link: 目标链接路径。
        mode: ``symlink``、``hardlink`` 或 ``copy``。
    """

    source = source.resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    normalized_mode = mode.lower()
    if normalized_mode != "copy":
        try:
            if link.exists() and source.samefile(link):
                print(f"skip existing {link}")
                return
        except OSError:
            pass

    if normalized_mode == "copy":
        copy_file(source, link)
        return

    remove_existing_file(link)
    if normalized_mode == "hardlink":
        try:
            os.link(source, link)
            print(f"create hardlink {link} -> {source}")
        except OSError:
            copy_file(source, link)
        return

    try:
        os.symlink(source, link)
        print(f"create symlink {link} -> {source}")
    except OSError:
        try:
            os.link(source, link)
            print(f"create hardlink {link} -> {source}")
        except OSError:
            copy_file(source, link)


def remove_existing_dir_link(path: Path) -> None:
    """删除已存在的目录链接或 Windows junction。

    Args:
        path: 目标目录链接路径。

    Raises:
        RuntimeError: 目标是真实目录时抛出。
    """

    if not path.exists() and not path.is_symlink():
        return
    is_junction = bool(getattr(path, "is_junction", lambda: False)())
    if is_junction:
        path.rmdir()
        return
    if path.is_symlink():
        path.unlink()
        return
    raise RuntimeError(f"existing path is not a link, refusing to overwrite: {path}")


def link_dir(source: Path, link: Path) -> None:
    """创建目录符号链接，Windows 下失败时回退为 junction。

    Args:
        source: 源目录路径。
        link: 目标目录链接路径。
    """

    source = source.resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link.exists() and source.samefile(link):
            print(f"skip existing {link}")
            return
    except OSError:
        pass
    remove_existing_dir_link(link)
    try:
        os.symlink(source, link, target_is_directory=True)
        print(f"create symlink {link} -> {source}")
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip() or f"failed to create junction {link} -> {source}")
        print(f"create junction {link} -> {source}")


def should_skip_config_dir(path: Path) -> bool:
    """判断相对路径是否位于 CMake 配置名目录中。

    Args:
        path: 相对路径。

    Returns:
        bool: 位于 ``Debug`` / ``Release`` 等目录中时返回 ``True``。
    """

    return any(part.lower() in CONFIG_DIR_NAMES for part in path.parts)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """按出现顺序去重路径。

    Args:
        paths: 输入路径序列。

    Returns:
        list[Path]: 去重后的路径列表。
    """

    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        result.append(path)
        seen.add(key)
    return result
