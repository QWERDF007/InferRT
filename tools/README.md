# 工具说明

`tools/` 目录里的依赖链接和运行时打包流程已经改成 Python 脚本。
默认使用的 Python 环境是：

`D:\Software\anaconda3\envs\py312\python.exe`

脚本依赖 `pyyaml` 读取 `tools/dependencies.yaml`。

## 1. 链接依赖

开发阶段使用，把第三方运行时文件链接到 `build/bin`。

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py
```

常用参数：

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py --config Release
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py --config Debug
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py --mode hardlink
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py --mode copy
```

说明：

- 默认目标是 `build/bin`
- 默认只处理 `release` 配置
- `--mode symlink` 先尝试符号链接，再回退到硬链接和复制
- `--mode hardlink` 只尝试硬链接，失败后复制
- `--mode copy` 直接复制

## 2. 打包运行时文件

用于把 `build/bin` 里的可执行文件、`.pyd` 和第三方依赖复制到安装目录。

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py
```

默认规则：

- 安装目录优先读取 `build/CMakeCache.txt` 里的 `CMAKE_INSTALL_PREFIX`
- 如果没有，回退到仓库根目录下的 `InferRT-0.0.1`
- 实际复制目标是安装目录下的 `bin`
- 只复制 `build/bin` 下的 `inferrt*.dll` 和 `inferrt*_py*.pyd`
- 其余第三方运行时依赖从 `tools/dependencies.yaml` 读取

常用参数：

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py --install-dir install
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py --install-bin-dir install\bin
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py --skip-dependencies
```

## 3. 依赖清单

`tools/dependencies.yaml` 同时给链接脚本和打包脚本使用。

字段说明：

| 字段 | 说明 |
| --- | --- |
| `name` | 依赖名称，只用于日志 |
| `config` | 可选，`release` 或 `debug` |
| `cmake` | 可选，用于读取对应的 CMake 配置文件 |
| `root` | 依赖根目录，可写直接路径、环境变量名或 CMake 变量名 |
| `default` | `root` 解析失败时的回退路径 |
| `destinations` | 链接目标目录列表 |
| `windows` / `linux` / `macos` | 当前平台的文件模式列表 |
| `all` | 所有平台都使用的文件模式列表 |

文件模式支持 `*` 和 `?`。

## 4. Python 环境

建议直接使用完整路径运行脚本：

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' --version
& 'D:\Software\anaconda3\envs\py312\python.exe' -c "import yaml; print(yaml.__version__)"
```

如果当前环境没有 `pyyaml`，脚本会直接报错。
