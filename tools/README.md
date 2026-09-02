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
- `--mode copy` 复制真实文件，并替换目标处已有的符号链接

需要把构建目录作为可交付运行目录时，使用 `--mode copy`，避免程序依赖开发机绝对路径或符号链接权限：

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\link_dependencies.py --config Release --mode copy
```

## 2. 打包运行时文件

用于把 `build/bin` 里的 InferRT 运行库（DLL）、`.pyd` 和第三方依赖部署到安装目录。

可执行文件等 InferRT 自身安装目标仍由 `cmake --install` 安装；本脚本不参与
CMake 构建或安装钩子。

发布时按以下顺序执行：CMake 只安装 InferRT 自身产物，第三方运行库由用户显式运行 Python 脚本部署。

`cmake` 配置、构建和 CTest 注册不会调用本脚本，也不会从依赖根目录复制
DLL；CTest 只把构建输出目录加入进程搜索路径。运行 CTest 或发布程序前，
请先完成下面对应的显式部署步骤。

```powershell
cmake --install build\presets\full --config Release --prefix InferRT-0.0.2
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py `
    --build-dir build\presets\full `
    --install-dir InferRT-0.0.2 `
    --config Release `
    --strict
```

发布包的 CMake 配置默认只解析无第三方依赖的 `InferRT::core`。需要其他模块时，
在 `find_package` 中显式请求组件（例如 `COMPONENTS model`），并提供该组件所需的
第三方开发根；运行时 DLL 仍由上面的打包脚本部署。

ONNX Runtime 和 OpenVINO 是可选后端。是否构建由 CMake preset 决定；只有实际启用
对应后端的包才按 `tools/dependencies.yaml` 的 `requires` 条件部署运行库。只需
core-only 或 cuda 能力时，不需要部署这两个后端的运行库；后端验收以可行性验证为准，入口见
[`tests/python/test_backend_runtime.py`](../tests/python/test_backend_runtime.py)。

```cmake
find_package(InferRT CONFIG REQUIRED) # core-only consumer
# 或：find_package(InferRT CONFIG REQUIRED COMPONENTS model)
```

CTest 运行前也要显式把同一套第三方运行库部署到构建输出目录；CMake 不会
在构建钩子中复制 DLL。Windows 下这样可以避免系统目录中的同名 DLL 被加载：

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py `
    --build-dir build\presets\full `
    --install-bin-dir build\presets\full\bin `
    --config Release `
    --strict
```

```powershell
& 'D:\Software\anaconda3\envs\py312\python.exe' tools\package_runtime_dlls.py
```

默认规则：

- 安装目录优先读取 `build/CMakeCache.txt` 里的 `CMAKE_INSTALL_PREFIX`
- 如果没有，回退到仓库根目录下的 `InferRT-0.0.2`
- 实际复制目标是安装目录下的 `bin`
- 只复制 `build/bin` 下的 `inferrt*.dll` 和 `inferrt*_py*.pyd`
- 如果构建缓存包含 `PYTHON_MODULE_EXTENSION`，Python 扩展只部署与该 ABI 匹配的 `.pyd`
- 其余第三方运行时依赖从 `tools/dependencies.yaml` 读取
- `--strict` 会在已启用依赖的根目录或清单模式缺失时失败，发布验收必须使用该选项
- `--strict` 还会在构建输出目录或项目运行库为空时失败；普通模式只报告警告

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
| `requires` | 可选，CMake BOOL 变量列表；全部为 OFF 时跳过该依赖 |
| `cmake` | 可选，用于读取对应的 CMake 配置文件 |
| `root` | 依赖根目录，可写直接路径、环境变量名或 CMake 变量名 |
| `<platform>_root` | 可选，覆盖指定平台的 `root`，例如 `windows_root` |
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
