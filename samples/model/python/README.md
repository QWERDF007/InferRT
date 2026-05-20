# Python 模型绑定示例

这个目录展示如何使用 `pybind11` 导出的 `inferrt_model_py` 模块在 Python 中创建模型、执行分类推理，以及导出中间特征。

## 1. 配置与构建

推荐直接使用 `D:/Software/anaconda3/envs/py312` 作为 Python 环境：

```bash
cmake -S . -B build -DINFERRT_BUILD_PYTHON=ON -DINFERRT_PYTHON_ROOT=D:/Software/anaconda3/envs/py312
cmake --build build --config Debug --target inferrt_model_py
```

构建完成后，模块默认位于：

- `build/lib/inferrt_model_py.cp312-win_amd64.pyd`（Windows）

同时，运行时依赖的 InferRT DLL 位于：

- `build/bin/`

## 2. 运行分类示例

```bash
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/SamplePythonClassification.py
```

也可以显式指定模型和权重：

```bash
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/SamplePythonClassification.py ^
  --model resnet18 ^
  --weights samples/model/classification/resnet18.wts ^
  --image assets/pics/dog.jpg ^
  --labels assets/imagenet1000_clsidx_to_labels.txt ^
  --build-dir build_py312_final
```

## 3. 运行特征提取示例

```bash
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/python_feature_extract.py ^
  --build-dir build_py312_final ^
  --model resnet18 ^
  --weights samples/model/classification/resnet18.wts ^
  --features layer1,layer4 ^
  --image assets/pics/dog.jpg ^
  --output-dir build/feature_dump_py
```

输出目录会生成：

- `manifest.txt`
- `input.bin`
- 每个请求特征对应一个 `.bin` 文件

## 4. 说明

- Python 示例会自动把 `build/bin` 加入 `sys.path`，以便导入构建出的模块。
- 如果使用了非默认构建目录，记得通过 `--build-dir` 指向对应目录。
- 脚本会尝试从构建目录的 `CMakeCache.txt` 中推断 TensorRT / OpenCV DLL 目录，并补充 `CUDA_PATH/bin`。
- 示例侧预处理与现有 C++ 分类 sample 保持一致，输入为 `1x3x224x224` 的 `float32` 张量。
- 若首次运行没有对应 `.engine` 文件，模块会先从 `.wts` 构建 engine，之后优先复用已有 engine。
- `python_feature_extract.py` 会创建 `feature_only=True` 的模型配置，并调用 `forward_features()` 导出中间张量。
