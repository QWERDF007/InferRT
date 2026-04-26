# AlexNet Sample

这个目录提供了基于 `torchvision` 权重导出的 AlexNet 分类示例。

## 文件说明

- [gen_wts.py](/d:/Project/InferRT/samples/model/alexnet/gen_wts.py): 生成 `.wts` 权重文件
- [main.cpp](/d:/Project/InferRT/samples/model/alexnet/main.cpp): TensorRT 推理示例
- [CMakeLists.txt](/d:/Project/InferRT/samples/model/alexnet/CMakeLists.txt): sample 构建脚本

## 1. 生成权重

建议在安装了 `torch`, `torchvision`, `opencv-python` 的 Python 环境中执行：

```bash
cd samples/model/alexnet
python gen_wts.py
```

仅验证 PyTorch 预测结果而不导出权重：

```bash
python gen_wts.py -p
```

默认会生成：

```text
alexnet.wts
```

## 2. 编译示例

在工程根目录执行：

```bash
cmake --build build --config Debug --target inferrt_sample_alexnet
```

生成的可执行文件位于：

```text
build/bin/inferrt_sample_alexnet.exe
```

## 3. 运行推理

命令格式：

```bash
build/bin/inferrt_sample_alexnet.exe <weights_file.wts> <image_path> [label_file]
```

示例：

```bash
build/bin/inferrt_sample_alexnet.exe samples/model/alexnet/alexnet.wts assets/pics/dog.jpg
build/bin/inferrt_sample_alexnet.exe samples/model/alexnet/alexnet.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

程序会完成：

1. 读取 `.wts` 文件并构建或加载 TensorRT engine
2. 对输入图像执行 ImageNet 风格预处理
3. 输出 Top-3 分类结果

## 4. 预处理说明

示例中的图像预处理与 `gen_wts.py` 保持一致：

- BGR -> RGB
- resize 到 `224x224`
- 归一化到 `[0, 1]`
- 使用 ImageNet `mean/std`
- 转为 `NCHW`

## 5. 标签文件

可选标签文件通常使用：

```text
assets/imagenet1000_clsidx_to_labels.txt
```

如果不提供标签文件，程序仍会输出类别索引和对应分数。

## 6. 注意事项

- 当前输入尺寸固定为 `1x3x224x224`
- 首次运行会从 `.wts` 构建 engine，耗时通常明显高于后续直接加载 `.engine`
