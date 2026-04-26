# Model Samples

这里汇总了 `InferRT` 当前提供的分类模型示例。

## 当前可用示例

- [alexnet](./alexnet/README.md)
- [resnet](./resnet/README.md)

## 目录结构

```text
samples/model/
├─ alexnet/
│  ├─ CMakeLists.txt
│  ├─ gen_wts.py
│  ├─ main.cpp
│  └─ README.md
└─ resnet/
   ├─ CMakeLists.txt
   ├─ gen_wts.py
   ├─ main.cpp
   └─ README.md
```

## 通用流程

每个 sample 基本都分为两步：

1. 使用 `torchvision` 预训练模型生成 `.wts`
2. 使用对应的 sample 可执行文件加载 `.wts` 并运行 TensorRT 推理

## 1. 构建所有 model samples

在工程根目录执行：

```bash
cmake --build build --config Debug --target inferrt_sample_alexnet inferrt_sample_resnet
```

如果只想构建某一个 sample，也可以分别执行：

```bash
cmake --build build --config Debug --target inferrt_sample_alexnet
cmake --build build --config Debug --target inferrt_sample_resnet
```

## 2. AlexNet

文档入口：

- [AlexNet Sample README](./alexnet/README.md)

生成权重：

```bash
cd samples/model/alexnet
python gen_wts.py
```

运行推理：

```bash
build/bin/inferrt_sample_alexnet.exe samples/model/alexnet/alexnet.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

## 3. ResNet

文档入口：

- [ResNet Sample README](./resnet/README.md)

当前支持：

- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`

生成权重：

```bash
cd samples/model/resnet
python gen_wts.py -m resnet18
python gen_wts.py -m resnet34
python gen_wts.py -m resnet50
python gen_wts.py -m resnet101
python gen_wts.py -m resnet152
```

运行推理：

```bash
build/bin/inferrt_sample_resnet.exe resnet18 samples/model/resnet/resnet18.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_resnet.exe resnet50 samples/model/resnet/resnet50.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_resnet.exe resnet101 samples/model/resnet/resnet101.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_resnet.exe resnet152 samples/model/resnet/resnet152.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

## 4. 依赖说明

权重导出脚本通常需要以下 Python 依赖：

- `torch`
- `torchvision`
- `opencv-python`

其中 `samples/model/resnet/gen_wts.py` 还会用到：

- `timm`

## 5. 常见说明

- 输入图像预处理默认对齐 ImageNet 分类任务
- 当前示例默认输入尺寸为 `224x224`
- 首次运行 sample 时，如果不存在 `.engine` 文件，会先从 `.wts` 构建 TensorRT engine
- 后续再次运行时会优先加载已有 `.engine`
