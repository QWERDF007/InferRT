# InferRT Python 模型转换工具

本目录只负责把第三方 PyTorch checkpoint 转换为 InferRT 使用的模型文件：

- `.wts`：供 InferRT 原生 TensorRT 模型构建器使用；
- `.onnx`：供 ONNX Runtime、OpenVINO 或通用 ONNX 模型入口使用。

支持的模型族为分类、DINO、YOLO 检测、RF-DETR 和 SAM。推理、特征对比和 Python 绑定示例不放在这里。
所有命令都在仓库根目录 `F:/Projects/InferRT` 执行。

## 环境准备

基础依赖：

```bash
python -m pip install torch torchvision opencv-python onnx
```

按模型族安装可选依赖：

```bash
python -m pip install timm transformers
python -m pip install ultralytics
python -m pip install "rfdetr[onnx]"
```

SAM v1、SAM2/SAM2.1 和 SAM3 还需要各自官方仓库或 Python 包。离线源码目录通过
`--sam-root`、`--sam2-root` 和 `--edge-sam-root` 传入；SAM3 使用 Transformers 本地模型目录。

## 工具一览

| 模型族 | 生成 `.wts` | 导出 `.onnx` |
| --- | --- | --- |
| 分类 | `classification_gen_wts.py` | `classification_export_onnx.py` |
| DINOv2 / DINOv3 / LingBot-Vision | `dino_gen_wts.py` | `dino_export_onnx.py` |
| YOLO 检测 | `detection_gen_wts.py` | `detection_export_onnx.py` |
| RF-DETR | `rfdetr_gen_wts.py` | `rfdetr_export_onnx.py` |
| SAM v1 / SAM2 / SAM3 | `sam_gen_wts.py` | `sam_export_onnx.py` |

`classification_model_zoo.py`、`dino_model_zoo.py`、`yolo_model_zoo.py`、`rfdetr_compat.py` 和 `wts_utils.py` 是上述入口使用的内部适配模块，
不需要单独运行。

## 分类模型

生成 TensorRT `.wts`：

```bash
python samples/model/python/classification_gen_wts.py ^
  -b torchvision -m resnet18 ^
  -o build/models/resnet18.wts
```

导出普通分类输出的 ONNX：

```bash
python samples/model/python/classification_export_onnx.py ^
  -b torchvision -m resnet18 ^
  -o build/models/resnet18.onnx ^
  --exporter legacy
```

列出某个来源支持的模型：

```bash
python samples/model/python/classification_gen_wts.py -b torchvision -l
python samples/model/python/classification_export_onnx.py -b timm -l
```

`--exporter` 可选 `auto`、`dynamo` 或 `legacy`；在需要兼容旧版 PyTorch/ONNX 的环境中使用 `legacy`。

## DINOv2 / DINOv3

DINOv2 默认从 PyTorch Hub 加载。离线时指定本地仓库和本地 checkpoint：

```bash
python samples/model/python/dino_gen_wts.py ^
  -b torchhub -m dinov2_vits14 ^
  --hub-repo F:/Github/CV/dinov2 --hub-source local ^
  --hub-weights F:/models/dinov2/dinov2_vits14_pretrain.pth ^
  -o build/models/dinov2_vits14.wts
```

DINOv3 使用 Transformers：

```bash
python samples/model/python/dino_gen_wts.py ^
  -b transformers -m dinov3_vitb16 --local-files-only ^
  -o build/models/dinov3_vitb16.wts
```

如果只需要 DINO 的全局向量，可以导出 `x_norm_clstoken`：

```bash
python samples/model/python/dino_export_onnx.py ^
  -b torchhub -m dinov2_vits14 ^
  --hub-repo F:/Github/CV/dinov2 --hub-source local ^
  --hub-weights F:/models/dinov2/dinov2_vits14_pretrain.pth ^
  -f x_norm_clstoken ^
  -o build/models/dinov2_vits14.onnx ^
  --exporter legacy
```

图像搜索、图像聚类通常需要显式命名的特征输出，使用 `dino_export_onnx.py`：

```bash
python samples/model/python/dino_export_onnx.py ^
  -b torchhub -m dinov2_vits14 ^
  --hub-repo F:/Github/CV/dinov2 --hub-source local ^
  --hub-weights F:/models/dinov2/dinov2_vits14_pretrain.pth ^
  -f x_norm_clstoken ^
  -o build/models/dinov2_vits14.features.onnx ^
  --exporter legacy
```

`-f` 支持逗号分隔的多个输出，例如 `x_norm_clstoken,x_norm_patchtokens`。图像级检索优先使用
`x_norm_clstoken`；`x_norm_patchtokens` 是更大的 patch 特征。需要 OpenVINO IR 时增加
`--emit-openvino` 或 `--openvino-output <目录或.xml路径>`。

LingBot-Vision 使用 `lingbot` backend，默认模型配置和 checkpoint 目录分别为
`F:/Github/lingbot-vision`、`F:/models/lingbot-vision`，也可以显式传入 checkpoint：

```bash
python samples/model/python/dino_gen_wts.py ^
  -b lingbot -m lingbot_vision_vits16 ^
  -c F:/models/lingbot-vision/lingbot-vision-vit-small.pt ^
  --lingbot-root F:/Github/lingbot-vision ^
  -o build/models/lingbot_vision_vits16.wts --quiet

python samples/model/python/dino_export_onnx.py ^
  -b lingbot -m lingbot_vision_vits16 ^
  -c F:/models/lingbot-vision/lingbot-vision-vit-small.pt ^
  --lingbot-root F:/Github/lingbot-vision ^
  -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens ^
  -o build/models/lingbot_vision_vits16.features.onnx --exporter legacy
```

支持 `lingbot_vision_vits16`、`lingbot_vision_vitb16`、`lingbot_vision_vitl16` 和
`lingbot_vision_vitg16`。

## YOLO 检测

生成 InferRT 原生 `.wts`：

```bash
python samples/model/python/detection_gen_wts.py ^
  -m yolov8n ^
  --weights F:/models/yolo/yolov8n.pt ^
  --ultralytics-repo F:/Github/CV/ultralytics ^
  -o build/models/yolov8n.wts --quiet
```

导出 Ultralytics ONNX：

```bash
python samples/model/python/detection_export_onnx.py ^
  -m yolov8n ^
  --weights F:/models/yolo/yolov8n.pt ^
  --ultralytics-repo F:/Github/CV/ultralytics ^
  --input-size 640 --dynamic-batch ^
  -o build/models/yolov8n.onnx
```

旧版 YOLOv5 checkpoint 的 `.wts` 转换仍使用 `detection_gen_wts.py`；ONNX 导出请使用对应的
Ultralytics 模型/权重。

## RF-DETR

生成 InferRT 原生 `.wts`：

```bash
python samples/model/python/rfdetr_gen_wts.py ^
  -m rfdetr_nano ^
  -c F:/models/rfdetr/rf-detr-nano.pth ^
  --rfdetr-root F:/Github/CV/rf-detr ^
  -o build/models/rfdetr_nano.wts --quiet
```

调用 RF-DETR 官方 ONNX 导出流程：

```bash
python samples/model/python/rfdetr_export_onnx.py ^
  -m rfdetr_nano ^
  -c F:/models/rfdetr/rf-detr-nano.pth ^
  --rfdetr-root F:/Github/CV/rf-detr ^
  --shape 384 --dynamic-batch ^
  -o build/models/rfdetr_nano.onnx
```

RF-DETR ONNX 导出需要 `rfdetr[onnx]`。`--shape` 的高宽必须满足上游模型的 patch/window 整除约束。

## SAM

SAM v1：

```bash
python samples/model/python/sam_gen_wts.py ^
  -m vit_b ^
  -c F:/models/sam/sam_vit_b_01ec64.pth ^
  --sam-root F:/Github/SAM/segment-anything ^
  -o build/models/sam_vit_b.wts --skip-forward

python samples/model/python/sam_export_onnx.py ^
  -m sam_vit_b ^
  -c F:/models/sam/sam_vit_b_01ec64.pth ^
  --sam-root F:/Github/SAM/segment-anything ^
  -o build/models/sam_vit_b.onnx
```

SAM2/SAM2.1：

```bash
python samples/model/python/sam_gen_wts.py ^
  -m sam2_1_hiera_tiny ^
  -c F:/models/sam/sam2.1_hiera_tiny.pt ^
  --sam2-root F:/Github/SAM/sam2 ^
  -o build/models/sam2_1_hiera_tiny.wts --skip-forward

python samples/model/python/sam_export_onnx.py ^
  -m sam2_1_hiera_tiny ^
  -c F:/models/sam/sam2.1_hiera_tiny.pt ^
  --sam2-root F:/Github/SAM/sam2 ^
  -o build/models/sam2_1_hiera_tiny.onnx
```

SAM ONNX 的输入为 `image`、`point_coords`、`point_labels`、`mask_input`、`has_mask_input`，
输出为 `masks`、`iou_predictions`、`low_res_masks`，与 InferRT SAM sample 的图契约一致。

## 输出文件和后端

输出目录不存在时各脚本会自动创建。生成后的文件可以直接交给相应 C++ sample：

- `.wts` 通过原生 TensorRT 模型构建；
- `.onnx` 通过 `onnx` 模型或对应高级 API 加载；
- 运行时和设备在 C++/Python 的 `ModelConfig.runtime` 中配置，例如 `tensorrt:0`、
  `onnxruntime:0`、`openvino:cpu`，不在这些导出脚本中配置。

查看任意入口的完整参数：

```bash
python samples/model/python/<script>.py --help
```
