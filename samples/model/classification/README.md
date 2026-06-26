# Classification Sample

This directory contains the classification C++ sample under `samples/model`.
The Python weight exporter lives in `samples/model/python/classification_gen_wts.py`.

## Supported models

The sample uses the runtime model registry, so `--help` prints the full list built into the current binary. Major supported groups are:

- `alexnet`
- `googlenet`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `vgg11`
- `vgg13`
- `vgg16`
- `vgg19`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `wide_resnet50_2`
- `wide_resnet101_2`
- standard ViT keys such as `vit_base_patch16_224` and `vit_base_patch16_384`
- DINOv2 keys such as `dinov2_vits14`, `dinov2_vitb14_reg`, and timm aliases such as `vit_small_patch14_dinov2`
- DINOv3 keys such as `dinov3_vitb16`, `dinov3_vitl16plus`, and timm aliases such as `vit_base_patch16_dinov3`

## Build

```bash
cmake --build build --config Debug --target inferrt_sample_classification
```

## Generate weights

Run in a Python environment with `torch`, `torchvision`, and `opencv-python` installed:

```bash
python samples/model/python/classification_gen_wts.py -m alexnet
python samples/model/python/classification_gen_wts.py -m mobilenet_v2
python samples/model/python/classification_gen_wts.py -m mobilenet_v3_large
python samples/model/python/classification_gen_wts.py -m mobilenet_v3_small
python samples/model/python/classification_gen_wts.py -m resnet50
python samples/model/python/classification_gen_wts.py -m vgg16
```

For timm-backed ResNet, ViT, and DINO variants:

```bash
python samples/model/python/classification_gen_wts.py -b timm -m resnet18
python samples/model/python/classification_gen_wts.py -b timm -m vit_base_patch16_384 --input-size 384
python samples/model/python/classification_gen_wts.py -b timm -m vit_base_patch16_dinov3
```

For official DINO weights:

```bash
python samples/model/python/classification_gen_wts.py -b torchhub -m dinov2_vits14
python samples/model/python/classification_gen_wts.py -b torchhub -m dinov2_vits14 --hub-repo <local-dinov2-repo> --hub-source local
python samples/model/python/classification_gen_wts.py -b torchhub -m dinov2_vits14 --hub-repo <local-dinov2-repo> --hub-source local --hub-weights <dinov2-vits14-pretrain.pth>
python samples/model/python/classification_gen_wts.py -b transformers -m dinov3_vitb16
python samples/model/python/classification_gen_wts.py -b transformers -m dinov3_vitb16 --hf-model-id facebook/dinov3-vitb16-pretrain-lvd1689m
python samples/model/python/classification_gen_wts.py -b transformers -m dinov3_vitb16 --local-files-only
```

DINOv2 uses PyTorch Hub. DINOv3 uses Hugging Face `pipeline(model="facebook/dinov3-vitb16-pretrain-lvd1689m", task="image-feature-extraction")` by default and converts the Transformers state dict to the InferRT DINOv3 weight names during `.wts` export. In offline environments, use `--local-files-only` after the Hugging Face model is cached.

DINO models export feature-vector backbones rather than 1000-class logits. `classification_gen_wts.py` and the C++ sample print feature top values when the output dimension does not match the ImageNet label count.

List supported models for a backend:

```bash
python samples/model/python/classification_gen_wts.py -l
python samples/model/python/classification_gen_wts.py -b timm -l
python samples/model/python/classification_gen_wts.py -b transformers -l
```

## Run

```bash
build/bin/inferrt_sample_classification.exe --model <model_name> --weights-file <weights_or_model_file> [--image-path PATH] [--label-file PATH] [--backend tensorrt|openvino|onnxruntime] [--device cpu|gpu] [--warmup N] [--repeat N]
build/bin/inferrt_sample_classification.exe --help
```

Examples:

```bash
build/bin/inferrt_sample_classification.exe --model alexnet --weights-file assets/models/alexnet/alexnet.wts --image-path assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe --model mobilenet_v2 --weights-file assets/models/mobilenet/mobilenet_v2.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file assets/models/resnet/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file D:/Models/resnet/<checkpoint-stem-or-dir>/resnet50.onnx --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --backend openvino --device cpu
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file assets/models/resnet/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --backend tensorrt --device gpu --warmup 10 --repeat 100
```

Defaults:

- `image_path`: `assets/pics/dog.jpg`
- `label_file`: `assets/imagenet1000_clsidx_to_labels.txt`
- `backend`: `tensorrt`
- `device`: `gpu`
- `warmup`: `0`
- `repeat`: `1`

## Extend

When a new ImageNet-style classification model is added:

1. register the model in `inferrt_model`
2. add its builder to `TORCHVISION_MODEL_ZOO` or handle it in `create_model()` inside `samples/model/python/classification_model_zoo.py`
3. add a smoke test for the model key in `tests/model` and `tests/python`
