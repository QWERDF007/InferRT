# Classification Sample

This directory is the shared entry for classification weight export and inference under `samples/model`.

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
cd samples/model/classification
python gen_wts.py -m alexnet
python gen_wts.py -m mobilenet_v2
python gen_wts.py -m mobilenet_v3_large
python gen_wts.py -m mobilenet_v3_small
python gen_wts.py -m resnet50
python gen_wts.py -m vgg16
```

For timm-backed ResNet, ViT, and DINO variants:

```bash
python gen_wts.py -b timm -m resnet18
python gen_wts.py -b timm -m vit_base_patch16_384 --input-size 384
python gen_wts.py -b timm -m vit_base_patch16_dinov3
```

For official DINO weights:

```bash
python gen_wts.py -b torchhub -m dinov2_vits14
python gen_wts.py -b torchhub -m dinov2_vits14 --hub-repo <local-dinov2-repo> --hub-source local
python gen_wts.py -b torchhub -m dinov2_vits14 --hub-repo <local-dinov2-repo> --hub-source local --hub-weights <dinov2-vits14-pretrain.pth>
python gen_wts.py -b transformers -m dinov3_vitb16
python gen_wts.py -b transformers -m dinov3_vitb16 --hf-model-id facebook/dinov3-vitb16-pretrain-lvd1689m
python gen_wts.py -b transformers -m dinov3_vitb16 --local-files-only
```

DINOv2 uses PyTorch Hub. DINOv3 uses Hugging Face `pipeline(model="facebook/dinov3-vitb16-pretrain-lvd1689m", task="image-feature-extraction")` by default and converts the Transformers state dict to the InferRT DINOv3 weight names during `.wts` export. In offline environments, use `--local-files-only` after the Hugging Face model is cached.

DINO models export feature-vector backbones rather than 1000-class logits. `gen_wts.py` and the C++ sample print feature top values when the output dimension does not match the ImageNet label count.

List supported models for a backend:

```bash
python gen_wts.py -l
python gen_wts.py -b timm -l
python gen_wts.py -b transformers -l
```

## Run

```bash
build/bin/inferrt_sample_classification.exe <model_name> <weights_file.wts> [image_path] [label_file]
build/bin/inferrt_sample_classification.exe --help
```

Examples:

```bash
build/bin/inferrt_sample_classification.exe alexnet samples/model/classification/alexnet.wts assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe mobilenet_v2 samples/model/classification/mobilenet_v2.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe resnet50 samples/model/classification/resnet50.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe vgg16 samples/model/classification/vgg16.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

Defaults:

- `image_path`: `assets/pics/dog.jpg`
- `label_file`: `assets/imagenet1000_clsidx_to_labels.txt`

## Extend

When a new ImageNet-style classification model is added:

1. register the model in `inferrt_model`
2. add its builder to `TORCHVISION_MODEL_ZOO` or handle it in `create_model()` inside `model_zoo.py`
3. add a smoke test for the model key in `tests/model` and `tests/python`
