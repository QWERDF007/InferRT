# Classification Sample

This directory is the shared entry for classification weight export and inference under `samples/model`.

## Supported models

- `alexnet`
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

For timm-backed ResNet variants:

```bash
python gen_wts.py -b timm -m resnet18
```

List supported models for a backend:

```bash
python gen_wts.py -l
python gen_wts.py -b timm -l
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
2. add its name to `kSupportedModels` in `SampleModelClassification.cpp`
3. add its builder to `TORCHVISION_MODEL_ZOO` or handle it in `create_model()` inside `model_zoo.py`
