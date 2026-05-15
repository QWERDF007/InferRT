# Model Samples

This directory contains the ImageNet-style classification sample assets provided by InferRT.

## Layout

- `classification/`: shared weight export and inference entry for all supported classification models
- `onnx/`: ONNX export script and ONNX -> TensorRT inference sample

## Build

Build the shared sample from the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_classification
```

## Run

```bash
build/bin/inferrt_sample_classification.exe <model_name> <weights_file.wts> <image_path> [label_file]
```

Examples:

```bash
build/bin/inferrt_sample_classification.exe alexnet samples/model/classification/alexnet.wts assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe resnet50 samples/model/classification/resnet50.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe vgg16 samples/model/classification/vgg16.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

## Weight export

Generate weights with the shared script:

```bash
cd samples/model/classification
python gen_wts.py -m alexnet
python gen_wts.py -m resnet50
python gen_wts.py -m vgg16
```

## Notes

- Input preprocessing is aligned with standard ImageNet classification
- The shared sample assumes `1x3x224x224` input and `1000` output classes
- The first run builds an engine from `.wts`, and later runs reuse the generated `.engine`
