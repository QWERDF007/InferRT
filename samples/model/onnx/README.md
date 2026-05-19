# ONNX Sample

This directory demonstrates an ONNX -> TensorRT -> inference workflow.
The sample automatically reads engine tensor names, runtime input shape, and output size from the built TensorRT engine.

## Export ONNX

Run in a Python environment with `torch` and `torchvision` installed:

```bash
cd samples/model/onnx
python export_onnx.py -m alexnet
python export_onnx.py -m resnet50
python export_onnx.py -m vgg16
python export_onnx.py -m resnet50 --exporter legacy
```

List supported models:

```bash
python export_onnx.py -l
python export_onnx.py -b timm -l
```

Exporter options:

- `--exporter auto`: try the new dynamo exporter first, then fall back to the legacy exporter on common environment failures.
- `--exporter dynamo`: force the new exporter.
- `--exporter legacy`: force the legacy TorchScript-based exporter.

## Build sample

```bash
cmake --build build --config Debug --target inferrt_sample_onnx
```

## Run sample

```bash
build/bin/inferrt_sample_onnx.exe <model.onnx> <image_path> [label_file]
build/bin/inferrt_sample_onnx.exe samples/model/onnx/alexnet.onnx assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_onnx.exe samples/model/onnx/alexnet.onnx assets/pics/dog.jpg
build/bin/inferrt_sample_onnx.exe --help
```

Notes:

- The sample currently supports one input tensor.
- The sample currently expects a single input image from the command line.
- Image input tensors are currently supported for `float32` and `float16`.
- Output tensors can be inspected for `float32`, `float16`, `int8`, `uint8`, `int32`, `int64`, and `bool`.
- Multi-output models are printed per output tensor.
