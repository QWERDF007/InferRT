# ONNX Sample

This directory demonstrates an ONNX -> TensorRT -> inference workflow.
The sample automatically reads engine tensor names, runtime input shape, and output size from the built TensorRT engine.

## Export ONNX

Run in a Python environment with `torch` and `torchvision` installed:

```bash
python samples/model/python/classification_export_onnx.py -m alexnet
python samples/model/python/classification_export_onnx.py -m resnet50
python samples/model/python/classification_export_onnx.py -m vgg16
python samples/model/python/classification_export_onnx.py -m resnet50 --exporter legacy
```

List supported models:

```bash
python samples/model/python/classification_export_onnx.py -l
python samples/model/python/classification_export_onnx.py -b timm -l
```

Exporter options:

- `--exporter auto`: try the new dynamo exporter first, then fall back to the legacy exporter on common environment failures.
- `--exporter dynamo`: force the new exporter.
- `--exporter legacy`: force the legacy TorchScript-based exporter.

## Export Feature ONNX / OpenVINO IR

`dino_export_onnx.py` wraps DINO/LingBot-Vision `model.forward_features()` and exports selected feature keys as graph outputs.
Those output names can then be used by the ONNX Runtime or OpenVINO backend through `output_tensor_names` /
`feature_tensor_names`.

Examples:

```bash
python samples/model/python/dino_export_onnx.py -m dinov2_vits14 -b torchhub -f x_norm_clstoken,x_norm_patchtokens --input-size 518 -o dinov2_vits14_features.onnx
python samples/model/python/dino_export_onnx.py -m dinov2_vits14 -b torchhub -f x_norm_clstoken --input-size 518 -o dinov2_vits14_cls.onnx --emit-openvino
python samples/model/python/dino_export_onnx.py -m dinov3_vitb16 -b transformers -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens --local-files-only --openvino-output build/openvino_ir
```

OpenVINO conversion uses the Python OpenVINO API. By default the script also searches
`D:/Software/openvino_toolkit`; override it with `--openvino-root` when needed.

## Build sample

```bash
cmake --build build --config Debug --target inferrt_sample_onnx
```

## Run sample

```bash
build/bin/inferrt_sample_onnx.exe --onnx-file <model.onnx> --image-path <image_path> [--label-file PATH]
build/bin/inferrt_sample_onnx.exe -n samples/model/onnx/alexnet.onnx -i assets/pics/dog.jpg -l assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_onnx.exe --onnx-file samples/model/onnx/alexnet.onnx --image-path assets/pics/dog.jpg
build/bin/inferrt_sample_onnx.exe --help
```

Notes:

- The sample currently supports one input tensor.
- The sample currently expects a single input image from the command line.
- Image input tensors are currently supported for `float32` and `float16`.
- Output tensors can be inspected for `float32`, `float16`, `int8`, `uint8`, `int32`, `int64`, and `bool`.
- Multi-output models are printed per output tensor.
