# Feature Comparison Sample

This sample dumps InferRT feature tensors for one image so you can compare them with a PyTorch reference run.

## Build

From the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_feature_extract
```

## Run the C++ sample

```bash
build/bin/inferrt_sample_feature_extract.exe --model <name> --weights-file <weights_or_model_file> --features <feature_a,feature_b,...> [--image-path PATH] [--output-dir DIR] [--runtime tensorrt:0|onnxruntime:cpu|onnxruntime:0|openvino:cpu] [--warmup N] [--repeat N]
build/bin/inferrt_sample_feature_extract.exe --help
```

Example:

```bash
build/bin/inferrt_sample_feature_extract.exe -m resnet18 -w assets/models/resnet/resnet18.wts -f layer1,layer4 -i assets/pics/dog.jpg -o build/feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe --model alexnet --weights-file assets/models/alexnet/alexnet.wts --features pool1,fc2
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w assets/models/dinov2/dinov2_vits14.wts -f x_norm_clstoken,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov2_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx -f x_norm_clstoken -i assets/pics/dog.jpg -o build/dinov2_openvino_cpu_feature_dump --runtime openvino:cpu --warmup 10 --repeat 100
build/bin/inferrt_sample_feature_extract.exe -m dinov3_vitb16 -w assets/models/dinov3/dinov3_vitb16.wts -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov3_feature_dump_cpp
```

DINO weights can be exported from `samples/model/python/dino_gen_wts.py` with
`python samples/model/python/dino_gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts` and
`python samples/model/python/dino_gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts`.

The sample always builds the model instance as a truncated feature extractor via
`IModelConfig::setFeatureOnly(true)`.
It reads the input tensor shape from the built engine, so DINOv2 defaults such as `1,3,518,518`
and DINOv3 defaults such as `1,3,224,224` are both reflected in `manifest.txt`.

Defaults:

- `--image-path`: `assets/pics/dog.jpg`
- `--output-dir`: `feature_dump_cpp`
- `--runtime`: `tensorrt:0` by default; use `onnxruntime:0` for ONNX Runtime GPU or `onnxruntime:cpu`/`openvino:cpu` for CPU graph models
- `--warmup`: `0`
- `--repeat`: `1`

The sample writes:

- `manifest.txt`
- `input.bin`
- one `.bin` file per requested feature tensor

## Export feature ONNX

Python 侧只负责生成模型文件。需要将 DINO 的命名特征导出为 ONNX 图输出时，使用：

```bash
python samples/model/python/dino_export_onnx.py \
  -m dinov2_vits14 -b torchhub \
  -f x_norm_clstoken,x_norm_patchtokens \
  -o build/dinov2_vits14.features.onnx --exporter legacy
```

## DINO Feature Keys

- `dinov2*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens`
- `dinov3*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_storage_tokens`, `x_norm_patchtokens`
- For image-level retrieval, prefer `x_norm_clstoken`; `x_norm_patchtokens` is a dense patch-token tensor and is much larger.

## Notes

- Use `--local-files-only` for offline DINOv3 export after the Hugging Face model has been cached.
- Use `--emit-openvino` or `--openvino-output` to generate OpenVINO IR from the exported ONNX graph.
