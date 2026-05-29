# Feature Comparison Sample

This sample dumps InferRT feature tensors for one image so you can compare them with a PyTorch reference run.

## Build

From the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_feature_extract
```

## Run the C++ sample

```bash
build/bin/inferrt_sample_feature_extract.exe --model <name> --weights-file <weights_or_model_file> --features <feature_a,feature_b,...> [--image-path PATH] [--output-dir DIR] [--backend tensorrt|openvino|onnxruntime] [--device cpu|gpu] [--warmup N] [--repeat N]
build/bin/inferrt_sample_feature_extract.exe --help
```

Example:

```bash
build/bin/inferrt_sample_feature_extract.exe -m resnet18 -w samples/model/classification/resnet18.wts -f layer1,layer4 -i assets/pics/dog.jpg -o build/feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe --model alexnet --weights-file samples/model/classification/alexnet.wts --features pool1,fc2
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w samples/model/classification/dinov2_vits14.wts -f x_norm_clstoken,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov2_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w build/python_test_artifacts/model_dir_parity/dinov2/dinov2_vits14/<case-id>/dinov2_vits14.features.onnx -f x_norm_clstoken -i assets/pics/dog.jpg -o build/dinov2_openvino_cpu_feature_dump --backend openvino --device cpu --warmup 10 --repeat 100
build/bin/inferrt_sample_feature_extract.exe -m dinov3_vitb16 -w samples/model/classification/dinov3_vitb16.wts -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov3_feature_dump_cpp
```

DINO weights can be exported from `samples/model/classification` with
`python gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts` and
`python gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts`.

The sample always builds the model instance as a truncated feature extractor via
`IModelConfig::setFeatureOnly(true)`.
It reads the input tensor shape from the built engine, so DINOv2 defaults such as `1,3,518,518`
and DINOv3 defaults such as `1,3,224,224` are both reflected in `manifest.txt`.

Defaults:

- `--image-path`: `assets/pics/dog.jpg`
- `--output-dir`: `feature_dump_cpp`
- `--backend`: `tensorrt`
- `--device`: `gpu`
- `--warmup`: `0`
- `--repeat`: `1`

The sample writes:

- `manifest.txt`
- `input.bin`
- one `.bin` file per requested feature tensor

## Compare with Python

```bash
cd samples/model/feature_extract
python compare_features.py --compare_dir ../../../build/feature_dump_cpp
```

For TensorRT vs PyTorch intermediate features, the script defaults to `--rtol 1e-2 --atol 1.2e-1`.
Those defaults are intentionally looser than exact tensor checks because TensorRT may fuse layers and
use different FP32 kernels while still producing numerically aligned features.

You can also select the model and features explicitly:

```bash
python compare_features.py -m resnet18 -f layer1,layer4 -i ../../../assets/pics/dog.jpg --compare_dir ../../../build/feature_dump_cpp
```

Optional Python-side dump:

```bash
python compare_features.py -m resnet18 -f layer1,layer4 -i ../../../assets/pics/dog.jpg --dump_dir ../../../build/feature_dump_py
```

## DINO Feature Keys

- `dinov2*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens`
- `dinov3*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_storage_tokens`, `x_norm_patchtokens`
- For image-level retrieval, prefer `x_norm_clstoken`; `x_norm_patchtokens` is a dense patch-token tensor and is much larger.

## Notes

- The Python script reuses the same ImageNet preprocessing as `gen_wts.py`.
- `--compare_dir` expects the directory produced by `inferrt_sample_feature_extract`.
- For `timm` backend comparison, use `-b timm` with a matching timm-exported weight file.
- For DINO comparison, `compare_features.py` infers `torchhub` for DINOv2 and `transformers` for DINOv3
  when `--backend` is omitted; pass `--local-files-only` if you want Hugging Face to use only cached files.
- You can still override `--backend` explicitly if you want to compare a timm-alias DINO key.
- The comparison output reports both absolute and relative error: `max_abs`, `mean_abs`, `max_rel`,
  `mean_rel`, and `ref_abs_max`.
- Relative error is reported only where the PyTorch reference magnitude is above `1e-3`, so near-zero
  activations do not dominate the summary with meaningless ratios.
- For feature tensors, treat `mean_abs` and `mean_rel` as the primary signal; a single `max_abs` around
  `1e-1` can still be acceptable when the tensor value range is much larger.
