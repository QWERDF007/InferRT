# Feature Comparison Sample

This sample dumps InferRT feature tensors for one image so you can compare them with a PyTorch reference run.

## Build

From the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_features
```

## Run the C++ sample

```bash
build/bin/inferrt_sample_features.exe <model_name> <weights_file.wts> <feature_a,feature_b,...> [image_path] [output_dir]
```

Example:

```bash
build/bin/inferrt_sample_features.exe resnet18 samples/model/classification/resnet18.wts layer1,layer4 assets/pics/dog.jpg build/feature_dump_cpp
build/bin/inferrt_sample_features.exe alexnet samples/model/classification/alexnet.wts pool1,fc2 assets/pics/dog.jpg build/feature_dump_cpp
```

The sample always builds the model instance as a truncated feature extractor via
`IModelConfig::setFeatureOnly(true)`.

The sample writes:

- `manifest.txt`
- `input.bin`
- one `.bin` file per requested feature tensor

## Compare with Python

```bash
cd samples/model/features
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

## Notes

- The Python script reuses the same ImageNet preprocessing as `gen_wts.py`.
- `--compare_dir` expects the directory produced by `inferrt_sample_features`.
- For `timm` backend comparison, use `-b timm` with a matching timm-exported weight file.
- The comparison output reports both absolute and relative error: `max_abs`, `mean_abs`, `max_rel`,
  `mean_rel`, and `ref_abs_max`.
- Relative error is reported only where the PyTorch reference magnitude is above `1e-3`, so near-zero
  activations do not dominate the summary with meaningless ratios.
- For feature tensors, treat `mean_abs` and `mean_rel` as the primary signal; a single `max_abs` around
  `1e-1` can still be acceptable when the tensor value range is much larger.
