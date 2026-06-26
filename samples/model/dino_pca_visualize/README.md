# DINO PCA Visualization Sample

This C++ sample extracts a DINO patch-token feature tensor, reduces each patch token from channel dimension to 3
components with OpenCV PCA, and saves two visualizations:

- direct PCA to 3 channels
- background-zeroed PCA visualization using the same 1D PCA threshold idea as `vis.py`

The PCA input is one image's local feature matrix shaped as `patch_count x channel_count`.

## Build

```bash
cmake --build build --config Debug --target inferrt_sample_dino_pca_visualize
```

## Run

```bash
build/bin/inferrt_sample_dino_pca_visualize.exe ^
  --model dinov2_vitb14 ^
  --weights-file samples/model/classification/dinov2_vitb14.wts ^
  --feature x_norm_patchtokens ^
  --image-path assets/pics/dog.jpg ^
  --output-dir build/dino_pca_visualize
```

If `--weights-file` is omitted for TensorRT, the sample uses
`samples/model/classification/<model>.wts`.

Default values:

- `--model`: `dinov2_vits14`
- `--feature`: `x_norm_patchtokens`
- `--image-path`: `assets/pics/dog.jpg`
- `--output-dir`: `dino_pca_visualize_cpp`
- `--threshold`: `0.5`
- `--backend`: `tensorrt`
- `--device`: `gpu`

The sample writes:

- `<image>_pca3.png`
- `<image>_background_zeroed_pca3.png`
- `<image>_background_mask.png`
- `manifest.txt`

It also prints timings for build/load, image load, preprocessing, H2D copy, inference, D2H copy, PCA stages, saving,
and total runtime.
