# SAM Segmentation Sample

This sample runs the TensorRT-native SAM/SAM2 model path from `inferrt_model` on a single image.
It uses the official SAM-style five-input contract:

- `image`: `1x3xHxW`
- `point_coords`: `1x16x2x1`
- `point_labels`: `1x16x1x1`
- `mask_input`: `1x1x256x256`
- `has_mask_input`: `1x1x1x1`

Outputs are `masks`, `iou_predictions`, and `low_res_masks`. `point_labels` follows the official prompt encoder
semantics: `1` positive point, `0` negative point, `-1` padding, `2/3` box corners.

## Build

```bash
cmake --build build --config Debug --target inferrt_sample_segmentation
```

## Run

Export official SAM/SAM2 weights first:

```bash
cd samples/model/segmentation
python gen_sam_wts.py -m vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.wts
python gen_sam2_wts.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.wts
```

```bash
build/bin/inferrt_sample_segmentation.exe -m sam_vit_b -w samples/model/segmentation/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_mask.jpg --point-x 0.5 --point-y 0.5
build/bin/inferrt_sample_segmentation.exe -m sam_vit_b -w samples/model/segmentation/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_box_mask.jpg --box 0.2,0.2,0.8,0.8
build/bin/inferrt_sample_segmentation.exe -m sam2_1_hiera_tiny -w samples/model/segmentation/sam2_1_hiera_tiny.wts -i assets/pics/dog.jpg -o build/sam2_mask.jpg --point-x 0.5 --point-y 0.5
```

SAM v1 builds the official ViT image encoder, prompt encoder, and mask decoder graph from exported official weights.
SAM2/SAM2.1 builds the official Hiera image encoder, FPN neck, prompt encoder, and high-resolution mask decoder graph.
The legacy `sam_placeholder.wts` file is only useful for tests that validate missing-weight errors; it cannot build a
runnable SAM engine.

## Supported Keys

- SAM v1: `sam`, `sam_vit_b`, `sam_vit_l`, `sam_vit_h`
- SAM2: `sam2`, `sam2_hiera_tiny`, `sam2_hiera_small`, `sam2_hiera_base_plus`, `sam2_hiera_large`
- SAM2.1: `sam2_1_hiera_tiny`, `sam2_1_hiera_small`, `sam2_1_hiera_base_plus`, `sam2_1_hiera_large`
- SAM3: `sam3`, `sam3_image`

SAM3 keys are registered for API compatibility. Its native image encoder is not silently mapped to SAM v1/SAM2, so
`build()` currently returns `ERROR_NOT_IMPLEMENTED` for SAM3 variants.
