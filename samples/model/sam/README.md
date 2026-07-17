# SAM Segmentation Sample

This sample runs SAM/SAM2 through the selected InferRT backend on a single image.
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
cmake --build build --config Debug --target inferrt_sample_sam
```

## Run

Export official SAM/SAM2 weights first:

```bash
python samples/model/python/gen_sam_wts.py -m vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.wts
python samples/model/python/gen_sam_wts.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.wts
python samples/model/python/gen_sam_wts.py -m sam3 -c D:/Models/sam3 -o sam3.wts --skip-forward
```

Export SAM v1/SAM2/SAM3 to ONNX for ONNX Runtime or OpenVINO:

```bash
python samples/model/python/export_sam_onnx.py -m sam_vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.onnx
python samples/model/python/export_sam_onnx.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.onnx
python samples/model/python/export_sam_onnx.py -m sam3 -c D:/Models/sam3 -o sam3.onnx
```

For SAM3, `--checkpoint` is a Hugging Face `Sam3Model` id or local model directory, and the exporter uses
`transformers` to bake a static text embedding while keeping the runtime graph on the same SAM five-input contract.
SAM3 `.wts` export writes the Hugging Face `Sam3Model.state_dict()` for inspection and future native TensorRT
integration; the current SAM3 TensorRT model entry still returns `ERROR_NOT_IMPLEMENTED`.

```bash
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_mask.jpg --point-x 0.5 --point-y 0.5 --runtime tensorrt:0 --warmup 5 --repeat 20
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_box_mask.jpg --box 0.2,0.2,0.8,0.8
build/bin/inferrt_sample_sam.exe -m sam2_1_hiera_tiny -w samples/model/sam/sam2_1_hiera_tiny.wts -i assets/pics/dog.jpg -o build/sam2_mask.jpg --point-x 0.5 --point-y 0.5
build/bin/inferrt_sample_sam.exe -m sam2_1_hiera_tiny -w sam2_1_hiera_tiny.onnx -i assets/pics/dog.jpg -o build/sam2_onnx_mask.jpg --point-x 0.5 --point-y 0.5 --runtime onnxruntime:cpu
```

Runtime options:

- `--runtime`: combines backend and device, for example `tensorrt:0`, `onnxruntime:cpu`, `onnxruntime:0`, or `openvino:cpu`; shorthand `cpu`, `gpu:0`, and `cuda:0` is also supported.
- `--warmup`: iterations to run before measurement.
- `--repeat`: measured iterations used for total/avg/min/max timing.

The timing line reports `build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end, timed-loop wall time, and postprocess. ONNX Runtime and OpenVINO use graph inputs and outputs directly; the graph must expose the same five-input/three-output SAM contract listed above.

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
