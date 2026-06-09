# RF-DETR-Seg Segmentation

This sample runs image-only RF-DETR-Seg instance segmentation. The model input contract is a single image tensor
(`input`), and the native TensorRT graph returns `dets`, `labels`, and `masks`.

SAM/SAM2/SAM3 prompt segmentation has been moved to `samples/model/sam`.

## Build

```bash
cmake --build build --config Release --target inferrt_sample_segmentation
```

## Export Weights

Use the shared RF-DETR exporter:

```bash
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_seg_nano -c F:/models/rfdetr/rf-detr-seg-nano.pt --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts --quiet
```

## Run

```bash
build/bin/inferrt_sample_segmentation.exe ^
  -m rfdetr_seg_nano ^
  -w build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts ^
  -i assets/pics/dog.jpg ^
  -o build/rfdetr_seg_nano_result.jpg ^
  --backend tensorrt ^
  --device gpu ^
  --warmup 5 ^
  --repeat 20
```

Options:

- `--conf-threshold`: instance confidence threshold.
- `--mask-threshold`: mask logit threshold used for overlay.
- `--nms-threshold`: class-wise box NMS IoU threshold.
- `--max-instances`: maximum instances to draw.
- `--input-size`: optional square override; by default the registered RF-DETR-Seg size is used.

The timing line reports `build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end, timed-loop wall time, and
postprocess.
