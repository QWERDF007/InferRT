# Instance Segmentation

This sample runs image-only instance segmentation. The model input contract is a single image tensor (`input`).
RF-DETR-Seg returns `dets`, `labels`, and `masks`; YOLOv8-Seg returns `output0`, `output1`, `output2`, and `proto`.

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

Use the shared YOLO exporter for YOLOv8-Seg:

```bash
python samples/model/python/detection_gen_wts.py -m yolov8n_seg -w F:/models/yolov8/yolov8n-seg.pt --ultralytics-repo F:/Github/CV/ultralytics -o build/python_test_artifacts/yolo/yolov8n_seg.wts --quiet
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

build/bin/inferrt_sample_segmentation.exe ^
  -m yolov8n_seg ^
  -w build/python_test_artifacts/yolo/yolov8n_seg.wts ^
  -i assets/pics/dog.jpg ^
  -o build/yolov8n_seg_result.jpg ^
  --conf-threshold 0.10 ^
  --warmup 5 ^
  --repeat 20
```

Options:

- `--conf-threshold`: instance confidence threshold.
- `--mask-threshold`: mask logit threshold used for overlay.
- `--nms-threshold`: class-wise box NMS IoU threshold.
- `--max-instances`: maximum instances to draw.
- `--num-classes`: YOLO class count; default is COCO 80.
- `--input-size`: optional square override; by default the registered model size is used.

The timing line reports `build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end, timed-loop wall time, and
postprocess.
