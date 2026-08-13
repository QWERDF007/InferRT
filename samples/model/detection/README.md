# Detection

This directory contains the InferRT native YOLOv5, YOLOv8, and RF-DETR detection C++ sample.
YOLO weights are exported by `samples/model/python/detection_gen_wts.py`; RF-DETR weights are exported by
`samples/model/python/rfdetr_gen_wts.py`.

## Export Weights

Use a local Ultralytics checkout when you want the exporter to use `D:/Github/ultralytics` directly:

```bash
python samples/model/python/detection_gen_wts.py -m yolov5n --weights yolov5nu.pt --ultralytics-repo D:/Github/ultralytics
python samples/model/python/detection_gen_wts.py -m yolov8n --weights yolov8n.pt --ultralytics-repo D:/Github/ultralytics
```

If `ultralytics` is already installed, `--ultralytics-repo` is optional:

```bash
python samples/model/python/detection_gen_wts.py -m yolov8s --weights D:/models/yolov8s.pt -o yolov8s.wts
python samples/model/python/detection_gen_wts.py -l
```

Export RF-DETR weights from the upstream checkpoint:

```bash
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_nano -c F:/models/rfdetr/rf-detr-nano.pth --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_nano.wts --quiet
```

The native C++ builders expose three detection outputs:

- YOLOv5u from Ultralytics: DFL-decoded distance plus class tensors, shaped as `N x (4 + classes) x grid`.
- Legacy YOLOv5 from tensorrtx-style weights: raw P3/P4/P5 head tensors, shaped as `N x (3 * (classes + 5)) x H x W`.
- YOLOv8: DFL-decoded distance plus class tensors, shaped as `N x (4 + classes) x grid`.

Default model config is `1x3x640x640`, COCO `80` classes, and output names `output0`, `output1`, `output2`.

## Run Detection

Build the sample target:

```bash
cmake --build build --target inferrt_sample_detection --config Release
```

Run YOLOv8 or Ultralytics YOLOv5u weights:

```bash
build/bin/inferrt_sample_detection.exe ^
  -m yolov8n ^
  -w samples/model/detection/yolov8n.wts ^
  -i assets/pics/dog.jpg ^
  -l assets/coco80.names ^
  -o build/yolov8n_result.jpg ^
  --runtime tensorrt:0 ^
  --warmup 10 ^
  --repeat 100

build/bin/inferrt_sample_detection.exe ^
  -m yolov5n ^
  -w samples/model/detection/yolov5n.wts ^
  --conf-threshold 0.25 ^
  --nms-threshold 0.45
```

Run RF-DETR weights through the same detection sample:

```bash
build/bin/inferrt_sample_detection.exe ^
  -m rfdetr_nano ^
  -w build/python_test_artifacts/rfdetr/rfdetr_nano.wts ^
  -i assets/pics/dog.jpg ^
  -l assets/coco80.names ^
  -o build/rfdetr_nano_result.jpg ^
  --runtime tensorrt:0 ^
  --warmup 5 ^
  --repeat 20
```

Run RF-DETR through the ONNX graph (onnx -> TensorRT parser, `ONNXModel`) instead of the native `.wts` builder.
The ONNX graph must expose one `images` input and `dets`/`labels` float outputs (official `rf-detr` `exporter.py`
outputs). The sample routes `.onnx` weights to `ONNXModel` for any backend:

```bash
build/bin/inferrt_sample_detection.exe ^
  -m rfdetr_nano ^
  -w build/python_test_artifacts/rfdetr/rfdetr_nano.onnx ^
  -i assets/pics/dog.jpg ^
  -l assets/coco80.names ^
  -o build/rfdetr_nano_onnx_result.jpg ^
  --runtime tensorrt:0 ^
  --warmup 5 ^
  --repeat 20
```

For `.onnx` weights the input size and output tensor names are taken from the graph, so `--input-size` is ignored.

RF-DETR segmentation variants such as `rfdetr_seg_nano` use `samples/model/segmentation` because they produce masks in
addition to boxes and labels.

YOLO uses letterbox preprocessing. RF-DETR uses the model's registered square input size and ImageNet normalization.
Both paths apply class-wise NMS, print detections, and optionally write a visualized image. The timing line reports
`build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end, timed-loop wall time, and postprocess.

Runtime options:

- `--runtime`: combines backend and device, for example `tensorrt:0`, `onnxruntime:cpu`, `onnxruntime:0`, or `openvino:cpu`; shorthand `cpu`, `gpu:0`, and `cuda:0` is also supported.
- `--warmup`: iterations to run before measurement.
- `--repeat`: measured iterations used for total/avg/min/max timing.
- `--batch-min`, `--batch-opt`, `--batch-max`: native YOLO TensorRT dynamic batch profile. They must satisfy `1 <= min <= opt <= max`; this single-image sample requires `min=1`, so `--batch-min 1 --batch-opt 4 --batch-max 8` builds the profile used by `InferenceEngine`.
- `--batch-size`: inference batch. For native RF-DETR it must be `<= --batch-max`; one dynamic engine (profile `1..batch-max`) serves runs at `1, 2, 4, ...` (e.g. `--batch-min 1 --batch-opt 8 --batch-max 8 --batch-size 2`). For RF-DETR ONNX the graph is static, so export one `.onnx` per batch size and pass `--batch-size` matching the graph; `--batch-size > 1` prints an extra `Per-image` timing line.
- `--precision`: `fp32` (default) or `fp16`; enables TensorRT FP16 tactics (applies to native and ONNX paths, engine cache keyed by precision).

ONNX Runtime and OpenVINO use graph inputs and outputs directly for YOLO. The graph must expose one image input and the same three YOLO output tensors expected by this sample. RF-DETR native `.wts` weights require the TensorRT backend; RF-DETR `.onnx` graphs run on any backend through `ONNXModel`.

For legacy YOLOv5 weights with `model.24.m.*` outputs, the sample uses the standard COCO anchors by default. Custom anchors can be supplied as 18 comma-separated numbers:

```bash
build/bin/inferrt_sample_detection.exe -m yolov5s -w yolov5s_legacy.wts --legacy-anchors "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"
```
