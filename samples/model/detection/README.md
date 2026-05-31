# YOLO Detection

This directory contains the InferRT native YOLOv5 and YOLOv8 detection C++ sample.
The Python weight exporter lives in `samples/model/python/detection_gen_wts.py`.

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
  --backend tensorrt ^
  --device gpu ^
  --warmup 10 ^
  --repeat 100

build/bin/inferrt_sample_detection.exe ^
  -m yolov5n ^
  -w samples/model/detection/yolov5n.wts ^
  --conf-threshold 0.25 ^
  --nms-threshold 0.45
```

The sample uses letterbox preprocessing, runs the selected backend, decodes YOLO heads, applies class-wise NMS, prints detections, and optionally writes a visualized image. The timing line reports `build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end, timed-loop wall time, and postprocess.

Backend options:

- `--backend`: `tensorrt`, `openvino`, or `onnxruntime`; `onnx` and `ort` are accepted aliases for ONNX Runtime.
- `--device`: `cpu` or `gpu`; TensorRT requires `gpu`.
- `--warmup`: iterations to run before measurement.
- `--repeat`: measured iterations used for total/avg/min/max timing.

ONNX Runtime and OpenVINO use graph inputs and outputs directly. The graph must expose one image input and the same three YOLO output tensors expected by this sample.

For legacy YOLOv5 weights with `model.24.m.*` outputs, the sample uses the standard COCO anchors by default. Custom anchors can be supplied as 18 comma-separated numbers:

```bash
build/bin/inferrt_sample_detection.exe -m yolov5s -w yolov5s_legacy.wts --legacy-anchors "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"
```
