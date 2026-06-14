# Model Samples

This directory contains the ImageNet-style classification sample assets provided by InferRT.

## Layout

- `classification/`: classification inference sample for all supported classification models
- `detection/`: YOLOv5/YOLOv8/RF-DETR single-image detection sample with decode + NMS
- `feature_extract/`: feature dump sample for checking InferRT vs PyTorch feature consistency
- `image_search/`: Faiss-based image retrieval sample, including CNN and DINO feature tensors
- `onnx/`: ONNX -> TensorRT inference sample
- `python/`: centralized Python scripts, including weight exporters, ONNX exporters, comparators, and pybind11 samples
- `sam/`: SAM/SAM2/SAM3 prompt segmentation sample
- `segmentation/`: RF-DETR-Seg and YOLOv8-Seg image-only instance segmentation sample

## Build

Build the shared sample from the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_classification
cmake --build build --config Debug --target inferrt_sample_detection
cmake --build build --config Debug --target inferrt_sample_feature_extract
cmake --build build --config Debug --target inferrt_sample_image_search
cmake --build build --config Debug --target inferrt_sample_sam
cmake --build build --config Debug --target inferrt_sample_segmentation
cmake --build build --config Debug --target inferrt_model_py
```

## Run

```bash
build/bin/inferrt_sample_classification.exe --model <model_name> --weights-file <weights_or_model_file> [--image-path PATH] [--label-file PATH] [--backend tensorrt|openvino|onnxruntime] [--device cpu|gpu] [--warmup N] [--repeat N]
build/bin/inferrt_sample_classification.exe --help
```

Examples:

```bash
build/bin/inferrt_sample_classification.exe --model alexnet --weights-file samples/model/classification/alexnet.wts --image-path assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file samples/model/classification/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file D:/Models/resnet/<checkpoint-stem-or-dir>/resnet50.onnx --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --backend openvino --device cpu
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file samples/model/classification/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --backend tensorrt --device gpu --warmup 10 --repeat 100
build/bin/inferrt_sample_detection.exe -m yolov8n -w samples/model/detection/yolov8n.wts -i assets/pics/dog.jpg -l assets/coco80.names -o build/yolov8n_result.jpg --backend tensorrt --device gpu --warmup 10 --repeat 100
build/bin/inferrt_sample_detection.exe -m rfdetr_nano -w build/python_test_artifacts/rfdetr/rfdetr_nano.wts -i assets/pics/dog.jpg -l assets/coco80.names -o build/rfdetr_nano_result.jpg --backend tensorrt --device gpu --warmup 5 --repeat 20
build/bin/inferrt_sample_segmentation.exe -m rfdetr_seg_nano -w build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts -i assets/pics/dog.jpg -o build/rfdetr_seg_nano_result.jpg --backend tensorrt --device gpu --warmup 5 --repeat 20
build/bin/inferrt_sample_segmentation.exe -m yolov8n_seg -w build/python_test_artifacts/yolo/yolov8n_seg.wts -i assets/pics/dog.jpg -o build/yolov8n_seg_result.jpg --backend tensorrt --device gpu --conf-threshold 0.10 --warmup 5 --repeat 20
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_mask.jpg --point-x 0.5 --point-y 0.5 --backend tensorrt --device gpu --warmup 5 --repeat 20
```

## Weight export

Generate weights with the shared script:

```bash
python samples/model/python/classification_gen_wts.py -m alexnet
python samples/model/python/classification_gen_wts.py -m resnet50
python samples/model/python/classification_gen_wts.py -m vgg16
python samples/model/python/classification_gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts
python samples/model/python/classification_gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts
python samples/model/python/gen_sam_wts.py -m vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.wts
python samples/model/python/gen_sam_wts.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.wts
python samples/model/python/gen_sam_wts.py -m sam3 -c D:/Models/sam3 -o sam3.wts --skip-forward
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_nano -c F:/models/rfdetr/rf-detr-nano.pth --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_nano.wts --quiet
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_seg_nano -c F:/models/rfdetr/rf-detr-seg-nano.pt --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts --quiet
python samples/model/python/detection_gen_wts.py -m yolov8n_seg -w F:/models/yolov8/yolov8n-seg.pt --ultralytics-repo F:/Github/CV/ultralytics -o build/python_test_artifacts/yolo/yolov8n_seg.wts --quiet
python samples/model/python/export_sam_onnx.py -m sam_vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.onnx
python samples/model/python/export_sam_onnx.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.onnx
python samples/model/python/export_sam_onnx.py -m sam3 -c D:/Models/sam3 -o sam3.onnx
```

SAM3 ONNX export uses `transformers.Sam3Model`; pass a Hugging Face model id or local model directory as
`--checkpoint`.
SAM3 `.wts` export writes the Hugging Face `Sam3Model.state_dict()` for inspection and future native TensorRT
integration; the current SAM3 TensorRT model entry still returns `ERROR_NOT_IMPLEMENTED`.

## Notes

- Input preprocessing is aligned with standard ImageNet classification
- The shared sample reads input and output tensor shapes from the built engine, so ViT/DINO variants can use their registered default sizes or a custom size exported by `classification_gen_wts.py --input-size`
- DINO backbones output feature vectors; the sample prints feature top values when the primary output is not a 1000-class logits tensor
- The first run builds an engine from `.wts`, and later runs reuse the generated `.engine`

## Feature Extraction API

Built-in classification models can now expose intermediate feature tensors through `IModelConfig`.
This is an API-level capability in `inferrt_model`; the existing samples still run the default single-output path unless you provide a custom config in your own code.

Example:

```cpp
auto config = std::make_unique<irt::model::IModelConfig>();
config->setFeatureTensorNames({"layer1", "layer4"});   // layer keys used while building the network
config->setOutputTensorNames({"feat_low", "feat_high"}); // TRT output tensor names (same count as above)
config->setFeatureOnly(true);

auto model = irt::model::CreateModel("resnet50", std::move(config));
model->buildOrLoad("samples/model/classification/resnet50.wts");
model->forwardFeatures(feature_buffers);
```

Behavior:

- default mode: `output_tensor_names` is typically `{"output"}`; `buildOrLoad` builds a full classifier for `infer(...)`
- `featureOnly`: set `feature_tensor_names` and `output_tensor_names` with the same length; runtime binding always uses `output_tensor_names`
- `forwardFeatures(...)` expects buffers ordered as inputs followed by feature outputs listed in `output_tensor_names`

Common feature keys exposed by built-in models:

- `alexnet`: `conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `conv4`, `conv5`, `pool3`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `resnet*`: `stem.conv1`, `stem.relu`, `stem.pool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `flatten`, `logits`
- `mobilenet_v2`: `stem`, `features.1` ... `features.18`, `flatten`, `logits`
- `mobilenet_v3_large` / `mobilenet_v3_small`: `stem`, `features.1` ... final feature block, `flatten`, `classifier.0`, `logits`
- `vgg*`: `block1`, `block2`, `block3`, `block4`, `block5`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `vit*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `norm`, `cls`, `pre_logits`, `logits`
- `dinov2*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens`
- `dinov3*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_storage_tokens`, `x_norm_patchtokens`

ONNX / OpenVINO note:

- ONNX graph backends cannot select hidden tensors after export; export the desired features as graph outputs first.
- Use `python/export_feature_onnx.py` to turn `forward_features()` keys such as `x_norm_clstoken` into ONNX/OpenVINO output tensors.

## Instance Segmentation Sample

The `segmentation` sample is image-only and does not accept point or box prompts. RF-DETR-Seg returns
`dets`/`labels`/`masks`; YOLOv8-Seg returns three DFL outputs plus `proto`.

```bash
build/bin/inferrt_sample_segmentation.exe -m rfdetr_seg_nano -w build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts -i assets/pics/dog.jpg -o build/rfdetr_seg_nano_result.jpg --backend tensorrt --device gpu
build/bin/inferrt_sample_segmentation.exe -m yolov8n_seg -w build/python_test_artifacts/yolo/yolov8n_seg.wts -i assets/pics/dog.jpg -o build/yolov8n_seg_result.jpg --backend tensorrt --device gpu --conf-threshold 0.10
build/bin/inferrt_sample_segmentation.exe --help
```

## SAM Prompt Sample

The SAM sample exercises the selected InferRT backend and uses the default prompt contract:
`image`, `point_coords`, `point_labels`, `mask_input`, `has_mask_input` -> `masks`, `iou_predictions`, `low_res_masks`.
SAM v1 builds the official ViT image encoder, prompt encoder, and mask decoder from official `segment_anything`
checkpoints exported by `python/gen_sam_wts.py`. SAM2/SAM2.1 builds the official Hiera image encoder, FPN
neck, prompt encoder, and high-resolution mask decoder from checkpoints exported by `python/gen_sam_wts.py`.
SAM3 keys are registered, but their native backbone currently fails explicitly with `ERROR_NOT_IMPLEMENTED`.

```bash
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_vit_b_mask.jpg --box 0.2,0.2,0.8,0.8 --backend tensorrt --device gpu --warmup 5 --repeat 20
build/bin/inferrt_sample_sam.exe -m sam2_1_hiera_tiny -w samples/model/sam/sam2_1_hiera_tiny.wts -i assets/pics/dog.jpg -o build/sam2_mask.jpg --backend tensorrt --device gpu
build/bin/inferrt_sample_sam.exe --help
```

Detection, segmentation, and SAM samples report `build_or_load`, `preprocess`, H2D, inference, D2H, end-to-end,
timed-loop wall time, and postprocess. See [`detection/README.md`](detection/README.md),
[`segmentation/README.md`](segmentation/README.md), and [`sam/README.md`](sam/README.md) for options and runtime
contracts.

## Feature Comparison Sample

Use the dedicated feature sample to dump InferRT tensors and compare them with a PyTorch reference:

```bash
build/bin/inferrt_sample_feature_extract.exe -m resnet18 -w samples/model/classification/resnet18.wts -f layer1,layer4 -i assets/pics/dog.jpg -o build/feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w samples/model/classification/dinov2_vits14.wts -f x_norm_clstoken,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov2_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx -f x_norm_clstoken -i assets/pics/dog.jpg -o build/dinov2_openvino_cpu_feature_dump --backend openvino --device cpu --warmup 10 --repeat 100
build/bin/inferrt_sample_feature_extract.exe -m dinov3_vitb16 -w samples/model/classification/dinov3_vitb16.wts -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov3_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe --help
python samples/model/python/compare_features.py --compare_dir build/feature_dump_cpp
```

The dedicated feature sample always configures the model as `featureOnly=true`, so it builds/loads the truncated
feature extractor directly.

See [`feature_extract/README.md`](feature_extract/README.md) for the dump format and more usage examples.

## Faiss Image Search Sample

Use the dedicated image search sample to build an image retrieval index from a gallery directory and query top-k similar images:

```bash
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe -w samples/model/classification/resnet18.wts -g assets/pics -q assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --faiss-backend cpu --index-storage disk --disk-build-batch-size 128 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend gpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --backend onnxruntime --device cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --backend openvino --device cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet50.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model resnet50 --feature layer3
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --index build/gallery/dinov2_vits14_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/dinov3_vitb16.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov3_vitb16 --feature x_norm_clstoken --index build/gallery/dinov3_vitb16_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --help
```

Behavior:

- default model: `resnet18`
- default feature tensor: `layer4`
- `--model` and `--feature` can be used to switch to other built-in classification, ViT, and DINO feature tensors
- `--backend` selects feature extraction runtime: `tensorrt`, `openvino`, or `onnxruntime`; graph backends require the requested feature to be exported as a graph output
- `--device` selects `cpu` or `gpu`; TensorRT requires `gpu`
- `--norm` selects `l2`, `l1`, or `none`
- `--preprocess-backend` selects `cpu` or `gpu`; GPU preprocessing is reserved and currently reports not implemented
- `--faiss-backend` selects `cpu` or `gpu`
- `--index-storage` selects `ram` or `disk` for CPU Faiss search; `disk` uses IVF with an on-disk inverted-list sidecar for large galleries; GPU Faiss currently keeps RAM behavior
- `--disk-build-batch-size` controls CPU disk index build memory; default is `256`
- default `top_k`: `5`
- if the target Faiss index already exists, the sample reuses it by default
- pass `--rebuild-index` to rescan the gallery and include newly added images
- the sample prints index build progress through the new batch completion callback

See [`image_search/README.md`](image_search/README.md) for details.

## Python Binding Sample

The pybind11-based Python samples show how to create an InferRT model, run NumPy inference, and dump intermediate features directly from Python:

```bash
cmake -S . -B build -DINFERRT_BUILD_PYTHON=ON -DINFERRT_PYTHON_ROOT=D:/Software/anaconda3/envs/py312
cmake --build build --config Debug --target inferrt_model_py
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/SamplePythonClassification.py
```

The Python extension is generated under `build/bin` together with the dependent InferRT DLLs.

Feature extraction from Python:

```bash
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/python_feature_extract.py --build-dir build_py312_final --model resnet18 --weights samples/model/classification/resnet18.wts --features layer1,layer4 --output-dir build/feature_dump_py
```

See [`python/README.md`](python/README.md) for details.
