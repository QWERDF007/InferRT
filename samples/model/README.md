# Model 示例

本目录包含 InferRT 提供的 ImageNet 风格分类示例资源。

## 目录结构

- `classification/`：所有已支持分类模型的分类推理示例
- `detection/`：YOLOv5/YOLOv8/RF-DETR 单图检测示例（含 decode + NMS）
- `feature_extract/`：特征导出示例，用于对比 InferRT 与 PyTorch 的特征一致性
- `image_search/`：基于 Faiss 的图像检索示例，涵盖 CNN 与 DINO 特征张量
- `onnx/`：ONNX -> TensorRT 推理示例
- `python/`：Python 脚本集中目录，包括权重导出、ONNX 导出、对比工具及 pybind11 示例
- `sam/`：SAM/SAM2/SAM3 提示分割示例
- `segmentation/`：RF-DETR-Seg 和 YOLOv8-Seg 纯图像实例分割示例

## 构建

在项目根目录下构建各示例：

```bash
cmake --build build --config Debug --target inferrt_sample_classification
cmake --build build --config Debug --target inferrt_sample_detection
cmake --build build --config Debug --target inferrt_sample_feature_extract
cmake --build build --config Debug --target inferrt_sample_image_search
cmake --build build --config Debug --target inferrt_sample_sam
cmake --build build --config Debug --target inferrt_sample_segmentation
cmake --build build --config Debug --target inferrt_model_py
```

## 运行

```bash
build/bin/inferrt_sample_classification.exe --model <model_name> --weights-file <weights_or_model_file> [--image-path PATH] [--label-file PATH] [--runtime tensorrt:0|onnxruntime:cpu|onnxruntime:0|openvino:cpu] [--warmup N] [--repeat N]
build/bin/inferrt_sample_classification.exe --help
```

示例：

```bash
build/bin/inferrt_sample_classification.exe --model alexnet --weights-file assets/models/alexnet/alexnet.wts --image-path assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file assets/models/resnet/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file D:/Models/resnet/<checkpoint-stem-or-dir>/resnet50.onnx --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --runtime openvino:cpu
build/bin/inferrt_sample_classification.exe --model resnet50 --weights-file assets/models/resnet/resnet50.wts --image-path assets/pics/dog.jpg --label-file assets/imagenet1000_clsidx_to_labels.txt --runtime tensorrt:0 --warmup 10 --repeat 100
build/bin/inferrt_sample_detection.exe -m yolov8n -w samples/model/detection/yolov8n.wts -i assets/pics/dog.jpg -l assets/coco80.names -o build/yolov8n_result.jpg --runtime tensorrt:0 --warmup 10 --repeat 100
build/bin/inferrt_sample_detection.exe -m rfdetr_nano -w build/python_test_artifacts/rfdetr/rfdetr_nano.wts -i assets/pics/dog.jpg -l assets/coco80.names -o build/rfdetr_nano_result.jpg --runtime tensorrt:0 --warmup 5 --repeat 20
build/bin/inferrt_sample_segmentation.exe -m rfdetr_seg_nano -w build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts -i assets/pics/dog.jpg -o build/rfdetr_seg_nano_result.jpg --runtime tensorrt:0 --warmup 5 --repeat 20
build/bin/inferrt_sample_segmentation.exe -m yolov8n_seg -w build/python_test_artifacts/yolo/yolov8n_seg.wts -i assets/pics/dog.jpg -o build/yolov8n_seg_result.jpg --runtime tensorrt:0 --conf-threshold 0.10 --warmup 5 --repeat 20
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_mask.jpg --point-x 0.5 --point-y 0.5 --runtime tensorrt:0 --warmup 5 --repeat 20
```

## 权重导出

使用统一脚本导出权重：

```bash
python samples/model/python/classification_gen_wts.py -m alexnet
python samples/model/python/classification_gen_wts.py -m resnet50
python samples/model/python/classification_gen_wts.py -m vgg16
python samples/model/python/dino_gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts
python samples/model/python/dino_gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts
python samples/model/python/sam_gen_wts.py -m vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.wts
python samples/model/python/sam_gen_wts.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.wts
python samples/model/python/sam_gen_wts.py -m sam3 -c D:/Models/sam3 -o sam3.wts --skip-forward
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_nano -c F:/models/rfdetr/rf-detr-nano.pth --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_nano.wts --quiet
python samples/model/python/rfdetr_gen_wts.py -m rfdetr_seg_nano -c F:/models/rfdetr/rf-detr-seg-nano.pt --rfdetr-root F:/Github/CV/rf-detr -o build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts --quiet
python samples/model/python/detection_gen_wts.py -m yolov8n_seg -w F:/models/yolov8/yolov8n-seg.pt --ultralytics-repo F:/Github/CV/ultralytics -o build/python_test_artifacts/yolo/yolov8n_seg.wts --quiet
python samples/model/python/sam_export_onnx.py -m sam_vit_b -c D:/Models/sam_vit_b_01ec64.pth --sam-root D:/Github/SAM/segment-anything -o sam_vit_b.onnx
python samples/model/python/sam_export_onnx.py -m sam2_1_hiera_tiny -c D:/Models/sam2.1_hiera_tiny.pt --sam2-root D:/Github/SAM/sam2 -o sam2_1_hiera_tiny.onnx
python samples/model/python/sam_export_onnx.py -m sam3 -c D:/Models/sam3 -o sam3.onnx
```

SAM3 ONNX 导出使用 `transformers.Sam3Model`；通过 `--checkpoint` 传入 Hugging Face 模型 id 或本地模型目录。
SAM3 `.wts` 导出会写出 Hugging Face `Sam3Model.state_dict()` 供检查及后续原生 TensorRT 集成使用；当前 SAM3 TensorRT 模型条目仍返回 `ERROR_NOT_IMPLEMENTED`。

## 说明

- 输入预处理与标准 ImageNet 分类对齐
- ONNX/权重转换命令、DINO 特征输出和第三方依赖见 [`python/README.md`](python/README.md)
- DINO 主干输出特征向量；当主输出不是 1000 类 logits 张量时，示例会打印特征 top 值
- 首次运行会从 `.wts` 构建 engine，后续运行复用已生成的 `.engine`

## 特征提取 API

内置分类模型现可通过 `IModelConfig` 暴露中间特征张量。这是 `inferrt_model` 的 API 层能力；现有示例在不提供自定义配置的情况下仍走默认单输出路径。

示例：

```cpp
auto config = std::make_unique<irt::model::IModelConfig>();
config->setFeatureTensorNames({"layer1", "layer4"});   // 构建网络时使用的层 key
config->setOutputTensorNames({"feat_low", "feat_high"}); // TRT 输出张量名（与上面对应）
config->setFeatureOnly(true);

auto model = irt::model::CreateModel("resnet50", std::move(config));
model->buildOrLoad("assets/models/resnet/resnet50.wts");
model->forwardFeatures(feature_buffers);
```

行为：

- 默认模式：`output_tensor_names` 通常为 `{"output"}`；`buildOrLoad` 构建完整分类器供 `infer(...)` 使用
- `featureOnly`：设置 `feature_tensor_names` 和 `output_tensor_names` 为相同长度；运行时绑定始终使用 `output_tensor_names`
- `forwardFeatures(...)` 要求缓冲区按"输入在前、特征输出在后"的顺序排列，特征输出顺序与 `output_tensor_names` 一致

内置模型常用的特征 key：

- `alexnet`：`conv1`、`pool1`、`conv2`、`pool2`、`conv3`、`conv4`、`conv5`、`pool3`、`avgpool`、`flatten`、`fc1`、`fc2`、`logits`
- `resnet*`：`stem.conv1`、`stem.relu`、`stem.pool`、`layer1`、`layer2`、`layer3`、`layer4`、`avgpool`、`flatten`、`logits`
- `mobilenet_v2`：`stem`、`features.1` ... `features.18`、`flatten`、`logits`
- `mobilenet_v3_large` / `mobilenet_v3_small`：`stem`、`features.1` ... 最终特征块、`flatten`、`classifier.0`、`logits`
- `vgg*`：`block1`、`block2`、`block3`、`block4`、`block5`、`avgpool`、`flatten`、`fc1`、`fc2`、`logits`
- `vit*`：`patch_embed`、`tokens`、`blockN` / `blocks.N`、`norm`、`cls`、`pre_logits`、`logits`
- `dinov2*`：`patch_embed`、`tokens`、`blockN` / `blocks.N`、`x_prenorm`、`norm`、`cls`、`pre_logits`、`x_norm_clstoken`、`x_norm_regtokens`、`x_norm_patchtokens`
- `dinov3*`：`patch_embed`、`tokens`、`blockN` / `blocks.N`、`x_prenorm`、`norm`、`cls`、`pre_logits`、`x_norm_clstoken`、`x_storage_tokens`、`x_norm_patchtokens`

ONNX / OpenVINO 注意事项：

- ONNX 图后端无法在导出后选取隐藏张量；需要先将目标特征导出为图输出
- 使用 `python/dino_export_onnx.py` 将 DINO/LingBot-Vision `forward_features()` 的 key（如 `x_norm_clstoken`）转为 ONNX/OpenVINO 输出张量

## 实例分割示例

`segmentation` 示例为纯图像输入，不接受点或框提示。RF-DETR-Seg 返回 `dets`/`labels`/`masks`；YOLOv8-Seg 返回三个 DFL 输出加上 `proto`。

```bash
build/bin/inferrt_sample_segmentation.exe -m rfdetr_seg_nano -w build/python_test_artifacts/rfdetr/rfdetr_seg_nano.wts -i assets/pics/dog.jpg -o build/rfdetr_seg_nano_result.jpg --runtime tensorrt:0
build/bin/inferrt_sample_segmentation.exe -m yolov8n_seg -w build/python_test_artifacts/yolo/yolov8n_seg.wts -i assets/pics/dog.jpg -o build/yolov8n_seg_result.jpg --runtime tensorrt:0 --conf-threshold 0.10
build/bin/inferrt_sample_segmentation.exe --help
```

## SAM 提示分割示例

SAM 示例驱动所选 InferRT 后端，使用默认提示协议：
`image`、`point_coords`、`point_labels`、`mask_input`、`has_mask_input` -> `masks`、`iou_predictions`、`low_res_masks`。
SAM v1 基于 `python/sam_gen_wts.py` 导出的官方 `segment_anything` 权重构建 ViT 图像编码器、提示编码器和掩码解码器。SAM2/SAM2.1 基于 `python/sam_gen_wts.py` 导出的权重构建 Hiera 图像编码器、FPN 颈部、提示编码器和高分辨率掩码解码器。SAM3 key 已注册，但其原生主干目前显式返回 `ERROR_NOT_IMPLEMENTED`。

```bash
build/bin/inferrt_sample_sam.exe -m sam_vit_b -w samples/model/sam/sam_vit_b.wts -i assets/pics/dog.jpg -o build/sam_vit_b_mask.jpg --box 0.2,0.2,0.8,0.8 --runtime tensorrt:0 --warmup 5 --repeat 20
build/bin/inferrt_sample_sam.exe -m sam2_1_hiera_tiny -w samples/model/sam/sam2_1_hiera_tiny.wts -i assets/pics/dog.jpg -o build/sam2_mask.jpg --runtime tensorrt:0
build/bin/inferrt_sample_sam.exe --help
```

检测、分割和 SAM 示例均会报告 `build_or_load`、预处理、H2D、推理、D2H、端到端耗时、计时循环耗时及后处理耗时。选项和运行时协议详见 [`detection/README.md`](detection/README.md)、[`segmentation/README.md`](segmentation/README.md) 和 [`sam/README.md`](sam/README.md)。

## 特征对比示例

使用专用特征示例导出 InferRT 张量并与 PyTorch 参考输出对比：

```bash
build/bin/inferrt_sample_feature_extract.exe -m resnet18 -w assets/models/resnet/resnet18.wts -f layer1,layer4 -i assets/pics/dog.jpg -o build/feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w assets/models/dinov2/dinov2_vits14.wts -f x_norm_clstoken,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov2_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe -m dinov2_vits14 -w D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx -f x_norm_clstoken -i assets/pics/dog.jpg -o build/dinov2_openvino_cpu_feature_dump --runtime openvino:cpu --warmup 10 --repeat 100
build/bin/inferrt_sample_feature_extract.exe -m dinov3_vitb16 -w assets/models/dinov3/dinov3_vitb16.wts -f x_norm_clstoken,x_storage_tokens,x_norm_patchtokens -i assets/pics/dog.jpg -o build/dinov3_feature_dump_cpp
build/bin/inferrt_sample_feature_extract.exe --help
```

特征示例始终将模型配置为 `featureOnly=true`，因此直接构建/加载截断后的特征提取器。

导出格式和更多用法示例见 [`feature_extract/README.md`](feature_extract/README.md)。

## Faiss 图像搜索示例

使用专用图像搜索示例从图库目录构建图像检索索引并查询 top-k 相似图像：

```bash
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe -w assets/models/resnet/resnet18.wts -g assets/pics -q assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --faiss-backend cpu --index-storage disk --model-batch-size 4 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend gpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --runtime onnxruntime:cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --runtime openvino:cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet50.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model resnet50 --feature layer3
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/dinov2/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --index build/gallery/dinov2_vits14_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/dinov3/dinov3_vitb16.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov3_vitb16 --feature x_norm_clstoken --index build/gallery/dinov3_vitb16_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --help
```

行为：

- 默认模型：`resnet18`
- 默认特征张量：`layer4`
- `--model` 和 `--feature` 可切换至其他内置分类、ViT 和 DINO 特征张量
- `--runtime` 统一选择模型后端和设备：GPU 使用 `tensorrt:0`、`onnxruntime:0` 或 `openvino:0`，CPU 使用 `onnxruntime:cpu` 或 `openvino:cpu`；也支持裸设备简写 `cpu`、`gpu:0`、`cuda:0`
- `--norm` 选择归一化方式：`l2`、`l1` 或 `none`
- `--preprocess-backend` 选择预处理后端：`cpu` 或 `gpu`；GPU 预处理为预留项，当前报告未实现
- `--faiss-backend` 选择 Faiss 后端：`cpu` 或 `gpu`
- `--index-storage` 选择 CPU Faiss 搜索存储方式：`ram` 或 `disk`；`disk` 使用 IVF 配合磁盘倒排链表侧文件以支持大规模图库；GPU Faiss 当前保持 RAM 行为
- `--model-batch-size` 同时控制特征提取和 Faiss 索引构建的批大小；默认为 `1`
- 默认 `top_k`：`5`
- 若目标 Faiss 索引已存在，默认复用
- 传入 `--rebuild-index` 可重新扫描图库以纳入新增图像
- 示例通过批次完成回调打印索引构建进度

详见 [`image_search/README.md`](image_search/README.md)。

## Python 模型转换

Python 目录只保留分类、DINO、YOLO、RF-DETR 和 SAM 的 `.wts`/`.onnx` 转换工具。
完整命令和依赖见 [`python/README.md`](python/README.md)。
