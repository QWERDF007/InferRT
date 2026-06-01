# Faiss Image Search Sample

This sample uses InferRT built-in classification and DINO feature models for image retrieval.
By default it uses `resnet18` and the `layer4` feature tensor.

It builds a Faiss index from all images under a gallery directory, then extracts the query image feature and returns the top-k most similar gallery images.

## Build

```bash
cmake --build build --config Debug --target inferrt_sample_image_search
```

## Run

```bash
build/bin/inferrt_sample_image_search.exe --weights-file <weights_or_model_file> --gallery-dir <gallery_dir> --query-image <query_image> [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--backend tensorrt|openvino|onnxruntime] [--device cpu|gpu] [--norm l2|l1|none] [--preprocess-backend cpu|gpu] [--faiss-backend cpu|gpu] [--index-storage ram|disk] [--disk-build-batch-size N] [--model-batch-size N] [--rebuild-index]
build/bin/inferrt_sample_image_search.exe -w <weights_or_model_file> -g <gallery_dir> -q <query_image> [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--backend tensorrt|openvino|onnxruntime] [--device cpu|gpu] [--norm l2|l1|none] [--preprocess-backend cpu|gpu] [--faiss-backend cpu|gpu] [--index-storage ram|disk] [--disk-build-batch-size N] [--model-batch-size N] [--rebuild-index]
build/bin/inferrt_sample_image_search.exe --help
```

Example:

```bash
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe -w samples/model/classification/resnet18.wts -g assets/pics -q assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --index build/gallery/resnet18_layer4.faiss
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet50.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model resnet50 --feature layer3
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend cpu --index-storage disk --disk-build-batch-size 128 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --model-batch-size 4 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend gpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --backend onnxruntime --device cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --backend openvino --device cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --index build/gallery/dinov2_vits14_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/dinov3_vitb16.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov3_vitb16 --feature x_norm_clstoken --index build/gallery/dinov3_vitb16_x_norm_clstoken.faiss
```

DINO weights can be exported from `samples/model/python/classification_gen_wts.py` with
`python samples/model/python/classification_gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts` and
`python samples/model/python/classification_gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts`.

## Index Reuse

By default, if the specified Faiss index file and its companion path-mapping file already exist, the sample loads them directly and skips rebuilding.

Use `--rebuild-index` when the gallery directory has changed and you want to include newly added images.

## Model And Feature Selection

- `--model`: selects the built-in classification model, default is `resnet18`
- `--feature`: selects the feature tensor name used for retrieval, default is `layer4`
- `--backend`: selects the feature extraction backend, one of `tensorrt`, `openvino`, `onnxruntime`; `onnx` and `ort` are accepted aliases for ONNX Runtime
- `--device`: selects the feature extraction device, one of `cpu`, `gpu`; TensorRT requires `gpu`
- `--norm`: selects feature normalization, one of `l2`, `l1`, `none`; default is `l2`
- `--preprocess-backend`: selects preprocessing backend, one of `cpu`, `gpu`; default is `cpu`; GPU preprocessing is reserved and currently reports not implemented
- `--faiss-backend`: selects Faiss backend, one of `cpu`, `gpu`; default is `cpu`
- `--index-storage`: selects CPU Faiss search storage, one of `ram`, `disk`; default is `ram`; `disk` uses IVF with an on-disk inverted-list sidecar for large galleries; GPU Faiss currently keeps the default RAM behavior
- `--disk-build-batch-size`: controls the batch size used while building CPU disk indexes; default is `256`; lower it to reduce peak RAM during build
- `--model-batch-size`: controls the TensorRT feature extraction model batch size; default is `1`; increase it for gallery indexing when the selected model supports dynamic batch
- if `--index` is omitted, the sample writes `<gallery_dir>/<model>_<feature>.faiss`
- DINO models use the engine input size during preprocessing, so `dinov2_vits14` runs at its registered `518x518` default and `dinov3_*` official keys run at `224x224` unless the model config is overridden.
- ONNX Runtime and OpenVINO backends use graph outputs directly. Export the feature you want to search, such as `x_norm_clstoken`, as an ONNX/OpenVINO output first.
- During index construction the sample passes a progress callback to `ImageSearch::buildOrLoad` and prints the current stage plus counts when a stage has measurable progress.

Common feature keys:

- `resnet*`: `stem.conv1`, `stem.relu`, `stem.pool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `flatten`, `logits`
- `alexnet`: `conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `conv4`, `conv5`, `pool3`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `mobilenet_v2`: `stem`, `features.1` ... `features.18`, `flatten`, `logits`
- `mobilenet_v3_large` / `mobilenet_v3_small`: `stem`, `features.1` ... final feature block, `flatten`, `classifier.0`, `logits`
- `vgg*`: `block1`, `block2`, `block3`, `block4`, `block5`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `vit*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `norm`, `cls`, `pre_logits`, `logits`
- `dinov2*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_norm_regtokens`, `x_norm_patchtokens`
- `dinov3*`: `patch_embed`, `tokens`, `blockN` / `blocks.N`, `x_prenorm`, `norm`, `cls`, `pre_logits`, `x_norm_clstoken`, `x_storage_tokens`, `x_norm_patchtokens`

For DINO image retrieval, prefer `x_norm_clstoken` as the compact global vector.
Patch-token features can also be indexed, but they flatten to much larger vectors.

Generated files:

- `*.faiss`: Faiss index file
- `*.faiss.ivfdata`: CPU disk inverted-list sidecar
- `*.faiss.paths.txt`: line-by-line mapping from Faiss vector ids to image paths
- `*.faiss.meta.txt`: simple metadata about the sample configuration
