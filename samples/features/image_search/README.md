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
build/bin/inferrt_sample_image_search.exe --weights-file <weights_or_model_file> --gallery-dir <gallery_dir> --query-image <query_image> [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--runtime tensorrt:0|onnxruntime:cpu|onnxruntime:0|openvino:cpu] [--norm l2|l1|none] [--preprocess-backend cpu|gpu] [--faiss-backend cpu|gpu] [--index-storage ram|disk] [--model-batch-size N] [--rebuild-index]
build/bin/inferrt_sample_image_search.exe -w <weights_or_model_file> -g <gallery_dir> -q <query_image> [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--runtime tensorrt:0|onnxruntime:cpu|onnxruntime:0|openvino:cpu] [--norm l2|l1|none] [--preprocess-backend cpu|gpu] [--faiss-backend cpu|gpu] [--index-storage ram|disk] [--model-batch-size N] [--rebuild-index]
build/bin/inferrt_sample_image_search.exe --help
```

Example:

```bash
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe -w assets/models/resnet/resnet18.wts -g assets/pics -q assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --index build/gallery/resnet18_layer4.faiss
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet50.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model resnet50 --feature layer3
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend cpu --index-storage disk --model-batch-size 4 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/dinov2/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --model-batch-size 4 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/resnet/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --norm l2 --preprocess-backend cpu --faiss-backend gpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --runtime onnxruntime:cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file D:/Models/dinov2/<checkpoint-stem-or-dir>/dinov2_vits14.features.onnx --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --runtime openvino:cpu --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/dinov2/dinov2_vits14.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov2_vits14 --feature x_norm_clstoken --index build/gallery/dinov2_vits14_x_norm_clstoken.faiss
build/bin/inferrt_sample_image_search.exe --weights-file assets/models/dinov3/dinov3_vitb16.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model dinov3_vitb16 --feature x_norm_clstoken --index build/gallery/dinov3_vitb16_x_norm_clstoken.faiss
```

DINO weights can be exported from `samples/model/python/dino_gen_wts.py` with
`python samples/model/python/dino_gen_wts.py -m dinov2_vits14 -b torchhub -o dinov2_vits14.wts` and
`python samples/model/python/dino_gen_wts.py -m dinov3_vitb16 -b transformers -o dinov3_vitb16.wts`.

## Index Reuse

By default, if the specified Faiss index file and its companion path-mapping file already exist, the sample loads them directly and skips rebuilding.

Use `--rebuild-index` when the gallery directory has changed and you want to include newly added images.

## Model And Feature Selection

- `--model`: selects the built-in classification model, default is `resnet18`
- `--feature`: selects the feature tensor name used for retrieval, default is `layer4`
- `--runtime`: combines the feature extraction backend and device, for example `tensorrt:0`, `onnxruntime:cpu`, `onnxruntime:0`, or `openvino:cpu`; shorthand `cpu`, `gpu:0`, and `cuda:0` is also supported
- `--norm`: selects feature normalization, one of `l2`, `l1`, `none`; default is `l2`
- `--preprocess-backend`: selects preprocessing backend, one of `cpu`, `gpu`; default is `cpu`; GPU mode uses CVCUDA for device-side color conversion, resize, padding/crop, and normalization
- `--faiss-backend`: selects Faiss backend, one of `cpu`, `gpu`; default is `cpu`
- `--index-storage`: selects CPU Faiss search storage, one of `ram`, `disk`; default is `ram`; `disk` uses IVF with an on-disk inverted-list sidecar for large galleries; GPU Faiss currently keeps the default RAM behavior
- `--model-batch-size`: controls both feature extraction and Faiss index build batch size; default is `1`; TensorRT uses a dynamic profile, while ONNX Runtime/OpenVINO require an exported graph with dynamic batch
- if `--index` is omitted, the sample writes `<gallery_dir>/<timestamp>.faiss`
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
- `*.manifest.yaml`: model, feature, gallery mapping, and configuration metadata
- `*.faiss.ivfdata`: CPU disk inverted-list data, only when `--index-storage disk` is used
