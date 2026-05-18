# Model Samples

This directory contains the ImageNet-style classification sample assets provided by InferRT.

## Layout

- `classification/`: shared weight export and inference entry for all supported classification models
- `features/`: feature dump sample plus a Python comparator for checking InferRT vs PyTorch feature consistency
- `image_search/`: ResNet18 `layer4` feature extraction plus Faiss-based image retrieval sample
- `onnx/`: ONNX export script and ONNX -> TensorRT inference sample

## Build

Build the shared sample from the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_classification
cmake --build build --config Debug --target inferrt_sample_features
cmake --build build --config Debug --target inferrt_sample_image_search
```

## Run

```bash
build/bin/inferrt_sample_classification.exe <model_name> <weights_file.wts> <image_path> [label_file]
```

Examples:

```bash
build/bin/inferrt_sample_classification.exe alexnet samples/model/classification/alexnet.wts assets/pics/dog.jpg
build/bin/inferrt_sample_classification.exe resnet50 samples/model/classification/resnet50.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_classification.exe vgg16 samples/model/classification/vgg16.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

## Weight export

Generate weights with the shared script:

```bash
cd samples/model/classification
python gen_wts.py -m alexnet
python gen_wts.py -m resnet50
python gen_wts.py -m vgg16
```

## Notes

- Input preprocessing is aligned with standard ImageNet classification
- The shared sample assumes `1x3x224x224` input and `1000` output classes
- The first run builds an engine from `.wts`, and later runs reuse the generated `.engine`

## Feature Extraction API

Built-in classification models can now expose intermediate feature tensors through `IModelConfig`.
This is an API-level capability in `inferrt_model`; the existing samples still run the default single-output path unless you provide a custom config in your own code.

Example:

```cpp
auto config = std::make_unique<irt::model::IModelConfig>();
config->setFeatureTensorNames({"layer1", "layer4"});
config->setFeatureOutputTensorNames({"feat_low", "feat_high"});
config->setFeatureOnly(true); // optional: build this model instance as a truncated feature extractor

auto model = irt::model::CreateModel("resnet50", std::move(config));
model->buildOrLoad("samples/model/classification/resnet50.wts");
model->forwardFeatures(feature_buffers);
```

Behavior:

- default mode: `buildOrLoad(...)` prepares the normal inference engine and, when feature tensors are requested, a separate truncated feature engine
- `config->setFeatureOnly(true)` builds the model instance itself as a truncated feature extractor and stores/loads the feature engine path directly
- `infer(...)` rejects feature-only engines to prevent running classification on an incomplete network
- `forwardFeatures(...)` uses the truncated feature engine and expects buffers ordered as inputs followed by requested feature outputs
- if `featureOutputTensorNames()` is empty, the feature layer keys themselves are used as output tensor names

Common feature keys exposed by built-in models:

- `alexnet`: `conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `conv4`, `conv5`, `pool3`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `resnet*`: `stem.conv1`, `stem.relu`, `stem.pool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `flatten`, `logits`
- `mobilenet_v2`: `stem`, `features.1` ... `features.18`, `flatten`, `logits`
- `mobilenet_v3_large` / `mobilenet_v3_small`: `stem`, `features.1` ... final feature block, `flatten`, `classifier.0`, `logits`
- `vgg*`: `block1`, `block2`, `block3`, `block4`, `block5`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`

Current limitation:

- `onnx` models do not support selecting internal feature tensors through this API

## Feature Comparison Sample

Use the dedicated feature sample to dump InferRT tensors and compare them with a PyTorch reference:

```bash
build/bin/inferrt_sample_features.exe resnet18 samples/model/classification/resnet18.wts layer1,layer4 assets/pics/dog.jpg build/feature_dump_cpp
cd samples/model/features
python compare_features.py --compare_dir ../../../build/feature_dump_cpp
```

The dedicated feature sample always configures the model as `featureOnly=true`, so it builds/loads the truncated
feature extractor directly.

See [`features/README.md`](features/README.md) for the dump format and more usage examples.

## Faiss Image Search Sample

Use the dedicated image search sample to build an image retrieval index from a gallery directory and query top-k similar images:

```bash
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet18.wts assets/pics assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet18.wts assets/pics assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet50.wts assets/pics assets/pics/dog.jpg --model resnet50 --feature layer3
```

Behavior:

- default model: `resnet18`
- default feature tensor: `layer4`
- `--model` and `--feature` can be used to switch to other built-in classification models and feature tensors
- default `top_k`: `5`
- if the target Faiss index already exists, the sample reuses it by default
- pass `--rebuild-index` to rescan the gallery and include newly added images

See [`image_search/README.md`](image_search/README.md) for details.
