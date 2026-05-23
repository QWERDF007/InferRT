# Model Samples

This directory contains the ImageNet-style classification sample assets provided by InferRT.

## Layout

- `classification/`: shared weight export and inference entry for all supported classification models
- `feature_extract/`: feature dump sample plus a Python comparator for checking InferRT vs PyTorch feature consistency
- `image_search/`: ResNet18 `layer4` feature extraction plus Faiss-based image retrieval sample
- `onnx/`: ONNX export script and ONNX -> TensorRT inference sample
- `python/`: pybind11 Python binding sample for model creation and inference

## Build

Build the shared sample from the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_classification
cmake --build build --config Debug --target inferrt_sample_features
cmake --build build --config Debug --target inferrt_sample_image_search
cmake --build build --config Debug --target inferrt_model_py
```

## Run

```bash
build/bin/inferrt_sample_classification.exe <model_name> <weights_file.wts> [image_path] [label_file]
build/bin/inferrt_sample_classification.exe --help
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
- The shared sample reads input and output tensor shapes from the built engine, so ViT/DINO variants can use their registered default sizes or a custom size exported by `gen_wts.py --input-size`
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

Current limitation:

- `onnx` models do not support selecting internal feature tensors through this API

## Feature Comparison Sample

Use the dedicated feature sample to dump InferRT tensors and compare them with a PyTorch reference:

```bash
build/bin/inferrt_sample_features.exe -m resnet18 -w samples/model/classification/resnet18.wts -f layer1,layer4 -i assets/pics/dog.jpg -o build/feature_dump_cpp
build/bin/inferrt_sample_features.exe --help
cd samples/model/features
python compare_features.py --compare_dir ../../../build/feature_dump_cpp
```

The dedicated feature sample always configures the model as `featureOnly=true`, so it builds/loads the truncated
feature extractor directly.

See [`features/README.md`](features/README.md) for the dump format and more usage examples.

## Faiss Image Search Sample

Use the dedicated image search sample to build an image retrieval index from a gallery directory and query top-k similar images:

```bash
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe -w samples/model/classification/resnet18.wts -g assets/pics -q assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet18.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe --weights-file samples/model/classification/resnet50.wts --gallery-dir assets/pics --query-image assets/pics/dog.jpg --model resnet50 --feature layer3
build/bin/inferrt_sample_image_search.exe --help
```

Behavior:

- default model: `resnet18`
- default feature tensor: `layer4`
- `--model` and `--feature` can be used to switch to other built-in classification models and feature tensors
- default `top_k`: `5`
- if the target Faiss index already exists, the sample reuses it by default
- pass `--rebuild-index` to rescan the gallery and include newly added images

See [`image_search/README.md`](image_search/README.md) for details.

## Python Binding Sample

The pybind11-based Python samples show how to create an InferRT model, run NumPy inference, and dump intermediate features directly from Python:

```bash
cmake -S . -B build -DINFERRT_BUILD_PYTHON=ON -DINFERRT_PYTHON_ROOT=D:/Software/anaconda3/envs/py312
cmake --build build --config Debug --target inferrt_model_py
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/SamplePythonClassification.py
```

The Python extension is generated under `build/lib`, and the dependent InferRT DLLs remain under `build/bin`.

Feature extraction from Python:

```bash
D:/Software/anaconda3/envs/py312/python.exe samples/model/python/python_feature_extract.py --build-dir build_py312_final --model resnet18 --weights samples/model/classification/resnet18.wts --features layer1,layer4 --output-dir build/feature_dump_py
```

See [`python/README.md`](python/README.md) for details.
