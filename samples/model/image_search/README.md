# Faiss Image Search Sample

This sample uses InferRT built-in classification models for image retrieval. By default it uses `resnet18` and the `layer4` feature tensor.

It builds a Faiss index from all images under a gallery directory, then extracts the query image feature and returns the top-k most similar gallery images.

## Build

```bash
cmake --build build --config Debug --target inferrt_sample_image_search
```

## Run

```bash
build/bin/inferrt_sample_image_search.exe <weights_file.wts> <gallery_dir> <query_image> [--model NAME] [--feature NAME] [--topk N] [--index PATH] [--rebuild-index]
```

Example:

```bash
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet18.wts assets/pics assets/pics/dog.jpg
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet18.wts assets/pics assets/pics/dog.jpg --topk 5 --rebuild-index
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet18.wts assets/pics assets/pics/dog.jpg --index build/gallery/resnet18_layer4.faiss
build/bin/inferrt_sample_image_search.exe samples/model/classification/resnet50.wts assets/pics assets/pics/dog.jpg --model resnet50 --feature layer3
```

## Index Reuse

By default, if the specified Faiss index file and its companion path-mapping file already exist, the sample loads them directly and skips rebuilding.

Use `--rebuild-index` when the gallery directory has changed and you want to include newly added images.

## Model And Feature Selection

- `--model`: selects the built-in classification model, default is `resnet18`
- `--feature`: selects the feature tensor name used for retrieval, default is `layer4`
- if `--index` is omitted, the sample writes `<gallery_dir>/<model>_<feature>.faiss`

Common feature keys:

- `resnet*`: `stem.conv1`, `stem.relu`, `stem.pool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `flatten`, `logits`
- `alexnet`: `conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `conv4`, `conv5`, `pool3`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`
- `mobilenet_v2`: `stem`, `features.1` ... `features.18`, `flatten`, `logits`
- `mobilenet_v3_large` / `mobilenet_v3_small`: `stem`, `features.1` ... final feature block, `flatten`, `classifier.0`, `logits`
- `vgg*`: `block1`, `block2`, `block3`, `block4`, `block5`, `avgpool`, `flatten`, `fc1`, `fc2`, `logits`

Generated files:

- `*.faiss`: Faiss index file
- `*.faiss.paths.txt`: line-by-line mapping from Faiss vector ids to image paths
- `*.faiss.meta.txt`: simple metadata about the sample configuration
