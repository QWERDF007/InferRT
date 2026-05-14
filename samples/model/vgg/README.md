# VGG Sample

This directory provides classification samples for `vgg11`, `vgg13`, `vgg16`, and `vgg19` based on `torchvision` weights export.

## Files

- `gen_wts.py`: generate `.wts` weights
- `SampleModelVGG.cpp`: TensorRT inference sample
- `CMakeLists.txt`: sample build script

## 1. Generate weights

Run in a Python environment with `torch`, `torchvision`, and `opencv-python` installed:

```bash
cd samples/model/vgg
python gen_wts.py -m vgg11
python gen_wts.py -m vgg13
python gen_wts.py -m vgg16
python gen_wts.py -m vgg19
```

Only verify the PyTorch prediction result without exporting weights:

```bash
python gen_wts.py -m vgg16 -p
```

Specify the output file name if needed:

```bash
python gen_wts.py -m vgg19 -o vgg19_imagenet.wts
```

## 2. Build the sample

From the project root:

```bash
cmake --build build --config Debug --target inferrt_sample_vgg
```

## 3. Run inference

```bash
build/bin/inferrt_sample_vgg.exe <vgg11|vgg13|vgg16|vgg19> <weights_file.wts> <image_path> [label_file]
```

Examples:

```bash
build/bin/inferrt_sample_vgg.exe vgg11 samples/model/vgg/vgg11.wts assets/pics/dog.jpg
build/bin/inferrt_sample_vgg.exe vgg16 samples/model/vgg/vgg16.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
build/bin/inferrt_sample_vgg.exe vgg19 samples/model/vgg/vgg19.wts assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt
```

## 4. Notes

- The sample supports `vgg11`, `vgg13`, `vgg16`, and `vgg19`
- The default input shape is `1x3x224x224`
- The first run builds an engine from `.wts`, and later runs reuse the generated `.engine`
