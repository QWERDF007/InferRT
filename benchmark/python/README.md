  用法示例：

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\benchmark_clustering.py --benchmark_min_time=0.05s
  --benchmark_format=json --benchmark_out=benchmark\python\results\clustering.json --benchmark_out_format=json

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\benchmark_curve.py --benchmark_min_time=0.05s
  --benchmark_format=json --benchmark_out=benchmark\python\results\curve.json --benchmark_out_format=json

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\plot_benchmark_results.py
  benchmark\python\results\clustering.json benchmark\python\results\curve.json --output-dir benchmark\python\plots

不传 JSON 参数时，脚本默认读取 benchmark\python\results\*.json，图片默认保存到 benchmark\python\plots。
聚类图会按 D 维度拆分，N 轴默认使用 log scale。支持参数：
--image-format png|svg|pdf、--linear-x、--log-y、--time-field real_time|cpu_time。

## DINO / LingBot-Vision

`benchmark_vision.py` 比较 PyTorch 与 InferRT 的 GPU 常驻前向，默认覆盖：

- `dinov2_vits14`、`dinov2_vitb14`
- `dinov3_vits16`、`dinov3_vitb16`
- `lingbot_vision_vits16`、`lingbot_vision_vitb16`

每个模型会测试 batch `1,2,4,8` 和 FP32/FP16。默认模型、权重和源码路径分别对应：

- `F:\models\dinov2`、`F:\models\dinov3`、`F:\models\lingbot-vision`
- `F:\Github\DINO-based\dinov2`、`F:\Github\DINO-based\dinov3`
- `F:\Github\lingbot-vision`

首次运行会把 LingBot-Vision 的 `.pt` 转成 `.wts`，放在 `build\benchmark_vision_artifacts`，并由 C++ benchmark 复用：

```powershell
python benchmark\python\benchmark_vision.py `
  --benchmark_min_time=0.05s `
  --benchmark_format=json `
  --benchmark_out=benchmark\python\results\vision.json
```

可用 `--models dinov2_vits14,lingbot_vision_vits16`、`--batches 1,2`、
`--precisions fp32,fp16` 缩小测试范围；`--build-dir`、`--model-root`、
`--dino-root` 和 `--lingbot-root` 可覆盖默认路径。

## C++ TensorRT benchmark

构建目标为 `inferrt_benchmark_vision_models`。它只统计 CUDA event 包围的
InferRT 推理时间，支持相同的模型、动态 batch、FP32/FP16 参数：

```powershell
build\bin\inferrt_benchmark_vision_models.exe `
  --models all --batches 1,2,4,8 --precisions fp32,fp16 `
  --benchmark_min_time=0.05s
```

C++ benchmark 会递归搜索 `--weights-root`（默认 `F:\models`）和
`--artifact-root`（默认 `build\benchmark_vision_artifacts`）中的 `.wts`。
DINO 的现有 `.wts` 可直接使用；LingBot-Vision 需要先运行上面的 Python
benchmark 生成 `.wts`，或者手动传入 `--weights-file`。FP16/FP32 engine
会通过 manifest 中的 precision 字段区分缓存。
