# Vision model benchmark

本文记录 DINOv2、DINOv3 和 LingBot-Vision small/base 模型的 GPU 前向性能。C++
测试使用 InferRT CUDA event；Python 测试分别记录 PyTorch 和 InferRT Python
绑定的 GPU 常驻输入前向时间。

## 测试口径

- 输入图像数量为 N=100 时，表格单元格为顺序处理 100 张图的估算总耗时：`100 / items_per_second * 1000 ms`。
- N=1 时使用 `batch=1` 的单次前向 `real_time`。
- 每个单元格格式为 `耗时（相对倍率）`；相对倍率 = 当前耗时 / 同一 runtime、precision、batch 下的 DINOv2 ViT-S/14 耗时，DINOv2 ViT-S/14 固定为 `1.00x`。
- batch 测试：`1, 2, 4, 8`；精度测试：FP32、FP16；benchmark 参数：`--benchmark_min_time=0.2s --benchmark_repetitions=1`。
- C++ 和 Python 的 InferRT 结果分别列出，二者的计时边界不同，不应直接混作同一条曲线比较。

## 性能图

### N=100：batch 对总耗时的影响

![N=100 batch latency](plots/n100_batch_latency.png)

### N=1：FP32/FP16 延迟对比

![N=1 precision latency](plots/n1_precision_latency.png)

## N=100：不同 batch

| Runtime | Precision | Model | batch=1 | batch=2 | batch=4 | batch=8 |
|---|---|---|---:|---:|---:|---:|
| InferRT C++ | FP32 | DINOv2 ViT-S/14 | 1677.95 ms (1.00x) | 1587.88 ms (1.00x) | 1608.79 ms (1.00x) | 1529.13 ms (1.00x) |
| InferRT C++ | FP32 | DINOv2 ViT-B/14 | 3949.61 ms (2.35x) | 3981.10 ms (2.51x) | 3938.28 ms (2.45x) | 3758.72 ms (2.46x) |
| InferRT C++ | FP32 | DINOv3 ViT-S/16 | 229.48 ms (0.14x) | 163.01 ms (0.10x) | 136.53 ms (0.08x) | 120.67 ms (0.08x) |
| InferRT C++ | FP32 | DINOv3 ViT-B/16 | 480.24 ms (0.29x) | 402.20 ms (0.25x) | 363.20 ms (0.23x) | 340.32 ms (0.22x) |
| InferRT C++ | FP32 | LingBot-Vision ViT-S/16 | 1206.75 ms (0.72x) | 1122.43 ms (0.71x) | 1113.91 ms (0.69x) | 1069.18 ms (0.70x) |
| InferRT C++ | FP32 | LingBot-Vision ViT-B/16 | 2945.26 ms (1.76x) | 2793.32 ms (1.76x) | 2779.39 ms (1.73x) | 2749.34 ms (1.80x) |
| InferRT C++ | FP16 | DINOv2 ViT-S/14 | 310.03 ms (1.00x) | 271.01 ms (1.00x) | 267.35 ms (1.00x) | 266.67 ms (1.00x) |
| InferRT C++ | FP16 | DINOv2 ViT-B/14 | 811.94 ms (2.62x) | 754.19 ms (2.78x) | 761.38 ms (2.85x) | 761.36 ms (2.86x) |
| InferRT C++ | FP16 | DINOv3 ViT-S/16 | 246.08 ms (0.79x) | 176.66 ms (0.65x) | 159.28 ms (0.60x) | 137.72 ms (0.52x) |
| InferRT C++ | FP16 | DINOv3 ViT-B/16 | 162.40 ms (0.52x) | 136.54 ms (0.50x) | 121.03 ms (0.45x) | 109.07 ms (0.41x) |
| InferRT C++ | FP16 | LingBot-Vision ViT-S/16 | 265.80 ms (0.86x) | 226.26 ms (0.83x) | 210.69 ms (0.79x) | 200.55 ms (0.75x) |
| InferRT C++ | FP16 | LingBot-Vision ViT-B/16 | 633.94 ms (2.04x) | 586.47 ms (2.16x) | 585.10 ms (2.19x) | 579.66 ms (2.17x) |
| PyTorch Python | FP32 | DINOv2 ViT-S/14 | 1949.05 ms (1.00x) | 1843.81 ms (1.00x) | 1752.00 ms (1.00x) | 1797.80 ms (1.00x) |
| PyTorch Python | FP32 | DINOv2 ViT-B/14 | 5068.60 ms (2.60x) | 4883.84 ms (2.65x) | 4933.00 ms (2.82x) | 4963.25 ms (2.76x) |
| PyTorch Python | FP32 | DINOv3 ViT-S/16 | 695.14 ms (0.36x) | 365.48 ms (0.20x) | 220.45 ms (0.13x) | 194.36 ms (0.11x) |
| PyTorch Python | FP32 | DINOv3 ViT-B/16 | 866.98 ms (0.44x) | 675.88 ms (0.37x) | 613.51 ms (0.35x) | 582.86 ms (0.32x) |
| PyTorch Python | FP32 | LingBot-Vision ViT-S/16 | 1525.36 ms (0.78x) | 1361.18 ms (0.74x) | 1303.23 ms (0.74x) | 1339.12 ms (0.74x) |
| PyTorch Python | FP32 | LingBot-Vision ViT-B/16 | 3880.56 ms (1.99x) | 3684.52 ms (2.00x) | 3711.77 ms (2.12x) | 3701.11 ms (2.06x) |
| PyTorch Python | FP16 | DINOv2 ViT-S/14 | 523.16 ms (1.00x) | 473.00 ms (1.00x) | 469.98 ms (1.00x) | 467.88 ms (1.00x) |
| PyTorch Python | FP16 | DINOv2 ViT-B/14 | 1342.25 ms (2.57x) | 1299.98 ms (2.75x) | 1303.80 ms (2.77x) | 1329.52 ms (2.84x) |
| PyTorch Python | FP16 | DINOv3 ViT-S/16 | 639.54 ms (1.22x) | 343.19 ms (0.73x) | 166.75 ms (0.35x) | 90.86 ms (0.19x) |
| PyTorch Python | FP16 | DINOv3 ViT-B/16 | 642.05 ms (1.23x) | 318.69 ms (0.67x) | 215.94 ms (0.46x) | 195.63 ms (0.42x) |
| PyTorch Python | FP16 | LingBot-Vision ViT-S/16 | 647.59 ms (1.24x) | 487.20 ms (1.03x) | 476.72 ms (1.01x) | 521.84 ms (1.12x) |
| PyTorch Python | FP16 | LingBot-Vision ViT-B/16 | 1354.23 ms (2.59x) | 1266.20 ms (2.68x) | 1344.41 ms (2.86x) | 1409.91 ms (3.01x) |
| InferRT Python | FP32 | DINOv2 ViT-S/14 | 1675.57 ms (1.00x) | 1609.44 ms (1.00x) | 1609.48 ms (1.00x) | 1518.58 ms (1.00x) |
| InferRT Python | FP32 | DINOv2 ViT-B/14 | 3988.36 ms (2.38x) | 3964.64 ms (2.46x) | 3942.94 ms (2.45x) | 3777.61 ms (2.49x) |
| InferRT Python | FP32 | DINOv3 ViT-S/16 | 236.66 ms (0.14x) | 176.00 ms (0.11x) | 141.53 ms (0.09x) | 126.34 ms (0.08x) |
| InferRT Python | FP32 | DINOv3 ViT-B/16 | 463.43 ms (0.28x) | 398.67 ms (0.25x) | 360.70 ms (0.22x) | 336.84 ms (0.22x) |
| InferRT Python | FP32 | LingBot-Vision ViT-S/16 | 1152.89 ms (0.69x) | 1125.77 ms (0.70x) | 1104.60 ms (0.69x) | 1096.03 ms (0.72x) |
| InferRT Python | FP32 | LingBot-Vision ViT-B/16 | 2943.09 ms (1.76x) | 2961.53 ms (1.84x) | 5465.38 ms (3.40x) | 2690.91 ms (1.77x) |
| InferRT Python | FP16 | DINOv2 ViT-S/14 | 321.26 ms (1.00x) | 288.46 ms (1.00x) | 266.52 ms (1.00x) | 268.98 ms (1.00x) |
| InferRT Python | FP16 | DINOv2 ViT-B/14 | 836.86 ms (2.60x) | 775.07 ms (2.69x) | 762.17 ms (2.86x) | 754.95 ms (2.81x) |
| InferRT Python | FP16 | DINOv3 ViT-S/16 | 238.37 ms (0.74x) | 170.27 ms (0.59x) | 139.60 ms (0.52x) | 120.60 ms (0.45x) |
| InferRT Python | FP16 | DINOv3 ViT-B/16 | 166.99 ms (0.52x) | 144.49 ms (0.50x) | 122.20 ms (0.46x) | 107.24 ms (0.40x) |
| InferRT Python | FP16 | LingBot-Vision ViT-S/16 | 265.76 ms (0.83x) | 228.91 ms (0.79x) | 213.31 ms (0.80x) | 202.04 ms (0.75x) |
| InferRT Python | FP16 | LingBot-Vision ViT-B/16 | 650.18 ms (2.02x) | 624.85 ms (2.17x) | 591.01 ms (2.22x) | 594.44 ms (2.21x) |

## N=1：不同精度

| Runtime | Model | FP32（batch=1） | FP16（batch=1） |
|---|---|---:|---:|
| InferRT C++ | DINOv2 ViT-S/14 | 16.78 ms (1.00x) | 3.10 ms (1.00x) |
| InferRT C++ | DINOv2 ViT-B/14 | 39.50 ms (2.35x) | 8.12 ms (2.62x) |
| InferRT C++ | DINOv3 ViT-S/16 | 2.29 ms (0.14x) | 2.46 ms (0.79x) |
| InferRT C++ | DINOv3 ViT-B/16 | 4.80 ms (0.29x) | 1.62 ms (0.52x) |
| InferRT C++ | LingBot-Vision ViT-S/16 | 12.07 ms (0.72x) | 2.66 ms (0.86x) |
| InferRT C++ | LingBot-Vision ViT-B/16 | 29.45 ms (1.76x) | 6.34 ms (2.04x) |
| PyTorch Python | DINOv2 ViT-S/14 | 19.49 ms (1.00x) | 5.23 ms (1.00x) |
| PyTorch Python | DINOv2 ViT-B/14 | 50.69 ms (2.60x) | 13.42 ms (2.57x) |
| PyTorch Python | DINOv3 ViT-S/16 | 6.95 ms (0.36x) | 6.40 ms (1.22x) |
| PyTorch Python | DINOv3 ViT-B/16 | 8.67 ms (0.44x) | 6.42 ms (1.23x) |
| PyTorch Python | LingBot-Vision ViT-S/16 | 15.25 ms (0.78x) | 6.48 ms (1.24x) |
| PyTorch Python | LingBot-Vision ViT-B/16 | 38.81 ms (1.99x) | 13.54 ms (2.59x) |
| InferRT Python | DINOv2 ViT-S/14 | 16.76 ms (1.00x) | 3.21 ms (1.00x) |
| InferRT Python | DINOv2 ViT-B/14 | 39.88 ms (2.38x) | 8.37 ms (2.60x) |
| InferRT Python | DINOv3 ViT-S/16 | 2.37 ms (0.14x) | 2.38 ms (0.74x) |
| InferRT Python | DINOv3 ViT-B/16 | 4.63 ms (0.28x) | 1.67 ms (0.52x) |
| InferRT Python | LingBot-Vision ViT-S/16 | 11.53 ms (0.69x) | 2.66 ms (0.83x) |
| InferRT Python | LingBot-Vision ViT-B/16 | 29.43 ms (1.76x) | 6.50 ms (2.02x) |

## 原始结果

C++ JSON `F:\tmp\vision_cpp.json`，时间 2026-07-16T15:00:48+08:00  
C++ JSON `F:\tmp\vision_cpp_base.json`，时间 2026-07-16T15:24:40+08:00  
Python JSON `F:\tmp\vision_python.json`，时间 2026-07-16T15:09:36+08:00，device `cuda`，PyTorch `2.8.0+cu128`  
Python JSON `F:\tmp\vision_python_base.json`，时间 2026-07-16T15:28:06+08:00，device `cuda`，PyTorch `2.8.0+cu128`
