  用法示例：

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\benchmark_clustering.py --benchmark_min_time=0.05s
  --benchmark_format=json --benchmark_out=benchmark\python\results\clustering.json --benchmark_out_format=json

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\benchmark_curve.py --benchmark_min_time=0.05s
  --benchmark_format=json --benchmark_out=benchmark\python\results\curve.json --benchmark_out_format=json

  'D:\Software\anaconda3\envs\py312\python.exe' benchmark\python\plot_benchmark_results.py
  benchmark\python\results\clustering.json benchmark\python\results\curve.json --output-dir benchmark\python\plots

  不传 JSON 参数时，脚本默认读取 benchmark\python\results\*.json，图片默认保存到 benchmark\python\plots。支持参
  数：--image-format png|svg|pdf、--log-y、--time-field real_time|cpu_time。