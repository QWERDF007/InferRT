add_inferrt_benchmark(cuda_ops
    PRIVATE_LIBS
        benchmark::benchmark
        inferrt_cvcuda
        opencv_core
        opencv_imgproc
)
