add_inferrt_benchmark(vision_models
    PRIVATE_LIBS
        benchmark::benchmark
        CUDA::cudart
        inferrt_core
        inferrt_model
        inferrt_util
)
