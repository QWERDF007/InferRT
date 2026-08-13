add_inferrt_benchmark(engine
    PRIVATE_LIBS
        cxxopts::cxxopts
        inferrt_engine
        inferrt_model
        opencv_imgcodecs
        opencv_imgproc
)

target_include_directories(${PROJECT_NAME_LOWER}_benchmark_engine
    PRIVATE
        ${PROJECT_SOURCE_DIR}/samples/engine/engine_basic
)
