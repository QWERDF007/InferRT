add_inferrt_test(model
    PRIVATE_LIBS
        GTest::gtest_main
        GTest::gmock
        inferrt_core
        inferrt_model
        CUDA::cudart
        TensorRT::TensorRT
)

target_include_directories(inferrt_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/model)
