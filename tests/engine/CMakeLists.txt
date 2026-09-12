add_inferrt_test(engine
    PRIVATE_LIBS
        GTest::gtest_main
        GTest::gmock
        inferrt_core
        inferrt_engine
        TensorRT::TensorRT
)
