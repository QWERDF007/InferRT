set(TARGET_NAME inferrt_model_test)
file(GLOB SRCS *.cpp)


add_executable(${TARGET_NAME}
    ${SRCS}
)

target_link_libraries(${TARGET_NAME} 
PRIVATE
    GTest::gtest_main GTest::gmock
    inferrt_core inferrt_model
    CUDA::cudart
    TensorRT::TensorRT
)

# 注册测试到 CTest
add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
