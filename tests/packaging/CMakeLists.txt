set(_inferrt_relocation_args
    "-DINFERRT_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DINFERRT_BUILD_DIR=${CMAKE_BINARY_DIR}"
    "-DINFERRT_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/relocation"
    "-DINFERRT_ENABLE_CUDA=${${PROJECT_NAME_UPPER}_ENABLE_CUDA}"
    "-DINFERRT_BUILD_TENSORRT=${${PROJECT_NAME_UPPER}_BUILD_TENSORRT}"
)

# Full installs resolve their external dependencies again from the consumer
# process. Forward explicitly selected roots so the test exercises the same
# dependency graph as the build rather than relying on a stale cache.
foreach(_inferrt_dependency_var IN ITEMS
        CUDAToolkit_ROOT
        TRT_ROOT
        TensorRT_ROOT
        ONNXRUNTIME_ROOT
        ONNXRuntime_ROOT
        INFERRT_OPENVINO_ROOT
        OpenVINO_DIR
        Faiss_HOME
        Faiss_ROOT
        MKL_ROOT
        OpenCV_DIR)
    if(DEFINED ${_inferrt_dependency_var} AND NOT "${${_inferrt_dependency_var}}" STREQUAL "")
        list(APPEND _inferrt_relocation_args
             "-D${_inferrt_dependency_var}=${${_inferrt_dependency_var}}")
    endif()
endforeach()

add_test(
    NAME inferrt_package_relocation
    COMMAND "${CMAKE_COMMAND}" ${_inferrt_relocation_args}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_relocation.cmake"
)
set_tests_properties(inferrt_package_relocation PROPERTIES LABELS "packaging;integration")
