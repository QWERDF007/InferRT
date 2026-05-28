# Python 配置现在由 cmake/ConfigPython.cmake 统一处理

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

pybind11_add_module(inferrt_model_py MODULE InferRTModelPybind.cpp)

target_link_libraries(inferrt_model_py
    PRIVATE
        inferrt_core
        inferrt_model
        inferrt_util
        CUDA::cudart
        Python::NumPy
        dlpack::dlpack
)

target_include_directories(inferrt_model_py
    PRIVATE
        ${Python_INCLUDE_DIRS}
)

if(INFERRT_WITH_ONNXRUNTIME AND WIN32)
    add_custom_command(TARGET inferrt_model_py POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${ONNXRUNTIME_BIN_DIR}"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model_py>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()

if(INFERRT_WITH_OPENVINO AND WIN32 AND INFERRT_OPENVINO_BIN_ROOT)
    add_custom_command(TARGET inferrt_model_py POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${INFERRT_OPENVINO_BIN_ROOT}/$<CONFIG>"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model_py>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()

if(INFERRT_WITH_OPENVINO AND WIN32 AND INFERRT_OPENVINO_TBB_BIN_DIR)
    add_custom_command(TARGET inferrt_model_py POST_BUILD
        COMMAND ${CMAKE_COMMAND}
                -DINFERRT_COPY_SRC_DIR="${INFERRT_OPENVINO_TBB_BIN_DIR}"
                -DINFERRT_COPY_DST_DIR="$<TARGET_FILE_DIR:inferrt_model_py>"
                -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeFiles.cmake")
endif()

install(
    TARGETS inferrt_model_py
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
