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
)

target_include_directories(inferrt_model_py
    PRIVATE
        ${Python_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/3rdparty/dlpack/include
)

install(
    TARGETS inferrt_model_py
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
