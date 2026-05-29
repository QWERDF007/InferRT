# Python 配置现在由 cmake/ConfigPython.cmake 统一处理

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

pybind11_add_module(inferrt_model_py MODULE InferRTModelPybind.cpp)

set_target_properties(inferrt_model_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

if(MSVC)
    set_target_properties(inferrt_model_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
endif()

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

install(
    TARGETS inferrt_model_py
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
