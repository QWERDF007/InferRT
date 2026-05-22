if(NOT Faiss_LIBS)
    message(STATUS "Faiss not found, skipping features module")
    return()
endif()

add_plugin_library(features
    PRIVATE_LIBS
        inferrt_core
        inferrt_model
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        CUDA::cudart
        ${Faiss_LIBS}
)

target_include_directories(${PROJECT_NAME_LOWER}_features
    PRIVATE
        ${Faiss_INCLUDE_DIRS}
)

if(WIN32)
    get_filename_component(_faiss_parent_dir "${Faiss_HOME}" DIRECTORY)
    set(_faiss_runtime_search_dirs
        "${Faiss_BIN_DIR}"
        "${_faiss_parent_dir}/mkl/mkl/bin"
        "${_faiss_parent_dir}/mkl/compiler/bin"
        "$ENV{CONDA_PREFIX}/Library/bin"
    )

    set(_faiss_runtime_dlls)
    foreach(_dll_name faiss.dll libiomp5md.dll mkl_intel_thread.2.dll mkl_core.2.dll)
        string(MAKE_C_IDENTIFIER "${_dll_name}" _dll_var_suffix)
        set(_dll_var "_faiss_runtime_${_dll_var_suffix}")
        find_file(${_dll_var}
            NAMES ${_dll_name}
            HINTS ${_faiss_runtime_search_dirs}
            NO_DEFAULT_PATH
        )
        if(${_dll_var})
            list(APPEND _faiss_runtime_dlls "${${_dll_var}}")
        endif()
    endforeach()

    foreach(_runtime_dll IN LISTS _faiss_runtime_dlls)
        add_custom_command(TARGET ${PROJECT_NAME_LOWER}_features POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_runtime_dll}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME_LOWER}_features>/"
            COMMENT "Copying Faiss runtime dependency ${_runtime_dll}"
        )
    endforeach()
endif()
