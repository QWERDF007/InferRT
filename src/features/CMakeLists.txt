if(NOT Faiss_LIBS)
    message(STATUS "Faiss not found, skipping features module")
    return()
endif()

add_plugin_library(features
    PRIVATE_LIBS

    PUBLIC_LIBS
        opencv_core opencv_imgcodecs opencv_imgproc
        CUDA::cudart
        ${Faiss_LIBS}

        inferrt_core
        inferrt_model
)

target_include_directories(${PROJECT_NAME_LOWER}_features
    PRIVATE
        ${Faiss_INCLUDE_DIRS}
)
