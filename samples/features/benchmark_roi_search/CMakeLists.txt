if(NOT TARGET ${PROJECT_NAME_LOWER}_features)
    message(WARNING "features module not available, skipping sample: benchmark_roi_search")
    return()
endif()

add_inferrt_sample(benchmark_roi_search
    PRIVATE_LIBS
        inferrt_features
        inferrt_model
        Faiss::faiss
        CUDA::cudart
        yaml-cpp::yaml-cpp
)
target_include_directories(${PROJECT_NAME_LOWER}_sample_benchmark_roi_search PRIVATE ${Faiss_INCLUDE_DIRS})
