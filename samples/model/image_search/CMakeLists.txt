if(NOT Faiss_LIBS)
    message(WARNING "Faiss not found, skipping sample: image_search")
    return()
endif()

add_inferrt_sample(image_search
    PRIVATE_LIBS
        inferrt_core
        inferrt_model
        inferrt_util
        ${Faiss_LIBS}
)

target_include_directories(${PROJECT_NAME_LOWER}_sample_image_search
    PRIVATE
        ${Faiss_INCLUDE_DIRS}
)
