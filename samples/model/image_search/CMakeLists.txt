if(NOT TARGET ${PROJECT_NAME_LOWER}_features)
    message(WARNING "features module not available, skipping sample: image_search")
    return()
endif()

add_inferrt_sample(image_search
    PRIVATE_LIBS
        inferrt_features
        inferrt_model
)
