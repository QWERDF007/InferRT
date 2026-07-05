if(NOT TARGET ${PROJECT_NAME_LOWER}_features)
    message(WARNING "features module not available, skipping sample: shape_template_matching")
    return()
endif()

add_inferrt_sample(shape_template_matching
    PRIVATE_LIBS
        inferrt_features
)
