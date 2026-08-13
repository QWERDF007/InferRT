add_inferrt_sample(engine_rfdetr
    PRIVATE_LIBS
        inferrt_engine
        inferrt_model
        opencv_imgcodecs
        opencv_imgproc
)

target_include_directories(${PROJECT_NAME_LOWER}_sample_engine_rfdetr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../engine_basic)
