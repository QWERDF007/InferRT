if(NOT TARGET ${PROJECT_NAME_LOWER}_features)
    message(STATUS "features module not available, skipping features tests")
    return()
endif()

add_inferrt_test(features
    PRIVATE_LIBS
        GTest::gtest_main
        GTest::gmock
        inferrt_core
        inferrt_features
        inferrt_model
        ${Faiss_LIBS}
)

target_include_directories(${PROJECT_NAME_LOWER}_test_features
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/features/priv
        ${Faiss_INCLUDE_DIRS}
)
