if(NOT ${PROJECT_NAME_UPPER}_BUILD_PYTHON OR NOT ${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
    return()
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

set(INFERRT_PYTHON_TEST_DIR "${CMAKE_SOURCE_DIR}/tests/python")

add_test(
    NAME ${PROJECT_NAME_LOWER}_python_api
    COMMAND ${Python_EXECUTABLE} -m pytest "${INFERRT_PYTHON_TEST_DIR}" -v --tb=short
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

set_tests_properties(${PROJECT_NAME_LOWER}_python_api PROPERTIES
    ENVIRONMENT "INFERRT_BUILD_DIR=${CMAKE_BINARY_DIR}"
    LABELS "python;integration"
)

inferrt_apply_test_environment(${PROJECT_NAME_LOWER}_python_api)
