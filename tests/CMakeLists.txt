add_subdirectory(core)
add_subdirectory(cvcuda)
add_subdirectory(model)
add_subdirectory(util)

if(${PROJECT_NAME_UPPER}_BUILD_PYTHON AND ${PROJECT_NAME_UPPER}_BUILD_SAMPLES)
    add_subdirectory(python)
endif()
