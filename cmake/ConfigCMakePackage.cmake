include(CMakePackageConfigHelpers)

if(NOT DEFINED INFERRT_INSTALL_CMAKEDIR)
    set(INFERRT_INSTALL_CMAKEDIR "lib/cmake/${PROJECT_NAME}" CACHE STRING "InferRT CMake package install directory")
endif()

set(_INFERRT_PACKAGE_COMPONENTS core util ops)
if(${PROJECT_NAME_UPPER}_ENABLE_CUDA)
    list(APPEND _INFERRT_PACKAGE_COMPONENTS cvcuda)
    if(${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
        list(APPEND _INFERRT_PACKAGE_COMPONENTS model engine features)
    endif()
endif()

foreach(_inferrt_component IN LISTS _INFERRT_PACKAGE_COMPONENTS)
    install(EXPORT "${PROJECT_NAME}_${_inferrt_component}Targets"
        FILE "${PROJECT_NAME}_${_inferrt_component}Targets.cmake"
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
    )
endforeach()

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

set(_INFERRT_PACKAGE_CONFIG_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
)

install(FILES ${_INFERRT_PACKAGE_CONFIG_FILES} DESTINATION "${INFERRT_INSTALL_CMAKEDIR}")

unset(_inferrt_component)
unset(_INFERRT_PACKAGE_COMPONENTS)
