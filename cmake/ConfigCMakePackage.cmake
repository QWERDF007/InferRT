include(CMakePackageConfigHelpers)

set(INFERRT_INSTALL_CMAKEDIR "cmake" CACHE STRING
    "InferRT CMake package install directory" FORCE)

install(EXPORT ${PROJECT_NAME}Targets
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)

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

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/ConfigONNXRuntime.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/ConfigOpenVINO.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/ConfigFaiss.cmake"
    DESTINATION "${INFERRT_INSTALL_CMAKEDIR}"
)
