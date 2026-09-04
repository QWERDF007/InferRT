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

if(${PROJECT_NAME_UPPER}_ENABLE_CUDA AND ${PROJECT_NAME_UPPER}_BUILD_TENSORRT)
    list(APPEND _INFERRT_PACKAGE_CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/FindTensorRT.cmake")
    list(APPEND _INFERRT_PACKAGE_CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")
    # features has a public Faiss target in its exported link interface. Keep
    # the resolver in every package that can export features, even if a
    # future build discovers Faiss through an external CMake package instead
    # of the legacy variables used by the build tree.
    list(APPEND _INFERRT_PACKAGE_CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/ConfigFaiss.cmake")
endif()

if(${PROJECT_NAME_UPPER}_BUILD_ONNX)
    list(APPEND _INFERRT_PACKAGE_CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/ConfigONNXRuntime.cmake")
endif()

if(${PROJECT_NAME_UPPER}_BUILD_OPENVINO)
    list(APPEND _INFERRT_PACKAGE_CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/ConfigOpenVINO.cmake")
endif()

install(FILES ${_INFERRT_PACKAGE_CONFIG_FILES} DESTINATION "${INFERRT_INSTALL_CMAKEDIR}")

unset(_inferrt_component)
unset(_INFERRT_PACKAGE_COMPONENTS)
