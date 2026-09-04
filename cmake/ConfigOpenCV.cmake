include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

set(_inferrt_opencv_home_from_default OFF)
set(_inferrt_opencv_default_home)
set(_inferrt_opencv_search_dir)

if(DEFINED OpenCV_DIR AND "${OpenCV_DIR}" MATCHES "-NOTFOUND$")
    unset(OpenCV_DIR CACHE)
    unset(OpenCV_DIR)
endif()

inferrt_dependency_variable_is_explicit(
    OpenCV_DIR INFERRT_DEPENDENCY_OPENCV_DIR _inferrt_opencv_dir_explicit)
if(NOT _inferrt_opencv_dir_explicit)
    unset(OpenCV_DIR CACHE)
    unset(OpenCV_DIR)
endif()

if(_inferrt_opencv_dir_explicit)
    set(_inferrt_opencv_search_dir "${OpenCV_DIR}")
elseif(DEFINED ENV{OpenCV_DIR} AND NOT "$ENV{OpenCV_DIR}" STREQUAL "")
    set(_inferrt_opencv_search_dir "$ENV{OpenCV_DIR}")
    inferrt_dependency_cache_set(
        OpenCV_DIR "${_inferrt_opencv_search_dir}" PATH
        "OpenCV CMake package directory"
        INFERRT_DEPENDENCY_OPENCV_DIR environment)
else()
    inferrt_dependency_resolve_path(
        _inferrt_opencv_home _inferrt_opencv_home_origin opencv
        VARIABLES OpenCV_HOME OpenCV_ROOT
        ENVIRONMENT_VARIABLES OpenCV_HOME OpenCV_ROOT
    )
    if(_inferrt_opencv_home)
        inferrt_dependency_cache_set(
            OpenCV_HOME "${_inferrt_opencv_home}" PATH
            "OpenCV installation directory"
            INFERRT_DEPENDENCY_OPENCV_HOME "${_inferrt_opencv_home_origin}")
        set(_inferrt_opencv_search_dir "${OpenCV_HOME}/lib")
        if(_inferrt_opencv_home_origin STREQUAL "project-default")
            set(_inferrt_opencv_home_from_default ON)
        endif()
    endif()
endif()

if(_inferrt_opencv_search_dir)
    if(_inferrt_opencv_dir_explicit)
        set(OpenCV_DIR "${_inferrt_opencv_search_dir}")
    elseif(NOT DEFINED OpenCV_DIR OR NOT OpenCV_DIR STREQUAL "${_inferrt_opencv_search_dir}")
        inferrt_dependency_cache_set(
            OpenCV_DIR "${_inferrt_opencv_search_dir}" PATH
            "OpenCV CMake package directory"
            INFERRT_DEPENDENCY_OPENCV_DIR
            "${_inferrt_opencv_home_origin}")
    endif()
endif()

if(DEFINED OpenCV_DIR AND NOT OpenCV_DIR STREQUAL "")
    set(OpenCV_LIBRARY_DIR "${OpenCV_DIR}")
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND AND (${PROJECT_NAME_UPPER}_ENABLE_CUDA OR ${PROJECT_NAME_UPPER}_BUILD_SAMPLES OR ${PROJECT_NAME_UPPER}_BUILD_BENCHMARK))
    find_package(OpenCV REQUIRED)
endif()

if(OpenCV_FOUND)
    if(_inferrt_opencv_home_from_default
       AND DEFINED OpenCV_LIB_PATH AND NOT OpenCV_LIB_PATH STREQUAL "")
        foreach(_inferrt_opencv_candidate IN ITEMS
                "${OpenCV_LIB_PATH}/.."
                "${OpenCV_LIB_PATH}/../.."
                "${OpenCV_LIB_PATH}/../../.."
                "${OpenCV_LIB_PATH}/../../../..")
            if(IS_DIRECTORY "${_inferrt_opencv_candidate}/include")
                get_filename_component(_inferrt_opencv_home "${_inferrt_opencv_candidate}" ABSOLUTE)
                unset(OpenCV_HOME CACHE)
                set(OpenCV_HOME "${_inferrt_opencv_home}" CACHE PATH
                    "OpenCV installation root")
                set(_inferrt_opencv_home_from_default OFF)
                break()
            endif()
        endforeach()
    endif()

    if(_inferrt_opencv_home_from_default AND OpenCV_DIR)
        foreach(_inferrt_opencv_candidate IN ITEMS
                "${OpenCV_DIR}"
                "${OpenCV_DIR}/.."
                "${OpenCV_DIR}/../.."
                "${OpenCV_DIR}/../../.."
                "${OpenCV_DIR}/../../../..")
            if(IS_DIRECTORY "${_inferrt_opencv_candidate}/bin")
                get_filename_component(_inferrt_opencv_home "${_inferrt_opencv_candidate}" ABSOLUTE)
                unset(OpenCV_HOME CACHE)
                set(OpenCV_HOME "${_inferrt_opencv_home}" CACHE PATH
                    "OpenCV installation root")
                break()
            endif()
        endforeach()
    endif()

    # Keep the package metadata in sync with the resolved CMake package. The
    # package directory and the runtime directory are not always siblings on
    # Windows (the official archive uses x64/<runtime>/lib and bin).
    if((NOT DEFINED OpenCV_HOME OR OpenCV_HOME STREQUAL "") AND OpenCV_DIR)
        foreach(_inferrt_opencv_candidate IN ITEMS
                "${OpenCV_DIR}"
                "${OpenCV_DIR}/.."
                "${OpenCV_DIR}/../.."
                "${OpenCV_DIR}/../../.."
                "${OpenCV_DIR}/../../../..")
            if(IS_DIRECTORY "${_inferrt_opencv_candidate}/bin")
                get_filename_component(_inferrt_opencv_home "${_inferrt_opencv_candidate}" ABSOLUTE)
                set(OpenCV_HOME "${_inferrt_opencv_home}" CACHE PATH
                    "OpenCV installation root")
                break()
            endif()
        endforeach()
    endif()

    if((NOT DEFINED OpenCV_HOME OR OpenCV_HOME STREQUAL "")
       AND DEFINED OpenCV_LIB_PATH AND NOT OpenCV_LIB_PATH STREQUAL "")
        foreach(_inferrt_opencv_candidate IN ITEMS
                "${OpenCV_LIB_PATH}/.."
                "${OpenCV_LIB_PATH}/../.."
                "${OpenCV_LIB_PATH}/../../.."
                "${OpenCV_LIB_PATH}/../../../..")
            if(IS_DIRECTORY "${_inferrt_opencv_candidate}/include")
                get_filename_component(_inferrt_opencv_home "${_inferrt_opencv_candidate}" ABSOLUTE)
                set(OpenCV_HOME "${_inferrt_opencv_home}" CACHE PATH
                    "OpenCV installation root")
                break()
            endif()
        endforeach()
    endif()

    set(OpenCV_LIBRARY_DIR ${OpenCV_DIR})

    set(_inferrt_opencv_bin_dir)
    if(DEFINED OpenCV_LIB_PATH AND NOT OpenCV_LIB_PATH STREQUAL "")
        get_filename_component(_inferrt_opencv_bin_dir "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
        if(NOT IS_DIRECTORY "${_inferrt_opencv_bin_dir}")
            unset(_inferrt_opencv_bin_dir)
        endif()
    endif()
    if(NOT _inferrt_opencv_bin_dir AND DEFINED OpenCV_HOME
       AND IS_DIRECTORY "${OpenCV_HOME}/bin")
        get_filename_component(_inferrt_opencv_bin_dir "${OpenCV_HOME}/bin" ABSOLUTE)
    endif()
    if(NOT _inferrt_opencv_bin_dir AND OpenCV_DIR)
        foreach(_inferrt_opencv_candidate IN ITEMS
                "${OpenCV_DIR}/bin"
                "${OpenCV_DIR}/../bin"
                "${OpenCV_DIR}/../../bin"
                "${OpenCV_DIR}/../../../bin"
                "${OpenCV_DIR}/../../../../bin")
            if(IS_DIRECTORY "${_inferrt_opencv_candidate}")
                get_filename_component(_inferrt_opencv_bin_dir
                    "${_inferrt_opencv_candidate}" ABSOLUTE)
                break()
            endif()
        endforeach()
    endif()
    if(_inferrt_opencv_bin_dir
       AND (NOT DEFINED OpenCV_BIN_DIR OR OpenCV_BIN_DIR STREQUAL ""))
        set(OpenCV_BIN_DIR "${_inferrt_opencv_bin_dir}" CACHE PATH
            "OpenCV runtime binary directory")
    endif()

    unset(_inferrt_opencv_candidate)
    unset(_inferrt_opencv_home)
    unset(_inferrt_opencv_bin_dir)
endif()

unset(_inferrt_opencv_home_from_default)
unset(_inferrt_opencv_default_home)
unset(_inferrt_opencv_search_dir)
unset(_inferrt_opencv_home)
unset(_inferrt_opencv_home_origin)
unset(_inferrt_opencv_dir_explicit)
