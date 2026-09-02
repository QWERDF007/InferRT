if(NOT DEFINED OpenCV_DIR AND NOT DEFINED OpenCV_HOME)
    if(DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "$ENV{OpenCV_DIR}")
    elseif(DEFINED ENV{OpenCV_HOME})
        set(OpenCV_DIR "$ENV{OpenCV_HOME}/lib")
    endif()
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND AND (${PROJECT_NAME_UPPER}_ENABLE_CUDA OR ${PROJECT_NAME_UPPER}_BUILD_SAMPLES OR ${PROJECT_NAME_UPPER}_BUILD_BENCHMARK))
    find_package(OpenCV REQUIRED)
endif()

if(OpenCV_FOUND)
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
