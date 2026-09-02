function(inferrt_apply_test_environment TEST_NAME)
    # Runtime files are explicitly deployed to build/bin by
    # tools/package_runtime_dlls.py. CTest only exposes the build output
    # directories; it never discovers or copies dependency files itself.
    set(_runtime_dirs
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    set(_environment_modifications)
    foreach(_runtime_dir IN LISTS _runtime_dirs)
        if(NOT _runtime_dir)
            continue()
        endif()
        if(WIN32)
            list(APPEND _environment_modifications
                 "PATH=path_list_prepend:${_runtime_dir}")
        elseif(APPLE)
            list(APPEND _environment_modifications
                 "DYLD_LIBRARY_PATH=path_list_prepend:${_runtime_dir}")
        else()
            list(APPEND _environment_modifications
                 "LD_LIBRARY_PATH=path_list_prepend:${_runtime_dir}")
        endif()
    endforeach()
    if(NOT _environment_modifications)
        return()
    endif()
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT_MODIFICATION "${_environment_modifications}")
endfunction()

function(add_inferrt_test TEST_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TARGET_NAME "${PROJECT_NAME_LOWER}_test_${TEST_NAME}")

    file(GLOB SRCS CONFIGURE_DEPENDS *.cpp)
    file(GLOB HEADERS CONFIGURE_DEPENDS *.h *.hpp)

    add_executable(${TARGET_NAME}
        ${SRCS}
        ${HEADERS}
    )

    inferrt_apply_compile_options(${TARGET_NAME})

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    inferrt_apply_test_environment(${TARGET_NAME})
endfunction()
