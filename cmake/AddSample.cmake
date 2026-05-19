function(add_inferrt_sample SAMPLE_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TARGET_NAME "${PROJECT_NAME_LOWER}_sample_${SAMPLE_NAME}")

    file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
    file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)

    add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            cxxopts::cxxopts
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )

    install(
        TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endfunction()
