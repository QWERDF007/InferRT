function(add_inferrt_test TEST_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TARGET_NAME "${PROJECT_NAME_LOWER}_test_${TEST_NAME}")

    file(GLOB SRCS *.cpp)
    file(GLOB HEADERS *.h *.hpp)

    add_executable(${TARGET_NAME}
        ${SRCS}
        ${HEADERS}
    )

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()
