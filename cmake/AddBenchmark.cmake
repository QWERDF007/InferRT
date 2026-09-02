function(add_inferrt_benchmark BENCHMARK_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PRIVATE_LIBS PUBLIC_LIBS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TARGET_NAME "${PROJECT_NAME_LOWER}_benchmark_${BENCHMARK_NAME}")

    file(GLOB SOURCES CONFIGURE_DEPENDS *.cpp)
    file(GLOB HEADERS CONFIGURE_DEPENDS *.h *.hpp)

    add_executable(${TARGET_NAME}
        ${SOURCES}
        ${HEADERS}
    )

    inferrt_apply_compile_options(${TARGET_NAME})

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${ARG_PRIVATE_LIBS}
        PUBLIC
            ${ARG_PUBLIC_LIBS}
    )
endfunction()
