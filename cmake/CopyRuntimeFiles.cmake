if(NOT DEFINED INFERRT_COPY_SRC_DIR OR NOT IS_DIRECTORY "${INFERRT_COPY_SRC_DIR}")
    return()
endif()

if(NOT DEFINED INFERRT_COPY_DST_DIR)
    message(FATAL_ERROR "INFERRT_COPY_DST_DIR is required")
endif()

file(MAKE_DIRECTORY "${INFERRT_COPY_DST_DIR}")
file(GLOB _inferrt_runtime_files
    "${INFERRT_COPY_SRC_DIR}/*.dll"
    "${INFERRT_COPY_SRC_DIR}/cache.json")

foreach(_file IN LISTS _inferrt_runtime_files)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_file}" "${INFERRT_COPY_DST_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
