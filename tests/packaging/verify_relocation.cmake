cmake_minimum_required(VERSION 3.25)

foreach(_required INFERRT_SOURCE_DIR INFERRT_BUILD_DIR INFERRT_TEST_ROOT)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} must be provided")
    endif()
endforeach()

set(_consumer_source "${INFERRT_SOURCE_DIR}/tests/packaging/consumer")
set(_install_dir "${INFERRT_TEST_ROOT}/install")
set(_consumer_build "${INFERRT_TEST_ROOT}/consumer")
set(_relocated_dir "${INFERRT_TEST_ROOT}/relocated")
set(_relocated_consumer_build "${INFERRT_TEST_ROOT}/consumer-relocated")
set(_relocated_install_dir "${_relocated_dir}/InferRT")

file(REMOVE_RECURSE "${INFERRT_TEST_ROOT}")
file(MAKE_DIRECTORY "${INFERRT_TEST_ROOT}" "${_install_dir}" "${_consumer_build}" "${_relocated_dir}"
     "${_relocated_consumer_build}")

set(_consumer_dependency_prefix_paths)
if(DEFINED CMAKE_PREFIX_PATH AND NOT "${CMAKE_PREFIX_PATH}" STREQUAL "")
    list(APPEND _consumer_dependency_prefix_paths ${CMAKE_PREFIX_PATH})
endif()
foreach(_dependency_var IN ITEMS
        CUDAToolkit_ROOT
        TRT_ROOT
        TensorRT_ROOT
        ONNXRUNTIME_ROOT
        ONNXRuntime_ROOT
        INFERRT_OPENVINO_ROOT
        OpenVINO_DIR
        Faiss_HOME
        Faiss_ROOT
        MKL_ROOT
        OpenCV_DIR
        OpenCV_HOME
        OpenCV_ROOT)
    if(DEFINED ${_dependency_var} AND NOT "${${_dependency_var}}" STREQUAL "")
        list(APPEND _consumer_dependency_prefix_paths "${${_dependency_var}}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _consumer_dependency_prefix_paths)

function(run_checked)
    execute_process(
        COMMAND ${ARGV}
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR
                "Command failed (${_result}): ${ARGV}\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
    endif()
endfunction()

function(set_consumer_prefix_path prefix)
    set(_consumer_prefix_path "${prefix}")
    list(APPEND _consumer_prefix_path ${_consumer_dependency_prefix_paths})
    list(REMOVE_DUPLICATES _consumer_prefix_path)
    # Keep the prefix list out of execute_process()'s command argument list.
    # CMAKE_PREFIX_PATH uses the platform-native separator in the environment
    # (a colon on Unix, a semicolon on Windows), unlike a CMake list.
    cmake_path(CONVERT "${_consumer_prefix_path}" TO_NATIVE_PATH_LIST
               _consumer_prefix_path)
    set(ENV{CMAKE_PREFIX_PATH} "${_consumer_prefix_path}")
endfunction()

function(run_component_consumer prefix label component)
    set(_component_build "${INFERRT_TEST_ROOT}/consumer-${label}-${component}")
    set_consumer_prefix_path("${prefix}")
    run_checked("${CMAKE_COMMAND}" -S "${INFERRT_SOURCE_DIR}/tests/packaging/component_consumer"
                -B "${_component_build}" "-DINFERRT_COMPONENT=${component}")
    run_checked("${CMAKE_COMMAND}" --build "${_component_build}" --config Release --parallel 2)
    # The install tree intentionally contains InferRT targets only. Third-party
    # DLLs are deployed by the explicit Python packaging command, so component
    # consumers are compile/link checks here; the core consumer below remains a
    # self-contained runtime check.
endfunction()

function(run_plain_features_consumer prefix label)
    set(_consumer_build "${INFERRT_TEST_ROOT}/consumer-${label}-plain-features")
    set_consumer_prefix_path("${prefix}")
    run_checked("${CMAKE_COMMAND}" -S "${INFERRT_SOURCE_DIR}/tests/packaging/component_consumer"
                -B "${_consumer_build}" -DINFERRT_PLAIN_FEATURES=ON)
    run_checked("${CMAKE_COMMAND}" --build "${_consumer_build}" --config Release --parallel 2)
endfunction()

run_checked("${CMAKE_COMMAND}" --install "${INFERRT_BUILD_DIR}" --config Release --prefix "${_install_dir}")

foreach(_legacy_shape_matcher_isa IN ITEMS v0 v1 v2)
    if(EXISTS "${_install_dir}/include/inferrt/features/${_legacy_shape_matcher_isa}")
        message(FATAL_ERROR
                "Installed public headers expose removed shape matcher ISA directory: "
                "${_legacy_shape_matcher_isa}")
    endif()
endforeach()

set_consumer_prefix_path("${_install_dir}")
run_checked("${CMAKE_COMMAND}" -S "${_consumer_source}" -B "${_consumer_build}")
run_checked("${CMAKE_COMMAND}" --build "${_consumer_build}" --config Release --parallel 2)

set(_components core util ops)
if(INFERRT_ENABLE_CUDA)
    list(APPEND _components cvcuda)
    if(INFERRT_BUILD_TENSORRT)
        list(APPEND _components model engine features)
    endif()
endif()
foreach(_component IN LISTS _components)
    run_component_consumer("${_install_dir}" installed "${_component}")
endforeach()
if(INFERRT_ENABLE_CUDA AND INFERRT_BUILD_TENSORRT)
    run_plain_features_consumer("${_install_dir}" installed)
endif()

if(WIN32)
    set(_consumer_exe "${_consumer_build}/Release/inferrt_consumer.exe")
    # Windows searches system directories independently of PATH; keeping the
    # test environment to the relocated package prevents a developer checkout
    # from satisfying the InferRT DLL dependency accidentally.
    set(_runtime_variable "PATH=${_install_dir}/bin")
else()
    set(_consumer_exe "${_consumer_build}/inferrt_consumer")
    set(_runtime_variable "LD_LIBRARY_PATH=${_install_dir}/lib:$ENV{LD_LIBRARY_PATH}")
endif()
if(NOT EXISTS "${_consumer_exe}")
    message(FATAL_ERROR "Consumer executable was not generated: ${_consumer_exe}")
endif()
run_checked("${CMAKE_COMMAND}" -E env "${_runtime_variable}" "${_consumer_exe}")

# Move the installed prefix before the second consumer configure.  This tests
# relocation without making a second copy of any runtime file.
file(RENAME "${_install_dir}" "${_relocated_install_dir}")
set_consumer_prefix_path("${_relocated_install_dir}")
run_checked("${CMAKE_COMMAND}" -S "${_consumer_source}" -B "${_relocated_consumer_build}")
run_checked("${CMAKE_COMMAND}" --build "${_relocated_consumer_build}" --config Release --parallel 2)

if(INFERRT_ENABLE_CUDA AND INFERRT_BUILD_TENSORRT)
    run_plain_features_consumer("${_relocated_install_dir}" relocated)
endif()

if(WIN32)
    set(_relocated_consumer_exe "${_relocated_consumer_build}/Release/inferrt_consumer.exe")
    set(_relocated_runtime_variable "PATH=${_relocated_install_dir}/bin")
else()
    set(_relocated_consumer_exe "${_relocated_consumer_build}/inferrt_consumer")
    set(_relocated_runtime_variable "LD_LIBRARY_PATH=${_relocated_install_dir}/lib:$ENV{LD_LIBRARY_PATH}")
endif()
if(NOT EXISTS "${_relocated_consumer_exe}")
    message(FATAL_ERROR "Relocated consumer executable was not generated: ${_relocated_consumer_exe}")
endif()
run_checked("${CMAKE_COMMAND}" -E env "${_relocated_runtime_variable}" "${_relocated_consumer_exe}")

file(TO_CMAKE_PATH "${INFERRT_SOURCE_DIR}" _source_path)
file(TO_CMAKE_PATH "${INFERRT_BUILD_DIR}" _build_path)
file(GLOB_RECURSE _installed_cmake_files LIST_DIRECTORIES FALSE
     "${_install_dir}/*.cmake" "${_relocated_dir}/*.cmake")
foreach(_cmake_file IN LISTS _installed_cmake_files)
    file(READ "${_cmake_file}" _contents)
    string(FIND "${_contents}" "${_source_path}" _source_index)
    string(FIND "${_contents}" "${_build_path}" _build_index)
    if(NOT _source_index EQUAL -1 OR NOT _build_index EQUAL -1)
        message(FATAL_ERROR "Installed CMake file contains a build/source absolute path: ${_cmake_file}")
    endif()
endforeach()

message(STATUS "InferRT core-only install relocation verification passed")
