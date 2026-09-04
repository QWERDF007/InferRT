cmake_minimum_required(VERSION 3.20)

if(INFERRT_PLAIN_FEATURES)
    set(_consumer_component features)
elseif(NOT DEFINED INFERRT_COMPONENT OR INFERRT_COMPONENT STREQUAL "")
    message(FATAL_ERROR "INFERRT_COMPONENT must be provided")
else()
    set(_consumer_component "${INFERRT_COMPONENT}")
endif()

project(InferRTComponentConsumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(INFERRT_PLAIN_FEATURES)
    find_package(InferRT CONFIG REQUIRED COMPONENTS features)
else()
    find_package(InferRT CONFIG REQUIRED COMPONENTS "${_consumer_component}")
endif()

add_executable(inferrt_component_consumer main.cpp)
target_link_libraries(inferrt_component_consumer PRIVATE InferRT::${_consumer_component})
string(MAKE_C_IDENTIFIER "${_consumer_component}" _component_id)
target_compile_definitions(inferrt_component_consumer PRIVATE
    INFERRT_COMPONENT_NAME="${_consumer_component}"
    INFERRT_COMPONENT_${_component_id})
