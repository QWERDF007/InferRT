cmake_minimum_required(VERSION 3.20)
project(InferRTConsumer LANGUAGES CXX)

find_package(InferRT CONFIG REQUIRED)

add_executable(inferrt_consumer main.cpp)
target_link_libraries(inferrt_consumer PRIVATE InferRT::core)
target_compile_features(inferrt_consumer PRIVATE cxx_std_20)
