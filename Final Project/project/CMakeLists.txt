cmake_minimum_required (VERSION 3.5)

# set(CMAKE_BUILD_TYPE Debug) # Do not set if profiling using NSight
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_STANDARD 11)

project (ece408project)
include(FindCUDA)
# list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70; -std=c++11; -lineinfo")
# list( APPEND CUDA_NVCC_FLAGS "; -std=c++11; -lineinfo")
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=[sm_86,compute_86]; -std=c++11; -lineinfo")

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

cuda_add_executable(m1 m1.cc)
target_link_libraries(m1 ece408net MiniDNNLib)
# Set compiler flags for debug builds
target_compile_options(m1 PRIVATE "$<$<CONFIG:DEBUG>:-gdwarf-5>")

cuda_add_executable(m2 m2.cc)
target_link_libraries(m2 ece408net MiniDNNLib)

cuda_add_executable(m3 m3.cc)
target_link_libraries(m3 ece408net MiniDNNLib)

cuda_add_executable(final final.cc)
target_link_libraries(final ece408net MiniDNNLib)

