cmake_minimum_required(VERSION 3.21)
project(BeeCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(BeeCuda main.cu)

set_target_properties(BeeCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
