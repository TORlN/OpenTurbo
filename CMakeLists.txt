cmake_minimum_required(VERSION 3.26)

project(OpenTurbo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX})
        set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
    elseif(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

enable_language(CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_openturbo_cuda MODULE
    src/openturbo/_pybind_shim.cpp
    kernels/encoder.cu
    kernels/scan.cu
)

target_include_directories(_openturbo_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(_openturbo_cuda
    PRIVATE
        CUDA::cudart
)

set_target_properties(_openturbo_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS _openturbo_cuda
    LIBRARY DESTINATION openturbo
    RUNTIME DESTINATION openturbo
)