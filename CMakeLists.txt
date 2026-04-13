cmake_minimum_required(VERSION 3.26)

project(OpenTurbo LANGUAGES CXX)

option(OPENTURBO_BUILD_CUDA "Build the native CUDA extension" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(OPENTURBO_BUILD_CUDA)
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

    find_package(CUDAToolkit REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    add_library(openturbo_cuda_core STATIC
        kernels/encoder.cu
        kernels/scan.cu
    )

    target_include_directories(openturbo_cuda_core
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(openturbo_cuda_core
        PRIVATE
            CUDA::cudart
    )

    set_target_properties(openturbo_cuda_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    add_library(openturbo_c_api SHARED
        kernels/openturbo_c_api.cu
        kernels/openturbo_ggml_adapter.cpp
        kernels/openturbo_llama_bridge.cpp
    )

    target_include_directories(openturbo_c_api
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    )

    target_link_libraries(openturbo_c_api
        PRIVATE
            openturbo_cuda_core
            CUDA::cudart
    )

    target_compile_definitions(openturbo_c_api PRIVATE OPENTURBO_CAPI_EXPORTS)

    pybind11_add_module(_openturbo_cuda MODULE
        src/openturbo/_pybind_shim.cpp
    )

    target_link_libraries(_openturbo_cuda
        PRIVATE
            openturbo_cuda_core
            CUDA::cudart
    )

    install(TARGETS _openturbo_cuda openturbo_c_api
        LIBRARY DESTINATION openturbo
        RUNTIME DESTINATION openturbo
        ARCHIVE DESTINATION openturbo
    )

    install(FILES include/openturbo/c_api.h include/openturbo/ggml_adapter.h include/openturbo/llama_bridge.h DESTINATION include/openturbo)
endif()