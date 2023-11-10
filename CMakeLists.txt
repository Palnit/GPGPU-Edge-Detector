cmake_minimum_required(VERSION 3.22)
project(GPGPU_Edge_Detector CXX CUDA)


set(CMAKE_CUDA_STANDARD 17)

set(HEADER_FILES

        )
set(SOURCE_FILES
        src/main.cpp
        )
set(CUDA_HEADER_FILES

        )
set(CUDA_SOURCE_FILES

        )

find_package(CUDAToolkit REQUIRED)


add_executable(${CMAKE_PROJECT_NAME}
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${CUDA_SOURCE_FILES}
        ${CUDA_HEADER_FILES}
        )

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        CUDA::cudart
        )
