cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(vcpkg
        GIT_REPOSITORY "https://github.com/microsoft/vcpkg/"
        )
FetchContent_MakeAvailable(vcpkg)

set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
set(VCPKG_TRACE_FIND_PACKAGE true)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

project(GPGPU_Edge_Detector
        VERSION 0.1
        LANGUAGES CUDA CXX)

set(HEADER_FILES
        include/basic_window.h
        include/gpu_info.h
        include/SDL_GLDebugMessageCallback.h
        include/main_window.h)

set(SOURCE_FILES
        src/main.cpp
        src/gpu_info.cpp
        src/basic_window.cpp
        src/SDL_GLDebugMessageCallback.cpp
        src/main_window.cpp)

set(CUDA_HEADER_FILES
        cuda_include/cuda_edge_detection.cuh
        )

set(CUDA_SOURCE_FILES
        cuda_src/cuda_edge_detection.cu)

set(OTHER_FILES
        pictures/img.jpg
        pictures/test.jpg
        )


add_library(cuda_comp STATIC
        ${CUDA_HEADER_FILES}
        ${CUDA_SOURCE_FILES}
        )
target_compile_features(cuda_comp PRIVATE cuda_std_14)

set_target_properties(cuda_comp
        PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        # CUDA_STANDARD 14 # this one cannot be changed by CMake
        CUDA_SEPARABLE_COMPILATION ON # not needed for this example
        )

find_package(CUDAToolkit)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(implot CONFIG REQUIRED)


add_executable(${CMAKE_PROJECT_NAME}
        ${SOURCE_FILES}
        ${HEADER_FILES}
        )

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        cuda_comp
        CUDA::toolkit
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>
        imgui::imgui
        GLEW::GLEW
        glm::glm
        OpenGL::GL
        implot::implot
        )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})