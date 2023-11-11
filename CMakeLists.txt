cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(vcpkg
        GIT_REPOSITORY "https://github.com/microsoft/vcpkg/"
        )
FetchContent_MakeAvailable(vcpkg)

set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
set(VCPKG_TRACE_FIND_PACKAGE true)

project(GPGPU_Edge_Detector
        VERSION 0.1
        LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES
        include/gpu_info.h
        include/basic_window.h
        include/SDL_GLDebugMessageCallback.h
        include/main_window.h)

set(SOURCE_FILES
        src/main.cpp
        src/gpu_info.cpp
        src/basic_window.cpp
        src/SDL_GLDebugMessageCallback.cpp
        src/main_window.cpp)

set(CUDA_HEADER_FILES

        )

set(CUDA_SOURCE_FILES

        )

find_package(CUDAToolkit REQUIRED)
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
        ${CUDA_SOURCE_FILES}
        ${CUDA_HEADER_FILES}
        )

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        CUDA::cudart
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