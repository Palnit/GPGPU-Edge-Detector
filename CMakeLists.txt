cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(vcpkg
        GIT_REPOSITORY "https://github.com/microsoft/vcpkg/"
)
FetchContent_MakeAvailable(vcpkg)

set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
set(VCPKG_TRACE_FIND_PACKAGE true)


#if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES 86)
#endif ()

project(GPGPU_Edge_Detector
        VERSION 0.1
        LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(HEADER_FILES
        include/general/OpenGL_SDL/basic_window.h
        include/cuda/gpu_info.h
        include/general/OpenGL_SDL/SDL_GLDebugMessageCallback.h
        include/general/main_window.h
        include/general/detector_base.h
        include/general/OpenGL_SDL/file_handling.h
        include/general/OpenGL_SDL/generic_structs.h
        include/general/OpenGL_SDL/vertex_buffer_object.h
        include/general/OpenGL_SDL/vertex_array_object.h
        include/general/OpenGL_SDL/element_buffer_object.h
        include/general/OpenGL_SDL/shader_program.h
        include/cuda/detector_cuda.h
        include/general/imgui_display.h
)

set(SOURCE_FILES
        src/main.cpp
        src/cuda/gpu_info.cpp
        src/general/OpenGL_SDL/basic_window.cpp
        src/general/OpenGL_SDL/SDL_GLDebugMessageCallback.cpp
        src/general/main_window.cpp
        src/general/detector_base.cpp
        src/general/OpenGL_SDL/file_handling.cpp
        src/general/OpenGL_SDL/generic_structs.cpp
        src/general/OpenGL_SDL/vertex_buffer_object.cpp
        src/general/OpenGL_SDL/vertex_array_object.cpp
        src/general/OpenGL_SDL/element_buffer_object.cpp
        src/general/OpenGL_SDL/shader_program.cpp
        src/cuda/detector_cuda.cpp
        src/general/imgui_display.cpp
)

set(CUDA_HEADER_FILES
        include/cuda/cuda_edge_detection.cuh
)

set(CUDA_SOURCE_FILES
        src/cuda/cuda_edge_detection.cu
)

set(RESOURCES
        pictures/img.jpg
        pictures/test.jpg
        pictures/d2.png
        pictures/d2_90.jpg
        pictures/github.PNG

        shaders/default_fragment.frag
        shaders/default_vertex.vert
)

find_package(CUDAToolkit)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(implot CONFIG REQUIRED)


add_library(cuda_comp STATIC
        ${CUDA_HEADER_FILES}
        ${CUDA_SOURCE_FILES}
)
target_compile_features(cuda_comp PRIVATE
        cuda_std_17
)


set_target_properties(cuda_comp
        PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 17 # this one cannot be changed by CMake
        CUDA_SEPARABLE_COMPILATION ON # not needed for this example
)

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

foreach (file ${RESOURCES})
    string(REPLACE "/" "_" n_file ${file})
    ADD_CUSTOM_TARGET(
            copy_${n_file}_file
            ${CMAKE_COMMAND}
            -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -D DESTINATION_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -D file=${file}
            -P copy.cmake
            COMMENT "Running custom target for file: ${file} in project: ${CMAKE_PROJECT_NAME}" VERBATIM
    )
    ADD_DEPENDENCIES(${CMAKE_PROJECT_NAME} copy_${n_file}_file)
endforeach ()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/copy.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR})